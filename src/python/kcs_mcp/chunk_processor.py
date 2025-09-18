"""
Chunk processor for KCS chunk processing with resume capability.

Provides functionality to process chunk files into the database with
proper error handling, retry logic, and resume capability for large-scale
kernel indexing operations.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from .bridge import RUST_BRIDGE_AVAILABLE
from .chunk_loader import ChecksumMismatchError, ChunkLoader
from .database.chunk_queries import ChunkQueries
from .models.chunk_models import ChunkManifest, ChunkMetadata

if RUST_BRIDGE_AVAILABLE:
    try:
        import kcs_python_bridge  # noqa: F401
    except ImportError:
        RUST_BRIDGE_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ChunkProcessingError(Exception):
    """Base exception for chunk processing errors."""

    def __init__(
        self, message: str, chunk_id: str | None = None, retryable: bool = True
    ):
        super().__init__(message)
        self.chunk_id = chunk_id
        self.retryable = retryable


class ChunkProcessingFatalError(ChunkProcessingError):
    """Non-retryable chunk processing error."""

    def __init__(self, message: str, chunk_id: str | None = None):
        super().__init__(message, chunk_id, retryable=False)


class ChunkProcessor:
    """
    Processes chunk files into the database with resume capability.

    Handles loading chunks, verifying checksums, parsing symbols and entry points,
    and storing results in the database with proper transaction boundaries.
    """

    def __init__(
        self,
        database_queries: ChunkQueries,
        chunk_loader: ChunkLoader | None = None,
        max_retry_count: int = 3,
        retry_delay_seconds: float = 1.0,
        verify_checksums: bool = True,
    ):
        """
        Initialize chunk processor.

        Args:
            database_queries: Database queries instance for persistence
            chunk_loader: Optional custom chunk loader (creates default if None)
            max_retry_count: Maximum number of retry attempts per chunk
            retry_delay_seconds: Delay between retry attempts
            verify_checksums: Whether to verify SHA256 checksums
        """
        self.database_queries = database_queries
        self.chunk_loader = chunk_loader or ChunkLoader(
            verify_checksums=verify_checksums
        )
        self.max_retry_count = max_retry_count
        self.retry_delay_seconds = retry_delay_seconds
        self.verify_checksums = verify_checksums

    async def process_chunk(
        self,
        chunk_metadata: ChunkMetadata,
        manifest_version: str,
        base_path: Path | str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Process a single chunk into the database.

        Args:
            chunk_metadata: Metadata for the chunk to process
            manifest_version: Version of manifest this chunk belongs to
            base_path: Base directory path for chunk files
            force: Whether to force reprocessing if already completed

        Returns:
            Processing result with status and metadata

        Raises:
            ChunkProcessingError: If processing fails after retries
            ChunkProcessingFatalError: If error is non-retryable
        """
        chunk_id = chunk_metadata.id

        logger.info(
            "Starting chunk processing",
            chunk_id=chunk_id,
            manifest_version=manifest_version,
            force=force,
        )

        # Check existing status
        existing_status = await self.database_queries.get_chunk_status(chunk_id)

        if existing_status and not force:
            current_status = existing_status["status"]
            if current_status == "completed":
                logger.info(
                    "Chunk already completed, skipping",
                    chunk_id=chunk_id,
                    completed_at=existing_status.get("completed_at"),
                )
                return {
                    "chunk_id": chunk_id,
                    "status": "completed",
                    "message": "Chunk already processed successfully",
                    "symbols_processed": existing_status.get("symbols_processed", 0),
                    "checksum_verified": existing_status.get(
                        "checksum_verified", False
                    ),
                }

            if current_status == "processing":
                raise ChunkProcessingError(
                    f"Chunk {chunk_id} is already being processed", chunk_id
                )

        # Create or update status record to "processing"
        await self._update_chunk_status(
            chunk_id,
            manifest_version,
            status="processing",
            started_at=datetime.now(),
            error_message=None,
        )

        try:
            # Load and validate chunk
            chunk_data = await self.chunk_loader.load_chunk(
                chunk_metadata, base_path, verify_checksum=self.verify_checksums
            )

            # Process chunk data into database
            processing_result = await self._process_chunk_data(
                chunk_id, chunk_data, chunk_metadata
            )

            # Mark as completed
            await self._update_chunk_status(
                chunk_id,
                manifest_version,
                status="completed",
                completed_at=datetime.now(),
                symbols_processed=processing_result["symbols_processed"],
                checksum_verified=self.verify_checksums,
            )

            logger.info(
                "Chunk processing completed successfully",
                chunk_id=chunk_id,
                symbols_processed=processing_result["symbols_processed"],
                entry_points_processed=processing_result["entry_points_processed"],
            )

            return {
                "chunk_id": chunk_id,
                "status": "completed",
                "message": "Chunk processed successfully",
                "symbols_processed": processing_result["symbols_processed"],
                "entry_points_processed": processing_result["entry_points_processed"],
                "checksum_verified": self.verify_checksums,
            }

        except (ChecksumMismatchError, ChunkProcessingFatalError) as e:
            # Non-retryable errors
            error_msg = str(e)
            logger.error(
                "Chunk processing failed with non-retryable error",
                chunk_id=chunk_id,
                error=error_msg,
            )

            await self._update_chunk_status(
                chunk_id,
                manifest_version,
                status="failed",
                completed_at=datetime.now(),
                error_message=error_msg,
            )

            raise ChunkProcessingFatalError(error_msg, chunk_id) from e

        except Exception as e:
            # Potentially retryable error
            error_msg = str(e)
            logger.error(
                "Chunk processing failed with retryable error",
                chunk_id=chunk_id,
                error=error_msg,
            )

            await self._update_chunk_status(
                chunk_id,
                manifest_version,
                status="failed",
                completed_at=datetime.now(),
                error_message=error_msg,
                increment_retry=True,
            )

            raise ChunkProcessingError(error_msg, chunk_id) from e

    async def process_chunks_batch(
        self,
        chunks: list[ChunkMetadata],
        manifest_version: str,
        base_path: Path | str | None = None,
        max_parallelism: int = 4,
        force: bool = False,
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        """
        Process multiple chunks in parallel with resume capability.

        Args:
            chunks: List of chunk metadata to process
            manifest_version: Version of manifest chunks belong to
            base_path: Base directory path for chunk files
            max_parallelism: Maximum number of chunks to process concurrently
            force: Whether to force reprocessing of completed chunks
            resume_from: Chunk ID to resume from (skips chunks before this one)

        Returns:
            Batch processing result with overall status and per-chunk results
        """
        logger.info(
            "Starting batch chunk processing",
            total_chunks=len(chunks),
            max_parallelism=max_parallelism,
            resume_from=resume_from,
            force=force,
        )

        # Filter chunks for processing based on resume_from
        chunks_to_process = self._filter_chunks_for_resume(chunks, resume_from)

        if not chunks_to_process:
            logger.info("No chunks to process after filtering")
            return {
                "total_chunks": len(chunks),
                "processed_chunks": 0,
                "successful_chunks": 0,
                "failed_chunks": 0,
                "skipped_chunks": len(chunks),
                "results": {},
            }

        logger.info(
            "Processing chunks after resume filtering",
            chunks_to_process=len(chunks_to_process),
            skipped_chunks=len(chunks) - len(chunks_to_process),
        )

        # Create semaphore to limit parallelism
        semaphore = asyncio.Semaphore(max_parallelism)
        results: dict[str, dict[str, Any]] = {}

        async def process_single_chunk(chunk_metadata: ChunkMetadata) -> None:
            async with semaphore:
                chunk_id = chunk_metadata.id
                try:
                    result = await self._process_chunk_with_retry(
                        chunk_metadata, manifest_version, base_path, force
                    )
                    results[chunk_id] = {"status": "success", "result": result}
                    logger.debug("Chunk processed successfully", chunk_id=chunk_id)

                except ChunkProcessingFatalError as e:
                    results[chunk_id] = {
                        "status": "failed_fatal",
                        "error": str(e),
                        "retryable": False,
                    }
                    logger.error(
                        "Chunk failed with fatal error", chunk_id=chunk_id, error=str(e)
                    )

                except ChunkProcessingError as e:
                    results[chunk_id] = {
                        "status": "failed_retryable",
                        "error": str(e),
                        "retryable": True,
                    }
                    logger.error(
                        "Chunk failed after retries", chunk_id=chunk_id, error=str(e)
                    )

                except Exception as e:
                    # Unexpected error - treat as retryable
                    results[chunk_id] = {
                        "status": "failed_unexpected",
                        "error": str(e),
                        "retryable": True,
                    }
                    logger.error(
                        "Chunk failed with unexpected error",
                        chunk_id=chunk_id,
                        error=str(e),
                    )

        # Execute all chunk processing tasks concurrently
        tasks = [process_single_chunk(chunk) for chunk in chunks_to_process]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Compute batch statistics
        successful_count = sum(
            1 for result in results.values() if result["status"] == "success"
        )
        failed_count = len(results) - successful_count
        skipped_count = len(chunks) - len(chunks_to_process)

        batch_result = {
            "total_chunks": len(chunks),
            "processed_chunks": len(chunks_to_process),
            "successful_chunks": successful_count,
            "failed_chunks": failed_count,
            "skipped_chunks": skipped_count,
            "results": results,
        }

        logger.info(
            "Batch chunk processing completed",
            total_chunks=len(chunks),
            successful_chunks=successful_count,
            failed_chunks=failed_count,
            skipped_chunks=skipped_count,
        )

        return batch_result

    async def resume_processing(
        self,
        manifest: ChunkManifest,
        base_path: Path | str | None = None,
        max_parallelism: int = 4,
        status_filter: str = "failed",
    ) -> dict[str, Any]:
        """
        Resume processing of failed or pending chunks from a manifest.

        Args:
            manifest: Chunk manifest to resume processing for
            base_path: Base directory path for chunk files
            max_parallelism: Maximum number of chunks to process concurrently
            status_filter: Status of chunks to resume ("failed", "pending", or "all")

        Returns:
            Resume processing result
        """
        logger.info(
            "Starting resume processing",
            manifest_version=manifest.version,
            total_chunks=manifest.total_chunks,
            status_filter=status_filter,
        )

        # Get chunks that need processing based on status filter
        if status_filter == "all":
            chunks_needing_processing = await self._get_all_unfinished_chunks(manifest)
        else:
            chunks_needing_processing = await self._get_chunks_by_status(
                manifest, status_filter
            )

        if not chunks_needing_processing:
            logger.info("No chunks need resume processing")
            return {
                "total_chunks": manifest.total_chunks,
                "chunks_needing_processing": 0,
                "successful_chunks": 0,
                "failed_chunks": 0,
                "results": {},
            }

        logger.info(
            "Found chunks needing resume processing",
            chunks_needing_processing=len(chunks_needing_processing),
        )

        # Process the chunks that need work
        return await self.process_chunks_batch(
            chunks_needing_processing,
            manifest.version,
            base_path,
            max_parallelism,
            force=False,  # Don't force reprocess completed chunks in resume
        )

    async def _process_chunk_with_retry(
        self,
        chunk_metadata: ChunkMetadata,
        manifest_version: str,
        base_path: Path | str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Process a chunk with retry logic."""
        chunk_id = chunk_metadata.id
        retry_count = 0

        while retry_count <= self.max_retry_count:
            try:
                return await self.process_chunk(
                    chunk_metadata, manifest_version, base_path, force
                )

            except ChunkProcessingFatalError:
                # Don't retry fatal errors
                raise

            except ChunkProcessingError as e:
                retry_count += 1
                if retry_count > self.max_retry_count:
                    logger.error(
                        "Chunk processing failed after max retries",
                        chunk_id=chunk_id,
                        retry_count=retry_count,
                        error=str(e),
                    )
                    raise

                logger.warning(
                    "Chunk processing failed, retrying",
                    chunk_id=chunk_id,
                    retry_count=retry_count,
                    max_retries=self.max_retry_count,
                    error=str(e),
                )

                # Exponential backoff
                delay = self.retry_delay_seconds * (2 ** (retry_count - 1))
                await asyncio.sleep(delay)

        # Should never reach here due to logic above
        raise ChunkProcessingError(f"Unexpected retry logic error for chunk {chunk_id}")

    async def _process_chunk_data(
        self, chunk_id: str, chunk_data: dict[str, Any], chunk_metadata: ChunkMetadata
    ) -> dict[str, Any]:
        """
        Process chunk data into database tables.

        This is where the actual database operations would happen to store
        symbols, entry points, files, etc. For now, we simulate processing.
        """
        # TODO: Implement actual database storage of symbols, entry points, etc.
        # This would involve:
        # 1. Parsing symbols from chunk_data["symbols"]
        # 2. Parsing entry_points from chunk_data["entry_points"]
        # 3. Parsing files from chunk_data["files"]
        # 4. Storing them in the appropriate database tables
        # 5. Creating relationships and updating indexes

        # For now, simulate processing by counting items
        symbols_count = len(chunk_data.get("symbols", []))
        entry_points_count = len(chunk_data.get("entry_points", []))
        files_count = len(chunk_data.get("files", []))

        logger.debug(
            "Processing chunk data",
            chunk_id=chunk_id,
            symbols_count=symbols_count,
            entry_points_count=entry_points_count,
            files_count=files_count,
        )

        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Validate chunk data structure
        required_fields = ["chunk_id", "manifest_version", "subsystem"]
        for field in required_fields:
            if field not in chunk_data:
                raise ChunkProcessingFatalError(
                    f"Missing required field '{field}' in chunk data", chunk_id
                )

        # Verify chunk_id matches
        if chunk_data["chunk_id"] != chunk_id:
            raise ChunkProcessingFatalError(
                f"Chunk ID mismatch: expected {chunk_id}, got {chunk_data['chunk_id']}",
                chunk_id,
            )

        return {
            "symbols_processed": symbols_count,
            "entry_points_processed": entry_points_count,
            "files_processed": files_count,
        }

    async def _update_chunk_status(
        self,
        chunk_id: str,
        manifest_version: str,
        status: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_message: str | None = None,
        symbols_processed: int | None = None,
        checksum_verified: bool | None = None,
        increment_retry: bool = False,
    ) -> dict[str, Any]:
        """Update or create chunk processing status."""
        # First try to update existing record
        updated_status = await self.database_queries.update_chunk_status(
            chunk_id=chunk_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            error_message=error_message,
            symbols_processed=symbols_processed,
            checksum_verified=checksum_verified,
            increment_retry=increment_retry,
        )

        if updated_status:
            return updated_status

        # If no existing record, create new one
        logger.debug("Creating new chunk status record", chunk_id=chunk_id)
        return await self.database_queries.create_chunk_status(
            chunk_id=chunk_id,
            manifest_version=manifest_version,
            status=status or "pending",
        )

    def _filter_chunks_for_resume(
        self, chunks: list[ChunkMetadata], resume_from: str | None
    ) -> list[ChunkMetadata]:
        """Filter chunks based on resume_from parameter."""
        if not resume_from:
            return chunks

        # Find the index of the resume_from chunk
        resume_index = None
        for i, chunk in enumerate(chunks):
            if chunk.id == resume_from:
                resume_index = i
                break

        if resume_index is None:
            logger.warning(
                "Resume chunk ID not found in manifest, processing all chunks",
                resume_from=resume_from,
            )
            return chunks

        logger.info(
            "Resuming from chunk",
            resume_from=resume_from,
            resume_index=resume_index,
            chunks_skipped=resume_index,
        )

        return chunks[resume_index:]

    async def _get_chunks_by_status(
        self, manifest: ChunkManifest, status: str
    ) -> list[ChunkMetadata]:
        """Get chunks from manifest that have a specific processing status."""
        # Get chunk status records from database
        status_records = await self.database_queries.get_chunks_by_status(
            status=status, manifest_version=manifest.version
        )

        # Create a set of chunk IDs with the desired status
        chunk_ids_with_status = {record["chunk_id"] for record in status_records}

        # Return chunks from manifest that match the status
        return [chunk for chunk in manifest.chunks if chunk.id in chunk_ids_with_status]

    async def _get_all_unfinished_chunks(
        self, manifest: ChunkManifest
    ) -> list[ChunkMetadata]:
        """Get all chunks that are not completed."""
        # Get all status records for this manifest
        all_records = await self.database_queries.get_chunks_by_status(
            manifest_version=manifest.version
        )

        # Create a set of chunk IDs that are completed
        completed_chunk_ids = {
            record["chunk_id"]
            for record in all_records
            if record["status"] == "completed"
        }

        # Return chunks that are not completed
        return [
            chunk for chunk in manifest.chunks if chunk.id not in completed_chunk_ids
        ]


# Convenience functions for common operations


async def process_single_chunk(
    chunk_metadata: ChunkMetadata,
    manifest_version: str,
    database_queries: ChunkQueries,
    base_path: Path | str | None = None,
    force: bool = False,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """
    Process a single chunk with default settings.

    Args:
        chunk_metadata: Metadata for the chunk to process
        manifest_version: Version of manifest this chunk belongs to
        database_queries: Database queries instance
        base_path: Base directory path for chunk files
        force: Whether to force reprocessing if already completed
        verify_checksums: Whether to verify SHA256 checksums

    Returns:
        Processing result
    """
    processor = ChunkProcessor(
        database_queries=database_queries, verify_checksums=verify_checksums
    )
    return await processor.process_chunk(
        chunk_metadata, manifest_version, base_path, force
    )


async def resume_failed_chunks(
    manifest: ChunkManifest,
    database_queries: ChunkQueries,
    base_path: Path | str | None = None,
    max_parallelism: int = 4,
) -> dict[str, Any]:
    """
    Resume processing of failed chunks from a manifest.

    Args:
        manifest: Chunk manifest to resume processing for
        database_queries: Database queries instance
        base_path: Base directory path for chunk files
        max_parallelism: Maximum number of chunks to process concurrently

    Returns:
        Resume processing result
    """
    processor = ChunkProcessor(database_queries=database_queries)
    return await processor.resume_processing(
        manifest, base_path, max_parallelism, status_filter="failed"
    )
