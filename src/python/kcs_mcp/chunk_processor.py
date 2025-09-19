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

import aiofiles
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
        default_max_parallelism: int = 4,
        adaptive_parallelism: bool = True,
        max_memory_mb: int = 2048,
        checksum_verification_policy: str = "strict",
        pre_verify_checksums: bool = True,
    ):
        """
        Initialize chunk processor.

        Args:
            database_queries: Database queries instance for persistence
            chunk_loader: Optional custom chunk loader (creates default if None)
            max_retry_count: Maximum number of retry attempts per chunk
            retry_delay_seconds: Delay between retry attempts
            verify_checksums: Whether to verify SHA256 checksums
            default_max_parallelism: Default parallelism when not specified
            adaptive_parallelism: Whether to adapt parallelism based on system resources
            max_memory_mb: Maximum memory usage hint for adaptive parallelism
            checksum_verification_policy: Verification policy ("strict", "warn", "skip")
            pre_verify_checksums: Whether to verify checksums before processing starts
        """
        self.database_queries = database_queries
        self.chunk_loader = chunk_loader or ChunkLoader(
            verify_checksums=verify_checksums
        )
        self.max_retry_count = max_retry_count
        self.retry_delay_seconds = retry_delay_seconds
        self.verify_checksums = verify_checksums
        self.default_max_parallelism = default_max_parallelism
        self.adaptive_parallelism = adaptive_parallelism
        self.max_memory_mb = max_memory_mb
        self.checksum_verification_policy = checksum_verification_policy
        self.pre_verify_checksums = pre_verify_checksums

        # Validate checksum verification policy
        valid_policies = ["strict", "warn", "skip"]
        if checksum_verification_policy not in valid_policies:
            raise ValueError(
                f"Invalid checksum verification policy: {checksum_verification_policy}. "
                f"Must be one of: {valid_policies}"
            )

    def _calculate_optimal_parallelism(
        self,
        chunks: list[ChunkMetadata],
        requested_parallelism: int | None = None,
    ) -> int:
        """
        Calculate optimal parallelism based on system resources and chunk characteristics.

        Args:
            chunks: List of chunks to be processed
            requested_parallelism: User-requested parallelism (overrides adaptive)

        Returns:
            Optimal parallelism level
        """
        if requested_parallelism is not None:
            logger.debug(
                "Using requested parallelism",
                requested_parallelism=requested_parallelism,
                adaptive_disabled=True,
            )
            return requested_parallelism

        if not self.adaptive_parallelism:
            logger.debug(
                "Using default parallelism",
                default_parallelism=self.default_max_parallelism,
                adaptive_disabled=True,
            )
            return self.default_max_parallelism

        # Calculate adaptive parallelism based on chunk size and available memory
        total_chunks = len(chunks)
        if total_chunks == 0:
            return self.default_max_parallelism

        # Estimate memory per chunk (conservative estimate)
        avg_chunk_size_mb = sum(chunk.size_bytes for chunk in chunks) / (
            len(chunks) * 1024 * 1024
        )
        estimated_memory_per_chunk = max(
            avg_chunk_size_mb * 2, 32
        )  # At least 32MB per chunk

        # Calculate max parallelism based on memory constraints
        memory_based_parallelism = max(
            1, int(self.max_memory_mb / estimated_memory_per_chunk)
        )

        # Don't exceed the number of chunks
        chunk_based_parallelism = min(total_chunks, self.default_max_parallelism * 2)

        # Use the minimum of all constraints
        optimal_parallelism = min(
            memory_based_parallelism,
            chunk_based_parallelism,
            self.default_max_parallelism * 3,  # Cap at 3x default
        )

        # Ensure at least 1
        optimal_parallelism = max(1, optimal_parallelism)

        logger.info(
            "Calculated adaptive parallelism",
            optimal_parallelism=optimal_parallelism,
            avg_chunk_size_mb=round(avg_chunk_size_mb, 2),
            estimated_memory_per_chunk_mb=round(estimated_memory_per_chunk, 2),
            memory_based_limit=memory_based_parallelism,
            chunk_based_limit=chunk_based_parallelism,
            total_chunks=total_chunks,
            max_memory_mb=self.max_memory_mb,
        )

        return optimal_parallelism

    async def _pre_verify_chunk_checksums(
        self,
        chunks: list[ChunkMetadata],
        base_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Pre-verify checksums for all chunks before processing begins.

        This allows fast failure detection and provides early feedback
        about data integrity issues before resource-intensive processing.

        Args:
            chunks: List of chunks to verify
            base_path: Base path for chunk files

        Returns:
            Verification results with details per chunk
        """
        if not self.pre_verify_checksums or self.checksum_verification_policy == "skip":
            logger.debug("Checksum pre-verification disabled, skipping")
            return {
                "verified": True,
                "total_chunks": len(chunks),
                "verified_chunks": 0,
                "failed_chunks": 0,
                "skipped_chunks": len(chunks),
                "policy": self.checksum_verification_policy,
                "errors": {},
            }

        logger.info(
            "Starting pre-verification of chunk checksums",
            total_chunks=len(chunks),
            policy=self.checksum_verification_policy,
            verify_checksums=self.verify_checksums,
        )

        verification_results: dict[str, Any] = {
            "verified": True,
            "total_chunks": len(chunks),
            "verified_chunks": 0,
            "failed_chunks": 0,
            "skipped_chunks": 0,
            "policy": self.checksum_verification_policy,
            "errors": {},
        }

        # Create a temporary chunk loader for verification only
        verification_loader = ChunkLoader(
            verify_checksums=True,  # Always verify for pre-verification
            database=None,  # No database logging for pre-verification
        )

        for chunk in chunks:
            try:
                # Resolve chunk file path
                if base_path:
                    chunk_path = Path(base_path) / chunk.file
                else:
                    chunk_path = Path(chunk.file)

                # Quick file existence check
                if not chunk_path.exists():
                    verification_results["errors"][chunk.id] = (
                        f"File not found: {chunk_path}"
                    )
                    verification_results["failed_chunks"] += 1
                    continue

                # Verify file size matches
                file_size = chunk_path.stat().st_size
                if file_size != chunk.size_bytes:
                    error_msg = (
                        f"Size mismatch: expected {chunk.size_bytes}, got {file_size}"
                    )
                    verification_results["errors"][chunk.id] = error_msg
                    verification_results["failed_chunks"] += 1
                    continue

                # Verify checksum by reading content
                async with aiofiles.open(chunk_path, encoding="utf-8") as f:
                    content = await f.read()

                await verification_loader._verify_chunk_checksum(
                    content, chunk.checksum_sha256, chunk.id
                )

                verification_results["verified_chunks"] += 1

                logger.debug(
                    "Chunk checksum verified successfully",
                    chunk_id=chunk.id,
                    file_size=file_size,
                    checksum=chunk.checksum_sha256[:8] + "...",
                )

            except Exception as e:
                error_msg = str(e)
                verification_results["errors"][chunk.id] = error_msg
                verification_results["failed_chunks"] += 1

                if self.checksum_verification_policy == "strict":
                    logger.error(
                        "Checksum pre-verification failed (strict mode)",
                        chunk_id=chunk.id,
                        error=error_msg,
                    )
                elif self.checksum_verification_policy == "warn":
                    logger.warning(
                        "Checksum pre-verification failed (warn mode)",
                        chunk_id=chunk.id,
                        error=error_msg,
                    )

        # Determine overall verification status
        if verification_results["failed_chunks"] > 0:
            verification_results["verified"] = False

        logger.info(
            "Checksum pre-verification completed",
            total_chunks=verification_results["total_chunks"],
            verified_chunks=verification_results["verified_chunks"],
            failed_chunks=verification_results["failed_chunks"],
            success_rate=round(
                (verification_results["verified_chunks"] / len(chunks)) * 100, 1
            )
            if chunks
            else 0,
            policy=self.checksum_verification_policy,
        )

        return verification_results

    async def _process_chunk_with_transaction(
        self,
        chunk_id: str,
        chunk_metadata: ChunkMetadata,
        manifest_version: str,
        base_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Process chunk data within a database transaction for atomic operations.

        This ensures that all database operations for a chunk (data insertion,
        status updates) either all succeed or all fail together.

        Args:
            chunk_id: Chunk identifier
            chunk_metadata: Chunk metadata
            manifest_version: Manifest version
            base_path: Base path for chunk files

        Returns:
            Processing result with counts

        Raises:
            ChunkProcessingError: If processing fails
            ChunkProcessingFatalError: If error is non-retryable
        """
        # Load chunk data outside of transaction (file I/O)
        logger.debug(
            "Loading chunk data outside transaction",
            chunk_id=chunk_id,
            base_path=str(base_path) if base_path else None,
            verify_checksums=self.verify_checksums,
        )

        chunk_data = await self.chunk_loader.load_chunk(
            chunk_metadata, base_path, verify_checksum=self.verify_checksums
        )

        logger.debug(
            "Chunk data loaded successfully",
            chunk_id=chunk_id,
            symbols_count=len(chunk_data.get("symbols", [])),
            entry_points_count=len(chunk_data.get("entry_points", [])),
            files_count=len(chunk_data.get("files", [])),
        )

        # Process chunk within database transaction
        async with self.database_queries.database.acquire() as conn:
            async with conn.transaction():
                logger.debug(
                    "Starting database transaction for chunk processing",
                    chunk_id=chunk_id,
                    manifest_version=manifest_version,
                )

                try:
                    # Process chunk data into database within transaction
                    processing_result = await self._process_chunk_data_transactional(
                        conn, chunk_id, chunk_data, chunk_metadata
                    )

                    # Update chunk status to completed within same transaction
                    await self._update_chunk_status_transactional(
                        conn,
                        chunk_id,
                        manifest_version,
                        status="completed",
                        completed_at=datetime.now(),
                        symbols_processed=processing_result["symbols_processed"],
                        checksum_verified=self.verify_checksums,
                    )

                    logger.debug(
                        "Database transaction completed successfully",
                        chunk_id=chunk_id,
                        symbols_processed=processing_result["symbols_processed"],
                        entry_points_processed=processing_result[
                            "entry_points_processed"
                        ],
                    )

                    return processing_result

                except Exception as e:
                    logger.error(
                        "Database transaction failed, rolling back",
                        chunk_id=chunk_id,
                        error=str(e),
                    )
                    # Transaction will automatically rollback on exception
                    raise

    def _get_semaphore_stats(self, semaphore: asyncio.Semaphore) -> dict[str, int]:
        """Get current semaphore usage statistics."""
        # Note: These are internal attributes and may not be available in all Python versions
        try:
            total_permits = getattr(semaphore, "_value", 0) + len(
                getattr(semaphore, "_waiters", [])
            )
            available_permits = getattr(semaphore, "_value", 0)
            active_tasks = total_permits - available_permits
            waiting_tasks = len(getattr(semaphore, "_waiters", []))

            return {
                "total_permits": total_permits,
                "available_permits": available_permits,
                "active_tasks": active_tasks,
                "waiting_tasks": waiting_tasks,
            }
        except (AttributeError, TypeError):
            # Fallback if semaphore internals are not accessible
            return {
                "total_permits": -1,
                "available_permits": -1,
                "active_tasks": -1,
                "waiting_tasks": -1,
            }

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
            "Starting chunk processing with transaction boundaries",
            chunk_id=chunk_id,
            manifest_version=manifest_version,
            force=force,
            chunk_size_bytes=chunk_metadata.size_bytes,
            chunk_size_mb=round(chunk_metadata.size_bytes / (1024 * 1024), 2),
            chunk_subsystem=getattr(chunk_metadata, "subsystem", "unknown"),
            chunk_file=chunk_metadata.file,
            transaction_enabled=True,
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
        logger.debug(
            "Updating chunk status to processing",
            chunk_id=chunk_id,
            manifest_version=manifest_version,
        )

        await self._update_chunk_status(
            chunk_id,
            manifest_version,
            status="processing",
            started_at=datetime.now(),
            error_message=None,
        )

        try:
            # Execute chunk processing within a database transaction
            result = await self._process_chunk_with_transaction(
                chunk_id, chunk_metadata, manifest_version, base_path
            )

            logger.info(
                "Chunk processing completed successfully",
                chunk_id=chunk_id,
                symbols_processed=result["symbols_processed"],
                entry_points_processed=result["entry_points_processed"],
            )

            return {
                "chunk_id": chunk_id,
                "status": "completed",
                "message": "Chunk processed successfully",
                "symbols_processed": result["symbols_processed"],
                "entry_points_processed": result["entry_points_processed"],
                "checksum_verified": self.verify_checksums,
            }

        except (ChecksumMismatchError, ChunkProcessingFatalError) as e:
            # Non-retryable errors - update status outside transaction since processing failed
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
            # Potentially retryable error - update status outside transaction since processing failed
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
        max_parallelism: int | None = None,
        force: bool = False,
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        """
        Process multiple chunks in parallel with resume capability.

        Args:
            chunks: List of chunk metadata to process
            manifest_version: Version of manifest chunks belong to
            base_path: Base directory path for chunk files
            max_parallelism: Maximum number of chunks to process concurrently (None for adaptive)
            force: Whether to force reprocessing of completed chunks
            resume_from: Chunk ID to resume from (skips chunks before this one)

        Returns:
            Batch processing result with overall status and per-chunk results
        """
        import time

        start_time = time.time()
        processed_count = 0

        logger.info(
            "Starting batch chunk processing",
            total_chunks=len(chunks),
            max_parallelism=max_parallelism,
            resume_from=resume_from,
            force=force,
            manifest_version=manifest_version,
            batch_start_time=start_time,
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

        # Pre-verify checksums before processing begins
        verification_result = await self._pre_verify_chunk_checksums(
            chunks_to_process, base_path
        )

        # Handle verification failures based on policy
        if not verification_result["verified"]:
            if self.checksum_verification_policy == "strict":
                # In strict mode, fail fast if any checksums are invalid
                error_summary = f"Pre-verification failed for {verification_result['failed_chunks']} chunks"
                if verification_result["errors"]:
                    sample_errors = list(verification_result["errors"].items())[:3]
                    error_summary += f". Sample errors: {sample_errors}"

                logger.error(
                    "Aborting batch processing due to checksum verification failures",
                    failed_chunks=verification_result["failed_chunks"],
                    total_chunks=len(chunks_to_process),
                    policy="strict",
                    errors=verification_result["errors"],
                )

                raise ChunkProcessingFatalError(
                    f"Checksum verification failed in strict mode: {error_summary}"
                )
            elif self.checksum_verification_policy == "warn":
                # In warn mode, continue but log warnings
                logger.warning(
                    "Continuing batch processing despite checksum verification failures",
                    failed_chunks=verification_result["failed_chunks"],
                    total_chunks=len(chunks_to_process),
                    policy="warn",
                )
            # In skip mode, verification is already disabled, so we continue

        # Calculate optimal parallelism
        optimal_parallelism = self._calculate_optimal_parallelism(
            chunks_to_process, max_parallelism
        )

        logger.info(
            "Processing chunks after resume filtering",
            chunks_to_process=len(chunks_to_process),
            skipped_chunks=len(chunks) - len(chunks_to_process),
            estimated_total_size_mb=sum(chunk.size_bytes for chunk in chunks_to_process)
            / (1024 * 1024),
            parallelism_level=optimal_parallelism,
            adaptive_parallelism_enabled=self.adaptive_parallelism,
        )

        # Create semaphore to limit parallelism with optimal value
        semaphore = asyncio.Semaphore(optimal_parallelism)
        results: dict[str, dict[str, Any]] = {}

        async def process_single_chunk(chunk_metadata: ChunkMetadata) -> None:
            nonlocal processed_count
            chunk_id = chunk_metadata.id
            semaphore_wait_start = time.time()

            async with semaphore:
                # Log semaphore acquisition if it took significant time
                semaphore_wait_time = time.time() - semaphore_wait_start
                if semaphore_wait_time > 5.0:  # Log if waited more than 5 seconds
                    logger.debug(
                        "Semaphore acquisition took significant time",
                        chunk_id=chunk_id,
                        wait_time_sec=round(semaphore_wait_time, 2),
                        parallelism_level=optimal_parallelism,
                    )

                chunk_start_time = time.time()

                try:
                    result = await self._process_chunk_with_retry(
                        chunk_metadata, manifest_version, base_path, force
                    )
                    results[chunk_id] = {"status": "success", "result": result}

                    processed_count += 1
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunks_per_second = (
                        processed_count / elapsed_time if elapsed_time > 0 else 0
                    )
                    eta_seconds = (
                        ((len(chunks_to_process) - processed_count) / chunks_per_second)
                        if chunks_per_second > 0
                        else 0
                    )

                    logger.info(
                        "Chunk processed successfully",
                        chunk_id=chunk_id,
                        progress_completed=processed_count,
                        progress_total=len(chunks_to_process),
                        progress_percentage=round(
                            (processed_count / len(chunks_to_process)) * 100, 1
                        ),
                        processing_rate_chunks_per_sec=round(chunks_per_second, 2),
                        eta_seconds=round(eta_seconds, 0),
                        chunk_processing_time_sec=round(
                            current_time - chunk_start_time, 2
                        ),
                        symbols_processed=result.get("symbols_processed", 0),
                        manifest_version=manifest_version,
                    )

                except ChunkProcessingFatalError as e:
                    processed_count += 1
                    results[chunk_id] = {
                        "status": "failed_fatal",
                        "error": str(e),
                        "retryable": False,
                    }
                    logger.error(
                        "Chunk failed with fatal error",
                        chunk_id=chunk_id,
                        error=str(e),
                        progress_completed=processed_count,
                        progress_total=len(chunks_to_process),
                        progress_percentage=round(
                            (processed_count / len(chunks_to_process)) * 100, 1
                        ),
                        manifest_version=manifest_version,
                    )

                except ChunkProcessingError as e:
                    processed_count += 1
                    results[chunk_id] = {
                        "status": "failed_retryable",
                        "error": str(e),
                        "retryable": True,
                    }
                    logger.error(
                        "Chunk failed after retries",
                        chunk_id=chunk_id,
                        error=str(e),
                        progress_completed=processed_count,
                        progress_total=len(chunks_to_process),
                        progress_percentage=round(
                            (processed_count / len(chunks_to_process)) * 100, 1
                        ),
                        manifest_version=manifest_version,
                    )

                except Exception as e:
                    # Unexpected error - treat as retryable
                    processed_count += 1
                    results[chunk_id] = {
                        "status": "failed_unexpected",
                        "error": str(e),
                        "retryable": True,
                    }
                    logger.error(
                        "Chunk failed with unexpected error",
                        chunk_id=chunk_id,
                        error=str(e),
                        progress_completed=processed_count,
                        progress_total=len(chunks_to_process),
                        progress_percentage=round(
                            (processed_count / len(chunks_to_process)) * 100, 1
                        ),
                        manifest_version=manifest_version,
                    )

        # Progress reporting task for periodic updates
        async def report_progress() -> None:
            """Report progress every 30 seconds."""
            report_interval = 30  # seconds
            last_reported_count = 0

            while processed_count < len(chunks_to_process):
                await asyncio.sleep(report_interval)

                if processed_count > last_reported_count:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta = (
                        ((len(chunks_to_process) - processed_count) / rate)
                        if rate > 0
                        else 0
                    )

                    # Get semaphore statistics
                    semaphore_stats = self._get_semaphore_stats(semaphore)

                    logger.info(
                        "Batch processing progress update",
                        progress_completed=processed_count,
                        progress_total=len(chunks_to_process),
                        progress_percentage=round(
                            (processed_count / len(chunks_to_process)) * 100, 1
                        ),
                        elapsed_time_sec=round(elapsed, 1),
                        processing_rate_chunks_per_sec=round(rate, 2),
                        eta_minutes=round(eta / 60, 1),
                        manifest_version=manifest_version,
                        semaphore_active_tasks=semaphore_stats["active_tasks"],
                        semaphore_waiting_tasks=semaphore_stats["waiting_tasks"],
                        semaphore_total_permits=semaphore_stats["total_permits"],
                    )
                    last_reported_count = processed_count

        # Execute all chunk processing tasks concurrently with progress reporting
        tasks = [process_single_chunk(chunk) for chunk in chunks_to_process]

        # Start progress reporting task
        progress_task = asyncio.create_task(report_progress())

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Cancel progress reporting
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

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
            "checksum_verification": verification_result,
        }

        end_time = time.time()
        total_duration = end_time - start_time
        overall_processing_rate = (
            processed_count / total_duration if total_duration > 0 else 0
        )

        logger.info(
            "Batch chunk processing completed",
            total_chunks=len(chunks),
            successful_chunks=successful_count,
            failed_chunks=failed_count,
            skipped_chunks=skipped_count,
            total_duration_sec=round(total_duration, 2),
            overall_processing_rate_chunks_per_sec=round(overall_processing_rate, 2),
            manifest_version=manifest_version,
            success_rate_percentage=round(
                (successful_count / len(chunks_to_process)) * 100, 1
            )
            if chunks_to_process
            else 0,
            batch_size_mb=round(
                sum(chunk.size_bytes for chunk in chunks_to_process) / (1024 * 1024), 2
            ),
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

    async def _process_chunk_data_transactional(
        self,
        conn: Any,
        chunk_id: str,
        chunk_data: dict[str, Any],
        chunk_metadata: ChunkMetadata,
    ) -> dict[str, Any]:
        """
        Process chunk data into database tables within an existing transaction.

        Args:
            conn: Database connection with active transaction
            chunk_id: Chunk identifier
            chunk_data: Parsed chunk data
            chunk_metadata: Chunk metadata

        Returns:
            Processing result with counts
        """
        # Validate chunk data structure - should be chunk object with "files" key
        if not isinstance(chunk_data, dict):
            raise ChunkProcessingFatalError(
                f"Expected chunk data to be dict, got {type(chunk_data)}", chunk_id
            )

        # Extract files data from chunk
        files_data = chunk_data.get("files", [])
        if not isinstance(files_data, list):
            raise ChunkProcessingFatalError(
                f"Expected chunk files to be list, got {type(files_data)}", chunk_id
            )

        # Count items for processing
        files_count = len(files_data)
        symbols_count = sum(
            len(file_data.get("symbols", [])) for file_data in files_data
        )
        entry_points_count = sum(
            len(file_data.get("entry_points", [])) for file_data in files_data
        )

        logger.debug(
            "Processing chunk data within transaction",
            chunk_id=chunk_id,
            symbols_count=symbols_count,
            entry_points_count=entry_points_count,
            files_count=files_count,
        )

        # ACTUAL DATABASE INSERTION IMPLEMENTATION
        # Process the chunk data (list of parsed files) into database tables

        symbols_inserted = 0
        entry_points_inserted = 0
        files_inserted = 0

        try:
            # Process each parsed file in the chunk
            for file_data in files_data:
                # Insert file record
                file_id = await self._insert_file(conn, file_data)
                files_inserted += 1

                # Insert symbols for this file
                if file_data.get("symbols"):
                    for symbol_data in file_data["symbols"]:
                        await self._insert_symbol(
                            conn, symbol_data, file_id, file_data["config"]
                        )
                        symbols_inserted += 1

                # Insert entry points for this file (if present)
                if file_data.get("entry_points"):
                    for entry_point_data in file_data["entry_points"]:
                        await self._insert_entry_point(
                            conn, entry_point_data, file_id, file_data["config"]
                        )
                        entry_points_inserted += 1

            logger.info(
                "Database operations completed within transaction",
                chunk_id=chunk_id,
                files_inserted=files_inserted,
                symbols_inserted=symbols_inserted,
                entry_points_inserted=entry_points_inserted,
            )

            return {
                "symbols_processed": symbols_inserted,
                "entry_points_processed": entry_points_inserted,
                "files_processed": files_inserted,
            }

        except Exception as e:
            logger.error(
                "Database insertion failed",
                chunk_id=chunk_id,
                error=str(e),
                files_processed=files_inserted,
                symbols_processed=symbols_inserted,
            )

            # Re-raise as appropriate error type
            if (
                "unique constraint" in str(e).lower()
                or "duplicate key" in str(e).lower()
            ):
                raise ChunkProcessingError(
                    f"Database constraint violation in chunk {chunk_id}: {e}", chunk_id
                ) from e
            else:
                raise ChunkProcessingFatalError(
                    f"Database error in chunk {chunk_id}: {e}", chunk_id
                ) from e

    async def _update_chunk_status_transactional(
        self,
        conn: Any,
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
        """
        Update chunk status within an existing transaction.

        Args:
            conn: Database connection with active transaction
            chunk_id: Chunk identifier
            manifest_version: Manifest version
            status: Processing status
            started_at: Start time
            completed_at: Completion time
            error_message: Error message if any
            symbols_processed: Number of symbols processed
            checksum_verified: Whether checksum was verified
            increment_retry: Whether to increment retry count

        Returns:
            Updated chunk status
        """
        # Build dynamic SQL based on provided parameters
        updates = ["updated_at = NOW()"]
        values: list[Any] = [chunk_id]
        param_count = 2

        if status is not None:
            updates.append(f"status = ${param_count}")
            values.append(status)
            param_count += 1

        if started_at is not None:
            updates.append(f"started_at = ${param_count}")
            values.append(started_at)
            param_count += 1

        if completed_at is not None:
            updates.append(f"completed_at = ${param_count}")
            values.append(completed_at)
            param_count += 1

        if error_message is not None:
            updates.append(f"error_message = ${param_count}")
            values.append(error_message)
            param_count += 1

        if symbols_processed is not None:
            updates.append(f"symbols_processed = ${param_count}")
            values.append(symbols_processed)
            param_count += 1

        if checksum_verified is not None:
            updates.append(f"checksum_verified = ${param_count}")
            values.append(checksum_verified)
            param_count += 1

        if increment_retry:
            updates.append("retry_count = retry_count + 1")

        sql = f"""
        UPDATE chunk_processing
        SET {", ".join(updates)}
        WHERE chunk_id = $1
        RETURNING
            chunk_id,
            manifest_version,
            status,
            started_at,
            completed_at,
            error_message,
            retry_count,
            symbols_processed,
            checksum_verified,
            created_at,
            updated_at
        """

        logger.debug(
            "Updating chunk status within transaction",
            chunk_id=chunk_id,
            status=status,
            symbols_processed=symbols_processed,
        )

        row = await conn.fetchrow(sql, *values)
        if not row:
            # If no existing record, create new one within transaction
            create_sql = """
            INSERT INTO chunk_processing (
                chunk_id, manifest_version, status,
                started_at, completed_at, symbols_processed, checksum_verified
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING
                chunk_id, manifest_version, status,
                started_at, completed_at, error_message, retry_count,
                symbols_processed, checksum_verified, created_at, updated_at
            """
            row = await conn.fetchrow(
                create_sql,
                chunk_id,
                manifest_version,
                status or "pending",
                started_at,
                completed_at,
                symbols_processed,
                checksum_verified,
            )

        logger.debug(
            "Chunk status updated within transaction",
            chunk_id=chunk_id,
            status=row["status"] if row else None,
        )

        return {
            "chunk_id": row["chunk_id"],
            "manifest_version": row["manifest_version"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
            "retry_count": row["retry_count"],
            "symbols_processed": row["symbols_processed"],
            "checksum_verified": row["checksum_verified"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
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

    async def _insert_file(self, conn: Any, file_data: dict[str, Any]) -> int:
        """Insert file record and return file_id."""
        sha = file_data["sha"]

        query = """
            INSERT INTO file (path, sha, config, indexed_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (path, config)
            DO UPDATE SET sha = EXCLUDED.sha, indexed_at = NOW()
            RETURNING id;
        """

        result = await conn.fetchrow(query, file_data["path"], sha, file_data["config"])
        if result is None:
            raise ChunkProcessingFatalError(
                f"Failed to insert file: {file_data['path']}", "unknown"
            )
        return int(result["id"])

    async def _insert_symbol(
        self, conn: Any, symbol_data: dict[str, Any], file_id: int, config: str
    ) -> None:
        """Insert symbol record into database."""
        query = """
            INSERT INTO symbol (
                name, kind, file_id, start_line, end_line, start_col, end_col,
                config, signature
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (name, file_id, config)
            DO UPDATE SET
                kind = EXCLUDED.kind,
                start_line = EXCLUDED.start_line,
                end_line = EXCLUDED.end_line,
                start_col = EXCLUDED.start_col,
                end_col = EXCLUDED.end_col,
                signature = EXCLUDED.signature;
        """

        # Map parser symbol kinds to database enum values
        kind_mapping = {
            "Function": "function",
            "Struct": "struct",
            "Variable": "variable",
            "Macro": "macro",
            "Typedef": "typedef",
            "Enum": "enum",
            "Union": "union",
            "Constant": "constant",
        }

        db_kind = kind_mapping.get(symbol_data.get("kind", ""), "function")

        await conn.execute(
            query,
            symbol_data["name"],
            db_kind,
            file_id,
            symbol_data.get("start_line", 1),
            symbol_data.get("end_line", 1),
            symbol_data.get("start_col", 0),
            symbol_data.get("end_col", 0),
            config,
            symbol_data.get("signature", ""),
        )

    async def _insert_entry_point(
        self, conn: Any, entry_point_data: dict[str, Any], file_id: int, config: str
    ) -> None:
        """Insert entry point record into database."""
        query = """
            INSERT INTO entrypoint (
                kind, key, symbol_id, file_id, details, config
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (kind, key, config)
            DO UPDATE SET
                symbol_id = EXCLUDED.symbol_id,
                file_id = EXCLUDED.file_id,
                details = EXCLUDED.details;
        """

        # Map entry point types to database enum values
        kind_mapping = {
            "syscall": "syscall",
            "ioctl": "ioctl",
            "file_ops": "file_ops",
            "sysfs": "sysfs",
        }

        db_kind = kind_mapping.get(entry_point_data.get("type", ""), "syscall")

        # Find symbol_id if we have symbol name
        symbol_id = None
        if "symbol_name" in entry_point_data:
            symbol_row = await conn.fetchrow(
                "SELECT id FROM symbol WHERE name = $1 AND file_id = $2 AND config = $3",
                entry_point_data["symbol_name"],
                file_id,
                config,
            )
            if symbol_row:
                symbol_id = symbol_row["id"]

        await conn.execute(
            query,
            db_kind,
            entry_point_data.get("key", entry_point_data.get("name", "")),
            symbol_id,
            file_id,
            entry_point_data.get("details", {}),
            config,
        )


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
