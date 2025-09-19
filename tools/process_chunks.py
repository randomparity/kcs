#!/usr/bin/env python3
"""
Process chunk files into the KCS database with resume capability.

This script loads a chunk manifest and processes the individual chunk files
into the PostgreSQL database, supporting parallel processing, resume from
failures, and comprehensive error handling.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import structlog
from kcs_mcp.chunk_loader import ChunkLoader, ChunkLoadError
from kcs_mcp.chunk_processor import ChunkProcessingError, ChunkProcessor
from kcs_mcp.chunk_tracker import ChunkTracker
from kcs_mcp.database import ChunkQueries, Database
from kcs_mcp.models.chunk_models import ChunkManifest

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ProcessingStats:
    """Track processing statistics."""

    def __init__(self) -> None:
        self.total_chunks = 0
        self.processed_chunks = 0
        self.failed_chunks = 0
        self.skipped_chunks = 0
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """Start timing."""
        import time
        self.start_time = time.time()

    def finish(self) -> None:
        """Finish timing."""
        import time
        self.end_time = time.time()

    @property
    def duration_seconds(self) -> float:
        """Get processing duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def chunks_per_second(self) -> float:
        """Get processing rate in chunks per second."""
        duration = self.duration_seconds
        if duration == 0:
            return 0.0
        return self.processed_chunks / duration

    def summary(self) -> dict[str, Any]:
        """Get processing summary."""
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "skipped_chunks": self.skipped_chunks,
            "duration_seconds": round(self.duration_seconds, 2),
            "chunks_per_second": round(self.chunks_per_second, 2),
        }


async def load_manifest(manifest_path: Path) -> ChunkManifest:
    """Load and validate chunk manifest."""
    logger.info("Loading chunk manifest", path=str(manifest_path))

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    chunk_loader = ChunkLoader()
    try:
        manifest = await chunk_loader.load_manifest(manifest_path)
        logger.info(
            "Loaded manifest",
            version=manifest.version,
            total_chunks=manifest.total_chunks,
            kernel_version=manifest.kernel_version,
        )
        return manifest
    except ChunkLoadError as e:
        logger.error("Failed to load manifest", error=str(e))
        raise


async def setup_database(database_url: str) -> tuple[Database, ChunkQueries]:
    """Setup database connection and queries."""
    logger.info("Setting up database connection")

    database = Database(database_url)
    await database.connect()

    chunk_queries = ChunkQueries(database)
    return database, chunk_queries


async def initialize_tracking(
    manifest: ChunkManifest,
    chunk_queries: ChunkQueries,
    force: bool = False
) -> dict[str, Any]:
    """Initialize chunk tracking for the manifest."""
    logger.info("Initializing chunk tracking", force=force)

    tracker = ChunkTracker(chunk_queries)
    result = await tracker.initialize_chunks_for_manifest(manifest, force)

    logger.info(
        "Chunk tracking initialized",
        initialized=result["initialized_chunks"],
        existing=result["existing_chunks"],
        errors=len(result["errors"]),
    )

    return result


async def get_chunks_to_process(
    manifest: ChunkManifest,
    chunk_queries: ChunkQueries,
    resume: bool = False,
    chunk_ids: list[str] | None = None,
) -> list[str]:
    """Get list of chunk IDs that need processing."""

    if chunk_ids:
        # Process specific chunks
        logger.info("Processing specific chunks", chunk_count=len(chunk_ids))
        return chunk_ids

    if resume:
        # Resume processing - get pending and failed chunks
        logger.info("Resume mode: finding chunks to process")

        pending_chunks = await chunk_queries.get_chunks_by_status(
            status="pending",
            manifest_version=manifest.version,
            limit=10000  # High limit to get all pending
        )

        failed_chunks = await chunk_queries.get_chunks_by_status(
            status="failed",
            manifest_version=manifest.version,
            limit=10000  # High limit to get all failed
        )

        chunk_ids_to_process = [c["chunk_id"] for c in pending_chunks + failed_chunks]

        logger.info(
            "Resume mode chunks identified",
            pending=len(pending_chunks),
            failed=len(failed_chunks),
            total_to_process=len(chunk_ids_to_process),
        )

        return chunk_ids_to_process

    else:
        # Process all chunks in manifest
        logger.info("Processing all chunks in manifest")
        return [chunk.id for chunk in manifest.chunks]


async def process_chunks_parallel(
    manifest: ChunkManifest,
    chunk_ids: list[str],
    chunk_processor: ChunkProcessor,
    parallel: int,
    stats: ProcessingStats,
) -> None:
    """Process chunks in parallel with semaphore limiting."""

    semaphore = asyncio.Semaphore(parallel)
    stats.total_chunks = len(chunk_ids)

    async def process_single_chunk(chunk_id: str) -> None:
        """Process a single chunk with semaphore."""
        async with semaphore:
            try:
                logger.debug("Processing chunk", chunk_id=chunk_id)

                # Find chunk metadata
                chunk_metadata = None
                for chunk in manifest.chunks:
                    if chunk.id == chunk_id:
                        chunk_metadata = chunk
                        break

                if not chunk_metadata:
                    logger.error("Chunk metadata not found", chunk_id=chunk_id)
                    stats.failed_chunks += 1
                    return

                # Process the chunk
                result = await chunk_processor.process_chunk(chunk_metadata)

                if result["status"] == "completed":
                    stats.processed_chunks += 1
                    logger.info(
                        "Chunk processed successfully",
                        chunk_id=chunk_id,
                        symbols_processed=result.get("symbols_processed", 0),
                    )
                elif result["status"] == "skipped":
                    stats.skipped_chunks += 1
                    logger.info("Chunk skipped", chunk_id=chunk_id, reason=result.get("message"))
                else:
                    stats.failed_chunks += 1
                    logger.error(
                        "Chunk processing failed",
                        chunk_id=chunk_id,
                        error=result.get("error_message"),
                    )

            except ChunkProcessingError as e:
                stats.failed_chunks += 1
                logger.error(
                    "Chunk processing error",
                    chunk_id=chunk_id,
                    error=str(e),
                    retryable=e.retryable,
                )
            except Exception as e:
                stats.failed_chunks += 1
                logger.error(
                    "Unexpected error processing chunk",
                    chunk_id=chunk_id,
                    error=str(e),
                )

    # Create tasks for all chunks
    tasks = [process_single_chunk(chunk_id) for chunk_id in chunk_ids]

    # Process all tasks
    logger.info(
        "Starting parallel chunk processing",
        total_chunks=len(chunk_ids),
        parallelism=parallel,
    )

    await asyncio.gather(*tasks, return_exceptions=True)


async def cleanup_stuck_chunks(
    chunk_queries: ChunkQueries,
    manifest_version: str,
    max_age_minutes: int = 60,
) -> None:
    """Reset chunks stuck in processing status."""
    logger.info("Cleaning up stuck processing chunks", max_age_minutes=max_age_minutes)

    tracker = ChunkTracker(chunk_queries)
    result = await tracker.reset_processing_chunks(manifest_version, max_age_minutes)

    if result["reset_chunks"] > 0:
        logger.warning(
            "Reset stuck processing chunks",
            reset_count=result["reset_chunks"],
            total_processing=result["total_processing_chunks"],
        )


async def print_processing_status(
    chunk_queries: ChunkQueries,
    manifest_version: str,
) -> None:
    """Print current processing status."""
    tracker = ChunkTracker(chunk_queries)
    progress = await tracker.get_processing_progress(manifest_version)

    print("\n=== Processing Status ===")
    print(f"Manifest Version: {manifest_version}")
    print(f"Total Chunks: {progress['total_chunks']}")

    if progress["status_breakdown"]:
        print("\nStatus Breakdown:")
        for status, data in progress["status_breakdown"].items():
            count = data["count"]
            percentage = progress["progress_percentages"].get(status, 0)
            print(f"  {status.capitalize()}: {count} ({percentage}%)")

    if progress.get("overall"):
        overall = progress["overall"]
        print(f"\nOverall Completion: {overall['completion_percentage']}%")
        print(f"Failure Rate: {overall['failure_percentage']}%")

    print()


async def main() -> None:
    """Main entry point for chunk processing."""
    parser = argparse.ArgumentParser(
        description="Process chunk files into KCS database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to chunk manifest JSON file",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of chunks to process in parallel",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing from previous failures (process pending and failed chunks only)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinitialization of chunk tracking (resets all statuses)",
    )

    parser.add_argument(
        "--chunk-ids",
        nargs="*",
        help="Process only specific chunk IDs (space-separated)",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show processing status and exit",
    )

    parser.add_argument(
        "--cleanup-stuck",
        type=int,
        metavar="MINUTES",
        help="Reset chunks stuck in processing status for more than MINUTES",
    )

    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://kcs:kcs@localhost/kcs"),
        help="PostgreSQL connection URL",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Load manifest
        manifest = await load_manifest(args.manifest)

        # Setup database
        database, chunk_queries = await setup_database(args.database_url)

        # Handle status-only request
        if args.status:
            await print_processing_status(chunk_queries, manifest.version)
            return

        # Handle cleanup request
        if args.cleanup_stuck is not None:
            await cleanup_stuck_chunks(chunk_queries, manifest.version, args.cleanup_stuck)
            return

        # Initialize chunk tracking if needed
        if not args.resume or args.force:
            await initialize_tracking(manifest, chunk_queries, args.force)

        # Get chunks to process
        chunk_ids = await get_chunks_to_process(
            manifest,
            chunk_queries,
            args.resume,
            args.chunk_ids
        )

        if not chunk_ids:
            logger.info("No chunks to process")
            return

        # Setup chunk processor
        chunk_loader = ChunkLoader()
        chunk_processor = ChunkProcessor(
            database_queries=chunk_queries,
            chunk_loader=chunk_loader,
        )

        # Process chunks
        stats = ProcessingStats()
        stats.start()

        try:
            await process_chunks_parallel(
                manifest,
                chunk_ids,
                chunk_processor,
                args.parallel,
                stats,
            )
        finally:
            stats.finish()

        # Print final statistics
        summary = stats.summary()
        logger.info("Processing completed", **summary)

        print("\n=== Processing Summary ===")
        print(f"Total Chunks: {summary['total_chunks']}")
        print(f"Processed: {summary['processed_chunks']}")
        print(f"Failed: {summary['failed_chunks']}")
        print(f"Skipped: {summary['skipped_chunks']}")
        print(f"Duration: {summary['duration_seconds']} seconds")
        print(f"Rate: {summary['chunks_per_second']} chunks/second")

        # Show final status
        await print_processing_status(chunk_queries, manifest.version)

        # Exit with error code if there were failures
        if stats.failed_chunks > 0:
            logger.error(f"Processing completed with {stats.failed_chunks} failures")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Processing failed", error=str(e))
        sys.exit(1)
    finally:
        # Close database connection
        if "database" in locals() and database.pool:
            await database.close()


if __name__ == "__main__":
    asyncio.run(main())
