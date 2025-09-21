"""
Chunk status tracking for KCS chunk processing operations.

Provides high-level interfaces for tracking and managing chunk processing
status across the entire system, including progress monitoring, status
aggregation, and processing coordination.
"""

from datetime import datetime
from typing import Any

import structlog

from .database.chunk_queries import ChunkQueryService
from .models.chunk_models import ChunkManifest

logger = structlog.get_logger(__name__)


class ChunkStateTracker:
    """
    High-level chunk status tracking and management.

    Provides convenient interfaces for monitoring chunk processing progress,
    managing status transitions, and coordinating processing operations
    across the entire system.
    """

    def __init__(self, database_queries: ChunkQueryService):
        """
        Initialize chunk tracker.

        Args:
            database_queries: Database queries instance for persistence
        """
        self.database_queries = database_queries

    async def initialize_chunks_for_manifest(
        self, manifest: ChunkManifest, force: bool = False
    ) -> dict[str, Any]:
        """
        Initialize chunk status records for all chunks in a manifest.

        Args:
            manifest: Chunk manifest to initialize status for
            force: Whether to reset existing status records

        Returns:
            Initialization result with counts and status
        """
        logger.info(
            "Initializing chunk status records",
            manifest_version=manifest.version,
            total_chunks=manifest.total_chunks,
            force=force,
        )

        results: dict[str, Any] = {
            "manifest_version": manifest.version,
            "total_chunks": manifest.total_chunks,
            "initialized_chunks": 0,
            "existing_chunks": 0,
            "errors": [],
        }

        for chunk in manifest.chunks:
            try:
                # Check if chunk status already exists
                existing_status = await self.database_queries.get_chunk_status(chunk.id)

                if existing_status and not force:
                    results["existing_chunks"] += 1
                    logger.debug(
                        "Chunk status already exists, skipping",
                        chunk_id=chunk.id,
                        current_status=existing_status["status"],
                    )
                    continue

                # Create or reset chunk status
                await self.database_queries.create_chunk_status(
                    chunk_id=chunk.id,
                    manifest_version=manifest.version,
                    status="pending",
                )

                results["initialized_chunks"] += 1
                logger.debug("Initialized chunk status", chunk_id=chunk.id)

            except Exception as e:
                error_msg = f"Failed to initialize chunk {chunk.id}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(
                    "Failed to initialize chunk status", chunk_id=chunk.id, error=str(e)
                )

        logger.info(
            "Completed chunk status initialization",
            manifest_version=manifest.version,
            initialized_chunks=results["initialized_chunks"],
            existing_chunks=results["existing_chunks"],
            errors=len(results["errors"]),
        )

        return results

    async def get_processing_progress(
        self, manifest_version: str | None = None
    ) -> dict[str, Any]:
        """
        Get comprehensive processing progress for chunks.

        Args:
            manifest_version: Specific manifest version or None for latest

        Returns:
            Processing progress with detailed statistics
        """
        logger.debug("Getting processing progress", manifest_version=manifest_version)

        # Get processing statistics from database
        stats = await self.database_queries.get_processing_stats(manifest_version)

        # Calculate progress percentages
        total_chunks = stats["totals"]["total_chunks"]
        status_counts = stats["by_status"]

        progress = {
            "manifest_version": manifest_version,
            "total_chunks": total_chunks,
            "status_breakdown": status_counts,
            "totals": stats["totals"],
            "progress_percentages": {},
        }

        if total_chunks > 0:
            for status, status_data in status_counts.items():
                percentage = (status_data["count"] / total_chunks) * 100
                progress["progress_percentages"][status] = round(percentage, 2)

            # Calculate overall completion
            completed_count = status_counts.get("completed", {}).get("count", 0)
            failed_count = status_counts.get("failed", {}).get("count", 0)
            processing_count = status_counts.get("processing", {}).get("count", 0)
            pending_count = status_counts.get("pending", {}).get("count", 0)

            progress["overall"] = {
                "completion_percentage": round(
                    (completed_count / total_chunks) * 100, 2
                ),
                "failure_percentage": round((failed_count / total_chunks) * 100, 2),
                "active_percentage": round((processing_count / total_chunks) * 100, 2),
                "pending_percentage": round((pending_count / total_chunks) * 100, 2),
            }

        return progress

    async def get_failed_chunks(
        self, manifest_version: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get list of failed chunks with error details.

        Args:
            manifest_version: Filter by manifest version
            limit: Maximum number of results

        Returns:
            List of failed chunk status records
        """
        logger.debug(
            "Getting failed chunks",
            manifest_version=manifest_version,
            limit=limit,
        )

        failed_chunks = await self.database_queries.get_chunks_by_status(
            status="failed",
            manifest_version=manifest_version,
            limit=limit,
        )

        # Enrich with retry information
        for chunk in failed_chunks:
            retry_count = chunk.get("retry_count", 0)
            chunk["can_retry"] = retry_count < 3  # Max retry limit
            chunk["next_retry_attempt"] = (
                retry_count + 1 if chunk["can_retry"] else None
            )

        logger.debug(
            "Retrieved failed chunks",
            count=len(failed_chunks),
            retryable_count=sum(1 for c in failed_chunks if c["can_retry"]),
        )

        return failed_chunks

    async def get_chunk_dependencies(self, chunk_id: str) -> dict[str, Any]:
        """
        Get dependency information for a chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Dependency information including blocking and blocked chunks
        """
        # For now, chunks are processed independently, but this could be extended
        # to handle dependencies based on subsystem ordering or call graph relationships
        logger.debug("Getting chunk dependencies", chunk_id=chunk_id)

        chunk_status = await self.database_queries.get_chunk_status(chunk_id)
        if not chunk_status:
            return {
                "chunk_id": chunk_id,
                "exists": False,
                "dependencies": [],
                "dependents": [],
                "blocking_issues": [],
            }

        return {
            "chunk_id": chunk_id,
            "exists": True,
            "current_status": chunk_status["status"],
            "dependencies": [],  # No dependencies in current implementation
            "dependents": [],  # No dependents in current implementation
            "blocking_issues": [],
        }

    async def mark_chunk_stale(
        self, chunk_id: str, reason: str
    ) -> dict[str, Any] | None:
        """
        Mark a chunk as needing reprocessing due to stale data.

        Args:
            chunk_id: Chunk identifier
            reason: Reason for marking as stale

        Returns:
            Updated chunk status or None if not found
        """
        logger.info(
            "Marking chunk as stale",
            chunk_id=chunk_id,
            reason=reason,
        )

        # Reset chunk to pending status with error message indicating staleness
        updated_status = await self.database_queries.update_chunk_status(
            chunk_id=chunk_id,
            status="pending",
            error_message=f"Marked stale: {reason}",
            completed_at=None,
            symbols_processed=None,
            checksum_verified=None,
        )

        if updated_status:
            logger.info(
                "Chunk marked as stale",
                chunk_id=chunk_id,
                previous_status=updated_status.get("status"),
            )

        return updated_status

    async def reset_processing_chunks(
        self, manifest_version: str | None = None, max_age_minutes: int = 60
    ) -> dict[str, Any]:
        """
        Reset chunks stuck in 'processing' status for too long.

        Args:
            manifest_version: Filter by manifest version
            max_age_minutes: Maximum age in minutes before considering stuck

        Returns:
            Reset operation result with counts
        """
        logger.info(
            "Resetting stuck processing chunks",
            manifest_version=manifest_version,
            max_age_minutes=max_age_minutes,
        )

        # Get chunks in processing status
        processing_chunks = await self.database_queries.get_chunks_by_status(
            status="processing",
            manifest_version=manifest_version,
            limit=1000,  # High limit to catch all stuck chunks
        )

        results: dict[str, Any] = {
            "total_processing_chunks": len(processing_chunks),
            "reset_chunks": 0,
            "recent_chunks": 0,
            "errors": [],
        }

        current_time = datetime.now()

        for chunk in processing_chunks:
            try:
                chunk_id = chunk["chunk_id"]
                started_at = chunk.get("started_at")

                if not started_at:
                    # No start time, definitely stuck
                    should_reset = True
                    age_minutes = float("inf")
                else:
                    # Parse ISO timestamp and calculate age
                    if isinstance(started_at, str):
                        started_time = datetime.fromisoformat(
                            started_at.replace("Z", "+00:00")
                        )
                    else:
                        started_time = started_at

                    age_delta = current_time - started_time.replace(tzinfo=None)
                    age_minutes = age_delta.total_seconds() / 60
                    should_reset = age_minutes > max_age_minutes

                if should_reset:
                    await self.database_queries.update_chunk_status(
                        chunk_id=chunk_id,
                        status="failed",
                        error_message=f"Reset after {age_minutes:.1f} minutes stuck in processing",
                        completed_at=current_time,
                    )
                    results["reset_chunks"] += 1
                    logger.debug(
                        "Reset stuck chunk",
                        chunk_id=chunk_id,
                        age_minutes=age_minutes,
                    )
                else:
                    results["recent_chunks"] += 1

            except Exception as e:
                error_msg = f"Failed to reset chunk {chunk['chunk_id']}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(
                    "Failed to reset stuck chunk",
                    chunk_id=chunk.get("chunk_id"),
                    error=str(e),
                )

        logger.info(
            "Completed reset of stuck processing chunks",
            total_processing=results["total_processing_chunks"],
            reset_count=results["reset_chunks"],
            recent_count=results["recent_chunks"],
            errors=len(results["errors"]),
        )

        return results

    async def get_manifest_summary(
        self, manifest_version: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get comprehensive summary for a manifest version.

        Args:
            manifest_version: Manifest version or None for latest

        Returns:
            Manifest summary with processing status
        """
        logger.debug("Getting manifest summary", manifest_version=manifest_version)

        # Get manifest data
        manifest_data = await self.database_queries.get_manifest(manifest_version)
        if not manifest_data:
            return None

        # Get processing progress
        progress = await self.get_processing_progress(manifest_data["version"])

        # Get failed chunks summary
        failed_chunks = await self.get_failed_chunks(manifest_data["version"], limit=10)

        summary = {
            "manifest": {
                "version": manifest_data["version"],
                "created": manifest_data["created"],
                "kernel_version": manifest_data.get("kernel_version"),
                "kernel_path": manifest_data.get("kernel_path"),
                "config": manifest_data.get("config"),
                "total_chunks": manifest_data["total_chunks"],
                "total_size_bytes": manifest_data.get("total_size_bytes"),
            },
            "processing": progress,
            "recent_failures": failed_chunks[:5],  # Top 5 recent failures
            "health": {
                "has_failures": len(failed_chunks) > 0,
                "stuck_processing": progress["status_breakdown"]
                .get("processing", {})
                .get("count", 0),
                "completion_rate": progress.get("overall", {}).get(
                    "completion_percentage", 0
                ),
            },
        }

        return summary

    async def cleanup_manifest(
        self, manifest_version: str, cleanup_status: str = "failed"
    ) -> dict[str, Any]:
        """
        Clean up chunk records for a manifest version.

        Args:
            manifest_version: Manifest version to clean up
            cleanup_status: Status of chunks to clean up

        Returns:
            Cleanup operation result
        """
        logger.info(
            "Cleaning up manifest chunks",
            manifest_version=manifest_version,
            cleanup_status=cleanup_status,
        )

        deleted_count = await self.database_queries.cleanup_old_chunks(
            manifest_version, cleanup_status
        )

        result = {
            "manifest_version": manifest_version,
            "cleanup_status": cleanup_status,
            "deleted_count": deleted_count,
        }

        logger.info(
            "Completed manifest cleanup",
            manifest_version=manifest_version,
            deleted_count=deleted_count,
        )

        return result


# Convenience functions for common operations


async def initialize_manifest_tracking(
    manifest: ChunkManifest,
    database_queries: ChunkQueryService,
    force: bool = False,
) -> dict[str, Any]:
    """
    Initialize chunk tracking for a manifest.

    Args:
        manifest: Chunk manifest to initialize
        database_queries: Database queries instance
        force: Whether to reset existing status records

    Returns:
        Initialization result
    """
    tracker = ChunkStateTracker(database_queries)
    return await tracker.initialize_chunks_for_manifest(manifest, force)


async def get_processing_overview(
    database_queries: ChunkQueryService,
    manifest_version: str | None = None,
) -> dict[str, Any]:
    """
    Get processing overview for a manifest version.

    Args:
        database_queries: Database queries instance
        manifest_version: Manifest version or None for latest

    Returns:
        Processing overview
    """
    tracker = ChunkStateTracker(database_queries)
    return await tracker.get_processing_progress(manifest_version)


async def reset_stuck_chunks(
    database_queries: ChunkQueryService,
    manifest_version: str | None = None,
    max_age_minutes: int = 60,
) -> dict[str, Any]:
    """
    Reset chunks stuck in processing status.

    Args:
        database_queries: Database queries instance
        manifest_version: Manifest version filter
        max_age_minutes: Maximum age before considering stuck

    Returns:
        Reset operation result
    """
    tracker = ChunkStateTracker(database_queries)
    return await tracker.reset_processing_chunks(manifest_version, max_age_minutes)
