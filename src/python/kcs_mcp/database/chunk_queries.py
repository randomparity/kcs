"""
Database queries for chunk processing operations.

This module provides database operations for managing chunk processing
status and manifest data. Designed to work with the existing Database
class and connection pool.
"""

import json
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ChunkQueryService:
    """Database queries for chunk operations."""

    def __init__(self, database: Any) -> None:
        """Initialize with a Database instance."""
        self.database = database

    async def get_manifest(self, version: str | None = None) -> dict[str, Any] | None:
        """
        Get chunk manifest by version or the latest manifest.

        Args:
            version: Specific manifest version, or None for latest

        Returns:
            Manifest data dict or None if not found
        """
        async with self.database.acquire() as conn:
            if version:
                sql = """
                SELECT
                    version,
                    created,
                    kernel_version,
                    kernel_path,
                    config,
                    total_chunks,
                    total_size_bytes,
                    manifest_data
                FROM indexing_manifest
                WHERE version = $1
                """
                row = await conn.fetchrow(sql, version)
            else:
                sql = """
                SELECT
                    version,
                    created,
                    kernel_version,
                    kernel_path,
                    config,
                    total_chunks,
                    total_size_bytes,
                    manifest_data
                FROM indexing_manifest
                ORDER BY created DESC
                LIMIT 1
                """
                row = await conn.fetchrow(sql)

            if not row:
                return None

            # Return the stored manifest_data as-is since it contains the full manifest
            manifest = dict(row["manifest_data"])

            # Ensure database fields are consistent with the stored data
            manifest.update(
                {
                    "version": row["version"],
                    "created": row["created"].isoformat() if row["created"] else None,
                    "kernel_version": row["kernel_version"],
                    "kernel_path": row["kernel_path"],
                    "config": row["config"],
                    "total_chunks": row["total_chunks"],
                    "total_size_bytes": row["total_size_bytes"],
                }
            )

            return manifest

    async def store_manifest(
        self,
        version: str,
        created: datetime,
        manifest_data: dict[str, Any],
        kernel_version: str | None = None,
        kernel_path: str | None = None,
        config: str | None = None,
        total_chunks: int | None = None,
        total_size_bytes: int | None = None,
    ) -> str:
        """
        Store a new chunk manifest.

        Args:
            version: Manifest version (semver)
            created: Manifest creation timestamp
            manifest_data: Complete manifest data as dict
            kernel_version: Kernel version string
            kernel_path: Path to kernel source
            config: Kernel configuration (e.g., "x86_64:defconfig")
            total_chunks: Total number of chunks
            total_size_bytes: Total size of all chunks

        Returns:
            The manifest version string
        """
        async with self.database.acquire() as conn:
            # Extract fields from manifest_data if not provided
            if total_chunks is None:
                total_chunks = manifest_data.get("total_chunks")
            if total_size_bytes is None:
                total_size_bytes = manifest_data.get("total_size_bytes")
            if kernel_version is None:
                kernel_version = manifest_data.get("kernel_version")
            if kernel_path is None:
                kernel_path = manifest_data.get("kernel_path")
            if config is None:
                config = manifest_data.get("config")

            sql = """
            INSERT INTO indexing_manifest (
                version, created, kernel_version, kernel_path, config,
                total_chunks, total_size_bytes, manifest_data
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (version)
            DO UPDATE SET
                created = EXCLUDED.created,
                kernel_version = EXCLUDED.kernel_version,
                kernel_path = EXCLUDED.kernel_path,
                config = EXCLUDED.config,
                total_chunks = EXCLUDED.total_chunks,
                total_size_bytes = EXCLUDED.total_size_bytes,
                manifest_data = EXCLUDED.manifest_data,
                created_at = NOW()
            RETURNING version
            """

            row = await conn.fetchrow(
                sql,
                version,
                created,
                kernel_version,
                kernel_path,
                config,
                total_chunks,
                total_size_bytes,
                json.dumps(manifest_data),  # Store as JSONB
            )

            logger.info(
                "Stored manifest",
                version=version,
                total_chunks=total_chunks,
                kernel_version=kernel_version,
            )

            return str(row["version"]) if row else version

    async def get_chunk_status(self, chunk_id: str) -> dict[str, Any] | None:
        """
        Get processing status for a specific chunk.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            Chunk status dict or None if not found
        """
        async with self.database.acquire() as conn:
            sql = """
            SELECT
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
            FROM chunk_processing
            WHERE chunk_id = $1
            """

            row = await conn.fetchrow(sql, chunk_id)
            if not row:
                return None

            return {
                "chunk_id": row["chunk_id"],
                "manifest_version": row["manifest_version"],
                "status": row["status"],
                "started_at": row["started_at"].isoformat()
                if row["started_at"]
                else None,
                "completed_at": row["completed_at"].isoformat()
                if row["completed_at"]
                else None,
                "error_message": row["error_message"],
                "retry_count": row["retry_count"],
                "symbols_processed": row["symbols_processed"],
                "checksum_verified": row["checksum_verified"],
            }

    async def create_chunk_status(
        self,
        chunk_id: str,
        manifest_version: str,
        status: str = "pending",
    ) -> dict[str, Any]:
        """
        Create a new chunk processing status record.

        Args:
            chunk_id: Unique chunk identifier
            manifest_version: Version of manifest this chunk belongs to
            status: Initial status (default: "pending")

        Returns:
            Created chunk status dict
        """
        async with self.database.acquire() as conn:
            sql = """
            INSERT INTO chunk_processing (
                chunk_id, manifest_version, status, created_at, updated_at
            )
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (chunk_id)
            DO UPDATE SET
                manifest_version = EXCLUDED.manifest_version,
                status = EXCLUDED.status,
                updated_at = NOW()
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

            row = await conn.fetchrow(sql, chunk_id, manifest_version, status)

            logger.info(
                "Created chunk status",
                chunk_id=chunk_id,
                manifest_version=manifest_version,
                status=status,
            )

            return {
                "chunk_id": row["chunk_id"],
                "manifest_version": row["manifest_version"],
                "status": row["status"],
                "started_at": row["started_at"].isoformat()
                if row["started_at"]
                else None,
                "completed_at": row["completed_at"].isoformat()
                if row["completed_at"]
                else None,
                "error_message": row["error_message"],
                "retry_count": row["retry_count"],
                "symbols_processed": row["symbols_processed"],
                "checksum_verified": row["checksum_verified"],
            }

    async def update_chunk_status(
        self,
        chunk_id: str,
        status: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_message: str | None = None,
        symbols_processed: int | None = None,
        checksum_verified: bool | None = None,
        increment_retry: bool = False,
    ) -> dict[str, Any] | None:
        """
        Update chunk processing status.

        Args:
            chunk_id: Chunk identifier to update
            status: New status
            started_at: Processing start time
            completed_at: Processing completion time
            error_message: Error message if failed
            symbols_processed: Number of symbols processed
            checksum_verified: Whether checksum was verified
            increment_retry: Whether to increment retry count

        Returns:
            Updated chunk status dict or None if not found
        """
        async with self.database.acquire() as conn:
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

            if len(updates) == 1:  # Only updated_at
                logger.warning(
                    "No updates provided for chunk status", chunk_id=chunk_id
                )
                return await self.get_chunk_status(chunk_id)

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

            row = await conn.fetchrow(sql, *values)
            if not row:
                return None

            logger.info(
                "Updated chunk status",
                chunk_id=chunk_id,
                status=row["status"],
                retry_count=row["retry_count"],
            )

            return {
                "chunk_id": row["chunk_id"],
                "manifest_version": row["manifest_version"],
                "status": row["status"],
                "started_at": row["started_at"].isoformat()
                if row["started_at"]
                else None,
                "completed_at": row["completed_at"].isoformat()
                if row["completed_at"]
                else None,
                "error_message": row["error_message"],
                "retry_count": row["retry_count"],
                "symbols_processed": row["symbols_processed"],
                "checksum_verified": row["checksum_verified"],
            }

    async def get_chunks_by_status(
        self,
        status: str | None = None,
        manifest_version: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get chunks filtered by status and/or manifest version.

        Args:
            status: Filter by processing status
            manifest_version: Filter by manifest version
            limit: Maximum number of results

        Returns:
            List of chunk status dicts
        """
        async with self.database.acquire() as conn:
            conditions = []
            values: list[Any] = []
            param_count = 1

            if status is not None:
                conditions.append(f"status = ${param_count}")
                values.append(status)
                param_count += 1

            if manifest_version is not None:
                conditions.append(f"manifest_version = ${param_count}")
                values.append(manifest_version)
                param_count += 1

            where_clause = ""
            if conditions:
                where_clause = f"WHERE {' AND '.join(conditions)}"

            sql = f"""
            SELECT
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
            FROM chunk_processing
            {where_clause}
            ORDER BY created_at ASC
            LIMIT ${param_count}
            """

            values.append(limit)
            rows = await conn.fetch(sql, *values)

            return [
                {
                    "chunk_id": row["chunk_id"],
                    "manifest_version": row["manifest_version"],
                    "status": row["status"],
                    "started_at": row["started_at"].isoformat()
                    if row["started_at"]
                    else None,
                    "completed_at": row["completed_at"].isoformat()
                    if row["completed_at"]
                    else None,
                    "error_message": row["error_message"],
                    "retry_count": row["retry_count"],
                    "symbols_processed": row["symbols_processed"],
                    "checksum_verified": row["checksum_verified"],
                }
                for row in rows
            ]

    async def get_processing_stats(
        self, manifest_version: str | None = None
    ) -> dict[str, Any]:
        """
        Get processing statistics for chunks.

        Args:
            manifest_version: Filter by manifest version

        Returns:
            Dict with processing statistics
        """
        async with self.database.acquire() as conn:
            where_clause = ""
            values: list[Any] = []
            if manifest_version:
                where_clause = "WHERE manifest_version = $1"
                values = [manifest_version]

            sql = f"""
            SELECT
                status,
                COUNT(*) as count,
                COALESCE(SUM(symbols_processed), 0) as total_symbols,
                COALESCE(AVG(symbols_processed), 0) as avg_symbols_per_chunk
            FROM chunk_processing
            {where_clause}
            GROUP BY status
            ORDER BY status
            """

            rows = await conn.fetch(sql, *values)

            stats = {
                "by_status": {
                    row["status"]: {
                        "count": row["count"],
                        "total_symbols": row["total_symbols"],
                        "avg_symbols_per_chunk": float(row["avg_symbols_per_chunk"]),
                    }
                    for row in rows
                },
                "totals": {
                    "total_chunks": sum(row["count"] for row in rows),
                    "total_symbols": sum(row["total_symbols"] for row in rows),
                },
            }

            return stats

    async def cleanup_old_chunks(
        self, manifest_version: str, status_to_clean: str = "failed"
    ) -> int:
        """
        Clean up old chunk records for a manifest version.

        Args:
            manifest_version: Manifest version to clean
            status_to_clean: Status of chunks to remove

        Returns:
            Number of records deleted
        """
        async with self.database.acquire() as conn:
            sql = """
            DELETE FROM chunk_processing
            WHERE manifest_version = $1 AND status = $2
            """

            result = await conn.execute(sql, manifest_version, status_to_clean)

            # Extract count from result string like "DELETE 5"
            deleted_count = int(result.split()[-1]) if result.split() else 0

            logger.info(
                "Cleaned up chunk records",
                manifest_version=manifest_version,
                status=status_to_clean,
                deleted_count=deleted_count,
            )

            return deleted_count
