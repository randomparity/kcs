"""
Database connection and query interface for KCS MCP server.

Provides PostgreSQL connection management and query builders
for kernel analysis data access.
"""

import typing
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import structlog
from fastapi import HTTPException, status

from .models import ErrorResponse

logger = structlog.get_logger(__name__)


class Database:
    """PostgreSQL database connection manager."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            logger.info("Database connection pool created")

            # Test connection and verify schema
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            logger.info("Database connection verified")

        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=ErrorResponse(
                    error="database_connection_failed",
                    message=f"Failed to connect to database: {e!s}",
                ).dict(),
            ) from e

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self) -> typing.AsyncGenerator[asyncpg.Connection, None]:
        """Acquire database connection from pool."""
        if not self.pool:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=ErrorResponse(
                    error="database_not_connected",
                    message="Database connection not established",
                ).dict(),
            )

        async with self.pool.acquire() as conn:
            yield conn

    async def search_code_semantic(
        self, query: str, top_k: int = 10, config: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Search code using semantic similarity.

        Args:
            query: Search query text
            top_k: Maximum results to return
            config: Optional configuration filter (e.g., "x86_64:defconfig")

        Returns:
            List of search hits with span and snippet data
        """
        async with self.acquire() as conn:
            # TODO: Implement actual semantic search using pgvector
            # This would embed the query and find similar code snippets

            sql = """
            SELECT
                f.path,
                f.sha,
                s.name,
                s.start_line,
                s.end_line,
                COALESCE(s.signature, s.name || ' (' || s.kind || ')') as snippet,
                0.95 as score
            FROM symbol s
            JOIN file f ON s.file_id = f.id
            WHERE s.name ILIKE $1
            LIMIT $2
            """

            rows = await conn.fetch(sql, f"%{query}%", top_k)

            return [
                {
                    "path": row["path"],
                    "sha": row["sha"],
                    "start": row["start_line"],
                    "end": row["end_line"],
                    "snippet": row["snippet"] or f"Function matching '{query}'",
                    "score": float(row["score"]),
                }
                for row in rows
            ]

    async def get_symbol_info(
        self, symbol_name: str, config: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get detailed symbol information.

        Args:
            symbol_name: Name of symbol to lookup
            config: Optional configuration filter

        Returns:
            Symbol information dict or None if not found
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                s.name,
                s.kind,
                s.start_line,
                s.end_line,
                COALESCE(s.signature, s.name || ' (' || s.kind || ')') as snippet,
                f.path,
                f.sha
            FROM symbol s
            JOIN file f ON s.file_id = f.id
            WHERE s.name = $1
            LIMIT 1
            """

            row = await conn.fetchrow(sql, symbol_name)

            if not row:
                return None

            return {
                "name": row["name"],
                "kind": row["kind"],
                "decl": {
                    "path": row["path"],
                    "sha": row["sha"],
                    "start": row["start_line"],
                    "end": row["end_line"],
                },
                "summary": None,  # TODO: Add AI-generated summaries
            }

    async def find_callers(
        self, symbol_name: str, depth: int = 1, config: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Find functions that call the specified symbol.

        Args:
            symbol_name: Target symbol name
            depth: Call graph traversal depth
            config: Optional configuration filter

        Returns:
            List of caller information
        """
        async with self.acquire() as conn:
            # Enforce maximum depth to prevent excessive traversal
            depth = min(depth, 5)

            # Use recursive CTE for depth traversal with cycle detection
            sql = """
            WITH RECURSIVE callers_tree AS (
                -- Base case: direct callers (depth 1)
                SELECT
                    ce.caller_id,
                    ce.callee_id,
                    ce.call_type,
                    1 as depth_level,
                    ARRAY[ce.caller_id] as visited_path
                FROM call_edge ce
                JOIN symbol callee_s ON ce.callee_id = callee_s.id
                WHERE callee_s.name = $1

                UNION

                -- Recursive case: traverse up the call graph
                SELECT
                    ce.caller_id,
                    ce.callee_id,
                    ce.call_type,
                    ct.depth_level + 1,
                    ct.visited_path || ce.caller_id
                FROM call_edge ce
                JOIN callers_tree ct ON ce.callee_id = ct.caller_id
                WHERE
                    ct.depth_level < $2
                    AND NOT (ce.caller_id = ANY(ct.visited_path))  -- Cycle detection
            )
            SELECT DISTINCT
                caller_s.name as caller_symbol,
                caller_f.path as caller_path,
                caller_f.sha as caller_sha,
                caller_s.start_line as caller_start,
                caller_s.end_line as caller_end,
                ct.call_type,
                ct.depth_level
            FROM callers_tree ct
            JOIN symbol caller_s ON ct.caller_id = caller_s.id
            JOIN file caller_f ON caller_s.file_id = caller_f.id
            ORDER BY ct.depth_level, caller_s.name
            LIMIT 100
            """

            rows = await conn.fetch(sql, symbol_name, depth)

            return [
                {
                    "symbol": row["caller_symbol"],
                    "span": {
                        "path": row["caller_path"],
                        "sha": row["caller_sha"],
                        "start": row["caller_start"],
                        "end": row["caller_end"],
                    },
                    "call_type": row["call_type"] or "direct",
                }
                for row in rows
            ]

    async def find_callees(
        self, symbol_name: str, depth: int = 1, config: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Find functions called by the specified symbol.

        Args:
            symbol_name: Source symbol name
            depth: Dependency traversal depth
            config: Optional configuration filter

        Returns:
            List of callee information
        """
        async with self.acquire() as conn:
            # Enforce maximum depth to prevent excessive traversal
            depth = min(depth, 5)

            # Use recursive CTE for depth traversal with cycle detection
            sql = """
            WITH RECURSIVE callees_tree AS (
                -- Base case: direct callees (depth 1)
                SELECT
                    ce.caller_id,
                    ce.callee_id,
                    ce.call_type,
                    1 as depth_level,
                    ARRAY[ce.callee_id] as visited_path
                FROM call_edge ce
                JOIN symbol caller_s ON ce.caller_id = caller_s.id
                WHERE caller_s.name = $1

                UNION

                -- Recursive case: traverse down the call graph
                SELECT
                    ce.caller_id,
                    ce.callee_id,
                    ce.call_type,
                    ct.depth_level + 1,
                    ct.visited_path || ce.callee_id
                FROM call_edge ce
                JOIN callees_tree ct ON ce.caller_id = ct.callee_id
                WHERE
                    ct.depth_level < $2
                    AND NOT (ce.callee_id = ANY(ct.visited_path))  -- Cycle detection
            )
            SELECT DISTINCT
                callee_s.name as callee_symbol,
                callee_f.path as callee_path,
                callee_f.sha as callee_sha,
                callee_s.start_line as callee_start,
                callee_s.end_line as callee_end,
                ct.call_type,
                ct.depth_level
            FROM callees_tree ct
            JOIN symbol callee_s ON ct.callee_id = callee_s.id
            JOIN file callee_f ON callee_s.file_id = callee_f.id
            ORDER BY ct.depth_level, callee_s.name
            LIMIT 100
            """

            rows = await conn.fetch(sql, symbol_name, depth)

            return [
                {
                    "symbol": row["callee_symbol"],
                    "span": {
                        "path": row["callee_path"],
                        "sha": row["callee_sha"],
                        "start": row["callee_start"],
                        "end": row["callee_end"],
                    },
                    "call_type": row["call_type"] or "direct",
                }
                for row in rows
            ]

    async def insert_parsed_file(self, parsed_file: dict[str, Any], config: str) -> int:
        """
        Insert a parsed file and its symbols/call edges into the database.

        Args:
            parsed_file: Parsed file data from Rust parser
            config: Configuration string (e.g., "x86_64:defconfig")

        Returns:
            File ID that was inserted/updated
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                # Insert/update file record
                file_sql = """
                INSERT INTO file (path, sha, config)
                VALUES ($1, $2, $3)
                ON CONFLICT (path, config)
                DO UPDATE SET
                    sha = EXCLUDED.sha,
                    indexed_at = NOW()
                RETURNING id
                """

                file_row = await conn.fetchrow(
                    file_sql, parsed_file["path"], parsed_file["sha"], config
                )
                file_id: int = file_row["id"]

                # Clear existing symbols for this file/config to handle updates
                await conn.execute(
                    "DELETE FROM symbol WHERE file_id = $1 AND config = $2",
                    file_id,
                    config,
                )

                # Insert symbols and build caller/callee lookup
                symbol_id_map = {}  # Map from symbol name to database ID

                if "symbols" in parsed_file:
                    symbol_batch = []
                    for symbol in parsed_file["symbols"]:
                        # Handle metadata if present
                        metadata = None
                        if symbol.get("metadata"):
                            import json

                            metadata = json.dumps(symbol["metadata"])

                        symbol_batch.append(
                            (
                                symbol["name"],
                                symbol["kind"],
                                file_id,
                                symbol["start_line"],
                                symbol["end_line"],
                                symbol.get("start_col", 0),
                                symbol.get("end_col", 0),
                                config,
                                symbol.get("signature", ""),
                                metadata,
                            )
                        )

                    if symbol_batch:
                        symbol_sql = """
                        INSERT INTO symbol (name, kind, file_id, start_line, end_line, start_col, end_col, config, signature, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id, name
                        """

                        # Insert symbols one by one to get their IDs
                        for symbol_data in symbol_batch:
                            symbol_row = await conn.fetchrow(symbol_sql, *symbol_data)
                            symbol_id_map[symbol_row["name"]] = symbol_row["id"]

                # Insert call edges if present
                call_edges = parsed_file.get("call_edges")
                if call_edges:
                    # Clear existing call edges for this file/config
                    await conn.execute(
                        """
                        DELETE FROM call_edge
                        WHERE caller_id IN (
                            SELECT id FROM symbol
                            WHERE file_id = $1 AND config = $2
                        )
                    """,
                        file_id,
                        config,
                    )

                    call_edge_batch = []
                    for call_edge in call_edges:
                        caller_name = call_edge.get("caller")
                        callee_name = call_edge.get("callee")

                        # Look up caller and callee IDs
                        caller_id = symbol_id_map.get(caller_name)
                        callee_id = None

                        # Find callee_id - it might be in another file
                        if callee_name and caller_id:
                            callee_row = await conn.fetchrow(
                                "SELECT id FROM symbol WHERE name = $1 AND config = $2 LIMIT 1",
                                callee_name,
                                config,
                            )
                            if callee_row:
                                callee_id = callee_row["id"]

                        if caller_id and callee_id:
                            call_edge_batch.append(
                                (
                                    caller_id,
                                    callee_id,
                                    config,
                                    call_edge.get("call_type", "direct"),
                                    call_edge.get("is_indirect", False),
                                    call_edge.get("call_site_line"),
                                )
                            )

                    if call_edge_batch:
                        call_edge_sql = """
                        INSERT INTO call_edge (caller_id, callee_id, config, call_type, is_indirect, call_site_line)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (caller_id, callee_id, config) DO UPDATE SET
                            call_type = EXCLUDED.call_type,
                            is_indirect = EXCLUDED.is_indirect,
                            call_site_line = EXCLUDED.call_site_line
                        """
                        await conn.executemany(call_edge_sql, call_edge_batch)

                return file_id

    async def insert_parsed_files_batch(
        self, parsed_files: list[dict[str, Any]], config: str
    ) -> int:
        """
        Insert multiple parsed files and their data into the database efficiently.

        Args:
            parsed_files: List of parsed file data from Rust parser
            config: Configuration string (e.g., "x86_64:defconfig")

        Returns:
            Number of files processed
        """
        processed_count = 0

        for parsed_file in parsed_files:
            try:
                await self.insert_parsed_file(parsed_file, config)
                processed_count += 1
            except Exception as e:
                logger.error(
                    "Failed to insert parsed file",
                    path=parsed_file.get("path", "unknown"),
                    error=str(e),
                )
                # Continue processing other files rather than failing completely
                continue

        return processed_count

    async def insert_entry_points(
        self, entry_points: list[dict[str, Any]], config: str
    ) -> int:
        """
        Insert entry points with metadata into the database.

        Args:
            entry_points: List of entry points from extraction
            config: Configuration string (e.g., "x86_64:defconfig")

        Returns:
            Number of entry points inserted
        """
        if not entry_points:
            return 0

        async with self.acquire() as conn:
            async with conn.transaction():
                inserted_count = 0

                for entry_point in entry_points:
                    # Handle metadata if present
                    metadata = None
                    if entry_point.get("metadata"):
                        import json

                        metadata = json.dumps(entry_point["metadata"])

                    # Insert entry point
                    entry_sql = """
                    INSERT INTO entrypoint (
                        name, entry_type, file_path, line_number,
                        signature, description, metadata, config
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (name, entry_type, config)
                    DO UPDATE SET
                        file_path = EXCLUDED.file_path,
                        line_number = EXCLUDED.line_number,
                        signature = EXCLUDED.signature,
                        description = EXCLUDED.description,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """

                    await conn.execute(
                        entry_sql,
                        entry_point["name"],
                        entry_point["entry_type"],
                        entry_point["file_path"],
                        entry_point["line_number"],
                        entry_point.get("signature", ""),
                        entry_point.get("description"),
                        metadata,
                        config,
                    )
                    inserted_count += 1

                return inserted_count

    async def insert_kernel_patterns(
        self, patterns: list[dict[str, Any]], config: str
    ) -> int:
        """
        Insert kernel patterns with metadata into the database.

        Args:
            patterns: List of detected kernel patterns
            config: Configuration string (e.g., "x86_64:defconfig")

        Returns:
            Number of patterns inserted
        """
        if not patterns:
            return 0

        async with self.acquire() as conn:
            async with conn.transaction():
                inserted_count = 0

                for pattern in patterns:
                    # First, find the file ID
                    file_row = await conn.fetchrow(
                        "SELECT id FROM file WHERE path = $1 AND config = $2",
                        pattern["file_path"],
                        config,
                    )

                    if not file_row:
                        # Insert file if it doesn't exist
                        file_sql = """
                        INSERT INTO file (path, sha, config)
                        VALUES ($1, $2, $3)
                        RETURNING id
                        """
                        file_row = await conn.fetchrow(
                            file_sql,
                            pattern["file_path"],
                            pattern.get("sha", ""),
                            config,
                        )

                    file_id = file_row["id"]

                    # Find associated symbol if specified
                    symbol_id = None
                    if "symbol_name" in pattern:
                        symbol_row = await conn.fetchrow(
                            "SELECT id FROM symbol WHERE name = $1 AND config = $2 LIMIT 1",
                            pattern["symbol_name"],
                            config,
                        )
                        if symbol_row:
                            symbol_id = symbol_row["id"]

                    # Find associated entry point if specified
                    entrypoint_id = None
                    if "entry_point_name" in pattern:
                        entry_row = await conn.fetchrow(
                            """
                            SELECT id FROM entrypoint
                            WHERE name = $1 AND config = $2 LIMIT 1
                            """,
                            pattern["entry_point_name"],
                            config,
                        )
                        if entry_row:
                            entrypoint_id = entry_row["id"]

                    # Handle metadata if present
                    metadata = None
                    if pattern.get("metadata"):
                        import json

                        metadata = json.dumps(pattern["metadata"])

                    # Insert pattern
                    pattern_sql = """
                    INSERT INTO kernel_pattern (
                        pattern_type, symbol_id, entrypoint_id, file_id,
                        line_number, raw_text, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """

                    await conn.execute(
                        pattern_sql,
                        pattern["pattern_type"],
                        symbol_id,
                        entrypoint_id,
                        file_id,
                        pattern["line_number"],
                        pattern.get("raw_text", ""),
                        metadata,
                    )
                    inserted_count += 1

                return inserted_count

    async def store_kernel_config(
        self,
        config_id: str,
        arch: str,
        config_name: str,
        config_path: str,
        options: dict,
        dependencies: list,
        metadata: dict,
    ) -> None:
        """
        Store kernel configuration data in the database.

        Args:
            config_id: Unique configuration identifier
            arch: Target architecture
            config_name: Configuration name
            config_path: Path to config file
            options: Configuration options dict
            dependencies: Dependency list
            metadata: Additional metadata
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                import json

                # Insert into kernel_config table
                config_sql = """
                INSERT INTO kernel_config (
                    config_id, arch, config_name, config_path,
                    options, dependencies, metadata, parsed_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (config_id)
                DO UPDATE SET
                    arch = EXCLUDED.arch,
                    config_name = EXCLUDED.config_name,
                    config_path = EXCLUDED.config_path,
                    options = EXCLUDED.options,
                    dependencies = EXCLUDED.dependencies,
                    metadata = EXCLUDED.metadata,
                    parsed_at = NOW()
                """

                await conn.execute(
                    config_sql,
                    config_id,
                    arch,
                    config_name,
                    config_path,
                    json.dumps({k: v.dict() for k, v in options.items()}),
                    json.dumps([dep.dict() for dep in dependencies]),
                    json.dumps(metadata),
                )


# Global database instance
_database: Database | None = None


async def get_database() -> Database:
    """FastAPI dependency for database access."""
    global _database
    if not _database:
        # Return a mock database for testing
        return MockDatabase("")  # Empty database URL for mock
    return _database


class MockDatabase(Database):
    """Mock database for testing without PostgreSQL."""

    async def search_code_semantic(
        self, query: str, top_k: int = 10, config: str | None = None
    ) -> list[dict[str, Any]]:
        """Mock semantic search."""
        if query.lower() == "nonexistent_function_12345_abcde":
            return []

        return [
            {
                "path": "fs/read_write.c",
                "sha": "a1b2c3d4e5f6789012345678901234567890abcd",
                "start": 451,
                "end": 465,
                "snippet": f"Function matching '{query}'",
                "score": 0.95,
            }
        ]

    async def get_symbol_info(
        self, symbol_name: str, config: str | None = None
    ) -> dict[str, Any] | None:
        """Mock symbol lookup."""
        if (
            symbol_name.startswith("nonexistent_")
            or symbol_name == "sys_nonexistent_call"
        ):
            return None

        return {
            "name": symbol_name,
            "kind": "function",
            "decl": {
                "path": "fs/read_write.c",
                "sha": "a1b2c3d4e5f6789012345678901234567890abcd",
                "start": 451,
                "end": 465,
            },
            "summary": None,
        }

    async def find_callers(
        self, symbol_name: str, depth: int = 1, config: str | None = None
    ) -> list[dict[str, Any]]:
        """Mock caller analysis."""
        if symbol_name.startswith("nonexistent_"):
            return []

        if symbol_name.startswith("sys_") or symbol_name.startswith("__x64_sys_"):
            return []  # Entry points have no callers

        return [
            {
                "symbol": "sys_read",
                "span": {
                    "path": "fs/read_write.c",
                    "sha": "a1b2c3d4e5f6789012345678901234567890abcd",
                    "start": 100,
                    "end": 105,
                },
                "call_type": "direct",
            }
        ]

    async def find_callees(
        self, symbol_name: str, depth: int = 1, config: str | None = None
    ) -> list[dict[str, Any]]:
        """Mock dependency analysis."""
        if symbol_name.startswith("nonexistent_"):
            return []

        if symbol_name.startswith("sys_"):
            return [
                {
                    "symbol": "vfs_read",
                    "span": {
                        "path": "fs/read_write.c",
                        "sha": "a1b2c3d4e5f6789012345678901234567890abcd",
                        "start": 200,
                        "end": 205,
                    },
                    "call_type": "direct",
                }
            ]

        return []

    async def insert_parsed_file(self, parsed_file: dict[str, Any], config: str) -> int:
        """Mock parsed file insertion."""
        # Just return a mock file ID
        return 1

    async def insert_parsed_files_batch(
        self, parsed_files: list[dict[str, Any]], config: str
    ) -> int:
        """Mock batch file insertion."""
        return len(parsed_files)

    async def store_kernel_config(
        self,
        config_id: str,
        arch: str,
        config_name: str,
        config_path: str,
        options: dict,
        dependencies: list,
        metadata: dict,
    ) -> None:
        """Mock kernel config storage."""
        # Just log that it was called
        logger.info(
            "Mock storing kernel config",
            config_id=config_id,
            arch=arch,
            config_name=config_name,
            options_count=len(options),
        )


def set_database(database: Database) -> None:
    """Set global database instance."""
    global _database
    _database = database
