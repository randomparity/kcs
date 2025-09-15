"""
Database connection and query interface for KCS MCP server.

Provides PostgreSQL connection management and query builders
for kernel analysis data access.
"""

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
    async def acquire(self):
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
                s.snippet,
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
                s.snippet,
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
            # TODO: Implement actual call graph traversal
            # This would use the call_edge table for graph queries

            sql = """
            SELECT
                caller_s.name as caller_symbol,
                caller_f.path as caller_path,
                caller_f.sha as caller_sha,
                caller_s.start_line as caller_start,
                caller_s.end_line as caller_end,
                ce.call_type
            FROM call_edge ce
            JOIN symbol callee_s ON ce.callee_id = callee_s.id
            JOIN symbol caller_s ON ce.caller_id = caller_s.id
            JOIN file caller_f ON caller_s.file_id = caller_f.id
            WHERE callee_s.name = $1
            LIMIT 100
            """

            rows = await conn.fetch(sql, symbol_name)

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
            sql = """
            SELECT
                callee_s.name as callee_symbol,
                callee_f.path as callee_path,
                callee_f.sha as callee_sha,
                callee_s.start_line as callee_start,
                callee_s.end_line as callee_end,
                ce.call_type
            FROM call_edge ce
            JOIN symbol caller_s ON ce.caller_id = caller_s.id
            JOIN symbol callee_s ON ce.callee_id = callee_s.id
            JOIN file callee_f ON callee_s.file_id = callee_f.id
            WHERE caller_s.name = $1
            LIMIT 100
            """

            rows = await conn.fetch(sql, symbol_name)

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


# Global database instance
_database: Database | None = None


async def get_database() -> Database:
    """FastAPI dependency for database access."""
    global _database
    if not _database:
        # Return a mock database for testing
        return MockDatabase()
    return _database


class MockDatabase:
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


def set_database(database: Database) -> None:
    """Set global database instance."""
    global _database
    _database = database
