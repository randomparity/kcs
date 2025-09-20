"""
Database operations for call graph extraction and analysis.

This module provides database operations for managing call graph data,
including call edges, function pointers, macro calls, and call paths.
Designed to work with the existing Database class and connection pool.
"""

import json
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)

# Type aliases for better type hints
CallType = Literal[
    "direct", "indirect", "macro", "callback", "conditional", "assembly", "syscall"
]
ConfidenceLevel = Literal["high", "medium", "low"]


class CallGraphQueries:
    """Database queries for call graph operations."""

    def __init__(self, database: Any) -> None:
        """Initialize with a Database instance."""
        self.database = database

    # ============================================================================
    # Call Edges Operations
    # ============================================================================

    async def insert_call_edge(
        self,
        caller_id: int,
        callee_id: int,
        file_path: str,
        line_number: int,
        call_type: CallType,
        confidence: ConfidenceLevel,
        conditional: bool = False,
        column_number: int | None = None,
        function_context: str | None = None,
        context_before: str | None = None,
        context_after: str | None = None,
        config_guard: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Insert a new call edge into the database.

        Args:
            caller_id: Symbol ID of the calling function
            callee_id: Symbol ID of the called function
            file_path: Path to the source file containing the call
            line_number: Line number of the call site
            call_type: Type of call mechanism
            confidence: Confidence level of the detection
            conditional: Whether the call is conditional
            column_number: Column number of the call site
            function_context: Name of the containing function
            context_before: Code context before the call
            context_after: Code context after the call
            config_guard: Configuration guard if conditional
            metadata: Additional metadata as JSON

        Returns:
            The ID of the inserted call edge

        Raises:
            ValueError: If conditional is True but config_guard is None
        """
        if conditional and config_guard is None:
            raise ValueError("config_guard must be provided when conditional is True")

        async with self.database.acquire() as conn:
            sql = """
            INSERT INTO call_edges (
                caller_id, callee_id, file_path, line_number, column_number,
                function_context, context_before, context_after,
                call_type, confidence, conditional, config_guard, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            """

            metadata_json = json.dumps(metadata or {})

            row = await conn.fetchrow(
                sql,
                caller_id,
                callee_id,
                file_path,
                line_number,
                column_number,
                function_context,
                context_before,
                context_after,
                call_type,
                confidence,
                conditional,
                config_guard,
                metadata_json,
            )

            call_edge_id: int = row["id"]

            logger.debug(
                "Inserted call edge",
                id=call_edge_id,
                caller_id=caller_id,
                callee_id=callee_id,
                call_type=call_type,
                confidence=confidence,
            )

            return call_edge_id

    async def insert_call_edges_batch(
        self, call_edges: list[dict[str, Any]]
    ) -> list[int]:
        """
        Insert multiple call edges in a single transaction.

        Args:
            call_edges: List of call edge dictionaries

        Returns:
            List of inserted call edge IDs
        """
        if not call_edges:
            return []

        async with self.database.acquire() as conn:
            async with conn.transaction():
                sql = """
                INSERT INTO call_edges (
                    caller_id, callee_id, file_path, line_number, column_number,
                    function_context, context_before, context_after,
                    call_type, confidence, conditional, config_guard, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
                """

                edge_ids = []
                for edge in call_edges:
                    # Validate conditional/config_guard constraint
                    if (
                        edge.get("conditional", False)
                        and edge.get("config_guard") is None
                    ):
                        raise ValueError(
                            f"config_guard must be provided when conditional is True for edge {edge}"
                        )

                    metadata_json = json.dumps(edge.get("metadata", {}))

                    row = await conn.fetchrow(
                        sql,
                        edge["caller_id"],
                        edge["callee_id"],
                        edge["file_path"],
                        edge["line_number"],
                        edge.get("column_number"),
                        edge.get("function_context"),
                        edge.get("context_before"),
                        edge.get("context_after"),
                        edge["call_type"],
                        edge["confidence"],
                        edge.get("conditional", False),
                        edge.get("config_guard"),
                        metadata_json,
                    )

                    edge_ids.append(row["id"])

                logger.info(
                    "Inserted call edges batch",
                    count=len(edge_ids),
                    ids=edge_ids[:5]
                    if len(edge_ids) > 5
                    else edge_ids,  # Log first 5 IDs
                )

                return edge_ids

    async def get_call_edges_for_caller(
        self,
        caller_id: int,
        call_types: list[CallType] | None = None,
        confidence_filter: ConfidenceLevel | None = None,
        include_conditional: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get all call edges for a specific caller function.

        Args:
            caller_id: Symbol ID of the calling function
            call_types: Filter by specific call types
            confidence_filter: Minimum confidence level
            include_conditional: Whether to include conditional calls

        Returns:
            List of call edge dictionaries
        """
        async with self.database.acquire() as conn:
            conditions = ["caller_id = $1"]
            values: list[Any] = [caller_id]
            param_count = 2

            if call_types:
                placeholders = ", ".join(
                    f"${i}" for i in range(param_count, param_count + len(call_types))
                )
                conditions.append(f"call_type IN ({placeholders})")
                values.extend(list(call_types))
                param_count += len(call_types)

            if confidence_filter:
                confidence_order = {"high": 3, "medium": 2, "low": 1}
                min_confidence = confidence_order[confidence_filter]
                confidence_cases = [
                    f"WHEN confidence = '{level}' THEN {order}"
                    for level, order in confidence_order.items()
                ]
                conditions.append(
                    f"(CASE {' '.join(confidence_cases)} END) >= {min_confidence}"
                )

            if not include_conditional:
                conditions.append("conditional = FALSE")

            where_clause = " AND ".join(conditions)

            sql = f"""
            SELECT
                id, caller_id, callee_id, file_path, line_number, column_number,
                function_context, context_before, context_after,
                call_type, confidence, conditional, config_guard, metadata,
                created_at
            FROM call_edges
            WHERE {where_clause}
            ORDER BY file_path, line_number
            """

            rows = await conn.fetch(sql, *values)

            return [
                {
                    "id": row["id"],
                    "caller_id": row["caller_id"],
                    "callee_id": row["callee_id"],
                    "file_path": row["file_path"],
                    "line_number": row["line_number"],
                    "column_number": row["column_number"],
                    "function_context": row["function_context"],
                    "context_before": row["context_before"],
                    "context_after": row["context_after"],
                    "call_type": row["call_type"],
                    "confidence": row["confidence"],
                    "conditional": row["conditional"],
                    "config_guard": row["config_guard"],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat()
                    if row["created_at"]
                    else None,
                }
                for row in rows
            ]

    async def get_call_edges_for_callee(
        self,
        callee_id: int,
        call_types: list[CallType] | None = None,
        confidence_filter: ConfidenceLevel | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get all call edges for a specific callee function (who calls this function).

        Args:
            callee_id: Symbol ID of the called function
            call_types: Filter by specific call types
            confidence_filter: Minimum confidence level

        Returns:
            List of call edge dictionaries
        """
        async with self.database.acquire() as conn:
            conditions = ["callee_id = $1"]
            values: list[Any] = [callee_id]
            param_count = 2

            if call_types:
                placeholders = ", ".join(
                    f"${i}" for i in range(param_count, param_count + len(call_types))
                )
                conditions.append(f"call_type IN ({placeholders})")
                values.extend(list(call_types))
                param_count += len(call_types)

            if confidence_filter:
                confidence_order = {"high": 3, "medium": 2, "low": 1}
                min_confidence = confidence_order[confidence_filter]
                confidence_cases = [
                    f"WHEN confidence = '{level}' THEN {order}"
                    for level, order in confidence_order.items()
                ]
                conditions.append(
                    f"(CASE {' '.join(confidence_cases)} END) >= {min_confidence}"
                )

            where_clause = " AND ".join(conditions)

            sql = f"""
            SELECT
                id, caller_id, callee_id, file_path, line_number, column_number,
                function_context, context_before, context_after,
                call_type, confidence, conditional, config_guard, metadata,
                created_at
            FROM call_edges
            WHERE {where_clause}
            ORDER BY file_path, line_number
            """

            rows = await conn.fetch(sql, *values)

            return [
                {
                    "id": row["id"],
                    "caller_id": row["caller_id"],
                    "callee_id": row["callee_id"],
                    "file_path": row["file_path"],
                    "line_number": row["line_number"],
                    "column_number": row["column_number"],
                    "function_context": row["function_context"],
                    "context_before": row["context_before"],
                    "context_after": row["context_after"],
                    "call_type": row["call_type"],
                    "confidence": row["confidence"],
                    "conditional": row["conditional"],
                    "config_guard": row["config_guard"],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat()
                    if row["created_at"]
                    else None,
                }
                for row in rows
            ]

    async def get_call_edges_in_file(
        self, file_path: str, line_start: int | None = None, line_end: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get all call edges in a specific file, optionally within a line range.

        Args:
            file_path: Path to the source file
            line_start: Start line number (inclusive)
            line_end: End line number (inclusive)

        Returns:
            List of call edge dictionaries
        """
        async with self.database.acquire() as conn:
            conditions = ["file_path = $1"]
            values: list[Any] = [file_path]
            param_count = 2

            if line_start is not None:
                conditions.append(f"line_number >= ${param_count}")
                values.append(line_start)
                param_count += 1

            if line_end is not None:
                conditions.append(f"line_number <= ${param_count}")
                values.append(line_end)
                param_count += 1

            where_clause = " AND ".join(conditions)

            sql = f"""
            SELECT
                id, caller_id, callee_id, file_path, line_number, column_number,
                function_context, context_before, context_after,
                call_type, confidence, conditional, config_guard, metadata,
                created_at
            FROM call_edges
            WHERE {where_clause}
            ORDER BY line_number, column_number
            """

            rows = await conn.fetch(sql, *values)

            return [
                {
                    "id": row["id"],
                    "caller_id": row["caller_id"],
                    "callee_id": row["callee_id"],
                    "file_path": row["file_path"],
                    "line_number": row["line_number"],
                    "column_number": row["column_number"],
                    "function_context": row["function_context"],
                    "context_before": row["context_before"],
                    "context_after": row["context_after"],
                    "call_type": row["call_type"],
                    "confidence": row["confidence"],
                    "conditional": row["conditional"],
                    "config_guard": row["config_guard"],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat()
                    if row["created_at"]
                    else None,
                }
                for row in rows
            ]

    # ============================================================================
    # Function Pointers Operations
    # ============================================================================

    async def insert_function_pointer(
        self,
        pointer_name: str,
        assignment_file: str,
        assignment_line: int,
        assigned_function_id: int | None = None,
        assignment_column: int | None = None,
        struct_context: str | None = None,
        assignment_context: str | None = None,
        usage_sites: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Insert a function pointer assignment record.

        Args:
            pointer_name: Name of the function pointer variable
            assignment_file: File where the assignment occurs
            assignment_line: Line number of the assignment
            assigned_function_id: Symbol ID of the assigned function (if known)
            assignment_column: Column number of the assignment
            struct_context: Struct name if pointer is a member
            assignment_context: Code context of the assignment
            usage_sites: List of locations where this pointer is used
            metadata: Additional metadata

        Returns:
            The ID of the inserted function pointer record
        """
        async with self.database.acquire() as conn:
            sql = """
            INSERT INTO function_pointers (
                pointer_name, assigned_function_id, assignment_file, assignment_line,
                assignment_column, struct_context, assignment_context, usage_sites, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """

            usage_sites_json = json.dumps(usage_sites or [])
            metadata_json = json.dumps(metadata or {})

            row = await conn.fetchrow(
                sql,
                pointer_name,
                assigned_function_id,
                assignment_file,
                assignment_line,
                assignment_column,
                struct_context,
                assignment_context,
                usage_sites_json,
                metadata_json,
            )

            pointer_id: int = row["id"]

            logger.debug(
                "Inserted function pointer",
                id=pointer_id,
                pointer_name=pointer_name,
                assigned_function_id=assigned_function_id,
            )

            return pointer_id

    async def get_function_pointers_by_name(
        self, pointer_name: str
    ) -> list[dict[str, Any]]:
        """
        Get function pointer assignments by pointer name.

        Args:
            pointer_name: Name of the function pointer

        Returns:
            List of function pointer assignment dictionaries
        """
        async with self.database.acquire() as conn:
            sql = """
            SELECT
                id, pointer_name, assigned_function_id, assignment_file, assignment_line,
                assignment_column, struct_context, assignment_context, usage_sites,
                metadata, created_at
            FROM function_pointers
            WHERE pointer_name = $1
            ORDER BY assignment_file, assignment_line
            """

            rows = await conn.fetch(sql, pointer_name)

            return [
                {
                    "id": row["id"],
                    "pointer_name": row["pointer_name"],
                    "assigned_function_id": row["assigned_function_id"],
                    "assignment_file": row["assignment_file"],
                    "assignment_line": row["assignment_line"],
                    "assignment_column": row["assignment_column"],
                    "struct_context": row["struct_context"],
                    "assignment_context": row["assignment_context"],
                    "usage_sites": list(row["usage_sites"])
                    if row["usage_sites"]
                    else [],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat()
                    if row["created_at"]
                    else None,
                }
                for row in rows
            ]

    # ============================================================================
    # Macro Calls Operations
    # ============================================================================

    async def insert_macro_call(
        self,
        macro_name: str,
        expansion_file: str,
        expansion_line: int,
        macro_definition: str | None = None,
        expansion_column: int | None = None,
        expanded_call_ids: list[int] | None = None,
        preprocessor_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Insert a macro call expansion record.

        Args:
            macro_name: Name of the macro being expanded
            expansion_file: File where the macro is expanded
            expansion_line: Line number of the expansion
            macro_definition: Definition of the macro (if available)
            expansion_column: Column number of the expansion
            expanded_call_ids: List of call edge IDs from the expansion
            preprocessor_context: Preprocessor context information
            metadata: Additional metadata

        Returns:
            The ID of the inserted macro call record
        """
        async with self.database.acquire() as conn:
            sql = """
            INSERT INTO macro_calls (
                macro_name, macro_definition, expansion_file, expansion_line,
                expansion_column, expanded_call_ids, preprocessor_context, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """

            metadata_json = json.dumps(metadata or {})

            row = await conn.fetchrow(
                sql,
                macro_name,
                macro_definition,
                expansion_file,
                expansion_line,
                expansion_column,
                expanded_call_ids or [],
                preprocessor_context,
                metadata_json,
            )

            macro_call_id: int = row["id"]

            logger.debug(
                "Inserted macro call",
                id=macro_call_id,
                macro_name=macro_name,
                expansion_file=expansion_file,
            )

            return macro_call_id

    # ============================================================================
    # Call Paths Operations
    # ============================================================================

    async def insert_call_path(
        self,
        entry_point_id: int,
        target_function_id: int,
        path_edge_ids: list[int],
        total_confidence: float,
        config_context: str | None = None,
        kernel_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Insert a pre-computed call path for caching.

        Args:
            entry_point_id: Symbol ID of the entry point function
            target_function_id: Symbol ID of the target function
            path_edge_ids: List of call edge IDs forming the path
            total_confidence: Combined confidence score for the entire path
            config_context: Configuration context for this path
            kernel_version: Kernel version this path applies to
            metadata: Additional metadata

        Returns:
            The ID of the inserted call path record
        """
        if entry_point_id == target_function_id:
            raise ValueError("Entry point and target function cannot be the same")

        if not path_edge_ids:
            raise ValueError("Path must contain at least one edge")

        if not (0.0 <= total_confidence <= 1.0):
            raise ValueError("Total confidence must be between 0.0 and 1.0")

        async with self.database.acquire() as conn:
            sql = """
            INSERT INTO call_paths (
                entry_point_id, target_function_id, path_edge_ids, path_length,
                total_confidence, config_context, kernel_version, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """

            metadata_json = json.dumps(metadata or {})

            row = await conn.fetchrow(
                sql,
                entry_point_id,
                target_function_id,
                path_edge_ids,
                len(path_edge_ids),
                total_confidence,
                config_context,
                kernel_version,
                metadata_json,
            )

            call_path_id: int = row["id"]

            logger.debug(
                "Inserted call path",
                id=call_path_id,
                entry_point_id=entry_point_id,
                target_function_id=target_function_id,
                path_length=len(path_edge_ids),
            )

            return call_path_id

    async def get_call_path(
        self,
        entry_point_id: int,
        target_function_id: int,
        config_context: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get a cached call path between two functions.

        Args:
            entry_point_id: Symbol ID of the entry point function
            target_function_id: Symbol ID of the target function
            config_context: Configuration context to match

        Returns:
            Call path dictionary or None if not found
        """
        async with self.database.acquire() as conn:
            conditions = ["entry_point_id = $1", "target_function_id = $2"]
            values: list[Any] = [entry_point_id, target_function_id]

            if config_context is not None:
                conditions.append("config_context = $3")
                values.append(config_context)

            where_clause = " AND ".join(conditions)

            sql = f"""
            SELECT
                id, entry_point_id, target_function_id, path_edge_ids, path_length,
                total_confidence, config_context, kernel_version, last_accessed,
                access_count, metadata, created_at
            FROM call_paths
            WHERE {where_clause}
            ORDER BY total_confidence DESC, last_accessed DESC
            LIMIT 1
            """

            row = await conn.fetchrow(sql, *values)

            if not row:
                return None

            # Update access statistics
            await conn.fetchval("SELECT update_call_path_access($1)", row["id"])

            return {
                "id": row["id"],
                "entry_point_id": row["entry_point_id"],
                "target_function_id": row["target_function_id"],
                "path_edge_ids": list(row["path_edge_ids"]),
                "path_length": row["path_length"],
                "total_confidence": row["total_confidence"],
                "config_context": row["config_context"],
                "kernel_version": row["kernel_version"],
                "last_accessed": row["last_accessed"].isoformat()
                if row["last_accessed"]
                else None,
                "access_count": row["access_count"],
                "metadata": dict(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
            }

    # ============================================================================
    # Statistics and Utility Operations
    # ============================================================================

    async def get_call_graph_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about the call graph data.

        Returns:
            Dictionary with call graph statistics
        """
        async with self.database.acquire() as conn:
            stats = await conn.fetchrow("SELECT * FROM get_call_graph_stats()")

            return {
                "total_call_edges": stats["total_call_edges"],
                "direct_calls": stats["direct_calls"],
                "indirect_calls": stats["indirect_calls"],
                "macro_calls": stats["macro_calls"],
                "function_pointers": stats["function_pointers"],
                "cached_paths": stats["cached_paths"],
                "avg_confidence": float(stats["avg_confidence"])
                if stats["avg_confidence"]
                else 0.0,
            }

    async def cleanup_old_call_paths(self, days_old: int = 30) -> int:
        """
        Clean up old, infrequently accessed call paths.

        Args:
            days_old: Number of days after which to consider paths old

        Returns:
            Number of paths deleted
        """
        async with self.database.acquire() as conn:
            deleted_count: int = await conn.fetchval(
                "SELECT cleanup_old_call_paths($1)", days_old
            )

            logger.info(
                "Cleaned up old call paths",
                days_old=days_old,
                deleted_count=deleted_count,
            )

            return deleted_count

    async def delete_call_edges_for_file(self, file_path: str) -> int:
        """
        Delete all call edges for a specific file (useful for re-processing).

        Args:
            file_path: Path to the source file

        Returns:
            Number of call edges deleted
        """
        async with self.database.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM call_edges WHERE file_path = $1", file_path
            )

            # Extract count from result string like "DELETE 5"
            deleted_count = int(result.split()[-1]) if result.split() else 0

            logger.info(
                "Deleted call edges for file",
                file_path=file_path,
                deleted_count=deleted_count,
            )

            return deleted_count
