"""
Database query functions for call graph relationship analysis.

This module provides high-level database operations for querying call graph
relationships, function analysis, and path tracing. These functions support
the MCP endpoint implementations and provide the core relationship queries.
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
RelationshipType = Literal["callers", "callees", "both"]


class CallGraphQueries:
    """High-level database queries for call graph analysis."""

    def __init__(self, database: Any) -> None:
        """Initialize with a Database instance."""
        self.database = database

    # ============================================================================
    # Function Relationship Queries
    # ============================================================================

    async def get_function_by_name(
        self, function_name: str, file_path: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get function information by name, optionally filtering by file.

        Args:
            function_name: Name of the function to find
            file_path: Optional file path to narrow search

        Returns:
            Function information dictionary or None if not found
        """
        async with self.database.acquire() as conn:
            conditions = ["s.name = $1", "s.symbol_type = 'function'"]
            values: list[Any] = [function_name]
            param_count = 2

            if file_path:
                conditions.append("s.file_path = $2")
                values.append(file_path)
                param_count += 1

            where_clause = " AND ".join(conditions)

            sql = f"""
            SELECT
                s.id, s.name, s.signature, s.file_path, s.line_number,
                s.symbol_type, s.config_dependencies, s.metadata
            FROM symbols s
            WHERE {where_clause}
            ORDER BY s.file_path, s.line_number
            LIMIT 1
            """

            row = await conn.fetchrow(sql, *values)

            if not row:
                return None

            return {
                "id": row["id"],
                "name": row["name"],
                "signature": row["signature"],
                "file_path": row["file_path"],
                "line_number": row["line_number"],
                "symbol_type": row["symbol_type"],
                "config_dependencies": list(row["config_dependencies"])
                if row["config_dependencies"]
                else [],
                "metadata": dict(row["metadata"]) if row["metadata"] else {},
            }

    async def get_call_relationships(
        self,
        function_name: str,
        relationship_type: RelationshipType = "both",
        max_depth: int = 1,
        config_context: str | None = None,
        call_types: list[CallType] | None = None,
        confidence_filter: ConfidenceLevel | None = None,
    ) -> dict[str, Any]:
        """
        Get call relationships for a function (callers, callees, or both).

        Args:
            function_name: Name of the function to analyze
            relationship_type: Type of relationships to retrieve
            max_depth: Maximum traversal depth
            config_context: Configuration context for filtering
            call_types: Filter by specific call types
            confidence_filter: Minimum confidence level

        Returns:
            Dictionary with callers and/or callees information
        """
        # First, find the function
        function = await self.get_function_by_name(function_name)
        if not function:
            return {
                "function_name": function_name,
                "relationships": {"callers": [], "callees": []},
                "error": "Function not found",
            }

        function_id = function["id"]
        relationships: dict[str, list[dict[str, Any]]] = {"callers": [], "callees": []}

        if relationship_type in ["callers", "both"]:
            callers = await self._get_callers_recursive(
                function_id,
                max_depth,
                config_context,
                call_types,
                confidence_filter,
            )
            relationships["callers"] = callers

        if relationship_type in ["callees", "both"]:
            callees = await self._get_callees_recursive(
                function_id,
                max_depth,
                config_context,
                call_types,
                confidence_filter,
            )
            relationships["callees"] = callees

        return {
            "function_name": function_name,
            "function": function,
            "relationships": relationships,
        }

    async def _get_callers_recursive(
        self,
        function_id: int,
        max_depth: int,
        config_context: str | None = None,
        call_types: list[CallType] | None = None,
        confidence_filter: ConfidenceLevel | None = None,
        current_depth: int = 0,
        visited: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recursively get all functions that call the given function.

        Args:
            function_id: Symbol ID of the function
            max_depth: Maximum recursion depth
            config_context: Configuration context for filtering
            call_types: Filter by specific call types
            confidence_filter: Minimum confidence level
            current_depth: Current recursion depth
            visited: Set of visited function IDs to prevent cycles

        Returns:
            List of caller relationship dictionaries
        """
        if current_depth >= max_depth:
            return []

        if visited is None:
            visited = set()

        if function_id in visited:
            return []

        visited.add(function_id)

        async with self.database.acquire() as conn:
            conditions = ["ce.callee_id = $1"]
            values: list[Any] = [function_id]
            param_count = 2

            if call_types:
                placeholders = ", ".join(
                    f"${i}" for i in range(param_count, param_count + len(call_types))
                )
                conditions.append(f"ce.call_type IN ({placeholders})")
                values.extend(list(call_types))
                param_count += len(call_types)

            if confidence_filter:
                confidence_order = {"high": 3, "medium": 2, "low": 1}
                min_confidence = confidence_order[confidence_filter]
                confidence_cases = [
                    f"WHEN ce.confidence = '{level}' THEN {order}"
                    for level, order in confidence_order.items()
                ]
                conditions.append(
                    f"(CASE {' '.join(confidence_cases)} END) >= {min_confidence}"
                )

            if config_context:
                conditions.append(
                    "(ce.config_guard IS NULL OR ce.config_guard = $"
                    + str(param_count)
                    + ")"
                )
                values.append(config_context)
                param_count += 1

            where_clause = " AND ".join(conditions)

            sql = f"""
            SELECT
                ce.id as call_edge_id,
                ce.caller_id,
                ce.callee_id,
                ce.file_path,
                ce.line_number,
                ce.column_number,
                ce.function_context,
                ce.context_before,
                ce.context_after,
                ce.call_type,
                ce.confidence,
                ce.conditional,
                ce.config_guard,
                ce.metadata as edge_metadata,
                s.name as caller_name,
                s.signature as caller_signature,
                s.file_path as caller_file,
                s.line_number as caller_line,
                s.symbol_type as caller_type,
                s.config_dependencies as caller_config_deps,
                s.metadata as caller_metadata
            FROM call_edges ce
            JOIN symbols s ON ce.caller_id = s.id
            WHERE {where_clause}
            ORDER BY ce.file_path, ce.line_number
            """

            rows = await conn.fetch(sql, *values)

            callers = []
            for row in rows:
                caller_function = {
                    "id": row["caller_id"],
                    "name": row["caller_name"],
                    "signature": row["caller_signature"],
                    "file_path": row["caller_file"],
                    "line_number": row["caller_line"],
                    "symbol_type": row["caller_type"],
                    "config_dependencies": list(row["caller_config_deps"])
                    if row["caller_config_deps"]
                    else [],
                    "metadata": dict(row["caller_metadata"])
                    if row["caller_metadata"]
                    else {},
                }

                call_edge = {
                    "id": row["call_edge_id"],
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
                    "metadata": dict(row["edge_metadata"])
                    if row["edge_metadata"]
                    else {},
                }

                relationship = {
                    "function": caller_function,
                    "call_edge": call_edge,
                    "depth": current_depth,
                }

                # Add recursive callers if depth allows
                if current_depth + 1 < max_depth:
                    relationship["callers"] = await self._get_callers_recursive(
                        row["caller_id"],
                        max_depth,
                        config_context,
                        call_types,
                        confidence_filter,
                        current_depth + 1,
                        visited.copy(),
                    )

                callers.append(relationship)

            return callers

    async def _get_callees_recursive(
        self,
        function_id: int,
        max_depth: int,
        config_context: str | None = None,
        call_types: list[CallType] | None = None,
        confidence_filter: ConfidenceLevel | None = None,
        current_depth: int = 0,
        visited: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recursively get all functions called by the given function.

        Args:
            function_id: Symbol ID of the function
            max_depth: Maximum recursion depth
            config_context: Configuration context for filtering
            call_types: Filter by specific call types
            confidence_filter: Minimum confidence level
            current_depth: Current recursion depth
            visited: Set of visited function IDs to prevent cycles

        Returns:
            List of callee relationship dictionaries
        """
        if current_depth >= max_depth:
            return []

        if visited is None:
            visited = set()

        if function_id in visited:
            return []

        visited.add(function_id)

        async with self.database.acquire() as conn:
            conditions = ["ce.caller_id = $1"]
            values: list[Any] = [function_id]
            param_count = 2

            if call_types:
                placeholders = ", ".join(
                    f"${i}" for i in range(param_count, param_count + len(call_types))
                )
                conditions.append(f"ce.call_type IN ({placeholders})")
                values.extend(list(call_types))
                param_count += len(call_types)

            if confidence_filter:
                confidence_order = {"high": 3, "medium": 2, "low": 1}
                min_confidence = confidence_order[confidence_filter]
                confidence_cases = [
                    f"WHEN ce.confidence = '{level}' THEN {order}"
                    for level, order in confidence_order.items()
                ]
                conditions.append(
                    f"(CASE {' '.join(confidence_cases)} END) >= {min_confidence}"
                )

            if config_context:
                conditions.append(
                    "(ce.config_guard IS NULL OR ce.config_guard = $"
                    + str(param_count)
                    + ")"
                )
                values.append(config_context)
                param_count += 1

            where_clause = " AND ".join(conditions)

            sql = f"""
            SELECT
                ce.id as call_edge_id,
                ce.caller_id,
                ce.callee_id,
                ce.file_path,
                ce.line_number,
                ce.column_number,
                ce.function_context,
                ce.context_before,
                ce.context_after,
                ce.call_type,
                ce.confidence,
                ce.conditional,
                ce.config_guard,
                ce.metadata as edge_metadata,
                s.name as callee_name,
                s.signature as callee_signature,
                s.file_path as callee_file,
                s.line_number as callee_line,
                s.symbol_type as callee_type,
                s.config_dependencies as callee_config_deps,
                s.metadata as callee_metadata
            FROM call_edges ce
            JOIN symbols s ON ce.callee_id = s.id
            WHERE {where_clause}
            ORDER BY ce.file_path, ce.line_number
            """

            rows = await conn.fetch(sql, *values)

            callees = []
            for row in rows:
                callee_function = {
                    "id": row["callee_id"],
                    "name": row["callee_name"],
                    "signature": row["callee_signature"],
                    "file_path": row["callee_file"],
                    "line_number": row["callee_line"],
                    "symbol_type": row["callee_type"],
                    "config_dependencies": list(row["callee_config_deps"])
                    if row["callee_config_deps"]
                    else [],
                    "metadata": dict(row["callee_metadata"])
                    if row["callee_metadata"]
                    else {},
                }

                call_edge = {
                    "id": row["call_edge_id"],
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
                    "metadata": dict(row["edge_metadata"])
                    if row["edge_metadata"]
                    else {},
                }

                relationship = {
                    "function": callee_function,
                    "call_edge": call_edge,
                    "depth": current_depth,
                }

                # Add recursive callees if depth allows
                if current_depth + 1 < max_depth:
                    relationship["callees"] = await self._get_callees_recursive(
                        row["callee_id"],
                        max_depth,
                        config_context,
                        call_types,
                        confidence_filter,
                        current_depth + 1,
                        visited.copy(),
                    )

                callees.append(relationship)

            return callees

    # ============================================================================
    # Call Path Tracing
    # ============================================================================

    async def trace_call_paths(
        self,
        from_function: str,
        to_function: str,
        max_paths: int = 3,
        max_depth: int = 5,
        config_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Find call paths between two functions using graph traversal.

        Args:
            from_function: Starting function name
            to_function: Target function name
            max_paths: Maximum number of paths to return
            max_depth: Maximum path length to consider
            config_context: Configuration context for filtering

        Returns:
            Dictionary with found paths and metadata
        """
        # Get function IDs
        from_func = await self.get_function_by_name(from_function)
        to_func = await self.get_function_by_name(to_function)

        if not from_func:
            return {
                "from_function": from_function,
                "to_function": to_function,
                "paths": [],
                "error": f"From function '{from_function}' not found",
            }

        if not to_func:
            return {
                "from_function": from_function,
                "to_function": to_function,
                "paths": [],
                "error": f"To function '{to_function}' not found",
            }

        if from_func["id"] == to_func["id"]:
            return {
                "from_function": from_function,
                "to_function": to_function,
                "paths": [],
                "error": "From and to functions are the same",
            }

        # Check for cached paths first
        cached_path = await self._get_cached_call_path(
            from_func["id"], to_func["id"], config_context
        )

        if cached_path:
            logger.debug(
                "Found cached call path",
                from_function=from_function,
                to_function=to_function,
                path_id=cached_path["id"],
            )

            # Convert cached path to full path representation
            paths = [await self._expand_cached_path(cached_path)]
            return {
                "from_function": from_function,
                "to_function": to_function,
                "from_function_info": from_func,
                "to_function_info": to_func,
                "paths": paths,
                "cached": True,
            }

        # Perform graph traversal to find new paths
        paths = await self._find_paths_dfs(
            from_func["id"],
            to_func["id"],
            max_paths,
            max_depth,
            config_context,
        )

        # Cache the best path if found
        if paths:
            best_path = max(paths, key=lambda p: p["total_confidence"])
            await self._cache_call_path(
                from_func["id"],
                to_func["id"],
                best_path["path_edge_ids"],
                best_path["total_confidence"],
                config_context,
            )

        return {
            "from_function": from_function,
            "to_function": to_function,
            "from_function_info": from_func,
            "to_function_info": to_func,
            "paths": paths,
            "cached": False,
        }

    async def _get_cached_call_path(
        self,
        from_id: int,
        to_id: int,
        config_context: str | None = None,
    ) -> dict[str, Any] | None:
        """Get a cached call path if it exists."""
        async with self.database.acquire() as conn:
            conditions = ["entry_point_id = $1", "target_function_id = $2"]
            values: list[Any] = [from_id, to_id]

            if config_context:
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

    async def _expand_cached_path(self, cached_path: dict[str, Any]) -> dict[str, Any]:
        """Expand a cached path into full call edge and function information."""
        edge_ids = cached_path["path_edge_ids"]

        if not edge_ids:
            return {
                "path_edges": [],
                "path_length": 0,
                "total_confidence": cached_path["total_confidence"],
                "config_context": cached_path["config_context"],
            }

        async with self.database.acquire() as conn:
            placeholders = ", ".join(f"${i + 1}" for i in range(len(edge_ids)))
            sql = f"""
            SELECT
                ce.id, ce.caller_id, ce.callee_id, ce.file_path, ce.line_number,
                ce.column_number, ce.function_context, ce.context_before, ce.context_after,
                ce.call_type, ce.confidence, ce.conditional, ce.config_guard,
                ce.metadata as edge_metadata,
                caller.name as caller_name, caller.signature as caller_signature,
                caller.file_path as caller_file, caller.line_number as caller_line,
                callee.name as callee_name, callee.signature as callee_signature,
                callee.file_path as callee_file, callee.line_number as callee_line
            FROM call_edges ce
            JOIN symbols caller ON ce.caller_id = caller.id
            JOIN symbols callee ON ce.callee_id = callee.id
            WHERE ce.id IN ({placeholders})
            ORDER BY array_position($#{len(edge_ids) + 1}, ce.id)
            """

            rows = await conn.fetch(sql, *edge_ids, edge_ids)

            path_edges = []
            for row in rows:
                call_edge = {
                    "caller": {
                        "name": row["caller_name"],
                        "signature": row["caller_signature"],
                        "file_path": row["caller_file"],
                        "line_number": row["caller_line"],
                    },
                    "callee": {
                        "name": row["callee_name"],
                        "signature": row["callee_signature"],
                        "file_path": row["callee_file"],
                        "line_number": row["callee_line"],
                    },
                    "call_site": {
                        "file_path": row["file_path"],
                        "line_number": row["line_number"],
                        "column_number": row["column_number"],
                        "function_context": row["function_context"],
                        "context_before": row["context_before"],
                        "context_after": row["context_after"],
                    },
                    "call_type": row["call_type"],
                    "confidence": row["confidence"],
                    "conditional": row["conditional"],
                    "config_guard": row["config_guard"],
                    "metadata": dict(row["edge_metadata"])
                    if row["edge_metadata"]
                    else {},
                }
                path_edges.append(call_edge)

            return {
                "path_edges": path_edges,
                "path_length": len(path_edges),
                "total_confidence": cached_path["total_confidence"],
                "config_context": cached_path["config_context"],
            }

    async def _find_paths_dfs(
        self,
        from_id: int,
        to_id: int,
        max_paths: int,
        max_depth: int,
        config_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find paths using depth-first search with memoization.

        Args:
            from_id: Starting function ID
            to_id: Target function ID
            max_paths: Maximum number of paths to find
            max_depth: Maximum path depth
            config_context: Configuration context for filtering

        Returns:
            List of path dictionaries
        """
        found_paths: list[dict[str, Any]] = []
        visited: set[int] = set()

        def calculate_path_confidence(edges: list[dict[str, Any]]) -> float:
            """Calculate combined confidence for a path."""
            if not edges:
                return 0.0

            confidence_values = {"high": 0.9, "medium": 0.7, "low": 0.5}
            total_confidence = 1.0

            for edge in edges:
                edge_confidence = confidence_values.get(edge["confidence"], 0.5)
                total_confidence *= edge_confidence

            return total_confidence

        async def dfs(
            current_id: int, target_id: int, path: list[dict[str, Any]], depth: int
        ) -> None:
            """Recursive DFS to find paths."""
            if len(found_paths) >= max_paths or depth >= max_depth:
                return

            if current_id == target_id and path:
                # Found a complete path
                path_confidence = calculate_path_confidence(path)
                found_paths.append(
                    {
                        "path_edges": [self._format_call_edge(edge) for edge in path],
                        "path_edge_ids": [edge["id"] for edge in path],
                        "path_length": len(path),
                        "total_confidence": path_confidence,
                        "config_context": config_context,
                    }
                )
                return

            if current_id in visited:
                return

            visited.add(current_id)

            # Get outgoing edges from current function
            async with self.database.acquire() as conn:
                conditions = ["ce.caller_id = $1"]
                values: list[Any] = [current_id]

                if config_context:
                    conditions.append(
                        "(ce.config_guard IS NULL OR ce.config_guard = $2)"
                    )
                    values.append(config_context)

                where_clause = " AND ".join(conditions)

                sql = f"""
                SELECT
                    ce.id, ce.caller_id, ce.callee_id, ce.file_path, ce.line_number,
                    ce.column_number, ce.function_context, ce.context_before, ce.context_after,
                    ce.call_type, ce.confidence, ce.conditional, ce.config_guard,
                    ce.metadata as edge_metadata,
                    caller.name as caller_name, caller.signature as caller_signature,
                    caller.file_path as caller_file, caller.line_number as caller_line,
                    callee.name as callee_name, callee.signature as callee_signature,
                    callee.file_path as callee_file, callee.line_number as callee_line
                FROM call_edges ce
                JOIN symbols caller ON ce.caller_id = caller.id
                JOIN symbols callee ON ce.callee_id = callee.id
                WHERE {where_clause}
                ORDER BY ce.confidence DESC, ce.call_type
                """

                rows = await conn.fetch(sql, *values)

                for row in rows:
                    if len(found_paths) >= max_paths:
                        break

                    edge = {
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
                        "metadata": dict(row["edge_metadata"])
                        if row["edge_metadata"]
                        else {},
                        "caller_name": row["caller_name"],
                        "caller_signature": row["caller_signature"],
                        "caller_file": row["caller_file"],
                        "caller_line": row["caller_line"],
                        "callee_name": row["callee_name"],
                        "callee_signature": row["callee_signature"],
                        "callee_file": row["callee_file"],
                        "callee_line": row["callee_line"],
                    }

                    new_path = [*path, edge]
                    await dfs(row["callee_id"], target_id, new_path, depth + 1)

            visited.remove(current_id)

        # Start DFS
        await dfs(from_id, to_id, [], 0)

        # Sort paths by confidence
        found_paths.sort(key=lambda p: float(p["total_confidence"]), reverse=True)

        return found_paths

    def _format_call_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        """Format a call edge for API response."""
        return {
            "caller": {
                "name": edge["caller_name"],
                "signature": edge["caller_signature"],
                "file_path": edge["caller_file"],
                "line_number": edge["caller_line"],
            },
            "callee": {
                "name": edge["callee_name"],
                "signature": edge["callee_signature"],
                "file_path": edge["callee_file"],
                "line_number": edge["callee_line"],
            },
            "call_site": {
                "file_path": edge["file_path"],
                "line_number": edge["line_number"],
                "column_number": edge["column_number"],
                "function_context": edge["function_context"],
                "context_before": edge["context_before"],
                "context_after": edge["context_after"],
            },
            "call_type": edge["call_type"],
            "confidence": edge["confidence"],
            "conditional": edge["conditional"],
            "config_guard": edge["config_guard"],
            "metadata": edge["metadata"],
        }

    async def _cache_call_path(
        self,
        from_id: int,
        to_id: int,
        path_edge_ids: list[int],
        total_confidence: float,
        config_context: str | None = None,
    ) -> int:
        """Cache a call path for future queries."""
        async with self.database.acquire() as conn:
            sql = """
            INSERT INTO call_paths (
                entry_point_id, target_function_id, path_edge_ids, path_length,
                total_confidence, config_context, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """

            metadata = {"cached_by": "trace_call_paths"}

            row = await conn.fetchrow(
                sql,
                from_id,
                to_id,
                path_edge_ids,
                len(path_edge_ids),
                total_confidence,
                config_context,
                json.dumps(metadata),
            )

            path_id: int = row["id"]

            logger.debug(
                "Cached call path",
                path_id=path_id,
                from_id=from_id,
                to_id=to_id,
                length=len(path_edge_ids),
                confidence=total_confidence,
            )

            return path_id

    # ============================================================================
    # Function Pointer Analysis
    # ============================================================================

    async def analyze_function_pointers(
        self,
        file_paths: list[str] | None = None,
        pointer_patterns: list[str] | None = None,
        config_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze function pointer assignments and usage patterns.

        Args:
            file_paths: Specific files to analyze (optional)
            pointer_patterns: Specific pointer patterns to search for
            config_context: Configuration context for filtering

        Returns:
            Dictionary with function pointer analysis results
        """
        async with self.database.acquire() as conn:
            conditions = []
            values: list[Any] = []
            param_count = 1

            if file_paths:
                placeholders = ", ".join(
                    f"${i}" for i in range(param_count, param_count + len(file_paths))
                )
                conditions.append(f"fp.assignment_file IN ({placeholders})")
                values.extend(file_paths)
                param_count += len(file_paths)

            if pointer_patterns:
                pattern_conditions = []
                for pattern in pointer_patterns:
                    pattern_conditions.append(f"fp.pointer_name ILIKE ${param_count}")
                    values.append(f"%{pattern}%")
                    param_count += 1
                conditions.append(f"({' OR '.join(pattern_conditions)})")

            where_clause = " AND ".join(conditions) if conditions else "TRUE"

            sql = f"""
            SELECT
                fp.id,
                fp.pointer_name,
                fp.assigned_function_id,
                fp.assignment_file,
                fp.assignment_line,
                fp.assignment_column,
                fp.struct_context,
                fp.assignment_context,
                fp.usage_sites,
                fp.metadata as fp_metadata,
                s.name as assigned_function_name,
                s.signature as assigned_function_signature,
                s.file_path as assigned_function_file,
                s.line_number as assigned_function_line,
                s.metadata as function_metadata
            FROM function_pointers fp
            LEFT JOIN symbols s ON fp.assigned_function_id = s.id
            WHERE {where_clause}
            ORDER BY fp.assignment_file, fp.assignment_line
            """

            rows = await conn.fetch(sql, *values)

            function_pointers = []
            callback_registrations = []

            for row in rows:
                pointer_data = {
                    "pointer_name": row["pointer_name"],
                    "assignment_site": {
                        "file_path": row["assignment_file"],
                        "line_number": row["assignment_line"],
                        "column_number": row["assignment_column"],
                        "context_before": None,
                        "context_after": None,
                        "function_context": row["assignment_context"],
                    },
                    "assigned_function": None,
                    "usage_sites": list(row["usage_sites"])
                    if row["usage_sites"]
                    else [],
                    "struct_context": row["struct_context"],
                    "metadata": dict(row["fp_metadata"]) if row["fp_metadata"] else {},
                }

                if row["assigned_function_id"]:
                    pointer_data["assigned_function"] = {
                        "name": row["assigned_function_name"],
                        "signature": row["assigned_function_signature"],
                        "file_path": row["assigned_function_file"],
                        "line_number": row["assigned_function_line"],
                        "symbol_type": "function",
                        "config_dependencies": [],
                    }

                function_pointers.append(pointer_data)

                # Detect callback registrations based on patterns
                if self._is_callback_registration(
                    row["pointer_name"], row["struct_context"]
                ):
                    callback_reg = {
                        "registration_site": pointer_data["assignment_site"],
                        "callback_function": pointer_data["assigned_function"],
                        "registration_pattern": self._detect_callback_pattern(
                            row["pointer_name"], row["struct_context"]
                        ),
                    }
                    callback_registrations.append(callback_reg)

            # Calculate analysis statistics
            analysis_stats = {
                "pointers_analyzed": len(function_pointers),
                "assignments_found": sum(
                    1 for fp in function_pointers if fp["assigned_function"]
                ),
                "usage_sites_found": sum(
                    len(fp["usage_sites"]) for fp in function_pointers
                ),
                "callback_patterns_matched": len(callback_registrations),
            }

            return {
                "function_pointers": function_pointers,
                "callback_registrations": callback_registrations,
                "analysis_stats": analysis_stats,
            }

    def _is_callback_registration(
        self, pointer_name: str, struct_context: str | None
    ) -> bool:
        """Detect if a function pointer assignment represents a callback registration."""
        callback_patterns = [
            "operations",
            "ops",
            "callbacks",
            "handlers",
            "hooks",
            "vtable",
            "dispatch",
        ]

        name_lower = pointer_name.lower()
        struct_lower = (struct_context or "").lower()

        return any(
            pattern in name_lower or pattern in struct_lower
            for pattern in callback_patterns
        )

    def _detect_callback_pattern(
        self, pointer_name: str, struct_context: str | None
    ) -> str:
        """Detect the specific callback pattern type."""
        name_lower = pointer_name.lower()
        struct_lower = (struct_context or "").lower()

        patterns = {
            "file_operations": "file_operations",
            "device_operations": "device_operations",
            "network_operations": "net_ops",
            "filesystem": "fs_ops",
            "driver": "driver_ops",
            "interrupt": "irq_handler",
            "timer": "timer_handler",
            "workqueue": "work_handler",
        }

        for pattern, category in patterns.items():
            if pattern in name_lower or pattern in struct_lower:
                return category

        return "generic_callback"

    # ============================================================================
    # Statistics and Utility Queries
    # ============================================================================

    async def get_function_call_statistics(self, function_name: str) -> dict[str, Any]:
        """
        Get detailed call statistics for a specific function.

        Args:
            function_name: Name of the function to analyze

        Returns:
            Dictionary with call statistics
        """
        function = await self.get_function_by_name(function_name)
        if not function:
            return {"error": "Function not found"}

        function_id = function["id"]

        async with self.database.acquire() as conn:
            # Count callers
            caller_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_callers,
                    COUNT(DISTINCT caller_id) as unique_callers,
                    COUNT(CASE WHEN call_type = 'direct' THEN 1 END) as direct_calls,
                    COUNT(CASE WHEN call_type = 'indirect' THEN 1 END) as indirect_calls,
                    COUNT(CASE WHEN call_type = 'macro' THEN 1 END) as macro_calls,
                    COUNT(CASE WHEN conditional = true THEN 1 END) as conditional_calls
                FROM call_edges
                WHERE callee_id = $1
                """,
                function_id,
            )

            # Count callees
            callee_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_callees,
                    COUNT(DISTINCT callee_id) as unique_callees,
                    COUNT(CASE WHEN call_type = 'direct' THEN 1 END) as direct_calls,
                    COUNT(CASE WHEN call_type = 'indirect' THEN 1 END) as indirect_calls,
                    COUNT(CASE WHEN call_type = 'macro' THEN 1 END) as macro_calls,
                    COUNT(CASE WHEN conditional = true THEN 1 END) as conditional_calls
                FROM call_edges
                WHERE caller_id = $1
                """,
                function_id,
            )

            # Get file distribution
            file_stats = await conn.fetch(
                """
                SELECT
                    file_path,
                    COUNT(*) as call_count
                FROM call_edges
                WHERE caller_id = $1 OR callee_id = $1
                GROUP BY file_path
                ORDER BY call_count DESC
                LIMIT 10
                """,
                function_id,
            )

            return {
                "function": function,
                "caller_statistics": {
                    "total_callers": caller_stats["total_callers"],
                    "unique_callers": caller_stats["unique_callers"],
                    "direct_calls": caller_stats["direct_calls"],
                    "indirect_calls": caller_stats["indirect_calls"],
                    "macro_calls": caller_stats["macro_calls"],
                    "conditional_calls": caller_stats["conditional_calls"],
                },
                "callee_statistics": {
                    "total_callees": callee_stats["total_callees"],
                    "unique_callees": callee_stats["unique_callees"],
                    "direct_calls": callee_stats["direct_calls"],
                    "indirect_calls": callee_stats["indirect_calls"],
                    "macro_calls": callee_stats["macro_calls"],
                    "conditional_calls": callee_stats["conditional_calls"],
                },
                "file_distribution": [
                    {"file_path": row["file_path"], "call_count": row["call_count"]}
                    for row in file_stats
                ],
            }

    async def search_functions_by_pattern(
        self,
        name_pattern: str,
        limit: int = 50,
        include_config_deps: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for functions matching a name pattern.

        Args:
            name_pattern: Pattern to match function names (supports SQL LIKE syntax)
            limit: Maximum number of results to return
            include_config_deps: Whether to include configuration dependencies

        Returns:
            List of matching function dictionaries
        """
        async with self.database.acquire() as conn:
            sql = """
            SELECT
                s.id, s.name, s.signature, s.file_path, s.line_number,
                s.symbol_type, s.config_dependencies, s.metadata
            FROM symbols s
            WHERE s.symbol_type = 'function' AND s.name ILIKE $1
            ORDER BY s.name, s.file_path
            LIMIT $2
            """

            rows = await conn.fetch(sql, f"%{name_pattern}%", limit)

            functions = []
            for row in rows:
                function_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "signature": row["signature"],
                    "file_path": row["file_path"],
                    "line_number": row["line_number"],
                    "symbol_type": row["symbol_type"],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                }

                if include_config_deps:
                    function_data["config_dependencies"] = (
                        list(row["config_dependencies"])
                        if row["config_dependencies"]
                        else []
                    )

                functions.append(function_data)

            return functions
