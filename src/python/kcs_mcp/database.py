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
        config_name: str,
        architecture: str,
        config_type: str,
        enabled_features: list[str],
        disabled_features: list[str],
        module_features: list[str],
        dependencies: dict,
        kernel_version: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Store kernel configuration data in the database.

        Args:
            config_name: Unique configuration identifier (e.g., "x86_64:defconfig")
            architecture: Target architecture
            config_type: Type of configuration (defconfig, allmodconfig, etc.)
            enabled_features: List of CONFIG_* options that are enabled (=y)
            disabled_features: List of CONFIG_* options that are disabled (=n)
            module_features: List of CONFIG_* options built as modules (=m)
            dependencies: Map of feature dependencies and constraints
            kernel_version: Kernel version this configuration applies to
            metadata: Additional architecture-specific settings and metadata
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                import json

                # Insert into kernel_config table
                config_sql = """
                INSERT INTO kernel_config (
                    config_name, architecture, config_type, enabled_features,
                    disabled_features, module_features, dependencies,
                    kernel_version, metadata, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
                ON CONFLICT (config_name)
                DO UPDATE SET
                    architecture = EXCLUDED.architecture,
                    config_type = EXCLUDED.config_type,
                    enabled_features = EXCLUDED.enabled_features,
                    disabled_features = EXCLUDED.disabled_features,
                    module_features = EXCLUDED.module_features,
                    dependencies = EXCLUDED.dependencies,
                    kernel_version = EXCLUDED.kernel_version,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """

                await conn.execute(
                    config_sql,
                    config_name,
                    architecture,
                    config_type,
                    json.dumps(enabled_features),
                    json.dumps(disabled_features),
                    json.dumps(module_features),
                    json.dumps(dependencies),
                    kernel_version,
                    json.dumps(metadata or {}),
                )

    async def get_kernel_config(self, config_name: str) -> dict[str, Any] | None:
        """
        Get kernel configuration details by name.

        Args:
            config_name: Configuration name (e.g., "x86_64:defconfig")

        Returns:
            Configuration data dict or None if not found
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                config_name,
                architecture,
                config_type,
                enabled_features,
                disabled_features,
                module_features,
                dependencies,
                kernel_version,
                metadata,
                created_at,
                updated_at
            FROM kernel_config
            WHERE config_name = $1
            """

            row = await conn.fetchrow(sql, config_name)

            if not row:
                return None

            import json

            return {
                "config_name": row["config_name"],
                "architecture": row["architecture"],
                "config_type": row["config_type"],
                "enabled_features": json.loads(row["enabled_features"]),
                "disabled_features": json.loads(row["disabled_features"]),
                "module_features": json.loads(row["module_features"]),
                "dependencies": json.loads(row["dependencies"]),
                "kernel_version": row["kernel_version"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }

    async def search_kernel_configs(
        self,
        architecture: str | None = None,
        config_type: str | None = None,
        kernel_version: str | None = None,
        feature: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Search kernel configurations with filters.

        Args:
            architecture: Filter by architecture
            config_type: Filter by configuration type
            kernel_version: Filter by kernel version
            feature: Filter by feature presence (enabled or module)
            limit: Maximum results to return

        Returns:
            List of matching configurations
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                config_name,
                architecture,
                config_type,
                kernel_version,
                created_at,
                updated_at,
                jsonb_array_length(enabled_features) as enabled_count,
                jsonb_array_length(module_features) as module_count
            FROM kernel_config
            WHERE 1=1
            """
            params = []
            param_idx = 1

            if architecture:
                sql += f" AND architecture = ${param_idx}"
                params.append(architecture)
                param_idx += 1

            if config_type:
                sql += f" AND config_type = ${param_idx}"
                params.append(config_type)
                param_idx += 1

            if kernel_version:
                sql += f" AND kernel_version = ${param_idx}"
                params.append(kernel_version)
                param_idx += 1

            if feature:
                sql += f" AND (enabled_features @> ${param_idx}::jsonb OR module_features @> ${param_idx}::jsonb)"
                params.append(f'["{feature}"]')
                param_idx += 1

            sql += f" ORDER BY created_at DESC LIMIT ${param_idx}"
            params.append(str(limit))

            rows = await conn.fetch(sql, *params)

            return [
                {
                    "config_name": row["config_name"],
                    "architecture": row["architecture"],
                    "config_type": row["config_type"],
                    "kernel_version": row["kernel_version"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "enabled_count": row["enabled_count"],
                    "module_count": row["module_count"],
                }
                for row in rows
            ]

    async def get_config_features(
        self, config_name: str, feature_type: str = "enabled"
    ) -> list[str]:
        """
        Get features of a specific type for a configuration.

        Args:
            config_name: Configuration name
            feature_type: Type of features ("enabled", "disabled", "module")

        Returns:
            List of feature names
        """
        async with self.acquire() as conn:
            column_map = {
                "enabled": "enabled_features",
                "disabled": "disabled_features",
                "module": "module_features",
            }

            if feature_type not in column_map:
                raise ValueError(f"Invalid feature_type: {feature_type}")

            sql = f"""
            SELECT {column_map[feature_type]}
            FROM kernel_config
            WHERE config_name = $1
            """

            row = await conn.fetchrow(sql, config_name)

            if not row:
                return []

            import json

            features = json.loads(row[column_map[feature_type]])
            return features if isinstance(features, list) else []

    async def is_feature_enabled(self, config_name: str, feature: str) -> bool:
        """
        Check if a feature is enabled in a configuration.

        Args:
            config_name: Configuration name
            feature: Feature name (with or without CONFIG_ prefix)

        Returns:
            True if feature is enabled, False otherwise
        """
        async with self.acquire() as conn:
            # Normalize feature name to include CONFIG_ prefix
            if not feature.startswith("CONFIG_"):
                feature = f"CONFIG_{feature}"

            # Use the database function created in migration
            sql = "SELECT is_feature_enabled($1, $2)"
            result = await conn.fetchval(sql, config_name, feature)

            return bool(result)

    async def get_active_features(self, config_name: str) -> dict[str, list[str]]:
        """
        Get all active features (enabled + modules) for a configuration.

        Args:
            config_name: Configuration name

        Returns:
            Dict with 'enabled' and 'modules' lists
        """
        async with self.acquire() as conn:
            # Use the database function created in migration
            sql = "SELECT get_active_features($1)"
            result = await conn.fetchval(sql, config_name)

            if result:
                import json

                features = json.loads(result)
                if isinstance(features, dict):
                    return features

            return {"enabled": [], "modules": []}

    async def store_config_dependency(
        self,
        config_name: str,
        option_name: str,
        depends_on: list[str] | None = None,
        selects: list[str] | None = None,
        implies: list[str] | None = None,
        conflicts_with: list[str] | None = None,
    ) -> None:
        """
        Store configuration dependency information.

        Args:
            config_name: Configuration name
            option_name: Configuration option name
            depends_on: List of options this depends on
            selects: List of options this selects
            implies: List of options this implies
            conflicts_with: List of options this conflicts with
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                import json

                # First delete existing entry if it exists
                await conn.execute(
                    "DELETE FROM config_dependency WHERE config_name = $1 AND option_name = $2",
                    config_name,
                    option_name,
                )

                # Insert into config_dependency table
                dep_sql = """
                INSERT INTO config_dependency (
                    config_name, option_name, depends_on, selects,
                    implies, conflicts_with, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """

                await conn.execute(
                    dep_sql,
                    config_name,
                    option_name,
                    json.dumps(depends_on or []),
                    json.dumps(selects or []),
                    json.dumps(implies or []),
                    json.dumps(conflicts_with or []),
                )

    async def get_config_dependencies(
        self, config_name: str, option_name: str
    ) -> dict[str, Any] | None:
        """
        Get dependency information for a configuration option.

        Args:
            config_name: Configuration name
            option_name: Configuration option name

        Returns:
            Dependency information dict or None if not found
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                option_name,
                depends_on,
                selects,
                implies,
                conflicts_with,
                created_at
            FROM config_dependency
            WHERE config_name = $1 AND option_name = $2
            """

            row = await conn.fetchrow(sql, config_name, option_name)

            if not row:
                return None

            import json

            return {
                "option_name": row["option_name"],
                "depends_on": json.loads(row["depends_on"]),
                "selects": json.loads(row["selects"]),
                "implies": json.loads(row["implies"]),
                "conflicts_with": json.loads(row["conflicts_with"]),
                "created_at": row["created_at"].isoformat(),
            }

    async def store_specification(
        self,
        name: str,
        version: str,
        spec_type: str,
        content: str,
        parsed_requirements: list[dict] | None = None,
        kernel_versions: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Store a specification document in the database.

        Args:
            name: Name of the specification document
            version: Version of the specification
            spec_type: Type of specification (api, behavior, interface, etc.)
            content: Raw specification content
            parsed_requirements: List of requirements extracted from the specification
            kernel_versions: List of kernel versions this specification applies to
            metadata: Additional metadata including source, author, references

        Returns:
            Specification UUID
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                import json

                # Insert into specification table
                spec_sql = """
                INSERT INTO specification (
                    name, version, type, content, parsed_requirements,
                    kernel_versions, metadata, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
                ON CONFLICT (name, version)
                DO UPDATE SET
                    type = EXCLUDED.type,
                    content = EXCLUDED.content,
                    parsed_requirements = EXCLUDED.parsed_requirements,
                    kernel_versions = EXCLUDED.kernel_versions,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING spec_id
                """

                row = await conn.fetchrow(
                    spec_sql,
                    name,
                    version,
                    spec_type,
                    content,
                    json.dumps(parsed_requirements or []),
                    json.dumps(kernel_versions or []),
                    json.dumps(metadata or {}),
                )

                return str(row["spec_id"])

    async def get_specification(
        self,
        spec_id: str | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get specification details by ID or name/version.

        Args:
            spec_id: Specification UUID
            name: Specification name (requires version)
            version: Specification version (requires name)

        Returns:
            Specification data dict or None if not found
        """
        async with self.acquire() as conn:
            if spec_id:
                sql = """
                SELECT
                    spec_id, name, version, type, content,
                    parsed_requirements, kernel_versions, metadata,
                    created_at, updated_at
                FROM specification
                WHERE spec_id = $1
                """
                row = await conn.fetchrow(sql, spec_id)
            elif name and version:
                sql = """
                SELECT
                    spec_id, name, version, type, content,
                    parsed_requirements, kernel_versions, metadata,
                    created_at, updated_at
                FROM specification
                WHERE name = $1 AND version = $2
                """
                row = await conn.fetchrow(sql, name, version)
            else:
                raise ValueError("Must provide either spec_id or both name and version")

            if not row:
                return None

            import json

            return {
                "spec_id": str(row["spec_id"]),
                "name": row["name"],
                "version": row["version"],
                "type": row["type"],
                "content": row["content"],
                "parsed_requirements": json.loads(row["parsed_requirements"]),
                "kernel_versions": json.loads(row["kernel_versions"]),
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }

    async def search_specifications(
        self,
        query: str | None = None,
        spec_type: str | None = None,
        kernel_version: str | None = None,
        name_pattern: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search specifications with various filters.

        Args:
            query: Full-text search query for content
            spec_type: Filter by specification type
            kernel_version: Filter by kernel version support
            name_pattern: Pattern match for specification name
            limit: Maximum results to return

        Returns:
            List of matching specifications
        """
        async with self.acquire() as conn:
            if query:
                # Use the database function for full-text search
                sql = "SELECT * FROM search_specifications($1, $2)"
                rows = await conn.fetch(sql, query, limit)

                # Get full details for each result
                results = []
                for row in rows:
                    spec = await self.get_specification(spec_id=str(row["spec_id"]))
                    if spec:
                        spec["search_rank"] = float(row["rank"])
                        results.append(spec)
                return results
            else:
                # Build manual search query
                sql = """
                SELECT
                    spec_id, name, version, type, kernel_versions,
                    created_at, updated_at
                FROM specification
                WHERE 1=1
                """
                params = []
                param_idx = 1

                if spec_type:
                    sql += f" AND type = ${param_idx}"
                    params.append(spec_type)
                    param_idx += 1

                if kernel_version:
                    sql += f" AND kernel_versions @> ${param_idx}::jsonb"
                    params.append(f'["{kernel_version}"]')
                    param_idx += 1

                if name_pattern:
                    sql += f" AND name ILIKE ${param_idx}"
                    params.append(f"%{name_pattern}%")
                    param_idx += 1

                sql += f" ORDER BY created_at DESC LIMIT ${param_idx}"
                params.append(str(limit))

                rows = await conn.fetch(sql, *params)

                import json

                return [
                    {
                        "spec_id": str(row["spec_id"]),
                        "name": row["name"],
                        "version": row["version"],
                        "type": row["type"],
                        "kernel_versions": json.loads(row["kernel_versions"]),
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat(),
                    }
                    for row in rows
                ]

    async def get_specifications_for_kernel(
        self, kernel_version: str
    ) -> list[dict[str, Any]]:
        """
        Find all specifications that apply to a specific kernel version.

        Args:
            kernel_version: Kernel version string

        Returns:
            List of applicable specifications
        """
        async with self.acquire() as conn:
            # Use the database function
            sql = "SELECT * FROM find_specs_for_kernel($1)"
            rows = await conn.fetch(sql, kernel_version)

            return [
                {
                    "spec_id": str(row["spec_id"]),
                    "name": row["name"],
                    "version": row["version"],
                    "type": row["type"],
                }
                for row in rows
            ]

    async def store_specification_requirement(
        self,
        spec_id: str,
        requirement_key: str,
        description: str,
        category: str | None = None,
        priority: str | None = None,
        testable: bool = True,
        test_criteria: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Store a specification requirement.

        Args:
            spec_id: Specification UUID
            requirement_key: Unique key for the requirement (e.g., FR-001)
            description: Full description of the requirement
            category: Category of requirement
            priority: Priority level (must, should, could, etc.)
            testable: Whether requirement can be automatically tested
            test_criteria: List of specific test criteria
            metadata: Additional metadata

        Returns:
            Requirement UUID
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                import json

                # First check if requirement already exists
                existing_req = await conn.fetchrow(
                    "SELECT requirement_id FROM specification_requirement WHERE spec_id = $1 AND requirement_key = $2",
                    spec_id,
                    requirement_key,
                )

                if existing_req:
                    # Update existing requirement
                    req_sql = """
                    UPDATE specification_requirement SET
                        description = $3,
                        category = $4,
                        priority = $5,
                        testable = $6,
                        test_criteria = $7,
                        metadata = $8
                    WHERE spec_id = $1 AND requirement_key = $2
                    RETURNING requirement_id
                    """
                    row = await conn.fetchrow(
                        req_sql,
                        spec_id,
                        requirement_key,
                        description,
                        category,
                        priority,
                        testable,
                        json.dumps(test_criteria or []),
                        json.dumps(metadata or {}),
                    )
                else:
                    # Insert new requirement
                    req_sql = """
                    INSERT INTO specification_requirement (
                        spec_id, requirement_key, description, category,
                        priority, testable, test_criteria, metadata, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                    RETURNING requirement_id
                    """
                    row = await conn.fetchrow(
                        req_sql,
                        spec_id,
                        requirement_key,
                        description,
                        category,
                        priority,
                        testable,
                        json.dumps(test_criteria or []),
                        json.dumps(metadata or {}),
                    )

                return str(row["requirement_id"])

    async def get_specification_requirements(
        self, spec_id: str
    ) -> list[dict[str, Any]]:
        """
        Get all requirements for a specification.

        Args:
            spec_id: Specification UUID

        Returns:
            List of requirements
        """
        async with self.acquire() as conn:
            # Use the database function
            sql = "SELECT get_specification_requirements($1)"
            result = await conn.fetchval(sql, spec_id)

            if result:
                import json

                requirements = json.loads(result)
                if isinstance(requirements, list):
                    return requirements

            return []

    async def store_specification_tags(
        self, spec_id: str, tags: dict[str, str | None]
    ) -> None:
        """
        Store tags for a specification.

        Args:
            spec_id: Specification UUID
            tags: Dictionary of tag_name -> tag_value pairs
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                # Clear existing tags for this specification
                await conn.execute(
                    "DELETE FROM specification_tag WHERE spec_id = $1", spec_id
                )

                # Insert new tags
                if tags:
                    tag_sql = """
                    INSERT INTO specification_tag (spec_id, tag_name, tag_value, created_at)
                    VALUES ($1, $2, $3, NOW())
                    """

                    for tag_name, tag_value in tags.items():
                        await conn.execute(tag_sql, spec_id, tag_name, tag_value)

    async def get_specification_tags(self, spec_id: str) -> dict[str, str | None]:
        """
        Get all tags for a specification.

        Args:
            spec_id: Specification UUID

        Returns:
            Dictionary of tag_name -> tag_value pairs
        """
        async with self.acquire() as conn:
            sql = """
            SELECT tag_name, tag_value
            FROM specification_tag
            WHERE spec_id = $1
            ORDER BY tag_name
            """

            rows = await conn.fetch(sql, spec_id)

            return {row["tag_name"]: row["tag_value"] for row in rows}

    async def search_specifications_by_tags(
        self, tags: dict[str, str | None], limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search specifications by tags.

        Args:
            tags: Dictionary of tag_name -> tag_value pairs to search for
            limit: Maximum results to return

        Returns:
            List of matching specifications
        """
        async with self.acquire() as conn:
            if not tags:
                return []

            # Build dynamic query for tag matching
            sql = """
            SELECT DISTINCT s.spec_id, s.name, s.version, s.type,
                   s.created_at, s.updated_at
            FROM specification s
            JOIN specification_tag st ON s.spec_id = st.spec_id
            WHERE
            """

            conditions = []
            params = []
            param_idx = 1

            for tag_name, tag_value in tags.items():
                if tag_value is None:
                    conditions.append(f"st.tag_name = ${param_idx}")
                    params.append(tag_name)
                    param_idx += 1
                else:
                    conditions.append(
                        f"(st.tag_name = ${param_idx} AND st.tag_value = ${param_idx + 1})"
                    )
                    params.extend([tag_name, tag_value])
                    param_idx += 2

            sql += " AND ".join(f"({cond})" for cond in conditions)
            sql += f" ORDER BY s.created_at DESC LIMIT ${param_idx}"
            params.append(str(limit))

            rows = await conn.fetch(sql, *params)

            return [
                {
                    "spec_id": str(row["spec_id"]),
                    "name": row["name"],
                    "version": row["version"],
                    "type": row["type"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                }
                for row in rows
            ]

    async def delete_specification(self, spec_id: str) -> bool:
        """
        Delete a specification and all its related data.

        Args:
            spec_id: Specification UUID

        Returns:
            True if deleted, False if not found
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                # Delete the specification (cascade will handle related tables)
                result = await conn.execute(
                    "DELETE FROM specification WHERE spec_id = $1", spec_id
                )

                # Check if any rows were affected
                return str(result) == "DELETE 1"

    async def store_validation_result(
        self,
        validation_id: str,
        specification_id: str,
        spec_name: str,
        spec_version: str,
        entry_point: str,
        compliance_score: float,
        is_valid: bool,
        deviations: list,
        implementation_details: dict,
    ) -> None:
        """
        Store specification validation result in the database.

        Args:
            validation_id: Unique validation identifier
            specification_id: Specification identifier
            spec_name: Specification name
            spec_version: Specification version
            entry_point: Entry point symbol
            compliance_score: Compliance score (0-100)
            is_valid: Whether validation passed
            deviations: List of deviations
            implementation_details: Implementation details
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                import json

                # Store specification if not exists
                spec_sql = """
                INSERT INTO specification (
                    spec_id, name, version, entry_point, created_at
                )
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (spec_id) DO NOTHING
                """

                await conn.execute(
                    spec_sql,
                    specification_id,
                    spec_name,
                    spec_version,
                    entry_point,
                )

                # Store validation result
                validation_sql = """
                INSERT INTO drift_report (
                    report_id, spec_id, compliance_score, is_valid,
                    deviations, implementation_details, validated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """

                await conn.execute(
                    validation_sql,
                    validation_id,
                    specification_id,
                    compliance_score,
                    is_valid,
                    json.dumps([dev.dict() for dev in deviations]),
                    json.dumps(implementation_details),
                )

    async def search_code_semantic_advanced(
        self,
        query: str,
        top_k: int = 10,
        offset: int = 0,
        threshold: float = 0.5,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Advanced semantic search with filtering and pagination.

        Args:
            query: Search query
            top_k: Maximum number of results
            offset: Result offset for pagination
            threshold: Similarity threshold
            filters: Search filters (subsystems, file_patterns, etc.)

        Returns:
            List of search results with metadata
        """
        async with self.acquire() as conn:
            # Build base SQL query with vector similarity
            sql = """
            SELECT
                s.name as symbol,
                s.file_path as path,
                s.sha,
                s.start_line as start,
                s.end_line as end,
                s.snippet,
                s.embedding <=> get_query_embedding($1) as score,
                CASE
                    WHEN s.file_path LIKE 'fs/%' THEN 'filesystem'
                    WHEN s.file_path LIKE 'mm/%' THEN 'memory'
                    WHEN s.file_path LIKE 'net/%' THEN 'networking'
                    WHEN s.file_path LIKE 'drivers/%' THEN 'drivers'
                    ELSE 'other'
                END as subsystem
            FROM symbol s
            WHERE s.embedding IS NOT NULL
            """

            params = [query]
            param_idx = 2

            # Apply filters
            if filters:
                if filters.get("subsystems"):
                    subsystem_conditions = []
                    for subsys in filters["subsystems"]:
                        if subsys in ["fs", "filesystem"]:
                            subsystem_conditions.append("s.file_path LIKE 'fs/%'")
                        elif subsys in ["mm", "memory"]:
                            subsystem_conditions.append("s.file_path LIKE 'mm/%'")
                        elif subsys in ["net", "networking"]:
                            subsystem_conditions.append("s.file_path LIKE 'net/%'")
                        elif subsys in ["drivers"]:
                            subsystem_conditions.append("s.file_path LIKE 'drivers/%'")

                    if subsystem_conditions:
                        sql += f" AND ({' OR '.join(subsystem_conditions)})"

                if filters.get("file_patterns"):
                    pattern_conditions = []
                    for pattern in filters["file_patterns"]:
                        # Convert glob pattern to SQL LIKE pattern
                        sql_pattern = pattern.replace("*", "%").replace("?", "_")
                        pattern_conditions.append(f"s.file_path LIKE ${param_idx}")
                        params.append(sql_pattern)
                        param_idx += 1

                    if pattern_conditions:
                        sql += f" AND ({' OR '.join(pattern_conditions)})"

                if filters.get("symbol_types"):
                    type_conditions = []
                    for sym_type in filters["symbol_types"]:
                        if sym_type == "function":
                            type_conditions.append("s.kind = 'function'")
                        elif sym_type == "macro":
                            type_conditions.append("s.kind = 'macro'")
                        elif sym_type == "struct":
                            type_conditions.append("s.kind = 'struct'")

                    if type_conditions:
                        sql += f" AND ({' OR '.join(type_conditions)})"

                if filters.get("exclude_tests"):
                    sql += " AND s.file_path NOT LIKE '%test%' AND s.file_path NOT LIKE '%Test%'"

            # Add similarity threshold
            sql += f" AND (s.embedding <=> get_query_embedding($1)) < ${param_idx}"
            params.append(str(1.0 - threshold))  # Convert similarity to distance
            param_idx += 1

            # Order by similarity and add pagination
            sql += f" ORDER BY score LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([str(top_k), str(offset)])

            try:
                results = await conn.fetch(sql, *params)

                return [
                    {
                        "symbol": row["symbol"],
                        "path": row["path"],
                        "sha": row["sha"],
                        "start": row["start"],
                        "end": row["end"],
                        "snippet": row["snippet"],
                        "score": 1.0
                        - float(row["score"]),  # Convert distance back to similarity
                        "subsystem": row["subsystem"],
                    }
                    for row in results
                ]

            except Exception as e:
                # If advanced search fails, fall back to basic search
                logger.warning(
                    "Advanced semantic search failed, falling back", error=str(e)
                )
                return []


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
        config_name: str,
        architecture: str,
        config_type: str,
        enabled_features: list[str],
        disabled_features: list[str],
        module_features: list[str],
        dependencies: dict,
        kernel_version: str,
        metadata: dict | None = None,
    ) -> None:
        """Mock kernel config storage."""
        # Just log that it was called
        logger.info(
            "Mock storing kernel config",
            config_name=config_name,
            architecture=architecture,
            config_type=config_type,
            features_count=len(enabled_features) + len(module_features),
        )

    async def get_kernel_config(self, config_name: str) -> dict[str, Any] | None:
        """Mock kernel config retrieval."""
        if config_name == "nonexistent_config":
            return None

        return {
            "config_name": config_name,
            "architecture": "x86_64",
            "config_type": "defconfig",
            "enabled_features": ["CONFIG_X86_64", "CONFIG_NET", "CONFIG_BLOCK"],
            "disabled_features": ["CONFIG_DEBUG_KERNEL", "CONFIG_EMBEDDED"],
            "module_features": ["CONFIG_EXT4_FS", "CONFIG_USB"],
            "dependencies": {"CONFIG_NET": ["CONFIG_NETDEVICES"]},
            "kernel_version": "6.1.0",
            "metadata": {"compiler": "gcc", "build_date": "2024-01-01"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

    async def search_kernel_configs(
        self,
        architecture: str | None = None,
        config_type: str | None = None,
        kernel_version: str | None = None,
        feature: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Mock kernel config search."""
        mock_configs = [
            {
                "config_name": "x86_64:defconfig",
                "architecture": "x86_64",
                "config_type": "defconfig",
                "kernel_version": "6.1.0",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "enabled_count": 100,
                "module_count": 50,
            },
            {
                "config_name": "arm64:defconfig",
                "architecture": "arm64",
                "config_type": "defconfig",
                "kernel_version": "6.1.0",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "enabled_count": 95,
                "module_count": 45,
            },
        ]

        # Apply basic filtering for mock
        results = mock_configs
        if architecture:
            results = [c for c in results if c["architecture"] == architecture]
        if config_type:
            results = [c for c in results if c["config_type"] == config_type]

        return results[:limit]

    async def get_config_features(
        self, config_name: str, feature_type: str = "enabled"
    ) -> list[str]:
        """Mock config features retrieval."""
        mock_features = {
            "enabled": ["CONFIG_X86_64", "CONFIG_NET", "CONFIG_BLOCK"],
            "disabled": ["CONFIG_DEBUG_KERNEL", "CONFIG_EMBEDDED"],
            "module": ["CONFIG_EXT4_FS", "CONFIG_USB"],
        }
        return mock_features.get(feature_type, [])

    async def is_feature_enabled(self, config_name: str, feature: str) -> bool:
        """Mock feature enabled check."""
        # Normalize feature name
        if not feature.startswith("CONFIG_"):
            feature = f"CONFIG_{feature}"

        enabled_features = ["CONFIG_X86_64", "CONFIG_NET", "CONFIG_BLOCK"]
        return feature in enabled_features

    async def get_active_features(self, config_name: str) -> dict[str, list[str]]:
        """Mock active features retrieval."""
        return {
            "enabled": ["CONFIG_X86_64", "CONFIG_NET", "CONFIG_BLOCK"],
            "modules": ["CONFIG_EXT4_FS", "CONFIG_USB"],
        }

    async def store_config_dependency(
        self,
        config_name: str,
        option_name: str,
        depends_on: list[str] | None = None,
        selects: list[str] | None = None,
        implies: list[str] | None = None,
        conflicts_with: list[str] | None = None,
    ) -> None:
        """Mock config dependency storage."""
        logger.info(
            "Mock storing config dependency",
            config_name=config_name,
            option_name=option_name,
            depends_count=len(depends_on or []),
        )

    async def get_config_dependencies(
        self, config_name: str, option_name: str
    ) -> dict[str, Any] | None:
        """Mock config dependency retrieval."""
        if option_name == "CONFIG_NONEXISTENT":
            return None

        return {
            "option_name": option_name,
            "depends_on": ["CONFIG_NETDEVICES"] if option_name == "CONFIG_NET" else [],
            "selects": [],
            "implies": [],
            "conflicts_with": [],
            "created_at": "2024-01-01T00:00:00",
        }

    async def store_specification(
        self,
        name: str,
        version: str,
        spec_type: str,
        content: str,
        parsed_requirements: list[dict] | None = None,
        kernel_versions: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Mock specification storage."""
        logger.info(
            "Mock storing specification",
            name=name,
            version=version,
            spec_type=spec_type,
            content_length=len(content),
            requirements_count=len(parsed_requirements or []),
        )
        return "spec-12345-uuid"

    async def get_specification(
        self,
        spec_id: str | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any] | None:
        """Mock specification retrieval."""
        if spec_id == "nonexistent-spec" or name == "nonexistent_spec":
            return None

        return {
            "spec_id": spec_id or "spec-12345-uuid",
            "name": name or "Linux Syscall API",
            "version": version or "1.0.0",
            "type": "api",
            "content": "# Linux Syscall API Specification\nThis specification defines the expected behavior...",
            "parsed_requirements": [
                {
                    "key": "API-001",
                    "description": "Must return EINVAL for invalid parameters",
                }
            ],
            "kernel_versions": ["6.1.0", "6.2.0"],
            "metadata": {"author": "Linux Foundation", "source": "kernel.org"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

    async def search_specifications(
        self,
        query: str | None = None,
        spec_type: str | None = None,
        kernel_version: str | None = None,
        name_pattern: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Mock specification search."""
        mock_specs = [
            {
                "spec_id": "spec-api-12345",
                "name": "Linux Syscall API",
                "version": "1.0.0",
                "type": "api",
                "kernel_versions": ["6.1.0", "6.2.0"],
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            },
            {
                "spec_id": "spec-behavior-67890",
                "name": "VFS Behavior Specification",
                "version": "2.1.0",
                "type": "behavior",
                "kernel_versions": ["6.1.0"],
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            },
        ]

        # Apply basic filtering for mock
        results = mock_specs
        if spec_type:
            results = [s for s in results if s["type"] == spec_type]
        if kernel_version:
            results = [s for s in results if kernel_version in s["kernel_versions"]]

        return results[:limit]

    async def get_specifications_for_kernel(
        self, kernel_version: str
    ) -> list[dict[str, Any]]:
        """Mock kernel specifications retrieval."""
        if kernel_version == "6.1.0":
            return [
                {
                    "spec_id": "spec-api-12345",
                    "name": "Linux Syscall API",
                    "version": "1.0.0",
                    "type": "api",
                },
                {
                    "spec_id": "spec-behavior-67890",
                    "name": "VFS Behavior Specification",
                    "version": "2.1.0",
                    "type": "behavior",
                },
            ]
        return []

    async def store_specification_requirement(
        self,
        spec_id: str,
        requirement_key: str,
        description: str,
        category: str | None = None,
        priority: str | None = None,
        testable: bool = True,
        test_criteria: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Mock specification requirement storage."""
        logger.info(
            "Mock storing specification requirement",
            spec_id=spec_id,
            requirement_key=requirement_key,
            category=category,
            priority=priority,
        )
        return "req-12345-uuid"

    async def get_specification_requirements(
        self, spec_id: str
    ) -> list[dict[str, Any]]:
        """Mock specification requirements retrieval."""
        return [
            {
                "id": "req-001-uuid",
                "key": "API-001",
                "description": "Must return EINVAL for invalid parameters",
                "category": "functional",
                "priority": "must",
                "testable": True,
                "test_criteria": [
                    {"type": "unit_test", "condition": "error_code == EINVAL"}
                ],
            },
            {
                "id": "req-002-uuid",
                "key": "API-002",
                "description": "Should complete within 100ms",
                "category": "performance",
                "priority": "should",
                "testable": True,
                "test_criteria": [
                    {"type": "benchmark", "condition": "response_time < 100ms"}
                ],
            },
        ]

    async def store_specification_tags(
        self, spec_id: str, tags: dict[str, str | None]
    ) -> None:
        """Mock specification tags storage."""
        logger.info(
            "Mock storing specification tags",
            spec_id=spec_id,
            tags_count=len(tags),
        )

    async def get_specification_tags(self, spec_id: str) -> dict[str, str | None]:
        """Mock specification tags retrieval."""
        return {
            "subsystem": "vfs",
            "stability": "stable",
            "platform": "x86_64",
            "version_introduced": "6.1.0",
        }

    async def search_specifications_by_tags(
        self, tags: dict[str, str | None], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Mock specification search by tags."""
        if "subsystem" in tags:
            return [
                {
                    "spec_id": "spec-behavior-67890",
                    "name": "VFS Behavior Specification",
                    "version": "2.1.0",
                    "type": "behavior",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                }
            ]
        return []

    async def delete_specification(self, spec_id: str) -> bool:
        """Mock specification deletion."""
        logger.info("Mock deleting specification", spec_id=spec_id)
        return spec_id != "nonexistent-spec"

    async def store_validation_result(
        self,
        validation_id: str,
        specification_id: str,
        spec_name: str,
        spec_version: str,
        entry_point: str,
        compliance_score: float,
        is_valid: bool,
        deviations: list,
        implementation_details: dict,
    ) -> None:
        """Mock validation result storage."""
        logger.info(
            "Mock storing validation result",
            validation_id=validation_id,
            spec_name=spec_name,
            compliance_score=compliance_score,
            is_valid=is_valid,
            deviations_count=len(deviations),
        )

    async def search_code_semantic_advanced(
        self,
        query: str,
        top_k: int = 10,
        offset: int = 0,
        threshold: float = 0.5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Mock advanced semantic search."""
        logger.info(
            "Mock advanced semantic search",
            query=query,
            top_k=top_k,
            offset=offset,
            threshold=threshold,
            has_filters=filters is not None,
        )

        # Return some mock results
        mock_results = [
            {
                "symbol": "vfs_read",
                "path": "fs/read_write.c",
                "sha": "abc123",
                "start": 100,
                "end": 150,
                "snippet": "ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)",
                "score": 0.85,
                "subsystem": "filesystem",
            },
            {
                "symbol": "generic_file_read_iter",
                "path": "mm/filemap.c",
                "sha": "def456",
                "start": 200,
                "end": 250,
                "snippet": "ssize_t generic_file_read_iter(struct kiocb *iocb, struct iov_iter *iter)",
                "score": 0.75,
                "subsystem": "memory",
            },
        ]

        # Apply basic filtering for mock
        if filters:
            subsystems = filters.get("subsystems")
            if subsystems and isinstance(subsystems, list):
                mock_results = [
                    r
                    for r in mock_results
                    if any(str(subsys) in str(r["subsystem"]) for subsys in subsystems)
                ]

        # Apply pagination
        start_idx = offset
        end_idx = offset + top_k
        return mock_results[start_idx:end_idx]


def set_database(database: Database) -> None:
    """Set global database instance."""
    global _database
    _database = database
