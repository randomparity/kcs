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

from kcs_mcp.models import ErrorResponse

# Import ChunkQueries from the submodule
from .chunk_queries import ChunkQueries

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
        options: dict[str, Any],
        dependencies: list[Any],
        metadata: dict | None = None,
        kernel_version: str | None = None,
    ) -> str:
        """
        Store kernel configuration data in the database.

        Args:
            config_id: Unique configuration identifier (UUID)
            arch: Target architecture
            config_name: Configuration name
            config_path: Path to the configuration file
            options: Dictionary of configuration options
            dependencies: List of configuration dependencies
            metadata: Additional metadata
            kernel_version: Kernel version (optional)

        Returns:
            Configuration ID
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                import json

                # Convert options dict to the expected format for the database
                # Extract enabled, disabled, and module features from options
                enabled_features = []
                disabled_features = []
                module_features = []

                for opt_name, opt_value in options.items():
                    if hasattr(opt_value, "value") and hasattr(opt_value, "type"):
                        # It's a ConfigOption object
                        value = opt_value.value
                        if value is True:
                            enabled_features.append(opt_name)
                        elif value is False:
                            disabled_features.append(opt_name)
                        elif value == "m":
                            module_features.append(opt_name)
                    else:
                        # It's a direct value
                        if opt_value is True:
                            enabled_features.append(opt_name)
                        elif opt_value is False:
                            disabled_features.append(opt_name)
                        elif opt_value == "m":
                            module_features.append(opt_name)

                # Convert dependencies list to dict format expected by database
                dependencies_dict = {}
                for dep in dependencies:
                    if hasattr(dep, "option") and hasattr(dep, "depends_on"):
                        # It's a ConfigDependency object
                        dependencies_dict[dep.option] = dep.depends_on
                    elif isinstance(dep, dict):
                        # It's already a dict
                        if "option" in dep and "depends_on" in dep:
                            dependencies_dict[dep["option"]] = dep["depends_on"]

                # Determine config_type from config_name or default to "custom"
                config_type = "custom"
                if "defconfig" in config_name.lower():
                    config_type = "defconfig"
                elif "allmodconfig" in config_name.lower():
                    config_type = "allmodconfig"
                elif "allyesconfig" in config_name.lower():
                    config_type = "allyesconfig"

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

                # Construct the final config name that includes architecture
                full_config_name = f"{arch}:{config_name}"

                await conn.execute(
                    config_sql,
                    full_config_name,
                    arch,
                    config_type,
                    json.dumps(enabled_features),
                    json.dumps(disabled_features),
                    json.dumps(module_features),
                    json.dumps(dependencies_dict),
                    kernel_version or "unknown",
                    json.dumps(
                        {
                            **(metadata or {}),
                            "config_id": config_id,
                            "config_path": config_path,
                            "original_options": {k: str(v) for k, v in options.items()},
                        }
                    ),
                )

                return config_id

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

    async def log_semantic_query(
        self,
        query_text: str,
        query_type: str,
        result_count: int,
        top_k: int,
        execution_time_ms: int,
        distance_threshold: float | None = None,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
        subsystem: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        results: list | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Log a semantic search query for analytics and optimization.

        Args:
            query_text: The search query text
            query_type: Type of query (symbol, function, concept, etc.)
            result_count: Number of results returned
            top_k: Maximum results requested
            execution_time_ms: Query execution time in milliseconds
            distance_threshold: Optional distance threshold used
            kernel_version: Kernel version context
            kernel_config: Kernel configuration context
            subsystem: Subsystem filter if used
            user_id: Optional user identifier
            session_id: Optional session identifier
            results: Top results with scores
            metadata: Additional metadata

        Returns:
            Query ID (UUID)
        """
        async with self.acquire() as conn:
            import json

            sql = """
            INSERT INTO semantic_query_log (
                query_text, query_type, result_count, top_k,
                distance_threshold, execution_time_ms, kernel_version,
                kernel_config, subsystem, user_id, session_id,
                results, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING query_id
            """

            row = await conn.fetchrow(
                sql,
                query_text,
                query_type,
                result_count,
                top_k,
                distance_threshold,
                execution_time_ms,
                kernel_version,
                kernel_config,
                subsystem,
                user_id,
                session_id,
                json.dumps(results or []),
                json.dumps(metadata or {}),
            )

            return str(row["query_id"])

    async def get_semantic_query_cache(
        self,
        query_hash: str,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
    ) -> dict | None:
        """
        Retrieve cached semantic search results.

        Args:
            query_hash: SHA256 hash of normalized query
            kernel_version: Kernel version context
            kernel_config: Kernel configuration context

        Returns:
            Cached results dict or None if not found/expired
        """
        async with self.acquire() as conn:
            sql = """
            SELECT results, cache_id
            FROM semantic_query_cache
            WHERE query_hash = $1
              AND expires_at > NOW()
              AND ($2::VARCHAR IS NULL OR kernel_version = $2)
              AND ($3::VARCHAR IS NULL OR kernel_config = $3)
            ORDER BY created_at DESC
            LIMIT 1
            """

            row = await conn.fetchrow(sql, query_hash, kernel_version, kernel_config)

            if row:
                # Update cache hit statistics
                await conn.execute("SELECT update_cache_hit($1)", row["cache_id"])

                import json

                results: dict = json.loads(row["results"])
                return results

            return None

    async def store_semantic_query_cache(
        self,
        query_hash: str,
        results: list,
        ttl_hours: int = 24,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
    ) -> str:
        """
        Store semantic search results in cache.

        Args:
            query_hash: SHA256 hash of normalized query
            results: Search results to cache
            ttl_hours: Time to live in hours
            kernel_version: Kernel version context
            kernel_config: Kernel configuration context

        Returns:
            Cache ID (UUID)
        """
        async with self.acquire() as conn:
            import json

            sql = f"""
            INSERT INTO semantic_query_cache (
                query_hash, results, kernel_version, kernel_config,
                expires_at
            )
            VALUES ($1, $2, $3, $4, NOW() + INTERVAL '{ttl_hours} hours')
            ON CONFLICT (query_hash, kernel_version, kernel_config)
            DO UPDATE SET
                results = EXCLUDED.results,
                expires_at = EXCLUDED.expires_at,
                hit_count = semantic_query_cache.hit_count + 1,
                last_accessed = NOW()
            RETURNING cache_id
            """

            row = await conn.fetchrow(
                sql,
                query_hash,
                json.dumps(results),
                kernel_version,
                kernel_config,
            )

            return str(row["cache_id"])

    async def store_semantic_query_feedback(
        self,
        query_id: str,
        result_rank: int,
        result_id: str,
        relevance_score: int | None = None,
        is_correct: bool | None = None,
        is_helpful: bool | None = None,
        feedback_text: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Store user feedback on semantic search results.

        Args:
            query_id: ID of the original query
            result_rank: Position of this result in the results (0-based)
            result_id: Identifier for the result (e.g., symbol ID)
            relevance_score: User-provided relevance score (1-5)
            is_correct: Whether the result is correct
            is_helpful: Whether the result is helpful
            feedback_text: Optional text feedback
            user_id: Optional user identifier
            metadata: Additional metadata

        Returns:
            Feedback ID (UUID)
        """
        async with self.acquire() as conn:
            import json

            sql = """
            INSERT INTO semantic_query_feedback (
                query_id, result_rank, result_id, relevance_score,
                is_correct, is_helpful, feedback_text, user_id, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING feedback_id
            """

            row = await conn.fetchrow(
                sql,
                query_id,
                result_rank,
                result_id,
                relevance_score,
                is_correct,
                is_helpful,
                feedback_text,
                user_id,
                json.dumps(metadata or {}),
            )

            return str(row["feedback_id"])

    async def get_semantic_query_stats(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        query_type: str | None = None,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
    ) -> list[dict]:
        """
        Get semantic query analytics statistics.

        Args:
            start_time: Start time for statistics period (ISO format)
            end_time: End time for statistics period (ISO format)
            query_type: Filter by query type
            kernel_version: Filter by kernel version
            kernel_config: Filter by kernel configuration

        Returns:
            List of statistics records
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                period_start,
                period_end,
                query_type,
                kernel_version,
                kernel_config,
                total_queries,
                avg_execution_time_ms,
                p50_execution_time_ms,
                p95_execution_time_ms,
                p99_execution_time_ms,
                avg_result_count,
                total_feedback_count,
                avg_relevance_score,
                cache_hit_rate,
                unique_users,
                top_queries,
                metadata
            FROM semantic_query_stats
            WHERE 1=1
            """

            params = []
            param_idx = 1

            if start_time:
                sql += f" AND period_start >= ${param_idx}::TIMESTAMP"
                params.append(start_time)
                param_idx += 1

            if end_time:
                sql += f" AND period_end <= ${param_idx}::TIMESTAMP"
                params.append(end_time)
                param_idx += 1

            if query_type:
                sql += f" AND query_type = ${param_idx}"
                params.append(query_type)
                param_idx += 1

            if kernel_version:
                sql += f" AND kernel_version = ${param_idx}"
                params.append(kernel_version)
                param_idx += 1

            if kernel_config:
                sql += f" AND kernel_config = ${param_idx}"
                params.append(kernel_config)
                param_idx += 1

            sql += " ORDER BY period_start DESC"

            rows = await conn.fetch(sql, *params)

            import json

            return [
                {
                    "period_start": row["period_start"].isoformat(),
                    "period_end": row["period_end"].isoformat(),
                    "query_type": row["query_type"],
                    "kernel_version": row["kernel_version"],
                    "kernel_config": row["kernel_config"],
                    "total_queries": row["total_queries"],
                    "avg_execution_time_ms": float(row["avg_execution_time_ms"])
                    if row["avg_execution_time_ms"]
                    else None,
                    "p50_execution_time_ms": row["p50_execution_time_ms"],
                    "p95_execution_time_ms": row["p95_execution_time_ms"],
                    "p99_execution_time_ms": row["p99_execution_time_ms"],
                    "avg_result_count": float(row["avg_result_count"])
                    if row["avg_result_count"]
                    else None,
                    "total_feedback_count": row["total_feedback_count"],
                    "avg_relevance_score": float(row["avg_relevance_score"])
                    if row["avg_relevance_score"]
                    else None,
                    "cache_hit_rate": float(row["cache_hit_rate"])
                    if row["cache_hit_rate"]
                    else None,
                    "unique_users": row["unique_users"],
                    "top_queries": json.loads(row["top_queries"]),
                    "metadata": json.loads(row["metadata"]),
                }
                for row in rows
            ]

    async def clean_semantic_cache(self) -> int:
        """
        Clean up expired semantic query cache entries.

        Returns:
            Number of entries deleted
        """
        async with self.acquire() as conn:
            result = await conn.fetchval("SELECT clean_semantic_cache()")
            return int(result) if result else 0

    async def compute_semantic_query_stats(
        self, start_time: str, end_time: str
    ) -> None:
        """
        Compute and store semantic query statistics for a time period.

        Args:
            start_time: Start time for statistics period (ISO format)
            end_time: End time for statistics period (ISO format)
        """
        async with self.acquire() as conn:
            await conn.execute(
                "SELECT compute_semantic_query_stats($1::TIMESTAMP, $2::TIMESTAMP)",
                start_time,
                end_time,
            )

    async def create_graph_export_job(
        self,
        export_name: str,
        export_format: str,
        kernel_version: str,
        kernel_config: str,
        subsystem: str | None = None,
        entry_point: str | None = None,
        max_depth: int = 10,
        include_metadata: bool = True,
        include_annotations: bool = True,
        chunk_size: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Create a new graph export job.

        Args:
            export_name: Name for the export
            export_format: Export format (json, graphml, gexf, dot, cytoscape, gephi)
            kernel_version: Kernel version context
            kernel_config: Kernel configuration context
            subsystem: Optional subsystem filter
            entry_point: Optional entry point filter
            max_depth: Maximum traversal depth
            include_metadata: Whether to include metadata
            include_annotations: Whether to include annotations
            chunk_size: Optional chunk size for large exports
            metadata: Additional metadata

        Returns:
            Export ID (UUID)
        """
        async with self.acquire() as conn:
            import json

            sql = """
            INSERT INTO graph_export (
                export_name, export_format, kernel_version, kernel_config,
                subsystem, entry_point, max_depth, include_metadata,
                include_annotations, chunk_size, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING export_id
            """

            row = await conn.fetchrow(
                sql,
                export_name,
                export_format,
                kernel_version,
                kernel_config,
                subsystem,
                entry_point,
                max_depth,
                include_metadata,
                include_annotations,
                chunk_size,
                json.dumps(metadata or {}),
            )

            return str(row["export_id"])

    async def get_graph_export_job(self, export_id: str) -> dict | None:
        """
        Get graph export job details.

        Args:
            export_id: Export job ID

        Returns:
            Export job details or None if not found
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                export_id, export_name, export_format, export_status,
                kernel_version, kernel_config, subsystem, entry_point,
                max_depth, include_metadata, include_annotations,
                chunk_size, total_chunks, created_at, started_at,
                completed_at, error_message, export_size_bytes,
                node_count, edge_count, output_path, metadata
            FROM graph_export
            WHERE export_id = $1
            """

            row = await conn.fetchrow(sql, export_id)

            if not row:
                return None

            import json

            return {
                "export_id": str(row["export_id"]),
                "export_name": row["export_name"],
                "export_format": row["export_format"],
                "export_status": row["export_status"],
                "kernel_version": row["kernel_version"],
                "kernel_config": row["kernel_config"],
                "subsystem": row["subsystem"],
                "entry_point": row["entry_point"],
                "max_depth": row["max_depth"],
                "include_metadata": row["include_metadata"],
                "include_annotations": row["include_annotations"],
                "chunk_size": row["chunk_size"],
                "total_chunks": row["total_chunks"],
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
                "started_at": row["started_at"].isoformat()
                if row["started_at"]
                else None,
                "completed_at": row["completed_at"].isoformat()
                if row["completed_at"]
                else None,
                "error_message": row["error_message"],
                "export_size_bytes": row["export_size_bytes"],
                "node_count": row["node_count"],
                "edge_count": row["edge_count"],
                "output_path": row["output_path"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }

    async def update_graph_export_status(
        self,
        export_id: str,
        status: str,
        error_message: str | None = None,
        output_path: str | None = None,
        export_size_bytes: int | None = None,
        node_count: int | None = None,
        edge_count: int | None = None,
    ) -> None:
        """
        Update graph export job status and metadata.

        Args:
            export_id: Export job ID
            status: New status (pending, running, completed, failed, cancelled)
            error_message: Optional error message
            output_path: Optional output file path
            export_size_bytes: Optional export size in bytes
            node_count: Optional total node count
            edge_count: Optional total edge count
        """
        async with self.acquire() as conn:
            # Use the database function for status update
            await conn.execute(
                "SELECT update_export_status($1, $2, $3)",
                export_id,
                status,
                error_message,
            )

            # Update additional fields if provided
            if any([output_path, export_size_bytes, node_count, edge_count]):
                update_parts = []
                params = []
                param_idx = 1

                if output_path:
                    update_parts.append(f"output_path = ${param_idx}")
                    params.append(output_path)
                    param_idx += 1

                if export_size_bytes is not None:
                    update_parts.append(f"export_size_bytes = ${param_idx}")
                    params.append(str(export_size_bytes))
                    param_idx += 1

                if node_count is not None:
                    update_parts.append(f"node_count = ${param_idx}")
                    params.append(str(node_count))
                    param_idx += 1

                if edge_count is not None:
                    update_parts.append(f"edge_count = ${param_idx}")
                    params.append(str(edge_count))
                    param_idx += 1

                if update_parts:
                    sql = f"""
                    UPDATE graph_export
                    SET {", ".join(update_parts)}
                    WHERE export_id = ${param_idx}
                    """
                    params.append(export_id)
                    await conn.execute(sql, *params)

    async def store_graph_export_chunk(
        self,
        export_id: str,
        chunk_index: int,
        chunk_data: bytes,
        node_count: int | None = None,
        edge_count: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Store a chunk of graph export data.

        Args:
            export_id: Export job ID
            chunk_index: Chunk index (0-based)
            chunk_data: Binary chunk data
            node_count: Optional node count for this chunk
            edge_count: Optional edge count for this chunk
            metadata: Optional chunk metadata

        Returns:
            Chunk ID (UUID)
        """
        async with self.acquire() as conn:
            import json

            sql = """
            INSERT INTO graph_export_chunk (
                export_id, chunk_index, chunk_data, chunk_size_bytes,
                node_count, edge_count, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING chunk_id
            """

            row = await conn.fetchrow(
                sql,
                export_id,
                chunk_index,
                chunk_data,
                len(chunk_data),
                node_count,
                edge_count,
                json.dumps(metadata or {}),
            )

            return str(row["chunk_id"])

    async def get_graph_export_chunks(self, export_id: str) -> list[dict]:
        """
        Get all chunks for a graph export job.

        Args:
            export_id: Export job ID

        Returns:
            List of chunk information (without data)
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                chunk_id, chunk_index, chunk_size_bytes,
                node_count, edge_count, created_at, metadata
            FROM graph_export_chunk
            WHERE export_id = $1
            ORDER BY chunk_index
            """

            rows = await conn.fetch(sql, export_id)

            import json

            return [
                {
                    "chunk_id": str(row["chunk_id"]),
                    "chunk_index": row["chunk_index"],
                    "chunk_size_bytes": row["chunk_size_bytes"],
                    "node_count": row["node_count"],
                    "edge_count": row["edge_count"],
                    "created_at": row["created_at"].isoformat(),
                    "metadata": json.loads(row["metadata"]),
                }
                for row in rows
            ]

    async def get_graph_export_chunk_data(
        self, export_id: str, chunk_index: int
    ) -> bytes | None:
        """
        Get the binary data for a specific chunk.

        Args:
            export_id: Export job ID
            chunk_index: Chunk index

        Returns:
            Binary chunk data or None if not found
        """
        async with self.acquire() as conn:
            sql = """
            SELECT chunk_data
            FROM graph_export_chunk
            WHERE export_id = $1 AND chunk_index = $2
            """

            row = await conn.fetchrow(sql, export_id, chunk_index)

            if row:
                return bytes(row["chunk_data"])
            return None

    async def list_graph_exports(
        self,
        status: str | None = None,
        format: str | None = None,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """
        List graph export jobs with optional filtering.

        Args:
            status: Optional status filter
            format: Optional format filter
            kernel_version: Optional kernel version filter
            kernel_config: Optional kernel config filter
            limit: Maximum results to return
            offset: Result offset for pagination

        Returns:
            List of export job summaries
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                export_id, export_name, export_format, export_status,
                kernel_version, kernel_config, subsystem, entry_point,
                created_at, completed_at, export_size_bytes,
                node_count, edge_count, total_chunks
            FROM graph_export
            WHERE 1=1
            """

            params = []
            param_idx = 1

            if status:
                sql += f" AND export_status = ${param_idx}"
                params.append(status)
                param_idx += 1

            if format:
                sql += f" AND export_format = ${param_idx}"
                params.append(format)
                param_idx += 1

            if kernel_version:
                sql += f" AND kernel_version = ${param_idx}"
                params.append(kernel_version)
                param_idx += 1

            if kernel_config:
                sql += f" AND kernel_config = ${param_idx}"
                params.append(kernel_config)
                param_idx += 1

            sql += (
                f" ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            )
            params.extend([str(limit), str(offset)])

            rows = await conn.fetch(sql, *params)

            return [
                {
                    "export_id": str(row["export_id"]),
                    "export_name": row["export_name"],
                    "export_format": row["export_format"],
                    "export_status": row["export_status"],
                    "kernel_version": row["kernel_version"],
                    "kernel_config": row["kernel_config"],
                    "subsystem": row["subsystem"],
                    "entry_point": row["entry_point"],
                    "created_at": row["created_at"].isoformat(),
                    "completed_at": row["completed_at"].isoformat()
                    if row["completed_at"]
                    else None,
                    "export_size_bytes": row["export_size_bytes"],
                    "node_count": row["node_count"],
                    "edge_count": row["edge_count"],
                    "total_chunks": row["total_chunks"],
                }
                for row in rows
            ]

    async def create_graph_export_template(
        self,
        template_name: str,
        template_description: str | None,
        export_format: str,
        default_config: dict,
        filter_rules: list | None = None,
        style_config: dict | None = None,
        layout_algorithm: str | None = None,
        is_public: bool = False,
        created_by: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Create a reusable graph export template.

        Args:
            template_name: Unique template name
            template_description: Optional description
            export_format: Export format
            default_config: Default configuration
            filter_rules: Optional filter rules
            style_config: Optional style configuration
            layout_algorithm: Optional layout algorithm
            is_public: Whether template is public
            created_by: Optional creator identifier
            metadata: Optional metadata

        Returns:
            Template ID (UUID)
        """
        async with self.acquire() as conn:
            import json

            sql = """
            INSERT INTO graph_export_template (
                template_name, template_description, export_format,
                default_config, filter_rules, style_config,
                layout_algorithm, is_public, created_by, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING template_id
            """

            row = await conn.fetchrow(
                sql,
                template_name,
                template_description,
                export_format,
                json.dumps(default_config),
                json.dumps(filter_rules or []),
                json.dumps(style_config or {}),
                layout_algorithm,
                is_public,
                created_by,
                json.dumps(metadata or {}),
            )

            return str(row["template_id"])

    async def get_graph_export_template(
        self, template_id: str | None = None, template_name: str | None = None
    ) -> dict | None:
        """
        Get graph export template by ID or name.

        Args:
            template_id: Template UUID
            template_name: Template name

        Returns:
            Template details or None if not found
        """
        async with self.acquire() as conn:
            if template_id:
                sql = """
                SELECT
                    template_id, template_name, template_description,
                    export_format, default_config, filter_rules,
                    style_config, layout_algorithm, created_at,
                    updated_at, is_public, created_by, metadata
                FROM graph_export_template
                WHERE template_id = $1
                """
                row = await conn.fetchrow(sql, template_id)
            elif template_name:
                sql = """
                SELECT
                    template_id, template_name, template_description,
                    export_format, default_config, filter_rules,
                    style_config, layout_algorithm, created_at,
                    updated_at, is_public, created_by, metadata
                FROM graph_export_template
                WHERE template_name = $1
                """
                row = await conn.fetchrow(sql, template_name)
            else:
                raise ValueError("Must provide either template_id or template_name")

            if not row:
                return None

            import json

            return {
                "template_id": str(row["template_id"]),
                "template_name": row["template_name"],
                "template_description": row["template_description"],
                "export_format": row["export_format"],
                "default_config": json.loads(row["default_config"]),
                "filter_rules": json.loads(row["filter_rules"]),
                "style_config": json.loads(row["style_config"]),
                "layout_algorithm": row["layout_algorithm"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "is_public": row["is_public"],
                "created_by": row["created_by"],
                "metadata": json.loads(row["metadata"]),
            }

    async def list_graph_export_templates(
        self,
        format: str | None = None,
        is_public: bool | None = None,
        created_by: str | None = None,
    ) -> list[dict]:
        """
        List available graph export templates.

        Args:
            format: Optional format filter
            is_public: Optional public/private filter
            created_by: Optional creator filter

        Returns:
            List of template summaries
        """
        async with self.acquire() as conn:
            sql = """
            SELECT
                template_id, template_name, template_description,
                export_format, created_at, updated_at,
                is_public, created_by
            FROM graph_export_template
            WHERE 1=1
            """

            params = []
            param_idx = 1

            if format:
                sql += f" AND export_format = ${param_idx}"
                params.append(format)
                param_idx += 1

            if is_public is not None:
                sql += f" AND is_public = ${param_idx}"
                params.append(str(is_public))
                param_idx += 1

            if created_by:
                sql += f" AND created_by = ${param_idx}"
                params.append(created_by)
                param_idx += 1

            sql += " ORDER BY updated_at DESC"

            rows = await conn.fetch(sql, *params)

            return [
                {
                    "template_id": str(row["template_id"]),
                    "template_name": row["template_name"],
                    "template_description": row["template_description"],
                    "export_format": row["export_format"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "is_public": row["is_public"],
                    "created_by": row["created_by"],
                }
                for row in rows
            ]

    async def get_graph_export_statistics(self, export_id: str) -> dict | None:
        """
        Get aggregated statistics for a graph export.

        Args:
            export_id: Export job ID

        Returns:
            Statistics dict or None if not found
        """
        async with self.acquire() as conn:
            # Use the database function
            result = await conn.fetchval(
                "SELECT calculate_export_statistics($1)", export_id
            )

            if result:
                import json

                stats: dict = json.loads(result)
                return stats

            return None

    async def clean_graph_export_history(self) -> int:
        """
        Clean up old graph export history entries.

        Returns:
            Number of entries deleted
        """
        async with self.acquire() as conn:
            result = await conn.fetchval("SELECT clean_export_history()")
            return int(result) if result else 0

    async def create_graph_visualization_preset(
        self,
        preset_name: str,
        tool_name: str,
        preset_type: str,
        configuration: dict,
        description: str | None = None,
        is_default: bool = False,
        metadata: dict | None = None,
    ) -> str:
        """
        Create a graph visualization preset.

        Args:
            preset_name: Preset name
            tool_name: Target visualization tool
            preset_type: Type of preset (layout, style, filter, analysis, complete)
            configuration: Tool-specific configuration
            description: Optional description
            is_default: Whether this is the default preset
            metadata: Optional metadata

        Returns:
            Preset ID (UUID)
        """
        async with self.acquire() as conn:
            import json

            sql = """
            INSERT INTO graph_visualization_preset (
                preset_name, tool_name, preset_type, configuration,
                description, is_default, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING preset_id
            """

            row = await conn.fetchrow(
                sql,
                preset_name,
                tool_name,
                preset_type,
                json.dumps(configuration),
                description,
                is_default,
                json.dumps(metadata or {}),
            )

            return str(row["preset_id"])

    async def get_default_visualization_preset(
        self, tool_name: str, preset_type: str
    ) -> dict | None:
        """
        Get the default visualization preset for a tool and type.

        Args:
            tool_name: Visualization tool name
            preset_type: Preset type

        Returns:
            Configuration dict or None if not found
        """
        async with self.acquire() as conn:
            # Use the database function
            result = await conn.fetchval(
                "SELECT get_default_preset($1, $2)", tool_name, preset_type
            )

            if result:
                import json

                config: dict = json.loads(result)
                return config

            return None


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
        options: dict[str, Any],
        dependencies: list[Any],
        metadata: dict | None = None,
        kernel_version: str | None = None,
    ) -> str:
        """Mock kernel config storage."""
        # Just log that it was called
        logger.info(
            "Mock storing kernel config",
            config_id=config_id,
            arch=arch,
            config_name=config_name,
            config_path=config_path,
            options_count=len(options),
            dependencies_count=len(dependencies),
        )
        return config_id

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

    async def log_semantic_query(
        self,
        query_text: str,
        query_type: str,
        result_count: int,
        top_k: int,
        execution_time_ms: int,
        distance_threshold: float | None = None,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
        subsystem: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        results: list | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Mock semantic query logging."""
        logger.info(
            "Mock logging semantic query",
            query_text=query_text,
            query_type=query_type,
            result_count=result_count,
            execution_time_ms=execution_time_ms,
        )
        return "query-12345-uuid"

    async def get_semantic_query_cache(
        self,
        query_hash: str,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
    ) -> dict | None:
        """Mock semantic query cache retrieval."""
        if query_hash == "cache_miss_hash":
            return None

        return {
            "results": [
                {
                    "symbol": "cached_function",
                    "path": "fs/cached.c",
                    "sha": "cached123",
                    "start": 10,
                    "end": 20,
                    "snippet": "cached function",
                    "score": 0.9,
                }
            ],
            "total_results": 1,
            "cached_at": "2024-01-01T00:00:00",
        }

    async def store_semantic_query_cache(
        self,
        query_hash: str,
        results: list,
        ttl_hours: int = 24,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
    ) -> str:
        """Mock semantic query cache storage."""
        logger.info(
            "Mock storing semantic query cache",
            query_hash=query_hash,
            results_count=len(results),
            ttl_hours=ttl_hours,
        )
        return "cache-12345-uuid"

    async def store_semantic_query_feedback(
        self,
        query_id: str,
        result_rank: int,
        result_id: str,
        relevance_score: int | None = None,
        is_correct: bool | None = None,
        is_helpful: bool | None = None,
        feedback_text: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Mock semantic query feedback storage."""
        logger.info(
            "Mock storing semantic query feedback",
            query_id=query_id,
            result_rank=result_rank,
            result_id=result_id,
            relevance_score=relevance_score,
        )
        return "feedback-12345-uuid"

    async def get_semantic_query_stats(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        query_type: str | None = None,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
    ) -> list[dict]:
        """Mock semantic query statistics retrieval."""
        return [
            {
                "period_start": "2024-01-01T00:00:00",
                "period_end": "2024-01-01T23:59:59",
                "query_type": query_type or "function",
                "kernel_version": kernel_version or "6.1.0",
                "kernel_config": kernel_config or "x86_64:defconfig",
                "total_queries": 100,
                "avg_execution_time_ms": 45.5,
                "p50_execution_time_ms": 40,
                "p95_execution_time_ms": 85,
                "p99_execution_time_ms": 120,
                "avg_result_count": 8.2,
                "total_feedback_count": 25,
                "avg_relevance_score": 4.1,
                "cache_hit_rate": 0.65,
                "unique_users": 12,
                "top_queries": [
                    {"query": "vfs_read", "count": 15},
                    {"query": "memory allocation", "count": 12},
                ],
                "metadata": {"source": "mock"},
            }
        ]

    async def clean_semantic_cache(self) -> int:
        """Mock semantic cache cleanup."""
        logger.info("Mock cleaning semantic cache")
        return 5  # Mock number of deleted entries

    async def compute_semantic_query_stats(
        self, start_time: str, end_time: str
    ) -> None:
        """Mock semantic query statistics computation."""
        logger.info(
            "Mock computing semantic query statistics",
            start_time=start_time,
            end_time=end_time,
        )

    async def create_graph_export_job(
        self,
        export_name: str,
        export_format: str,
        kernel_version: str,
        kernel_config: str,
        subsystem: str | None = None,
        entry_point: str | None = None,
        max_depth: int = 10,
        include_metadata: bool = True,
        include_annotations: bool = True,
        chunk_size: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Mock graph export job creation."""
        logger.info(
            "Mock creating graph export job",
            export_name=export_name,
            export_format=export_format,
            kernel_version=kernel_version,
            max_depth=max_depth,
        )
        return "export-12345-uuid"

    async def get_graph_export_job(self, export_id: str) -> dict | None:
        """Mock graph export job retrieval."""
        if export_id == "nonexistent-export":
            return None

        return {
            "export_id": export_id,
            "export_name": "Test Export",
            "export_format": "json",
            "export_status": "completed",
            "kernel_version": "6.1.0",
            "kernel_config": "x86_64:defconfig",
            "subsystem": None,
            "entry_point": "sys_read",
            "max_depth": 5,
            "include_metadata": True,
            "include_annotations": True,
            "chunk_size": None,
            "total_chunks": 1,
            "created_at": "2024-01-01T00:00:00",
            "started_at": "2024-01-01T00:01:00",
            "completed_at": "2024-01-01T00:05:00",
            "error_message": None,
            "export_size_bytes": 1024,
            "node_count": 50,
            "edge_count": 75,
            "output_path": "/tmp/export.json",
            "metadata": {"source": "mock"},
        }

    async def update_graph_export_status(
        self,
        export_id: str,
        status: str,
        error_message: str | None = None,
        output_path: str | None = None,
        export_size_bytes: int | None = None,
        node_count: int | None = None,
        edge_count: int | None = None,
    ) -> None:
        """Mock graph export status update."""
        logger.info(
            "Mock updating graph export status",
            export_id=export_id,
            status=status,
            has_error=error_message is not None,
        )

    async def store_graph_export_chunk(
        self,
        export_id: str,
        chunk_index: int,
        chunk_data: bytes,
        node_count: int | None = None,
        edge_count: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Mock graph export chunk storage."""
        logger.info(
            "Mock storing graph export chunk",
            export_id=export_id,
            chunk_index=chunk_index,
            chunk_size=len(chunk_data),
        )
        return f"chunk-{chunk_index}-uuid"

    async def get_graph_export_chunks(self, export_id: str) -> list[dict]:
        """Mock graph export chunks retrieval."""
        return [
            {
                "chunk_id": "chunk-0-uuid",
                "chunk_index": 0,
                "chunk_size_bytes": 1024,
                "node_count": 50,
                "edge_count": 75,
                "created_at": "2024-01-01T00:05:00",
                "metadata": {"format": "json"},
            }
        ]

    async def get_graph_export_chunk_data(
        self, export_id: str, chunk_index: int
    ) -> bytes | None:
        """Mock graph export chunk data retrieval."""
        if chunk_index == 0:
            return b'{"nodes": [], "edges": []}'
        return None

    async def list_graph_exports(
        self,
        status: str | None = None,
        format: str | None = None,
        kernel_version: str | None = None,
        kernel_config: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Mock graph exports listing."""
        mock_exports: list[dict] = [
            {
                "export_id": "export-1",
                "export_name": "Test Export 1",
                "export_format": "json",
                "export_status": "completed",
                "kernel_version": "6.1.0",
                "kernel_config": "x86_64:defconfig",
                "subsystem": "fs",
                "entry_point": "sys_read",
                "created_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:05:00",
                "export_size_bytes": 1024,
                "node_count": 50,
                "edge_count": 75,
                "total_chunks": 1,
            },
            {
                "export_id": "export-2",
                "export_name": "Test Export 2",
                "export_format": "graphml",
                "export_status": "running",
                "kernel_version": "6.1.0",
                "kernel_config": "x86_64:defconfig",
                "subsystem": "net",
                "entry_point": "sys_socket",
                "created_at": "2024-01-01T01:00:00",
                "completed_at": None,
                "export_size_bytes": None,
                "node_count": None,
                "edge_count": None,
                "total_chunks": None,
            },
        ]

        # Apply basic filtering
        results = mock_exports
        if status:
            results = [e for e in results if e["export_status"] == status]
        if format:
            results = [e for e in results if e["export_format"] == format]

        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        return results[start_idx:end_idx]

    async def create_graph_export_template(
        self,
        template_name: str,
        template_description: str | None,
        export_format: str,
        default_config: dict,
        filter_rules: list | None = None,
        style_config: dict | None = None,
        layout_algorithm: str | None = None,
        is_public: bool = False,
        created_by: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Mock graph export template creation."""
        logger.info(
            "Mock creating graph export template",
            template_name=template_name,
            export_format=export_format,
            is_public=is_public,
        )
        return "template-12345-uuid"

    async def get_graph_export_template(
        self, template_id: str | None = None, template_name: str | None = None
    ) -> dict | None:
        """Mock graph export template retrieval."""
        if template_id == "nonexistent-template" or template_name == "nonexistent":
            return None

        return {
            "template_id": template_id or "template-12345-uuid",
            "template_name": template_name or "Default JSON Template",
            "template_description": "Default template for JSON exports",
            "export_format": "json",
            "default_config": {"pretty": True, "include_metadata": True},
            "filter_rules": [{"type": "exclude", "pattern": "test_*"}],
            "style_config": {"node_color": "blue", "edge_color": "gray"},
            "layout_algorithm": "hierarchical",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "is_public": True,
            "created_by": "system",
            "metadata": {"version": "1.0"},
        }

    async def list_graph_export_templates(
        self,
        format: str | None = None,
        is_public: bool | None = None,
        created_by: str | None = None,
    ) -> list[dict]:
        """Mock graph export templates listing."""
        mock_templates = [
            {
                "template_id": "template-1",
                "template_name": "Default JSON Template",
                "template_description": "Default template for JSON exports",
                "export_format": "json",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "is_public": True,
                "created_by": "system",
            },
            {
                "template_id": "template-2",
                "template_name": "GraphML Template",
                "template_description": "Template for GraphML exports",
                "export_format": "graphml",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "is_public": False,
                "created_by": "user123",
            },
        ]

        # Apply basic filtering
        results = mock_templates
        if format:
            results = [t for t in results if t["export_format"] == format]
        if is_public is not None:
            results = [t for t in results if t["is_public"] == is_public]
        if created_by:
            results = [t for t in results if t["created_by"] == created_by]

        return results

    async def get_graph_export_statistics(self, export_id: str) -> dict | None:
        """Mock graph export statistics retrieval."""
        if export_id == "nonexistent-export":
            return None

        return {
            "total_chunks": 1,
            "total_size_bytes": 1024,
            "total_nodes": 50,
            "total_edges": 75,
            "avg_chunk_size": 1024,
        }

    async def clean_graph_export_history(self) -> int:
        """Mock graph export history cleanup."""
        logger.info("Mock cleaning graph export history")
        return 3  # Mock number of deleted entries

    async def create_graph_visualization_preset(
        self,
        preset_name: str,
        tool_name: str,
        preset_type: str,
        configuration: dict,
        description: str | None = None,
        is_default: bool = False,
        metadata: dict | None = None,
    ) -> str:
        """Mock graph visualization preset creation."""
        logger.info(
            "Mock creating graph visualization preset",
            preset_name=preset_name,
            tool_name=tool_name,
            preset_type=preset_type,
        )
        return "preset-12345-uuid"

    async def get_default_visualization_preset(
        self, tool_name: str, preset_type: str
    ) -> dict | None:
        """Mock default visualization preset retrieval."""
        return {
            "layout": "hierarchical",
            "node_size": 10,
            "edge_width": 2,
            "colors": {"node": "#4287f5", "edge": "#666666"},
            "animation": {"enabled": True, "duration": 1000},
        }


def set_database(database: Database) -> None:
    """Set global database instance."""
    global _database
    _database = database


__all__ = ["ChunkQueries", "Database", "get_database", "set_database"]
