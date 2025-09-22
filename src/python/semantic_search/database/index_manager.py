"""
Index management and optimization for semantic search.

Provides database index management, vector index optimization,
and performance tuning for efficient search operations.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from .connection import get_database_connection
from .index_optimizer import IndexOptimizer

logger = logging.getLogger(__name__)


class IndexStats(BaseModel):
    """Statistics for a database index."""

    index_name: str = Field(..., description="Name of the index")
    table_name: str = Field(..., description="Table the index belongs to")
    index_type: str = Field(..., description="Type of index (btree, hnsw, gin, etc.)")
    size_bytes: int = Field(..., description="Size of index in bytes")
    rows_count: int = Field(..., description="Number of rows in the table")
    last_used: str | None = Field(None, description="Last time index was used")
    usage_count: int = Field(0, description="Number of times index was used")


class VectorIndexConfig(BaseModel):
    """Configuration for pgvector HNSW index."""

    m: int = Field(16, ge=4, le=100, description="Max connections per node")
    ef_construction: int = Field(
        64, ge=1, le=1000, description="Construction time param"
    )
    ef_search: int = Field(40, ge=1, le=1000, description="Search time param")


class IndexManager:
    """
    Index management and optimization for semantic search.

    Manages database indexes, vector index optimization, and provides
    performance monitoring and tuning for efficient search operations.
    """

    def __init__(self) -> None:
        """Initialize index manager."""
        self._db = get_database_connection()
        self._optimizer = IndexOptimizer()

    async def analyze_table_stats(self, table_name: str) -> dict[str, Any]:
        """
        Analyze table statistics for optimization.

        Args:
            table_name: Name of table to analyze

        Returns:
            Dictionary with table statistics
        """
        try:
            # Get basic table stats
            stats = await self._db.fetch_one(
                """
                SELECT
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats
                WHERE tablename = $1
                ORDER BY attname
                """,
                table_name,
            )

            if not stats:
                logger.warning(f"No statistics found for table {table_name}")
                return {"error": f"Table {table_name} not found or no statistics"}

            # Get table size information
            size_info = await self._db.fetch_one(
                """
                SELECT
                    pg_size_pretty(pg_total_relation_size($1)) as total_size,
                    pg_size_pretty(pg_relation_size($1)) as table_size,
                    pg_size_pretty(pg_total_relation_size($1) - pg_relation_size($1)) as index_size,
                    (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1) as exists
                """,
                table_name,
            )

            # Get row count
            row_count = await self._db.fetch_val(f"SELECT COUNT(*) FROM {table_name}")

            return {
                "table_name": table_name,
                "row_count": row_count,
                "total_size": size_info["total_size"] if size_info else "unknown",
                "table_size": size_info["table_size"] if size_info else "unknown",
                "index_size": size_info["index_size"] if size_info else "unknown",
                "statistics_available": stats is not None,
            }

        except Exception as e:
            logger.error(f"Failed to analyze table stats for {table_name}: {e}")
            return {"error": str(e)}

    async def get_index_statistics(self) -> list[IndexStats]:
        """
        Get comprehensive index statistics.

        Returns:
            List of index statistics
        """
        try:
            # Query for index information
            results = await self._db.fetch_all(
                """
                SELECT
                    i.indexname as index_name,
                    i.tablename as table_name,
                    am.amname as index_type,
                    pg_relation_size(i.indexname::regclass) as size_bytes,
                    c.reltuples::bigint as rows_count,
                    s.idx_scan as usage_count
                FROM pg_indexes i
                JOIN pg_class c ON c.relname = i.tablename
                JOIN pg_am am ON am.oid = (
                    SELECT idx.indclass[0]
                    FROM pg_index idx
                    JOIN pg_class ic ON ic.oid = idx.indexrelid
                    WHERE ic.relname = i.indexname
                    LIMIT 1
                )
                LEFT JOIN pg_stat_user_indexes s ON s.indexrelname = i.indexname
                WHERE i.schemaname = 'public'
                AND (i.tablename LIKE '%indexed_content%'
                     OR i.tablename LIKE '%vector_embedding%'
                     OR i.tablename LIKE '%search_%')
                ORDER BY size_bytes DESC
                """
            )

            index_stats = []
            for row in results:
                stat = IndexStats(
                    index_name=row["index_name"],
                    table_name=row["table_name"],
                    index_type=row["index_type"],
                    size_bytes=row["size_bytes"] or 0,
                    rows_count=row["rows_count"] or 0,
                    usage_count=row["usage_count"] or 0,
                    last_used=None,  # This would require additional query to get actual usage time
                )
                index_stats.append(stat)

            return index_stats

        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            return []

    async def optimize_vector_index(
        self, config: VectorIndexConfig | None = None
    ) -> dict[str, Any]:
        """
        Optimize vector index parameters.

        Args:
            config: Vector index configuration

        Returns:
            Optimization results
        """
        config = config or VectorIndexConfig(
            m=16,
            ef_construction=64,
            ef_search=40,
        )

        try:
            # Check if vector extension is available
            has_vector = await self._db.fetch_val(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )

            if not has_vector:
                return {"error": "pgvector extension not available"}

            # Get current vector index information
            vector_indexes = await self._db.fetch_all(
                """
                SELECT
                    indexname,
                    tablename,
                    indexdef
                FROM pg_indexes
                WHERE indexdef LIKE '%vector%'
                AND tablename = 'vector_embedding'
                """
            )

            if not vector_indexes:
                return {"error": "No vector indexes found"}

            results = {}
            for idx in vector_indexes:
                index_name = idx["indexname"]

                # Set HNSW parameters for the index
                try:
                    # Update ef_search parameter
                    await self._db.execute(f"SET hnsw.ef_search = {config.ef_search}")

                    # Get index statistics
                    index_size = await self._db.fetch_val(
                        "SELECT pg_size_pretty(pg_relation_size($1::regclass))",
                        index_name,
                    )

                    results[index_name] = {
                        "optimized": True,
                        "ef_search": config.ef_search,
                        "size": index_size,
                        "table": idx["tablename"],
                    }

                    logger.info(f"Optimized vector index {index_name}")

                except Exception as idx_error:
                    logger.error(f"Failed to optimize index {index_name}: {idx_error}")
                    results[index_name] = {
                        "optimized": False,
                        "error": str(idx_error),
                    }

            return {"results": results, "config": config.dict()}

        except Exception as e:
            logger.error(f"Failed to optimize vector index: {e}")
            return {"error": str(e)}

    async def rebuild_index(self, index_name: str) -> dict[str, Any]:
        """
        Rebuild a specific index.

        Args:
            index_name: Name of index to rebuild

        Returns:
            Rebuild results
        """
        try:
            # Check if index exists
            exists = await self._db.fetch_val(
                "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = $1)",
                index_name,
            )

            if not exists:
                return {"error": f"Index {index_name} does not exist"}

            # Get index definition before rebuild
            index_def = await self._db.fetch_val(
                "SELECT indexdef FROM pg_indexes WHERE indexname = $1",
                index_name,
            )

            # Reindex
            await self._db.execute(f"REINDEX INDEX {index_name}")

            # Get post-rebuild statistics
            size_after = await self._db.fetch_val(
                "SELECT pg_size_pretty(pg_relation_size($1::regclass))",
                index_name,
            )

            logger.info(f"Successfully rebuilt index {index_name}")

            return {
                "success": "true",
                "index_name": index_name,
                "size_after": size_after or "unknown",
                "definition": index_def or "unknown",
            }

        except Exception as e:
            logger.error(f"Failed to rebuild index {index_name}: {e}")
            return {"error": str(e)}

    async def vacuum_analyze_tables(self) -> dict[str, Any]:
        """
        Perform VACUUM ANALYZE on semantic search tables.

        Returns:
            Vacuum results
        """
        tables = [
            "indexed_content",
            "vector_embedding",
            "search_query",
            "search_result",
        ]

        results = {}

        for table in tables:
            try:
                # Check if table exists
                exists = await self._db.fetch_val(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                    table,
                )

                if not exists:
                    results[table] = {"error": "Table does not exist"}
                    continue

                # Get size before vacuum
                size_before = await self._db.fetch_val(
                    f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))"
                )

                # Perform VACUUM ANALYZE
                await self._db.execute(f"VACUUM ANALYZE {table}")

                # Get size after vacuum
                size_after = await self._db.fetch_val(
                    f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))"
                )

                results[table] = {
                    "success": "true",
                    "size_before": size_before or "unknown",
                    "size_after": size_after or "unknown",
                }

                logger.info(f"VACUUM ANALYZE completed for {table}")

            except Exception as e:
                logger.error(f"VACUUM ANALYZE failed for {table}: {e}")
                results[table] = {"error": str(e)}

        return results

    async def check_index_usage(self) -> dict[str, Any]:
        """
        Check index usage patterns for optimization.

        Returns:
            Index usage analysis
        """
        try:
            # Get index usage statistics
            usage_stats = await self._db.fetch_all(
                """
                SELECT
                    schemaname,
                    tablename,
                    indexrelname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                AND (tablename LIKE '%indexed_content%'
                     OR tablename LIKE '%vector_embedding%'
                     OR tablename LIKE '%search_%')
                ORDER BY idx_scan DESC
                """
            )

            # Identify unused indexes
            unused_indexes = []
            low_usage_indexes = []

            for stat in usage_stats:
                if stat["idx_scan"] == 0:
                    unused_indexes.append(stat["indexrelname"])
                elif stat["idx_scan"] < 10:  # Arbitrary threshold
                    low_usage_indexes.append(
                        {
                            "name": stat["indexrelname"],
                            "table": stat["tablename"],
                            "scans": stat["idx_scan"],
                        }
                    )

            # Get table scan vs index scan ratios
            scan_ratios = await self._db.fetch_all(
                """
                SELECT
                    schemaname,
                    tablename,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch,
                    CASE
                        WHEN (seq_scan + idx_scan) > 0
                        THEN ROUND((idx_scan::float / (seq_scan + idx_scan)) * 100, 2)
                        ELSE 0
                    END as index_usage_percent
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
                AND (tablename LIKE '%indexed_content%'
                     OR tablename LIKE '%vector_embedding%'
                     OR tablename LIKE '%search_%')
                ORDER BY index_usage_percent DESC
                """
            )

            return {
                "usage_statistics": [dict(row) for row in usage_stats],
                "unused_indexes": unused_indexes,
                "low_usage_indexes": low_usage_indexes,
                "scan_ratios": [dict(row) for row in scan_ratios],
                "recommendations": self._generate_index_recommendations(
                    unused_indexes, low_usage_indexes, scan_ratios
                ),
            }

        except Exception as e:
            logger.error(f"Failed to check index usage: {e}")
            return {"error": str(e)}

    def _generate_index_recommendations(
        self,
        unused_indexes: list[str],
        low_usage_indexes: list[dict[str, Any]],
        scan_ratios: list[dict[str, Any]],
    ) -> list[str]:
        """
        Generate index optimization recommendations.

        Args:
            unused_indexes: List of unused index names
            low_usage_indexes: List of low usage indexes
            scan_ratios: Table scan ratio statistics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Unused index recommendations
        if unused_indexes:
            recommendations.append(
                f"Consider dropping {len(unused_indexes)} unused indexes: {', '.join(unused_indexes[:3])}..."
                if len(unused_indexes) > 3
                else f"Consider dropping unused indexes: {', '.join(unused_indexes)}"
            )

        # Low usage recommendations
        if low_usage_indexes:
            recommendations.append(
                f"Review {len(low_usage_indexes)} low-usage indexes for potential removal"
            )

        # Scan ratio recommendations
        for ratio in scan_ratios:
            if ratio["index_usage_percent"] < 50 and ratio["seq_scan"] > 100:
                recommendations.append(
                    f"Table {ratio['tablename']} has low index usage ({ratio['index_usage_percent']}%) - "
                    "consider adding indexes for common queries"
                )

        # Vector-specific recommendations
        recommendations.extend(
            [
                "Regularly monitor vector index performance with EXPLAIN ANALYZE",
                "Consider adjusting hnsw.ef_search parameter based on query patterns",
                "Use periodic VACUUM on vector_embedding table for optimal performance",
            ]
        )

        return recommendations

    async def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Performance metrics dictionary
        """
        try:
            # Database-wide metrics
            db_metrics = await self._db.fetch_one(
                """
                SELECT
                    pg_database_size(current_database()) as db_size,
                    (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT setting FROM pg_settings WHERE name = 'shared_buffers') as shared_buffers,
                    (SELECT setting FROM pg_settings WHERE name = 'effective_cache_size') as effective_cache_size
                """
            )

            # Cache hit ratios
            cache_hits = await self._db.fetch_one(
                """
                SELECT
                    ROUND(
                        (blks_hit::float / (blks_hit + blks_read)) * 100, 2
                    ) as cache_hit_ratio
                FROM pg_stat_database
                WHERE datname = current_database()
                """
            )

            # Index statistics
            index_stats = await self.get_index_statistics()

            # Table sizes
            table_sizes = {}
            for table in [
                "indexed_content",
                "vector_embedding",
                "search_query",
                "search_result",
            ]:
                try:
                    size = await self._db.fetch_val(
                        f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))"
                    )
                    count = await self._db.fetch_val(f"SELECT COUNT(*) FROM {table}")
                    table_sizes[table] = {"size": size, "row_count": count}
                except Exception:
                    table_sizes[table] = {"size": "unknown", "row_count": 0}

            return {
                "database_size": db_metrics["db_size"] if db_metrics else 0,
                "active_connections": db_metrics["active_connections"]
                if db_metrics
                else 0,
                "cache_hit_ratio": cache_hits["cache_hit_ratio"] if cache_hits else 0,
                "shared_buffers": db_metrics["shared_buffers"]
                if db_metrics
                else "unknown",
                "effective_cache_size": db_metrics["effective_cache_size"]
                if db_metrics
                else "unknown",
                "index_count": len(index_stats),
                "total_index_size": sum(stat.size_bytes for stat in index_stats),
                "table_sizes": table_sizes,
                "index_statistics": [stat.dict() for stat in index_stats],
            }

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    async def optimize_all_indexes(
        self, apply_optimizations: bool = False
    ) -> dict[str, Any]:
        """
        Perform comprehensive index optimization.

        Args:
            apply_optimizations: Whether to apply recommended optimizations

        Returns:
            Optimization results
        """
        try:
            results = {}

            # 1. Vacuum analyze all tables
            logger.info("Starting VACUUM ANALYZE on all tables")
            results["vacuum_results"] = await self.vacuum_analyze_tables()

            # 2. Advanced vector index optimization using new optimizer
            logger.info("Performing advanced vector index optimization")
            results[
                "advanced_vector_optimization"
            ] = await self._optimizer.optimize_index(dry_run=not apply_optimizations)

            # 3. Basic vector index optimization (legacy)
            logger.info("Optimizing vector indexes (legacy)")
            results["vector_optimization"] = await self.optimize_vector_index()

            # 4. Check index usage
            logger.info("Analyzing index usage patterns")
            results["usage_analysis"] = await self.check_index_usage()

            # 5. Get performance metrics
            logger.info("Collecting performance metrics")
            results["performance_metrics"] = await self.get_performance_metrics()

            # 6. Get index information
            logger.info("Gathering index configuration")
            results["index_info"] = await self._optimizer.get_index_info()

            # 7. Benchmark current performance
            logger.info("Benchmarking current performance")
            results[
                "performance_benchmark"
            ] = await self._optimizer.benchmark_current_index()

            # 8. Generate summary
            total_tables_vacuumed = sum(
                1
                for r in results["vacuum_results"].values()
                if isinstance(r, dict) and r.get("success")
            )

            advanced_opt = results["advanced_vector_optimization"]
            recommendations = results["usage_analysis"].get("recommendations", [])

            # Add advanced optimizer recommendations
            if "recommendations" in advanced_opt:
                recommendations.extend(
                    [
                        f"Advanced HNSW tuning: {advanced_opt['recommendations'].get('rationale', 'N/A')}",
                        f"Recommended parameters: m={advanced_opt['recommendations'].get('m', 'N/A')}, "
                        f"ef_construction={advanced_opt['recommendations'].get('ef_construction', 'N/A')}",
                    ]
                )

            results["summary"] = {
                "optimization_completed": True,
                "tables_vacuumed": total_tables_vacuumed,
                "vector_indexes_optimized": len(
                    results["vector_optimization"].get("results", {})
                ),
                "advanced_optimization_applied": apply_optimizations
                and advanced_opt.get("status") == "optimization_applied",
                "performance_meets_requirements": results["performance_benchmark"].get(
                    "meets_requirements", False
                ),
                "p95_response_time_ms": results["performance_benchmark"].get(
                    "p95_response_time_ms", "N/A"
                ),
                "recommendations": recommendations,
            }

            logger.info("Index optimization completed successfully")
            return results

        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return {"error": str(e), "optimization_completed": False}

    async def quick_performance_check(self) -> dict[str, Any]:
        """
        Perform a quick performance check of the vector search system.

        Returns:
            Performance check results
        """
        try:
            results = {}

            # Quick data analysis
            logger.info("Analyzing data characteristics")
            results[
                "data_analysis"
            ] = await self._optimizer.analyze_data_characteristics()

            # Performance benchmark
            logger.info("Running performance benchmark")
            results["benchmark"] = await self._optimizer.benchmark_current_index(
                test_queries=5
            )

            # Current index info
            logger.info("Getting current index configuration")
            results["index_config"] = await self._optimizer.get_index_info()

            # Generate recommendations
            if "error" not in results["data_analysis"]:
                results["recommendations"] = self._optimizer.recommend_hnsw_parameters(
                    results["data_analysis"]
                )

            # Summary
            meets_requirements = results["benchmark"].get("meets_requirements", False)
            p95_time = results["benchmark"].get("p95_response_time_ms", "N/A")

            results["summary"] = {
                "performance_grade": results["benchmark"].get(
                    "performance_grade", "unknown"
                ),
                "meets_600ms_requirement": meets_requirements,
                "p95_response_time_ms": p95_time,
                "total_embeddings": results["data_analysis"].get("total_embeddings", 0),
                "optimization_needed": not meets_requirements,
            }

            return results

        except Exception as e:
            logger.error(f"Quick performance check failed: {e}")
            return {"error": str(e)}
