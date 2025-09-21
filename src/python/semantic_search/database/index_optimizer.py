"""
Index optimization module for vector search performance tuning.

Provides dynamic HNSW index parameter optimization based on data characteristics,
performance requirements, and hardware constraints.
"""

import logging
import time
from typing import Any

from .connection import get_database_connection


class IndexOptimizer:
    """
    Optimizer for pgvector HNSW index parameters.

    Dynamically tunes index parameters based on:
    - Dataset size and characteristics
    - Query performance requirements (p95 ≤ 600ms target)
    - Hardware constraints (CPU-only operation)
    - Memory usage considerations
    """

    def __init__(self) -> None:
        """Initialize index optimizer."""
        self.logger = logging.getLogger(__name__)

    async def analyze_data_characteristics(self) -> dict[str, Any]:
        """
        Analyze dataset characteristics to inform index optimization.

        Returns:
            Dictionary with dataset analysis results
        """
        try:
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                # Get basic statistics
                stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_embeddings,
                        AVG(array_length(embedding::real[], 1)) as avg_dimensions,
                        COUNT(DISTINCT content_id) as unique_content_items,
                        MIN(created_at) as oldest_embedding,
                        MAX(created_at) as newest_embedding
                    FROM vector_embedding
                    WHERE embedding IS NOT NULL
                    """
                )

                if not stats or stats["total_embeddings"] == 0:
                    return {
                        "total_embeddings": 0,
                        "recommendation": "no_optimization_needed",
                        "reason": "No embeddings to optimize",
                    }

                # Estimate memory usage
                embedding_size_bytes = 384 * 4  # 384 dimensions * 4 bytes per float32
                total_memory_mb = (stats["total_embeddings"] * embedding_size_bytes) / (
                    1024 * 1024
                )

                # Calculate growth rate
                time_range = (
                    stats["newest_embedding"] - stats["oldest_embedding"]
                ).total_seconds()
                growth_rate = stats["total_embeddings"] / max(
                    time_range / 86400, 1
                )  # per day

                return {
                    "total_embeddings": stats["total_embeddings"],
                    "unique_content_items": stats["unique_content_items"],
                    "avg_dimensions": stats["avg_dimensions"],
                    "estimated_memory_mb": total_memory_mb,
                    "growth_rate_per_day": growth_rate,
                    "time_range_days": time_range / 86400,
                }

        except Exception as e:
            self.logger.error(f"Failed to analyze data characteristics: {e}")
            return {"error": str(e)}

    def recommend_hnsw_parameters(
        self, data_characteristics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Recommend optimal HNSW parameters based on data characteristics.

        Args:
            data_characteristics: Dataset analysis from analyze_data_characteristics

        Returns:
            Dictionary with recommended parameters and rationale
        """
        if "error" in data_characteristics:
            return {"error": data_characteristics["error"]}

        total_embeddings = data_characteristics.get("total_embeddings", 0)
        memory_mb = data_characteristics.get("estimated_memory_mb", 0)

        # Base parameters for different dataset sizes
        if total_embeddings < 1000:
            # Small dataset - prioritize build speed
            m = 8
            ef_construction = 32
            rationale = "Small dataset: optimized for fast indexing"
        elif total_embeddings < 10000:
            # Medium dataset - balanced parameters
            m = 16
            ef_construction = 64
            rationale = "Medium dataset: balanced performance/memory"
        elif total_embeddings < 100000:
            # Large dataset - prioritize search quality
            m = 24
            ef_construction = 128
            rationale = "Large dataset: optimized for search quality"
        else:
            # Very large dataset - high quality with memory constraints
            m = 32
            ef_construction = 200
            rationale = "Very large dataset: maximum search quality"

        # Adjust for memory constraints (assume 8GB available)
        max_memory_mb = 2000  # Conservative limit for index overhead
        if memory_mb > max_memory_mb:
            # Reduce parameters to fit memory constraints
            memory_ratio = max_memory_mb / memory_mb
            m = max(8, int(m * memory_ratio))
            ef_construction = max(32, int(ef_construction * memory_ratio))
            rationale += " (adjusted for memory constraints)"

        # Calculate maintenance intervals
        growth_rate = data_characteristics.get("growth_rate_per_day", 0)
        if growth_rate > 100:
            maintenance_interval_days = 1
        elif growth_rate > 10:
            maintenance_interval_days = 7
        else:
            maintenance_interval_days = 30

        return {
            "m": m,
            "ef_construction": ef_construction,
            "maintenance_interval_days": maintenance_interval_days,
            "rationale": rationale,
            "expected_build_time_minutes": self._estimate_build_time(
                total_embeddings, m, ef_construction
            ),
            "expected_memory_overhead_mb": self._estimate_memory_overhead(
                total_embeddings, m
            ),
        }

    def _estimate_build_time(
        self, total_embeddings: int, m: int, ef_construction: int
    ) -> float:
        """Estimate HNSW index build time in minutes."""
        # Empirical formula based on HNSW complexity
        base_time_seconds = total_embeddings * 0.001  # Base time per embedding
        complexity_factor = (m * ef_construction) / 1000  # Parameter complexity
        estimated_seconds = base_time_seconds * (1 + complexity_factor)
        return max(0.5, estimated_seconds / 60)  # Convert to minutes, minimum 30s

    def _estimate_memory_overhead(self, total_embeddings: int, m: int) -> float:
        """Estimate HNSW index memory overhead in MB."""
        # HNSW graph structure memory estimation
        avg_connections = m * 1.5  # Average connections per node
        bytes_per_connection = 8  # Node ID + distance
        overhead_bytes = total_embeddings * avg_connections * bytes_per_connection
        return overhead_bytes / (1024 * 1024)

    async def benchmark_current_index(self, test_queries: int = 10) -> dict[str, Any]:
        """
        Benchmark current index performance.

        Args:
            test_queries: Number of test queries to run

        Returns:
            Performance benchmark results
        """
        try:
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                # Get a sample embedding to use for testing
                sample_embedding = await conn.fetchval(
                    "SELECT embedding FROM vector_embedding LIMIT 1"
                )

                if not sample_embedding:
                    return {"error": "No embeddings available for benchmarking"}

                # Run benchmark queries
                query_times = []
                for _ in range(test_queries):
                    start_time = time.time()

                    await conn.fetch(
                        """
                        SELECT content_id, 1 - (embedding <=> $1) as similarity_score
                        FROM vector_embedding
                        ORDER BY embedding <=> $1
                        LIMIT 20
                        """,
                        sample_embedding,
                    )

                    query_time = (time.time() - start_time) * 1000  # Convert to ms
                    query_times.append(query_time)

                # Calculate statistics
                avg_time = sum(query_times) / len(query_times)
                p95_time = sorted(query_times)[int(0.95 * len(query_times))]
                p99_time = sorted(query_times)[int(0.99 * len(query_times))]

                # Check if performance meets requirements
                meets_requirements = p95_time <= 600  # 600ms target

                return {
                    "test_queries": test_queries,
                    "avg_response_time_ms": avg_time,
                    "p95_response_time_ms": p95_time,
                    "p99_response_time_ms": p99_time,
                    "meets_requirements": meets_requirements,
                    "performance_grade": "excellent"
                    if p95_time <= 200
                    else "good"
                    if p95_time <= 400
                    else "acceptable"
                    if p95_time <= 600
                    else "poor",
                }

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return {"error": str(e)}

    async def optimize_index(self, dry_run: bool = True) -> dict[str, Any]:
        """
        Optimize the vector index based on current data characteristics.

        Args:
            dry_run: If True, only return recommendations without applying changes

        Returns:
            Optimization results and recommendations
        """
        try:
            # Analyze current data
            data_chars = await self.analyze_data_characteristics()
            if "error" in data_chars:
                return data_chars

            # Get recommendations
            recommendations = self.recommend_hnsw_parameters(data_chars)
            if "error" in recommendations:
                return recommendations

            # Benchmark current performance
            current_benchmark = await self.benchmark_current_index()

            optimization_result = {
                "data_characteristics": data_chars,
                "recommendations": recommendations,
                "current_performance": current_benchmark,
                "dry_run": dry_run,
            }

            if dry_run:
                optimization_result["status"] = "analysis_complete"
                optimization_result["message"] = (
                    "Use dry_run=False to apply optimizations"
                )
                return optimization_result

            # Apply optimizations if not dry run
            if not current_benchmark.get("meets_requirements", False):
                await self._apply_index_optimization(recommendations)
                optimization_result["status"] = "optimization_applied"
                optimization_result["message"] = "Index optimization completed"
            else:
                optimization_result["status"] = "no_optimization_needed"
                optimization_result["message"] = (
                    "Current performance meets requirements"
                )

            return optimization_result

        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            return {"error": str(e)}

    async def _apply_index_optimization(self, recommendations: dict[str, Any]) -> None:
        """
        Apply the recommended index optimizations.

        Args:
            recommendations: Recommended parameters from recommend_hnsw_parameters
        """
        m = recommendations["m"]
        ef_construction = recommendations["ef_construction"]

        self.logger.info(
            f"Applying index optimization: m={m}, ef_construction={ef_construction}"
        )

        db_conn = get_database_connection()
        async with db_conn.acquire() as conn:
            # Drop existing index
            await conn.execute("DROP INDEX IF EXISTS idx_vector_embedding_hnsw")

            # Create optimized index
            await conn.execute(
                f"""
                CREATE INDEX idx_vector_embedding_hnsw ON vector_embedding
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction})
                """
            )

            self.logger.info("Index optimization applied successfully")

    async def get_index_info(self) -> dict[str, Any]:
        """
        Get current index configuration and statistics.

        Returns:
            Dictionary with index information
        """
        try:
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                # Get index information
                index_info = await conn.fetchrow(
                    """
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE indexname = 'idx_vector_embedding_hnsw'
                    """
                )

                if not index_info:
                    return {"error": "HNSW index not found"}

                # Extract parameters from index definition
                indexdef = index_info["indexdef"]
                m_match = None
                ef_construction_match = None

                # Parse parameters from index definition
                if "WITH (" in indexdef:
                    params_part = indexdef.split("WITH (")[1].split(")")[0]
                    for param in params_part.split(","):
                        param = param.strip()
                        if param.startswith("m ="):
                            m_match = int(param.split("=")[1].strip())
                        elif param.startswith("ef_construction ="):
                            ef_construction_match = int(param.split("=")[1].strip())

                # Get index size
                index_size = await conn.fetchval(
                    "SELECT pg_size_pretty(pg_relation_size('idx_vector_embedding_hnsw'))"
                )

                return {
                    "index_name": index_info["indexname"],
                    "table_name": index_info["tablename"],
                    "m": m_match,
                    "ef_construction": ef_construction_match,
                    "index_size": index_size,
                    "index_definition": indexdef,
                }

        except Exception as e:
            self.logger.error(f"Failed to get index info: {e}")
            return {"error": str(e)}


async def optimize_semantic_search_index(dry_run: bool = True) -> dict[str, Any]:
    """
    Convenience function to optimize semantic search vector index.

    Args:
        dry_run: If True, only analyze and recommend without applying changes

    Returns:
        Optimization results
    """
    optimizer = IndexOptimizer()
    return await optimizer.optimize_index(dry_run=dry_run)
