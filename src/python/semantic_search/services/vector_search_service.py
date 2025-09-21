"""
VectorSearchService for pgvector operations.

Provides semantic search capabilities using pgvector with PostgreSQL for
efficient vector similarity operations and hybrid search functionality.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from pydantic import BaseModel, Field

from ..models.search_result import SearchResult


class SearchFilters(BaseModel):
    """Filters for vector search operations."""

    content_types: list[str] | None = Field(None, description="Filter by content types")
    file_paths: list[str] | None = Field(
        None, description="Filter by specific file paths"
    )
    path_patterns: list[str] | None = Field(
        None, description="Filter by path patterns (LIKE matching)"
    )
    max_results: int = Field(20, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    include_context: bool = Field(True, description="Include surrounding context lines")


class VectorSearchService:
    """
    Vector search service with pgvector operations.

    Provides high-performance semantic search using PostgreSQL with pgvector
    extension. Supports similarity search, hybrid ranking, and advanced filtering.

    Performance targets:
    - Query response p95 ≤ 600ms
    - Support 10+ concurrent users
    - Efficient HNSW index utilization
    """

    def __init__(self, connection_string: str) -> None:
        """
        Initialize vector search service.

        Args:
            connection_string: PostgreSQL connection string with pgvector support
        """
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
        self._connection_pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize connection pool and verify pgvector extension."""
        try:
            self._connection_pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=20,  # Support concurrent users
                command_timeout=10,  # Fast timeout for performance
                server_settings={
                    "application_name": "kcs_semantic_search",
                },
            )

            # Verify pgvector extension is available
            async with self._connection_pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector'"
                )
                if not result:
                    raise RuntimeError("pgvector extension not installed")

            self.logger.info("VectorSearchService initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize VectorSearchService: {e}")
            raise

    async def close(self) -> None:
        """Close connection pool."""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
            self.logger.info("VectorSearchService connection pool closed")

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get database connection from pool."""
        if not self._connection_pool:
            raise RuntimeError("VectorSearchService not initialized")

        async with self._connection_pool.acquire() as connection:
            yield connection

    async def similarity_search(
        self,
        query_embedding: list[float],
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """
        Perform similarity search using vector embeddings.

        Uses pgvector cosine similarity with HNSW index for optimal performance.
        Supports filtering by content type, file paths, and similarity threshold.

        Args:
            query_embedding: 384-dimensional query vector
            filters: Optional filters for search refinement

        Returns:
            List of SearchResult objects ordered by similarity score

        Raises:
            ValueError: If query_embedding is invalid
            RuntimeError: If database operation fails
        """
        if not query_embedding or len(query_embedding) != 384:
            raise ValueError("query_embedding must be 384-dimensional vector")

        filters = filters or SearchFilters(
            content_types=None,
            file_paths=None,
            path_patterns=None,
            max_results=20,
            similarity_threshold=0.0,
            include_context=True,
        )

        try:
            async with self._get_connection() as conn:
                # Build dynamic SQL query with filters
                sql_parts = [
                    """
                    SELECT
                        ve.id as embedding_id,
                        ve.content_id,
                        ic.source_path,
                        ic.content_type,
                        ic.title,
                        ic.content,
                        ic.metadata,
                        1 - (ve.embedding <=> $1) as similarity_score
                    FROM vector_embedding ve
                    JOIN indexed_content ic ON ve.content_id = ic.id
                    WHERE ic.status = 'completed'
                """
                ]

                params: list[Any] = [query_embedding]
                param_idx = 2

                # Add content type filters
                if filters.content_types:
                    sql_parts.append(f"AND ic.content_type = ANY(${param_idx})")
                    params.append(filters.content_types)
                    param_idx += 1

                # Add file path filters
                if filters.file_paths:
                    sql_parts.append(f"AND ic.source_path = ANY(${param_idx})")
                    params.append(filters.file_paths)
                    param_idx += 1

                # Add path pattern filters
                if filters.path_patterns:
                    pattern_conditions = []
                    for pattern in filters.path_patterns:
                        pattern_conditions.append(f"ic.source_path LIKE ${param_idx}")
                        params.append(pattern)
                        param_idx += 1
                    sql_parts.append(f"AND ({' OR '.join(pattern_conditions)})")

                # Add similarity threshold
                if filters.similarity_threshold > 0:
                    sql_parts.append(f"AND (1 - (ve.embedding <=> $1)) >= ${param_idx}")
                    params.append(filters.similarity_threshold)
                    param_idx += 1

                # Add ordering and limit
                sql_parts.extend(
                    [
                        "ORDER BY ve.embedding <=> $1",  # pgvector distance ordering
                        f"LIMIT ${param_idx}",
                    ]
                )
                params.append(filters.max_results)

                sql_query = " ".join(sql_parts)

                # Execute the query with timeout for performance
                results = await asyncio.wait_for(
                    conn.fetch(sql_query, *params),
                    timeout=0.5,  # 500ms timeout for performance target
                )

                # Convert to SearchResult objects
                search_results = []
                for row in results:
                    context_lines = []
                    if filters.include_context:
                        context_lines = await self._get_context_lines(
                            conn, row["content"], 3
                        )

                    # Create SearchResult (BM25 will be handled by RankingService)
                    search_result = SearchResult(
                        query_id="",  # Will be set by calling service
                        content_id=str(row["content_id"]),
                        similarity_score=float(row["similarity_score"]),
                        bm25_score=0.0,  # Will be calculated by RankingService
                        combined_score=float(row["similarity_score"]),  # Initial score
                        confidence=self._calculate_confidence(
                            float(row["similarity_score"])
                        ),
                        context_lines=context_lines,
                        explanation=self._generate_explanation(
                            row["title"] or row["source_path"],
                            float(row["similarity_score"]),
                        ),
                    )
                    search_results.append(search_result)

                return search_results

        except TimeoutError:
            self.logger.warning("Similarity search timed out")
            raise RuntimeError(
                "Search operation timed out (performance limit exceeded)"
            ) from None
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}") from e

    async def batch_similarity_search(
        self,
        query_embeddings: list[list[float]],
        filters: SearchFilters | None = None,
    ) -> list[list[SearchResult]]:
        """
        Perform batch similarity search for multiple queries.

        Optimized for processing multiple queries efficiently with connection reuse.

        Args:
            query_embeddings: List of 384-dimensional query vectors
            filters: Optional filters applied to all queries

        Returns:
            List of SearchResult lists, one for each query

        Raises:
            ValueError: If any query_embedding is invalid
            RuntimeError: If database operation fails
        """
        if not query_embeddings:
            return []

        # Validate all embeddings
        for i, embedding in enumerate(query_embeddings):
            if not embedding or len(embedding) != 384:
                raise ValueError(f"query_embedding[{i}] must be 384-dimensional vector")

        results = []
        async with self._get_connection() as conn:
            # Process all queries within single connection for efficiency
            for embedding in query_embeddings:
                query_results = await self._single_similarity_search(
                    conn, embedding, filters
                )
                results.append(query_results)

        return results

    async def _single_similarity_search(
        self,
        conn: asyncpg.Connection,
        query_embedding: list[float],
        filters: SearchFilters | None,
    ) -> list[SearchResult]:
        """Helper method for single similarity search with existing connection."""
        # Implementation similar to similarity_search but using provided connection
        # This is a simplified version for batch processing
        sql = """
            SELECT
                ve.content_id,
                ic.source_path,
                ic.title,
                ic.content,
                1 - (ve.embedding <=> $1) as similarity_score
            FROM vector_embedding ve
            JOIN indexed_content ic ON ve.content_id = ic.id
            WHERE ic.status = 'completed'
            ORDER BY ve.embedding <=> $1
            LIMIT $2
        """

        max_results = filters.max_results if filters else 20
        results = await conn.fetch(sql, query_embedding, max_results)

        search_results = []
        for row in results:
            search_result = SearchResult(
                query_id="",
                content_id=str(row["content_id"]),
                similarity_score=float(row["similarity_score"]),
                bm25_score=0.0,
                combined_score=float(row["similarity_score"]),
                confidence=self._calculate_confidence(float(row["similarity_score"])),
                context_lines=[],  # Skip context for batch operations (performance)
                explanation=self._generate_explanation(
                    row["title"] or row["source_path"], float(row["similarity_score"])
                ),
            )
            search_results.append(search_result)

        return search_results

    async def find_similar_content(
        self,
        content_id: str,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """
        Find content similar to given content ID.

        Uses the embedding of existing content to find similar items.

        Args:
            content_id: ID of content to find similar items for
            filters: Optional filters for search refinement

        Returns:
            List of similar SearchResult objects

        Raises:
            ValueError: If content_id is invalid
            RuntimeError: If content not found or database operation fails
        """
        try:
            async with self._get_connection() as conn:
                # Get embedding for the content
                embedding_row = await conn.fetchrow(
                    """
                    SELECT embedding
                    FROM vector_embedding
                    WHERE content_id = $1
                    """,
                    int(content_id),
                )

                if not embedding_row:
                    raise ValueError(f"Content ID {content_id} not found")

                embedding = embedding_row["embedding"]

                # Perform similarity search excluding the original content
                filters = filters or SearchFilters(
                    content_types=None,
                    file_paths=None,
                    path_patterns=None,
                    max_results=20,
                    similarity_threshold=0.0,
                    include_context=True,
                )
                results = await self.similarity_search(embedding, filters)

                # Filter out the original content
                filtered_results = [
                    result for result in results if result.content_id != content_id
                ]

                return filtered_results

        except Exception as e:
            self.logger.error(f"Find similar content failed: {e}")
            raise RuntimeError(f"Similar content search failed: {e}") from e

    async def get_embedding_stats(self) -> dict[str, Any]:
        """
        Get statistics about indexed embeddings.

        Returns:
            Dictionary with embedding statistics and health metrics
        """
        try:
            async with self._get_connection() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_embeddings,
                        COUNT(DISTINCT ve.content_id) as unique_content,
                        COUNT(DISTINCT ic.content_type) as content_types,
                        AVG(ic.updated_at) as avg_update_time,
                        COUNT(*) FILTER (WHERE ic.status = 'completed') as completed_count,
                        COUNT(*) FILTER (WHERE ic.status = 'pending') as pending_count,
                        COUNT(*) FILTER (WHERE ic.status = 'failed') as failed_count
                    FROM vector_embedding ve
                    JOIN indexed_content ic ON ve.content_id = ic.id
                    """
                )

                return {
                    "total_embeddings": stats["total_embeddings"],
                    "unique_content": stats["unique_content"],
                    "content_types": stats["content_types"],
                    "completed_count": stats["completed_count"],
                    "pending_count": stats["pending_count"],
                    "failed_count": stats["failed_count"],
                    "health_status": "healthy"
                    if stats["failed_count"] == 0
                    else "degraded",
                }

        except Exception as e:
            self.logger.error(f"Get embedding stats failed: {e}")
            return {"error": str(e), "health_status": "error"}

    async def _get_context_lines(
        self, conn: asyncpg.Connection, content: str, context_size: int = 3
    ) -> list[str]:
        """
        Extract context lines around relevant content.

        Args:
            conn: Database connection
            content: Content text to extract context from
            context_size: Number of lines before/after to include

        Returns:
            List of context lines
        """
        try:
            lines = content.split("\n")
            # For now, return first few lines as context
            # In a full implementation, this would identify the most relevant lines
            return lines[: min(context_size * 2, len(lines))]
        except Exception:
            return []

    def _calculate_confidence(self, similarity_score: float) -> float:
        """
        Calculate confidence score based on similarity.

        Args:
            similarity_score: Cosine similarity score (0.0-1.0)

        Returns:
            Confidence score (0.0-1.0)
        """
        # Simple confidence calculation based on similarity thresholds
        if similarity_score >= 0.9:
            return 0.95
        elif similarity_score >= 0.8:
            return 0.85
        elif similarity_score >= 0.7:
            return 0.75
        elif similarity_score >= 0.6:
            return 0.65
        elif similarity_score >= 0.5:
            return 0.55
        else:
            return max(0.1, similarity_score * 0.8)

    def _generate_explanation(self, title: str, similarity_score: float) -> str:
        """
        Generate human-readable explanation for search result.

        Args:
            title: Content title or file path
            similarity_score: Similarity score

        Returns:
            Explanation string
        """
        if similarity_score >= 0.9:
            confidence_text = "very high"
        elif similarity_score >= 0.8:
            confidence_text = "high"
        elif similarity_score >= 0.7:
            confidence_text = "good"
        elif similarity_score >= 0.6:
            confidence_text = "moderate"
        else:
            confidence_text = "low"

        return f"Found in {title} with {confidence_text} semantic similarity ({similarity_score:.2f})"


class VectorSearchError(Exception):
    """Exception raised by VectorSearchService operations."""

    pass


class VectorSearchTimeoutError(VectorSearchError):
    """Exception raised when search operations exceed performance limits."""

    pass
