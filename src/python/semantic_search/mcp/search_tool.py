"""
semantic_search MCP tool implementation.

Provides semantic search functionality through MCP interface with hybrid
BM25+semantic ranking and performance-optimized query processing.
"""

import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..database.connection import get_database_connection
from ..models.indexed_content import ProcessingStatus
from ..services.embedding_service import EmbeddingService
from ..services.query_preprocessor import QueryPreprocessor
from ..services.ranking_service import RankingService
from ..services.vector_search_service import SearchFilters, VectorSearchService

logger = logging.getLogger(__name__)


class SemanticSearchRequest(BaseModel):
    """Input schema for semantic_search MCP tool."""

    query: str = Field(
        ..., min_length=1, max_length=1000, description="Natural language search query"
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results to return"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for results (0.0-1.0)",
    )
    content_types: list[str] = Field(
        default=["SOURCE_CODE", "HEADER"], description="Filter by content types"
    )
    config_context: list[str] = Field(
        default_factory=list, description="Kernel configuration context for filtering"
    )
    file_patterns: list[str] = Field(
        default_factory=list, description="Filter results by file path patterns"
    )


class SearchResultResponse(BaseModel):
    """Individual search result for MCP response."""

    file_path: str = Field(..., description="Absolute path to the source file")
    line_start: int = Field(..., description="Starting line number of the match")
    line_end: int = Field(..., description="Ending line number of the match")
    content: str = Field(..., description="The matched content")
    context_lines: list[str] = Field(
        default_factory=list, description="Surrounding lines for context"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for this result"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic similarity score"
    )
    explanation: str = Field(..., description="Why this result matches the query")
    content_type: str = Field(..., description="Type of content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context about the match"
    )


class SearchStatsResponse(BaseModel):
    """Search statistics for MCP response."""

    total_matches: int = Field(
        ..., description="Total number of potential matches found"
    )
    filtered_matches: int = Field(
        ..., description="Number of matches after confidence filtering"
    )
    search_time_ms: int = Field(
        ..., description="Time taken for search in milliseconds"
    )
    embedding_time_ms: int = Field(
        ..., description="Time taken for query embedding generation"
    )


class SemanticSearchResponse(BaseModel):
    """Response schema for semantic_search MCP tool."""

    query_id: str = Field(..., description="Unique identifier for this search query")
    results: list[SearchResultResponse] = Field(
        default_factory=list, description="Search results"
    )
    search_stats: SearchStatsResponse = Field(..., description="Search statistics")


class SemanticSearchTool:
    """
    MCP tool for semantic search functionality.

    Integrates query preprocessing, embedding generation, vector search,
    and hybrid ranking to provide high-quality semantic search results.
    """

    def __init__(self) -> None:
        """Initialize semantic search tool with required services."""
        self.embedding_service = EmbeddingService()
        self.query_preprocessor = QueryPreprocessor()
        self.ranking_service = RankingService()
        self._vector_search_service: VectorSearchService | None = None

    async def _get_vector_search_service(self) -> VectorSearchService:
        """Get or create vector search service instance."""
        if self._vector_search_service is None:
            db_conn = get_database_connection()
            connection_string = db_conn.config.to_url()
            self._vector_search_service = VectorSearchService(connection_string)
            await self._vector_search_service.initialize()

        return self._vector_search_service

    async def execute(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute semantic search MCP tool.

        Args:
            request_data: MCP tool request data

        Returns:
            Semantic search results following MCP contract

        Raises:
            ValueError: For validation errors
            RuntimeError: For processing errors
        """
        start_time = time.time()

        try:
            # Validate input
            request = SemanticSearchRequest(**request_data)

            # Generate unique query ID
            query_id = str(uuid.uuid4())

            # Track timing
            embedding_start = time.time()

            # Preprocess query
            processed_query = self.query_preprocessor.preprocess(request.query)

            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(processed_query)
            embedding_time_ms = int((time.time() - embedding_start) * 1000)

            # Perform vector search
            search_start = time.time()
            vector_service = await self._get_vector_search_service()

            # Configure search filters
            filters = SearchFilters(
                content_types=self._map_content_types(request.content_types),
                file_paths=None,  # No specific file path filtering
                path_patterns=request.file_patterns if request.file_patterns else None,
                max_results=request.max_results * 2,  # Get more for ranking
                similarity_threshold=0.1,  # Low threshold, filter by confidence later
                include_context=True,
            )

            # Execute similarity search
            search_results = await vector_service.similarity_search(
                query_embedding, filters
            )

            # Get content for ranking
            content_data = await self._get_content_for_results(search_results)

            # Apply hybrid ranking
            ranking_input = [
                (result.content_id, result.similarity_score, content)
                for result, content in zip(search_results, content_data, strict=False)
            ]

            ranked_results = self.ranking_service.rank_results(
                request.query, ranking_input, request.max_results
            )

            # Filter by confidence
            filtered_results = [
                result
                for result in ranked_results
                if result.confidence >= request.min_confidence
            ]

            search_time_ms = int((time.time() - search_start) * 1000)
            total_time_ms = int((time.time() - start_time) * 1000)

            # Convert to MCP response format
            response_results = []
            for result in filtered_results:
                # Get file path and content details
                file_info = await self._get_file_info(result.content_id)

                response_result = SearchResultResponse(
                    file_path=file_info["file_path"],
                    line_start=file_info.get("line_start", 1),
                    line_end=file_info.get("line_end", 1),
                    content=file_info["content"],
                    context_lines=result.context_lines,
                    confidence=result.confidence,
                    similarity_score=result.similarity_score,
                    explanation=result.explanation,
                    content_type=self._map_content_type_response(
                        file_info["content_type"]
                    ),
                    metadata=self._extract_metadata(file_info),
                )
                response_results.append(response_result)

            # Build search stats
            search_stats = SearchStatsResponse(
                total_matches=len(search_results),
                filtered_matches=len(filtered_results),
                search_time_ms=search_time_ms,
                embedding_time_ms=embedding_time_ms,
            )

            # Create response
            response = SemanticSearchResponse(
                query_id=query_id, results=response_results, search_stats=search_stats
            )

            logger.info(
                f"Semantic search completed: query_id={query_id}, "
                f"total_time={total_time_ms}ms, results={len(response_results)}"
            )

            return response.model_dump()

        except ValidationError as e:
            logger.error(f"Semantic search validation error: {e}")
            raise ValueError(f"Invalid request parameters: {e}") from e

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}") from e

    def _map_content_types(self, mcp_types: list[str]) -> list[str]:
        """Map MCP content types to internal content types."""
        type_mapping = {
            "SOURCE_CODE": ["C_SOURCE"],
            "HEADER": ["C_HEADER"],
            "DOCUMENTATION": ["DOCUMENTATION"],
            "COMMENT": ["C_SOURCE", "C_HEADER"],  # Comments in source files
        }

        internal_types = []
        for mcp_type in mcp_types:
            if mcp_type in type_mapping:
                internal_types.extend(type_mapping[mcp_type])

        return list(set(internal_types)) if internal_types else ["C_SOURCE", "C_HEADER"]

    def _map_content_type_response(self, internal_type: str) -> str:
        """Map internal content type to MCP response type."""
        type_mapping = {
            "C_SOURCE": "SOURCE_CODE",
            "C_HEADER": "HEADER",
            "DOCUMENTATION": "DOCUMENTATION",
            "MAKEFILE": "SOURCE_CODE",
        }

        return type_mapping.get(internal_type, "SOURCE_CODE")

    async def _get_content_for_results(self, results: list[Any]) -> list[str]:
        """Get content text for search results to enable ranking."""
        db_conn = get_database_connection()

        content_list = []
        for result in results:
            try:
                # Get content from indexed_content table
                content_row = await db_conn.fetch_one(
                    "SELECT content FROM indexed_content WHERE id = $1",
                    int(result.content_id),
                )

                if content_row:
                    content_list.append(content_row["content"])
                else:
                    content_list.append("")  # Fallback for missing content

            except Exception as e:
                logger.warning(f"Failed to get content for {result.content_id}: {e}")
                content_list.append("")

        return content_list

    async def _get_file_info(self, content_id: str) -> dict[str, Any]:
        """Get file information for a content ID."""
        db_conn = get_database_connection()

        try:
            # Get file information from indexed_content
            file_row = await db_conn.fetch_one(
                """
                SELECT
                    source_path as file_path,
                    content_type,
                    content,
                    metadata
                FROM indexed_content
                WHERE id = $1 AND status = $2
                """,
                int(content_id),
                ProcessingStatus.COMPLETED.value,
            )

            if not file_row:
                return {
                    "file_path": "/unknown",
                    "content_type": "C_SOURCE",
                    "content": "",
                    "metadata": {},
                }

            return {
                "file_path": file_row["source_path"],
                "content_type": file_row["content_type"],
                "content": file_row["content"],
                "metadata": file_row["metadata"] or {},
                "line_start": 1,  # TODO: Extract from chunk metadata
                "line_end": len(file_row["content"].split("\n")),
            }

        except Exception as e:
            logger.error(f"Failed to get file info for content {content_id}: {e}")
            return {
                "file_path": "/error",
                "content_type": "C_SOURCE",
                "content": "",
                "metadata": {},
            }

    def _extract_metadata(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata for MCP response."""
        metadata = file_info.get("metadata", {})

        # Extract common metadata fields
        extracted = {}

        if "function_name" in metadata:
            extracted["function_name"] = metadata["function_name"]

        if "symbols" in metadata:
            extracted["symbols"] = metadata["symbols"]

        if "config_guards" in metadata:
            extracted["config_guards"] = metadata["config_guards"]

        return extracted


# Create global tool instance
semantic_search_tool = SemanticSearchTool()


async def execute_semantic_search(request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Execute semantic search MCP tool function.

    This is the entry point called by the MCP server framework.

    Args:
        request_data: MCP tool request data

    Returns:
        Semantic search results following MCP contract
    """
    return await semantic_search_tool.execute(request_data)
