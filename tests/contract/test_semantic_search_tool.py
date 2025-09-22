"""
Contract test for semantic_search MCP tool.

This test validates the MCP tool contract as defined in
specs/008-semantic-search-engine/contracts/mcp-search-tools.json

Following TDD: This test MUST FAIL before implementation exists.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


class TestSemanticSearchToolContract:
    """Contract tests for semantic_search MCP tool."""

    @pytest.fixture
    def valid_search_input(self) -> dict[str, Any]:
        """Valid input conforming to inputSchema."""
        return {
            "query": "memory allocation functions that can fail",
            "max_results": 10,
            "min_confidence": 0.5,
            "content_types": ["SOURCE_CODE", "HEADER"],
            "config_context": ["CONFIG_NET", "!CONFIG_EMBEDDED"],
            "file_patterns": ["drivers/net/*", "kernel/sched/*"],
        }

    @pytest.fixture
    def minimal_search_input(self) -> dict[str, Any]:
        """Minimal valid input (only required fields)."""
        return {"query": "buffer overflow"}

    @pytest.fixture
    def expected_output_schema(self) -> dict[str, Any]:
        """Expected output schema structure."""
        return {
            "query_id": "test-query-123",
            "results": [
                {
                    "file_path": "/kernel/mm/slub.c",
                    "line_start": 1234,
                    "line_end": 1250,
                    "content": "static void *__slab_alloc(struct kmem_cache *s, gfp_t gfpflags, int node)",
                    "context_lines": ["/* Previous line */", "/* Next line */"],
                    "confidence": 0.85,
                    "similarity_score": 0.91,
                    "explanation": "Memory allocation function with failure handling",
                    "content_type": "SOURCE_CODE",
                    "metadata": {
                        "function_name": "__slab_alloc",
                        "symbols": ["kmem_cache", "gfp_t"],
                        "config_guards": ["CONFIG_SLUB"],
                    },
                }
            ],
            "search_stats": {
                "total_matches": 150,
                "filtered_matches": 12,
                "search_time_ms": 250,
                "embedding_time_ms": 50,
            },
        }

    async def test_semantic_search_tool_exists(self):
        """Test that semantic_search MCP tool is registered and callable."""
        # This should fail until the tool is implemented
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Tool should be importable
        assert callable(semantic_search)

    async def test_semantic_search_input_validation(self, valid_search_input):
        """Test input validation according to inputSchema."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Valid input should not raise validation errors
        result = await semantic_search(**valid_search_input)
        assert result is not None

    async def test_semantic_search_minimal_input(self, minimal_search_input):
        """Test with minimal required input (only query)."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Should work with only required field
        result = await semantic_search(**minimal_search_input)
        assert result is not None
        assert "query_id" in result
        assert "results" in result
        assert "search_stats" in result

    async def test_semantic_search_output_schema_compliance(
        self, minimal_search_input, expected_output_schema
    ):
        """Test that output matches the defined outputSchema."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        result = await semantic_search(**minimal_search_input)

        # Required top-level fields
        assert "query_id" in result
        assert "results" in result
        assert "search_stats" in result

        # query_id should be string
        assert isinstance(result["query_id"], str)

        # results should be array
        assert isinstance(result["results"], list)

        # Each result should have required fields
        for search_result in result["results"]:
            assert "file_path" in search_result
            assert "line_start" in search_result
            assert "line_end" in search_result
            assert "content" in search_result
            assert "confidence" in search_result
            assert "similarity_score" in search_result
            assert "content_type" in search_result

            # Validate types
            assert isinstance(search_result["file_path"], str)
            assert isinstance(search_result["line_start"], int)
            assert isinstance(search_result["line_end"], int)
            assert isinstance(search_result["content"], str)
            assert isinstance(search_result["confidence"], (int, float))
            assert isinstance(search_result["similarity_score"], (int, float))
            assert search_result["content_type"] in [
                "SOURCE_CODE",
                "DOCUMENTATION",
                "HEADER",
                "COMMENT",
            ]

            # Validate ranges
            assert 0.0 <= search_result["confidence"] <= 1.0
            assert 0.0 <= search_result["similarity_score"] <= 1.0

        # search_stats should have required fields
        stats = result["search_stats"]
        assert "total_matches" in stats
        assert "search_time_ms" in stats
        assert isinstance(stats["total_matches"], int)
        assert isinstance(stats["search_time_ms"], int)

    async def test_semantic_search_input_validation_errors(self):
        """Test input validation error cases."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Empty query should fail
        with pytest.raises((ValueError, TypeError)):
            await semantic_search(query="")

        # Query too long should fail (QueryTooLongError is an MCPError, not ValueError)
        from src.python.semantic_search.mcp.error_handlers import QueryTooLongError

        with pytest.raises(QueryTooLongError):
            await semantic_search(query="x" * 1001)

        # Invalid max_results should fail
        with pytest.raises(ValueError):
            await semantic_search(query="test", max_results=0)

        with pytest.raises(ValueError):
            await semantic_search(query="test", max_results=51)

        # Invalid min_confidence should fail
        with pytest.raises(ValueError):
            await semantic_search(query="test", min_confidence=-0.1)

        with pytest.raises(ValueError):
            await semantic_search(query="test", min_confidence=1.1)

        # Invalid content_types should fail
        with pytest.raises(ValueError):
            await semantic_search(query="test", content_types=["INVALID_TYPE"])

        # Invalid config_context pattern should fail
        with pytest.raises(ValueError):
            await semantic_search(query="test", config_contexts=["INVALID_CONFIG"])

    async def test_semantic_search_performance_constraint(self, minimal_search_input):
        """Test that search completes within 600ms performance requirement."""
        import time

        from src.python.semantic_search.mcp.search_tool import semantic_search

        start_time = time.time()
        result = await semantic_search(**minimal_search_input)
        end_time = time.time()

        # Performance requirement: p95 ≤ 600ms
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms <= 600, (
            f"Search took {execution_time_ms}ms, exceeds 600ms limit"
        )

        # Also check reported search_time_ms
        assert result["search_stats"]["search_time_ms"] <= 600

    async def test_semantic_search_error_handling(self):
        """Test error handling according to error_definitions."""
        from src.python.semantic_search.mcp.error_handlers import (
            EmbeddingFailedError,
            IndexUnavailableError,
            QueryTooLongError,
            SearchTimeoutError,
        )
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Test QUERY_TOO_LONG error
        with pytest.raises(QueryTooLongError) as exc_info:
            await semantic_search(query="x" * 1001)
        assert exc_info.value.code == "QUERY_TOO_LONG"
        assert not exc_info.value.retryable

        # Test other error types with mocked conditions
        # Note: In test mode with TESTING=true, we use mock data and don't call
        # the actual embedding service, so we can't test embedding failures this way.
        # This would need to be tested with integration tests that don't use mock mode.

    async def test_semantic_search_content_type_filtering(self):
        """Test content type filtering functionality."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Test SOURCE_CODE only filter
        result = await semantic_search(
            query="memory allocation", content_types=["SOURCE_CODE"]
        )

        for search_result in result["results"]:
            assert search_result["content_type"] == "SOURCE_CODE"

        # Test DOCUMENTATION only filter
        result = await semantic_search(
            query="memory allocation", content_types=["DOCUMENTATION"]
        )

        for search_result in result["results"]:
            assert search_result["content_type"] == "DOCUMENTATION"

    async def test_semantic_search_config_context_filtering(self):
        """Test kernel configuration context filtering."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Test positive config filter
        result = await semantic_search(
            query="network driver", config_context=["CONFIG_NET"]
        )

        # All results should be from files with CONFIG_NET context
        for search_result in result["results"]:
            if (
                "metadata" in search_result
                and "config_guards" in search_result["metadata"]
            ):
                config_guards = search_result["metadata"]["config_guards"]
                # Should have CONFIG_NET related guards
                assert any("NET" in guard for guard in config_guards)

        # Test negative config filter
        result = await semantic_search(
            query="network driver", config_context=["!CONFIG_EMBEDDED"]
        )

        # Results should not be from embedded-specific code
        for search_result in result["results"]:
            if (
                "metadata" in search_result
                and "config_guards" in search_result["metadata"]
            ):
                config_guards = search_result["metadata"]["config_guards"]
                assert "CONFIG_EMBEDDED" not in config_guards

    async def test_semantic_search_file_pattern_filtering(self):
        """Test file pattern filtering functionality."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        result = await semantic_search(
            query="memory allocation", file_patterns=["drivers/net/*", "kernel/mm/*"]
        )

        # All results should match the file patterns
        for search_result in result["results"]:
            file_path = search_result["file_path"]
            assert file_path.startswith("drivers/net/") or file_path.startswith(
                "kernel/mm/"
            ), f"File {file_path} doesn't match patterns"

    async def test_semantic_search_confidence_filtering(self):
        """Test minimum confidence filtering."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        min_confidence = 0.8
        result = await semantic_search(
            query="buffer overflow", min_confidence=min_confidence
        )

        # All results should meet minimum confidence
        for search_result in result["results"]:
            assert search_result["confidence"] >= min_confidence

    async def test_semantic_search_max_results_limit(self):
        """Test max_results parameter limits output correctly."""
        from src.python.semantic_search.mcp.search_tool import semantic_search

        max_results = 5
        result = await semantic_search(
            query="memory allocation", max_results=max_results
        )

        # Should return at most max_results
        assert len(result["results"]) <= max_results
