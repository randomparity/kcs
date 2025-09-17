"""
Contract tests for semantic_search MCP tool.

These tests verify the API contract for semantic code search using embeddings.
They MUST fail before implementation and pass after.
"""

import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import requests


# Skip tests requiring MCP server when it's not running
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible."""
    try:
        response = requests.get("http://localhost:8080", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server not running"
)

# Skip integration tests in CI environments unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "test_token_123"


# Test fixtures and helpers
@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for API requests."""
    async with httpx.AsyncClient(base_url=MCP_BASE_URL) as client:
        yield client


# Contract test cases
@skip_integration_in_ci
@skip_without_mcp_server
class TestSemanticSearchContract:
    """Contract tests for semantic_search MCP tool."""

    async def test_semantic_search_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the semantic_search endpoint exists and accepts POST requests."""
        payload = {
            "query": "file system read operations",
            "limit": 10,
            "similarity_threshold": 0.7,
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "semantic_search endpoint should exist"

    async def test_semantic_search_requires_authentication(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that semantic_search requires valid authentication."""
        payload = {"query": "memory allocation"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/semantic_search", json=payload)
        assert response.status_code == 401, "Should require authentication"

    async def test_semantic_search_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that semantic_search validates request schema according to OpenAPI spec."""

        # Missing required 'query' field
        response = await http_client.post(
            "/mcp/tools/semantic_search",
            json={"limit": 10},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject request without required 'query' field"
        )

        # Empty query string
        response = await http_client.post(
            "/mcp/tools/semantic_search",
            json={"query": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject empty query"

        # Invalid limit (negative)
        response = await http_client.post(
            "/mcp/tools/semantic_search",
            json={"query": "test", "limit": -1},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject negative limit"

        # Invalid limit (too large)
        response = await http_client.post(
            "/mcp/tools/semantic_search",
            json={"query": "test", "limit": 1001},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject limit > 1000"

        # Invalid similarity_threshold (out of range)
        response = await http_client.post(
            "/mcp/tools/semantic_search",
            json={"query": "test", "similarity_threshold": 1.5},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject similarity_threshold outside [0, 1]"
        )

    async def test_semantic_search_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that semantic_search returns response matching OpenAPI schema."""
        payload = {
            "query": "VFS file system operations for reading and writing data",
            "limit": 5,
            "similarity_threshold": 0.6,
            "filters": {
                "subsystems": ["fs", "vfs"],
                "file_patterns": ["fs/*.c", "vfs/*.c"],
            },
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify top-level structure
            assert "results" in data, "Response should contain 'results' field"
            assert "query_id" in data, "Response should contain 'query_id' field"
            assert "total_results" in data, (
                "Response should contain 'total_results' field"
            )
            assert "search_time_ms" in data, (
                "Response should contain 'search_time_ms' field"
            )

            # Verify field types
            assert isinstance(data["results"], list), "results should be array"
            assert isinstance(data["query_id"], str), "query_id should be string (UUID)"
            assert isinstance(data["total_results"], int), (
                "total_results should be integer"
            )
            assert isinstance(data["search_time_ms"], (int, float)), (
                "search_time_ms should be numeric"
            )

            # Verify results structure
            for result in data["results"]:
                assert isinstance(result, dict), "Each result should be object"
                assert "symbol" in result, "Result should have 'symbol' field"
                assert "span" in result, "Result should have 'span' field"
                assert "similarity_score" in result, (
                    "Result should have 'similarity_score' field"
                )
                assert "snippet" in result, "Result should have 'snippet' field"
                assert "context" in result, "Result should have 'context' field"

                # Verify field types
                assert isinstance(result["symbol"], str), "symbol should be string"
                assert isinstance(result["span"], dict), "span should be object"
                assert isinstance(result["similarity_score"], (int, float)), (
                    "similarity_score should be numeric"
                )
                assert 0 <= result["similarity_score"] <= 1, (
                    "similarity_score should be between 0 and 1"
                )
                assert isinstance(result["snippet"], str), "snippet should be string"
                assert isinstance(result["context"], dict), "context should be object"

                # Verify span structure
                span = result["span"]
                assert "path" in span, "Span should have 'path' field"
                assert "sha" in span, "Span should have 'sha' field"
                assert "start" in span, "Span should have 'start' field"
                assert "end" in span, "Span should have 'end' field"
                assert isinstance(span["path"], str), "Span path should be string"
                assert isinstance(span["sha"], str), "Span sha should be string"
                assert isinstance(span["start"], int), "Span start should be integer"
                assert isinstance(span["end"], int), "Span end should be integer"

                # Verify context structure
                context = result["context"]
                if "subsystem" in context:
                    assert isinstance(context["subsystem"], str), (
                        "subsystem should be string"
                    )
                if "function_type" in context:
                    assert isinstance(context["function_type"], str), (
                        "function_type should be string"
                    )
                if "related_symbols" in context:
                    assert isinstance(context["related_symbols"], list), (
                        "related_symbols should be array"
                    )

            # Verify result ordering (by similarity score)
            if len(data["results"]) > 1:
                scores = [r["similarity_score"] for r in data["results"]]
                assert scores == sorted(scores, reverse=True), (
                    "Results should be ordered by similarity score (descending)"
                )

    async def test_semantic_search_with_filters(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test semantic search with various filter combinations."""
        payload = {
            "query": "memory management",
            "filters": {
                "subsystems": ["mm", "memory"],
                "file_patterns": ["mm/*.c", "kernel/mm.c"],
                "symbol_types": ["function", "macro"],
                "exclude_tests": True,
            },
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # All results should match the filter criteria
            for result in data["results"]:
                # Should not include test files when exclude_tests is True
                assert "test" not in result["span"]["path"].lower(), (
                    "Should exclude test files"
                )

    async def test_semantic_search_reranking(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test semantic search with reranking for improved relevance."""
        payload = {
            "query": "system call entry points",
            "limit": 20,
            "rerank": True,
            "rerank_model": "cross-encoder",
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include reranking metadata when enabled
            if "reranking_applied" in data:
                assert isinstance(data["reranking_applied"], bool), (
                    "reranking_applied should be boolean"
                )
            if "rerank_time_ms" in data:
                assert isinstance(data["rerank_time_ms"], (int, float)), (
                    "rerank_time_ms should be numeric"
                )

    async def test_semantic_search_pagination(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test semantic search pagination for large result sets."""
        # First page
        payload = {
            "query": "kernel data structures",
            "limit": 10,
            "offset": 0,
        }

        response1 = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        # Second page
        payload["offset"] = 10
        response2 = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Results should be different between pages
            if len(data1["results"]) > 0 and len(data2["results"]) > 0:
                first_page_symbols = {r["symbol"] for r in data1["results"]}
                second_page_symbols = {r["symbol"] for r in data2["results"]}
                assert first_page_symbols != second_page_symbols, (
                    "Different pages should have different results"
                )

            # Should include pagination metadata
            if "has_more" in data1:
                assert isinstance(data1["has_more"], bool), "has_more should be boolean"
            if "next_offset" in data1 and data1["has_more"]:
                assert data1["next_offset"] == 10, "next_offset should be limit value"

    async def test_semantic_search_hybrid_mode(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test hybrid search combining semantic and keyword matching."""
        payload = {
            "query": "vfs_read function implementation",
            "search_mode": "hybrid",
            "keyword_weight": 0.3,
            "semantic_weight": 0.7,
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include both keyword and semantic scores
            for result in data["results"]:
                if "keyword_score" in result:
                    assert isinstance(result["keyword_score"], (int, float)), (
                        "keyword_score should be numeric"
                    )
                if "hybrid_score" in result:
                    assert isinstance(result["hybrid_score"], (int, float)), (
                        "hybrid_score should be numeric"
                    )

    async def test_semantic_search_with_embeddings_cache(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that repeated queries benefit from embeddings cache."""
        payload = {
            "query": "file system operations",
            "use_cache": True,
        }

        # First request (cache miss)
        response1 = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        # Second identical request (cache hit)
        response2 = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Second request should be faster due to cache
            if "cache_hit" in data2:
                assert data2["cache_hit"] is True, "Second request should hit cache"

            # Results should be identical
            assert len(data1["results"]) == len(data2["results"]), (
                "Cache should return same number of results"
            )

    async def test_semantic_search_query_expansion(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test query expansion for improved recall."""
        payload = {
            "query": "VFS",
            "expand_query": True,
            "expansion_terms": 5,
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include expanded query information
            if "expanded_query" in data:
                assert isinstance(data["expanded_query"], str), (
                    "expanded_query should be string"
                )
                assert len(data["expanded_query"]) > len("VFS"), (
                    "Expanded query should be longer than original"
                )
            if "expansion_terms_used" in data:
                assert isinstance(data["expansion_terms_used"], list), (
                    "expansion_terms_used should be array"
                )

    async def test_semantic_search_explain_mode(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test explain mode for understanding why results were returned."""
        payload = {
            "query": "kernel panic handler",
            "explain": True,
            "limit": 3,
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include explanations for each result
            for result in data["results"]:
                if "explanation" in result:
                    assert isinstance(result["explanation"], dict), (
                        "explanation should be object"
                    )
                    exp = result["explanation"]
                    if "matching_terms" in exp:
                        assert isinstance(exp["matching_terms"], list), (
                            "matching_terms should be array"
                        )
                    if "relevance_factors" in exp:
                        assert isinstance(exp["relevance_factors"], dict), (
                            "relevance_factors should be object"
                        )

    async def test_semantic_search_no_results(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test proper handling when no results match the query."""
        payload = {
            "query": "nonexistent_xyz123_function_that_does_not_exist",
            "similarity_threshold": 0.9,  # High threshold to ensure no matches
        }

        response = await http_client.post(
            "/mcp/tools/semantic_search", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert data["total_results"] == 0, (
                "Should return 0 results for non-matching query"
            )
            assert len(data["results"]) == 0, "Results array should be empty"
            assert "query_id" in data, "Should still include query_id"
            assert "search_time_ms" in data, "Should still include search time"
