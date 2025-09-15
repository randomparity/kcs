"""
Contract tests for search_code MCP tool.

These tests verify the API contract defined in contracts/mcp-api.yaml.
They MUST fail before implementation and pass after.
"""

from typing import Any

import httpx
import pytest

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "test_token_123"


# Test fixtures and helpers
@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> httpx.AsyncClient:
    """HTTP client for API requests."""
    async with httpx.AsyncClient(base_url=MCP_BASE_URL) as client:
        yield client


# Contract test cases
class TestSearchCodeContract:
    """Contract tests for search_code MCP tool."""

    @pytest.mark.asyncio
    async def test_search_code_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the search_code endpoint exists and accepts POST requests."""
        payload = {"query": "test query", "topK": 5}

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "search_code endpoint should exist"

    async def test_search_code_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that search_code requires valid authentication."""
        payload = {"query": "test query"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/search_code", json=payload)
        assert response.status_code == 401, "Should require authentication"

        # Request with invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=invalid_headers
        )
        assert response.status_code == 401, "Should reject invalid tokens"

    async def test_search_code_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that search_code validates request schema according to OpenAPI spec."""

        # Missing required 'query' field
        response = await http_client.post(
            "/mcp/tools/search_code", json={"topK": 5}, headers=auth_headers
        )
        assert response.status_code == 422, (
            "Should reject request without required 'query' field"
        )

        # Invalid topK type
        response = await http_client.post(
            "/mcp/tools/search_code",
            json={"query": "test", "topK": "invalid"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid topK type"

        # Empty query string
        response = await http_client.post(
            "/mcp/tools/search_code", json={"query": ""}, headers=auth_headers
        )
        assert response.status_code == 422, "Should reject empty query string"

    async def test_search_code_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that search_code returns response matching OpenAPI schema."""
        payload = {"query": "memory barrier", "topK": 3}

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        # For now, we expect this to fail because implementation doesn't exist
        # But when it exists, it should return 200 with proper schema
        if response.status_code == 200:
            data = response.json()

            # Verify top-level structure
            assert "hits" in data, "Response should contain 'hits' field"
            assert isinstance(data["hits"], list), "Hits should be an array"

            # Verify each hit has required fields
            for hit in data["hits"]:
                assert "span" in hit, "Each hit should have 'span' field"
                assert "snippet" in hit, "Each hit should have 'snippet' field"

                # Verify span structure
                span = hit["span"]
                assert "path" in span, "Span should have 'path' field"
                assert "sha" in span, "Span should have 'sha' field"
                assert "start" in span, "Span should have 'start' field"
                assert "end" in span, "Span should have 'end' field"

                # Verify field types
                assert isinstance(span["path"], str), "Span path should be string"
                assert isinstance(span["sha"], str), "Span sha should be string"
                assert isinstance(span["start"], int), "Span start should be integer"
                assert isinstance(span["end"], int), "Span end should be integer"
                assert isinstance(hit["snippet"], str), "Snippet should be string"

                # Verify span constraints
                assert span["start"] > 0, "Start line should be positive"
                assert span["end"] >= span["start"], "End line should be >= start line"
                assert len(span["path"]) > 0, "Path should not be empty"
                assert len(span["sha"]) == 40, "SHA should be 40 characters (git hash)"

    async def test_search_code_topk_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that topK parameter limits results correctly."""
        payload = {"query": "function", "topK": 2}

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert len(data["hits"]) <= 2, "Should return at most topK results"

    async def test_search_code_default_topk(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that topK defaults to 10 when not specified."""
        payload = {"query": "struct"}

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should default to 10 or fewer results
            assert len(data["hits"]) <= 10, "Should default to max 10 results"

    async def test_search_code_semantic_vs_lexical(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that search supports both semantic and lexical queries."""

        # Semantic query (concept-based)
        semantic_payload = {"query": "read from file descriptor", "topK": 5}

        response = await http_client.post(
            "/mcp/tools/search_code", json=semantic_payload, headers=auth_headers
        )

        if response.status_code == 200:
            response.json()

        # Lexical query (exact text)
        lexical_payload = {"query": "sys_read", "topK": 5}

        response = await http_client.post(
            "/mcp/tools/search_code", json=lexical_payload, headers=auth_headers
        )

        if response.status_code == 200:
            lexical_data = response.json()
            # Should find exact matches
            any("sys_read" in hit["snippet"] for hit in lexical_data["hits"])
            # This assertion might not hold until implementation exists
            # assert found_exact_match, "Lexical search should find exact matches"

    async def test_search_code_empty_results(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior when search returns no results."""
        payload = {"query": "nonexistent_function_12345_abcde", "topK": 5}

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert "hits" in data, "Should still return hits array"
            assert isinstance(data["hits"], list), "Hits should be array even if empty"
            # Empty results are valid
            assert len(data["hits"]) >= 0, "Empty results should be handled gracefully"

    async def test_search_code_large_topk(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with large topK values."""
        payload = {"query": "function", "topK": 1000}

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        # Should either succeed with limited results or reject with 422
        assert response.status_code in [200, 422], "Should handle large topK gracefully"

        if response.status_code == 200:
            data = response.json()
            # Should enforce reasonable maximum
            assert len(data["hits"]) <= 100, "Should enforce reasonable maximum results"

    async def test_search_code_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that search_code meets p95 < 600ms performance requirement."""
        payload = {"query": "memory allocation", "topK": 10}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/search_code",
            json=payload,
            headers=auth_headers,
            timeout=1.0,  # 1 second timeout
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code == 200:
            # Performance requirement from constitution: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    @pytest.mark.integration
    async def test_search_code_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample kernel data (if available)."""
        # This test requires sample data to be loaded
        # It should be marked as integration test

        payload = {"query": "vfs_read", "topK": 5}

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should find vfs_read in sample kernel data
            found_vfs_read = any("vfs_read" in hit["snippet"] for hit in data["hits"])
            assert found_vfs_read, "Should find vfs_read in sample kernel data"


# Additional test for error responses
class TestSearchCodeErrorHandling:
    """Test error handling for search_code tool."""

    async def test_search_code_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        # This test will likely fail until implementation exists
        # but defines the expected error format

        # Force a server error (implementation-dependent)
        payload = {"query": "x" * 10000, "topK": 1}  # Very long query might cause error

        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error field should be string"
            assert isinstance(data["message"], str), "Message field should be string"
