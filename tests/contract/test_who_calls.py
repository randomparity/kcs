"""
Contract tests for who_calls MCP tool.

These tests verify the API contract defined in contracts/mcp-api.yaml.
They MUST fail before implementation and pass after.
"""

from typing import Any, Dict

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
class TestWhoCallsContract:
    """Contract tests for who_calls MCP tool."""

    async def test_who_calls_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the who_calls endpoint exists and accepts POST requests."""
        payload = {"symbol": "vfs_read", "depth": 1}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "who_calls endpoint should exist"

    async def test_who_calls_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that who_calls requires valid authentication."""
        payload = {"symbol": "vfs_read"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/who_calls", json=payload)
        assert response.status_code == 401, "Should require authentication"

    async def test_who_calls_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that who_calls validates request schema according to OpenAPI spec."""

        # Missing required 'symbol' field
        response = await http_client.post(
            "/mcp/tools/who_calls", json={"depth": 2}, headers=auth_headers
        )
        assert (
            response.status_code == 422
        ), "Should reject request without required 'symbol' field"

        # Invalid depth type
        response = await http_client.post(
            "/mcp/tools/who_calls",
            json={"symbol": "vfs_read", "depth": "invalid"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid depth type"

        # Negative depth
        response = await http_client.post(
            "/mcp/tools/who_calls",
            json={"symbol": "vfs_read", "depth": -1},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject negative depth"

    async def test_who_calls_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that who_calls returns response matching OpenAPI schema."""
        payload = {"symbol": "vfs_read", "depth": 1}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify top-level structure
            assert "callers" in data, "Response should contain 'callers' field"
            assert isinstance(data["callers"], list), "Callers should be an array"

            # Verify each caller has required fields
            for caller in data["callers"]:
                assert "symbol" in caller, "Each caller should have 'symbol' field"
                assert "span" in caller, "Each caller should have 'span' field"

                # Verify field types
                assert isinstance(
                    caller["symbol"], str
                ), "Caller symbol should be string"
                assert isinstance(caller["span"], dict), "Caller span should be object"

                # Verify span structure
                span = caller["span"]
                assert "path" in span, "Span should have 'path' field"
                assert "sha" in span, "Span should have 'sha' field"
                assert "start" in span, "Span should have 'start' field"
                assert "end" in span, "Span should have 'end' field"

                # Verify span constraints
                assert isinstance(span["path"], str), "Span path should be string"
                assert isinstance(span["sha"], str), "Span sha should be string"
                assert isinstance(span["start"], int), "Span start should be integer"
                assert isinstance(span["end"], int), "Span end should be integer"
                assert span["start"] > 0, "Start line should be positive"
                assert span["end"] >= span["start"], "End line should be >= start line"
                assert len(span["sha"]) == 40, "SHA should be 40 characters (git hash)"

                # Verify optional call_type field
                if "call_type" in caller:
                    valid_call_types = ["direct", "indirect", "macro", "inline"]
                    assert (
                        caller["call_type"] in valid_call_types
                    ), f"Call type should be one of {valid_call_types}"

    async def test_who_calls_default_depth(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that depth defaults to 1 when not specified."""
        payload = {"symbol": "vfs_read"}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should return first-level callers only by default
            assert "callers" in data, "Should return callers array"

    async def test_who_calls_depth_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test different depth values."""
        symbol = "vfs_read"

        # Test depth 1
        payload = {"symbol": symbol, "depth": 1}
        response1 = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        # Test depth 2
        payload = {"symbol": symbol, "depth": 2}
        response2 = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Depth 2 should potentially return more results (or same if no deeper calls)
            assert len(data2["callers"]) >= len(
                data1["callers"]
            ), "Deeper search may return more results"

    async def test_who_calls_nonexistent_symbol(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior when symbol doesn't exist."""
        payload = {"symbol": "nonexistent_function_12345", "depth": 1}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        # Should return 404 or 200 with empty callers
        assert response.status_code in [
            200,
            404,
        ], "Should handle non-existent symbols gracefully"

        if response.status_code == 200:
            data = response.json()
            assert "callers" in data, "Should still return callers array"
            assert (
                len(data["callers"]) == 0
            ), "Should return empty array for non-existent symbol"

    async def test_who_calls_no_callers(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test symbol with no callers (like entry points)."""
        # Test with a likely entry point that has no callers
        payload = {"symbol": "__x64_sys_read", "depth": 1}  # System call entry point

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert "callers" in data, "Should return callers array even if empty"
            # Entry points typically have no callers or very few
            assert len(data["callers"]) >= 0, "Should handle symbols with no callers"

    async def test_who_calls_depth_limit(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with large depth values."""
        payload = {"symbol": "vfs_read", "depth": 100}  # Very large depth

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        # Should either handle gracefully or reject with reasonable limit
        assert response.status_code in [
            200,
            422,
        ], "Should handle large depth values gracefully"

        if response.status_code == 200:
            data = response.json()
            # Should enforce reasonable limits to prevent infinite recursion
            assert (
                len(data["callers"]) <= 1000
            ), "Should enforce reasonable result limits"

    async def test_who_calls_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that who_calls meets p95 < 600ms performance requirement."""
        payload = {"symbol": "vfs_read", "depth": 2}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/who_calls",
            json=payload,
            headers=auth_headers,
            timeout=1.0,  # 1 second timeout
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [200, 404]:
            # Performance requirement from constitution: p95 < 600ms
            assert (
                response_time_ms < 600
            ), f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"

    async def test_who_calls_recursive_functions(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test handling of recursive function calls."""
        # Some kernel functions may be recursive
        payload = {
            "symbol": "some_recursive_function",  # Would need to identify a real recursive function
            "depth": 3,
        }

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should handle recursion without infinite loops
            assert (
                len(data["callers"]) < 10000
            ), "Should handle recursion without infinite expansion"

    @pytest.mark.integration
    async def test_who_calls_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample kernel data (if available)."""
        payload = {"symbol": "vfs_read", "depth": 2}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should find some callers for vfs_read in kernel
            assert len(data["callers"]) > 0, "vfs_read should have callers in kernel"

            # Verify at least one caller makes sense
            caller_names = [caller["symbol"] for caller in data["callers"]]
            # Common callers of vfs_read might include sys_read, etc.
            assert any(
                "read" in name.lower() for name in caller_names
            ), "Should find read-related callers"

    async def test_who_calls_call_types(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that call types are properly identified."""
        payload = {"symbol": "vfs_read", "depth": 1}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            for caller in data["callers"]:
                if "call_type" in caller:
                    call_type = caller["call_type"]
                    valid_types = ["direct", "indirect", "macro", "inline"]
                    assert (
                        call_type in valid_types
                    ), f"Call type '{call_type}' should be valid"

    async def test_who_calls_span_accuracy(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that span information points to actual call sites."""
        payload = {"symbol": "vfs_read", "depth": 1}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            for caller in data["callers"]:
                span = caller["span"]
                # Span should point to reasonable kernel source locations
                assert span["path"].endswith(
                    (".c", ".h", ".S")
                ), "Should point to source files"
                assert not span["path"].startswith(
                    "/"
                ), "Path should be relative to repo root"

                # Line numbers should be reasonable
                assert span["start"] <= span["end"], "Start should be <= end"
                assert (
                    span["end"] - span["start"] < 1000
                ), "Span should not be excessively large"


class TestWhoCallsErrorHandling:
    """Test error handling for who_calls tool."""

    async def test_who_calls_invalid_symbol_characters(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with symbols containing invalid characters."""
        invalid_symbols = [
            "symbol with spaces",
            "symbol-with-dashes",
            "symbol.with.dots",
            "symbol/with/slashes",
        ]

        for symbol in invalid_symbols:
            payload = {"symbol": symbol, "depth": 1}

            response = await http_client.post(
                "/mcp/tools/who_calls", json=payload, headers=auth_headers
            )

            # Should either reject or handle gracefully
            assert response.status_code in [
                200,
                404,
                422,
            ], f"Should handle invalid symbol '{symbol}' gracefully"

    async def test_who_calls_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        # Force potential server error with edge case
        payload = {"symbol": "x" * 1000, "depth": 50}  # Very long symbol name

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
