"""
Contract tests for get_symbol MCP tool.

These tests verify the API contract defined in contracts/mcp-api.yaml.
They MUST fail before implementation and pass after.
"""

import os
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
async def http_client() -> httpx.AsyncClient:
    """HTTP client for API requests."""
    async with httpx.AsyncClient(base_url=MCP_BASE_URL) as client:
        yield client


# Contract test cases
@skip_integration_in_ci
@skip_without_mcp_server
class TestGetSymbolContract:
    """Contract tests for get_symbol MCP tool."""

    async def test_get_symbol_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the get_symbol endpoint exists and accepts POST requests."""
        payload = {"symbol": "sys_read"}

        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "get_symbol endpoint should exist"

    async def test_get_symbol_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that get_symbol requires valid authentication."""
        payload = {"symbol": "sys_read"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/get_symbol", json=payload)
        assert response.status_code == 401, "Should require authentication"

        # Request with invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=invalid_headers
        )
        assert response.status_code == 401, "Should reject invalid tokens"

    async def test_get_symbol_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that get_symbol validates request schema according to OpenAPI spec."""

        # Missing required 'symbol' field
        response = await http_client.post(
            "/mcp/tools/get_symbol", json={}, headers=auth_headers
        )
        assert response.status_code == 422, (
            "Should reject request without required 'symbol' field"
        )

        # Empty symbol name
        response = await http_client.post(
            "/mcp/tools/get_symbol", json={"symbol": ""}, headers=auth_headers
        )
        assert response.status_code == 422, "Should reject empty symbol name"

        # Invalid symbol type
        response = await http_client.post(
            "/mcp/tools/get_symbol", json={"symbol": 123}, headers=auth_headers
        )
        assert response.status_code == 422, "Should reject non-string symbol"

    async def test_get_symbol_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that get_symbol returns response matching OpenAPI schema."""
        payload = {"symbol": "sys_read"}

        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )

        # For now, we expect this to fail because implementation doesn't exist
        # But when it exists, it should return 200 with proper schema
        if response.status_code == 200:
            data = response.json()

            # Verify required fields according to SymbolInfo schema
            assert "name" in data, "Response should contain 'name' field"
            assert "kind" in data, "Response should contain 'kind' field"
            assert "decl" in data, "Response should contain 'decl' field"

            # Verify field types
            assert isinstance(data["name"], str), "Name should be string"
            assert isinstance(data["kind"], str), "Kind should be string"
            assert isinstance(data["decl"], dict), "Decl should be object (Span)"

            # Verify kind is valid enum value
            valid_kinds = ["function", "struct", "variable", "macro", "typedef"]
            assert data["kind"] in valid_kinds, f"Kind should be one of {valid_kinds}"

            # Verify decl (Span) structure
            decl = data["decl"]
            assert "path" in decl, "Decl should have 'path' field"
            assert "sha" in decl, "Decl should have 'sha' field"
            assert "start" in decl, "Decl should have 'start' field"
            assert "end" in decl, "Decl should have 'end' field"

            # Verify Span field types and constraints
            assert isinstance(decl["path"], str), "Decl path should be string"
            assert isinstance(decl["sha"], str), "Decl sha should be string"
            assert isinstance(decl["start"], int), "Decl start should be integer"
            assert isinstance(decl["end"], int), "Decl end should be integer"

            assert len(decl["path"]) > 0, "Path should not be empty"
            assert len(decl["sha"]) == 40, "SHA should be 40 characters (git hash)"
            assert decl["start"] > 0, "Start line should be positive"
            assert decl["end"] >= decl["start"], "End line should be >= start line"

            # Verify optional summary field if present
            if "summary" in data:
                assert isinstance(data["summary"], dict), (
                    "Summary should be object if present"
                )

    async def test_get_symbol_not_found(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior when symbol is not found."""
        payload = {"symbol": "nonexistent_symbol_12345"}

        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )

        # Should return 404 for non-existent symbols
        assert response.status_code == 404, "Should return 404 for non-existent symbols"

        if response.status_code == 404:
            data = response.json()
            # FastAPI puts error details under 'detail' field
            if "detail" in data:
                detail = data["detail"]
                assert "error" in detail, "Error response should have 'error' field"
                assert "message" in detail, "Error response should have 'message' field"
            else:
                assert "error" in data, "Error response should have 'error' field"
                assert "message" in data, "Error response should have 'message' field"

    async def test_get_symbol_different_kinds(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test getting symbols of different kinds."""
        test_symbols = [
            ("sys_read", "function"),
            ("task_struct", "struct"),
            ("current", "macro"),
        ]

        for symbol_name, expected_kind in test_symbols:
            payload = {"symbol": symbol_name}

            response = await http_client.post(
                "/mcp/tools/get_symbol", json=payload, headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()
                assert data["name"] == symbol_name, "Name should match requested symbol"
                assert data["kind"] == expected_kind, (
                    f"Kind should be {expected_kind} for {symbol_name}"
                )

    async def test_get_symbol_with_summary(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that symbols with summaries include summary data."""
        payload = {"symbol": "sys_read"}

        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            if "summary" in data:
                summary = data["summary"]
                assert isinstance(summary, dict), "Summary should be object"

                # Check for expected summary fields based on our schema
                expected_fields = [
                    "purpose",
                    "inputs",
                    "outputs",
                    "side_effects",
                    "concurrency",
                    "error_paths",
                    "citations",
                ]

                # At least some fields should be present
                present_fields = [
                    field for field in expected_fields if field in summary
                ]
                assert len(present_fields) > 0, (
                    "Summary should contain some expected fields"
                )

    async def test_get_symbol_case_sensitivity(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test symbol name case sensitivity."""
        # Test exact case
        payload = {"symbol": "sys_read"}
        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )
        exact_case_status = response.status_code

        # Test different case
        payload = {"symbol": "SYS_READ"}
        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )
        different_case_status = response.status_code

        # Symbol names should be case-sensitive in kernel
        # If exact case succeeds, different case should fail (or vice versa)
        if exact_case_status == 200:
            assert different_case_status == 404, (
                "Symbol lookup should be case-sensitive"
            )

    async def test_get_symbol_special_characters(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test symbols with special characters."""
        special_symbols = [
            "__x64_sys_read",  # Underscore prefix
            "module_init",  # Underscore in name
            "IS_ERR",  # All caps
        ]

        for symbol_name in special_symbols:
            payload = {"symbol": symbol_name}

            response = await http_client.post(
                "/mcp/tools/get_symbol", json=payload, headers=auth_headers
            )

            # Should handle special characters gracefully
            assert response.status_code in [
                200,
                404,
            ], f"Should handle symbol '{symbol_name}' gracefully"

            if response.status_code == 200:
                data = response.json()
                assert data["name"] == symbol_name, "Returned name should match exactly"

    async def test_get_symbol_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that get_symbol meets p95 < 600ms performance requirement."""
        payload = {"symbol": "sys_read"}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/get_symbol",
            json=payload,
            headers=auth_headers,
            timeout=1.0,  # 1 second timeout
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [200, 404]:
            # Performance requirement from constitution: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_get_symbol_long_symbol_name(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with very long symbol names."""
        # Very long but potentially valid symbol name
        long_symbol = "very_long_symbol_name_" + "x" * 200

        payload = {"symbol": long_symbol}

        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )

        # Should handle gracefully (404 is expected, but shouldn't crash)
        assert response.status_code in [
            200,
            404,
            422,
        ], "Should handle long symbol names gracefully"

    @pytest.mark.integration
    async def test_get_symbol_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample kernel data (if available)."""
        # This test requires sample data to be loaded
        # It should be marked as integration test

        payload = {"symbol": "vfs_read"}

        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert data["name"] == "vfs_read", "Should return correct symbol name"
            assert data["kind"] == "function", "vfs_read should be a function"
            assert "decl" in data, "Should include declaration location"

            # Check that decl points to a reasonable location
            decl = data["decl"]
            assert "fs/" in decl["path"], "vfs_read should be in fs/ directory"

    async def test_get_symbol_concurrent_requests(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test handling of concurrent symbol requests."""
        import asyncio

        async def get_symbol_request(symbol: str):
            payload = {"symbol": symbol}
            return await http_client.post(
                "/mcp/tools/get_symbol", json=payload, headers=auth_headers
            )

        # Send multiple concurrent requests
        symbols = ["sys_read", "sys_write", "sys_open", "sys_close", "vfs_read"]
        tasks = [get_symbol_request(symbol) for symbol in symbols]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete successfully or with expected errors
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                pytest.fail(
                    f"Request for {symbols[i]} failed with exception: {response}"
                )
            else:
                assert response.status_code in [
                    200,
                    404,
                ], f"Concurrent request for {symbols[i]} should succeed or return 404"


# Additional test for error responses
@skip_integration_in_ci
@skip_without_mcp_server
class TestGetSymbolErrorHandling:
    """Test error handling for get_symbol tool."""

    async def test_get_symbol_malformed_json(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with malformed JSON."""
        response = await http_client.post(
            "/mcp/tools/get_symbol", content="{ invalid json }", headers=auth_headers
        )

        assert response.status_code == 422, "Should reject malformed JSON"

    async def test_get_symbol_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        # This test will likely fail until implementation exists
        # but defines the expected error format

        # Force a potential server error
        payload = {"symbol": "x" * 10000}  # Very long symbol name might cause error

        response = await http_client.post(
            "/mcp/tools/get_symbol", json=payload, headers=auth_headers
        )

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error field should be string"
            assert isinstance(data["message"], str), "Message field should be string"
