"""
Contract tests for who_calls MCP endpoint with call graph support.

These tests verify the API contract defined in contracts/mcp-endpoints.json.
They MUST fail before implementation and pass after.

Enhanced contract with call graph features:
- function_name parameter (not symbol)
- callers array with caller_name and call_sites
- call_sites include call_type classification (Direct, Indirect, Macro)
- total_callers count
- config parameter support
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


# Enhanced contract test cases for call graph support
@skip_integration_in_ci
@skip_without_mcp_server
class TestWhoCallsEnhancedContract:
    """Contract tests for who_calls MCP endpoint with call graph data."""

    async def test_who_calls_enhanced_schema_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that who_calls accepts enhanced request schema."""
        # Enhanced schema uses function_name (not symbol)
        payload = {"function_name": "ext4_create", "config": "defconfig"}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists) or 422 (schema valid)
        assert response.status_code not in [404, 422], (
            f"Enhanced schema should be valid, got {response.status_code}"
        )

    async def test_who_calls_enhanced_response_structure(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that who_calls returns enhanced response schema."""
        payload = {"function_name": "vfs_read"}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Enhanced contract: must have callers and total_callers
            assert "callers" in data, "Response must contain 'callers' field"
            assert "total_callers" in data, (
                "Response must contain 'total_callers' field"
            )

            # total_callers should be integer
            assert isinstance(data["total_callers"], int), (
                "total_callers should be integer"
            )

            # total_callers should match callers length
            assert data["total_callers"] == len(data["callers"]), (
                "total_callers should match length of callers array"
            )

            # Verify caller structure
            for caller in data["callers"]:
                assert "caller_name" in caller, "Each caller must have 'caller_name'"
                assert "call_sites" in caller, "Each caller must have 'call_sites'"

                assert isinstance(caller["caller_name"], str), (
                    "caller_name should be string"
                )
                assert isinstance(caller["call_sites"], list), (
                    "call_sites should be array"
                )

                # Verify call site structure
                for site in caller["call_sites"]:
                    assert "file_path" in site, "Call site must have 'file_path'"
                    assert "line_number" in site, "Call site must have 'line_number'"
                    assert "call_type" in site, "Call site must have 'call_type'"

                    assert isinstance(site["file_path"], str), (
                        "file_path should be string"
                    )
                    assert isinstance(site["line_number"], int), (
                        "line_number should be integer"
                    )
                    assert isinstance(site["call_type"], str), (
                        "call_type should be string"
                    )

                    # Verify call_type is valid enum value
                    valid_call_types = ["Direct", "Indirect", "Macro"]
                    assert site["call_type"] in valid_call_types, (
                        f"call_type '{site['call_type']}' must be one of {valid_call_types}"
                    )

                    # line_number should be positive
                    assert site["line_number"] > 0, "line_number should be positive"

    async def test_who_calls_config_parameter_support(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test config parameter for kernel configuration context."""
        # Test without config
        payload1 = {"function_name": "vfs_read"}
        response1 = await http_client.post(
            "/mcp/tools/who_calls", json=payload1, headers=auth_headers
        )

        # Test with config
        payload2 = {"function_name": "vfs_read", "config": "defconfig"}
        response2 = await http_client.post(
            "/mcp/tools/who_calls", json=payload2, headers=auth_headers
        )

        # Both should be valid requests
        assert response1.status_code in [200, 404], (
            "Request without config should be valid"
        )
        assert response2.status_code in [200, 404], (
            "Request with config should be valid"
        )

        # If both succeed, results might differ based on configuration
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Both should have valid structure
            assert "callers" in data1 and "total_callers" in data1
            assert "callers" in data2 and "total_callers" in data2

    async def test_who_calls_call_type_classification(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that call types are properly classified in call sites."""
        payload = {"function_name": "vfs_read"}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Collect all call types from all call sites
            all_call_types = []
            for caller in data["callers"]:
                for site in caller["call_sites"]:
                    all_call_types.append(site["call_type"])

            # Should have at least some call types
            if all_call_types:
                # All call types should be valid
                valid_types = {"Direct", "Indirect", "Macro"}
                for call_type in all_call_types:
                    assert call_type in valid_types, f"Invalid call type: {call_type}"

                # Should ideally have some Direct calls (most common)
                direct_calls = [ct for ct in all_call_types if ct == "Direct"]
                assert len(direct_calls) > 0, (
                    "Should have at least some Direct function calls"
                )

    async def test_who_calls_multiple_call_sites_per_caller(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that callers can have multiple call sites."""
        # Use a commonly called function that might have multiple call sites
        payload = {"function_name": "kmalloc"}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Look for callers with multiple call sites
            multi_site_callers = [
                caller for caller in data["callers"] if len(caller["call_sites"]) > 1
            ]

            if multi_site_callers:
                caller = multi_site_callers[0]
                # Each call site should have different line numbers
                line_numbers = [site["line_number"] for site in caller["call_sites"]]
                assert len(set(line_numbers)) > 1, (
                    "Multiple call sites should have different line numbers"
                )

    async def test_who_calls_enhanced_error_handling(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test error handling with enhanced schema."""

        # Test missing required field (function_name)
        response = await http_client.post(
            "/mcp/tools/who_calls", json={"config": "defconfig"}, headers=auth_headers
        )
        assert response.status_code == 422, (
            "Should reject request without required function_name"
        )

        # Test invalid config value
        response = await http_client.post(
            "/mcp/tools/who_calls",
            json={"function_name": "test", "config": 123},  # Should be string
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject request with invalid config type"
        )

    async def test_who_calls_nonexistent_function_enhanced(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with nonexistent function in enhanced schema."""
        payload = {"function_name": "nonexistent_function_xyz_12345"}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        assert response.status_code in [200, 404], (
            "Should handle nonexistent function gracefully"
        )

        if response.status_code == 200:
            data = response.json()
            assert data["total_callers"] == 0, (
                "Nonexistent function should have 0 callers"
            )
            assert len(data["callers"]) == 0, (
                "Nonexistent function should have empty callers array"
            )

    async def test_who_calls_performance_with_call_graph(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test performance requirement with call graph data processing."""
        payload = {"function_name": "vfs_read", "config": "defconfig"}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers, timeout=1.0
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [200, 404]:
            # Constitutional requirement: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_who_calls_backward_compatibility(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that old 'symbol' parameter still works (backward compatibility)."""
        # Test old schema with 'symbol' parameter
        old_payload = {"symbol": "vfs_read", "depth": 1}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=old_payload, headers=auth_headers
        )

        # Should either:
        # 1. Work with old schema (backward compatible)
        # 2. Return 422 with clear error about using function_name instead
        assert response.status_code in [200, 404, 422], (
            "Should handle old schema gracefully"
        )

        if response.status_code == 422:
            error_data = response.json()
            # Error message should guide users to new schema
            error_msg = str(error_data).lower()
            assert any(word in error_msg for word in ["function_name", "schema"]), (
                "Error should mention new schema requirements"
            )


@skip_integration_in_ci
@skip_without_mcp_server
class TestWhoCallsIntegrationWithCallGraph:
    """Integration tests for who_calls with real call graph data."""

    async def test_who_calls_kernel_function_integration(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test with actual kernel function that should have callers."""
        payload = {"function_name": "ext4_create", "config": "defconfig"}

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # ext4_create should have some callers in kernel
            assert data["total_callers"] >= 0, "Should return caller count"

            if data["total_callers"] > 0:
                # Verify realistic kernel call patterns
                for caller in data["callers"]:
                    # Caller names should be reasonable kernel function names
                    assert caller["caller_name"], "Caller name should not be empty"
                    assert len(caller["call_sites"]) > 0, (
                        "Each caller should have at least one call site"
                    )

                    for site in caller["call_sites"]:
                        # Should point to real kernel source files
                        assert site["file_path"].endswith((".c", ".h")), (
                            "Should point to C/header files"
                        )
                        assert not site["file_path"].startswith("/"), (
                            "Path should be relative to kernel root"
                        )
                        assert site["line_number"] > 0, "Line number should be positive"

    async def test_who_calls_call_type_distribution(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test realistic distribution of call types in kernel."""
        payload = {"function_name": "kmalloc"}  # Commonly called function

        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            if data["total_callers"] > 0:
                # Count call types
                call_type_counts = {"Direct": 0, "Indirect": 0, "Macro": 0}

                for caller in data["callers"]:
                    for site in caller["call_sites"]:
                        call_type_counts[site["call_type"]] += 1

                total_calls = sum(call_type_counts.values())
                assert total_calls > 0, "Should have some function calls"

                # Most calls should be Direct in typical kernel code
                direct_ratio = call_type_counts["Direct"] / total_calls
                assert direct_ratio > 0.5, (
                    f"Expected majority direct calls, got {direct_ratio:.2%} direct"
                )
