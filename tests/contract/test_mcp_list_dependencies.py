"""
Contract tests for list_dependencies MCP endpoint with call graph support.

These tests verify the API contract defined in contracts/mcp-endpoints.json.
They MUST fail before implementation and pass after.

Enhanced contract with call graph features:
- function_name parameter for function to analyze
- dependencies array with callee_name and call_sites
- call_sites include call_type, depth, file_path, line_number
- max_depth parameter (default 5, range 1-20)
- total_dependencies count
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


# Contract test cases for list_dependencies
@skip_integration_in_ci
@skip_without_mcp_server
class TestListDependenciesContractIntegration:
    """Contract tests for list_dependencies MCP endpoint with call graph data."""

    async def test_list_dependencies_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that list_dependencies endpoint exists and accepts requests."""
        payload = {"function_name": "ext4_create"}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "list_dependencies endpoint should exist"

    async def test_list_dependencies_schema_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test request schema validation according to contract."""
        # Valid request with all parameters
        payload = {
            "function_name": "ext4_create",
            "config": "defconfig",
            "max_depth": 3,
        }

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        assert response.status_code not in [422], "Valid schema should be accepted"

        # Test missing required field
        invalid_payload = {"config": "defconfig", "max_depth": 2}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=invalid_payload, headers=auth_headers
        )

        assert response.status_code == 422, (
            "Should reject request without required function_name"
        )

    async def test_list_dependencies_max_depth_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test max_depth parameter validation."""
        # Test minimum boundary (1)
        response = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": "test", "max_depth": 1},
            headers=auth_headers,
        )
        assert response.status_code != 422, "max_depth=1 should be valid"

        # Test maximum boundary (20)
        response = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": "test", "max_depth": 20},
            headers=auth_headers,
        )
        assert response.status_code != 422, "max_depth=20 should be valid"

        # Test below minimum (0)
        response = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": "test", "max_depth": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422, "max_depth=0 should be rejected"

        # Test above maximum (21)
        response = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": "test", "max_depth": 21},
            headers=auth_headers,
        )
        assert response.status_code == 422, "max_depth=21 should be rejected"

        # Test invalid type
        response = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": "test", "max_depth": "invalid"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "max_depth should be integer"

    async def test_list_dependencies_response_structure(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test response schema matches contract specification."""
        payload = {"function_name": "vfs_write"}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Contract: must have dependencies and total_dependencies
            assert "dependencies" in data, "Response must contain 'dependencies' field"
            assert "total_dependencies" in data, (
                "Response must contain 'total_dependencies' field"
            )

            # total_dependencies should be integer
            assert isinstance(data["total_dependencies"], int), (
                "total_dependencies should be integer"
            )

            # dependencies should be array
            assert isinstance(data["dependencies"], list), (
                "dependencies should be array"
            )

            # total_dependencies should match dependencies length
            assert data["total_dependencies"] == len(data["dependencies"]), (
                "total_dependencies should match length of dependencies array"
            )

            # Verify dependency structure
            for dependency in data["dependencies"]:
                assert "callee_name" in dependency, (
                    "Each dependency must have 'callee_name'"
                )
                assert "call_sites" in dependency, (
                    "Each dependency must have 'call_sites'"
                )

                assert isinstance(dependency["callee_name"], str), (
                    "callee_name should be string"
                )
                assert isinstance(dependency["call_sites"], list), (
                    "call_sites should be array"
                )

                # Verify call site structure
                for site in dependency["call_sites"]:
                    required_fields = ["file_path", "line_number", "call_type", "depth"]
                    for field in required_fields:
                        assert field in site, f"Call site must have '{field}' field"

                    # Verify field types
                    assert isinstance(site["file_path"], str), (
                        "file_path should be string"
                    )
                    assert isinstance(site["line_number"], int), (
                        "line_number should be integer"
                    )
                    assert isinstance(site["call_type"], str), (
                        "call_type should be string"
                    )
                    assert isinstance(site["depth"], int), "depth should be integer"

                    # Verify call_type is valid
                    valid_call_types = ["Direct", "Indirect", "Macro"]
                    assert site["call_type"] in valid_call_types, (
                        f"call_type '{site['call_type']}' must be valid"
                    )

                    # Verify constraints
                    assert site["line_number"] > 0, "line_number should be positive"
                    assert site["depth"] >= 1, "depth should be >= 1"

    async def test_list_dependencies_max_depth_default(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that max_depth defaults to 5 when not specified."""
        payload = {"function_name": "vfs_write"}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should use default depth of 5
            if data["total_dependencies"] > 0:
                # Check that no call sites have depth > 5
                max_depth_found = 0
                for dependency in data["dependencies"]:
                    for site in dependency["call_sites"]:
                        max_depth_found = max(max_depth_found, site["depth"])

                assert max_depth_found <= 5, (
                    f"Default max_depth should be 5, found depth {max_depth_found}"
                )

    async def test_list_dependencies_depth_parameter_effect(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that max_depth parameter affects results depth."""
        function_name = "vfs_write"

        # Test depth 1
        response1 = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": function_name, "max_depth": 1},
            headers=auth_headers,
        )

        # Test depth 3
        response3 = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": function_name, "max_depth": 3},
            headers=auth_headers,
        )

        if response1.status_code == 200 and response3.status_code == 200:
            data1 = response1.json()
            data3 = response3.json()

            # Depth 3 should potentially have more results than depth 1
            assert data3["total_dependencies"] >= data1["total_dependencies"], (
                "Deeper search should find same or more dependencies"
            )

            # Verify depth constraints
            if data1["total_dependencies"] > 0:
                for dep in data1["dependencies"]:
                    for site in dep["call_sites"]:
                        assert site["depth"] <= 1, (
                            "Depth 1 search should only find depth 1 calls"
                        )

            if data3["total_dependencies"] > 0:
                for dep in data3["dependencies"]:
                    for site in dep["call_sites"]:
                        assert site["depth"] <= 3, (
                            "Depth 3 search should only find calls up to depth 3"
                        )

    async def test_list_dependencies_config_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test config parameter for kernel configuration context."""
        function_name = "ext4_create"

        # Test without config
        response1 = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": function_name},
            headers=auth_headers,
        )

        # Test with config
        response2 = await http_client.post(
            "/mcp/tools/list_dependencies",
            json={"function_name": function_name, "config": "defconfig"},
            headers=auth_headers,
        )

        # Both should be valid requests
        assert response1.status_code in [200, 404], (
            "Request without config should be valid"
        )
        assert response2.status_code in [200, 404], (
            "Request with config should be valid"
        )

        # Results might differ based on configuration context
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Both should have valid structure
            assert "dependencies" in data1 and "total_dependencies" in data1
            assert "dependencies" in data2 and "total_dependencies" in data2

    async def test_list_dependencies_nonexistent_function(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with nonexistent function."""
        payload = {"function_name": "nonexistent_function_xyz_999"}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        assert response.status_code in [200, 404], (
            "Should handle nonexistent function gracefully"
        )

        if response.status_code == 200:
            data = response.json()
            assert data["total_dependencies"] == 0, (
                "Nonexistent function should have 0 dependencies"
            )
            assert len(data["dependencies"]) == 0, (
                "Nonexistent function should have empty dependencies array"
            )

    async def test_list_dependencies_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test performance meets p95 < 600ms requirement."""
        payload = {"function_name": "vfs_write", "max_depth": 3}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/list_dependencies",
            json=payload,
            headers=auth_headers,
            timeout=1.0,
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [200, 404]:
            # Constitutional requirement: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_list_dependencies_call_type_classification(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that call types are properly classified."""
        payload = {"function_name": "kmalloc", "max_depth": 2}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Collect all call types
            all_call_types = []
            for dependency in data["dependencies"]:
                for site in dependency["call_sites"]:
                    all_call_types.append(site["call_type"])

            if all_call_types:
                # All should be valid
                valid_types = {"Direct", "Indirect", "Macro"}
                for call_type in all_call_types:
                    assert call_type in valid_types, f"Invalid call type: {call_type}"

                # Should have some Direct calls (most common)
                direct_calls = [ct for ct in all_call_types if ct == "Direct"]
                assert len(direct_calls) > 0, "Should have at least some Direct calls"

    async def test_list_dependencies_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that list_dependencies requires authentication."""
        payload = {"function_name": "test"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/list_dependencies", json=payload)

        assert response.status_code == 401, "Should require authentication"


@skip_integration_in_ci
@skip_without_mcp_server
class TestListDependenciesIntegration:
    """Integration tests for list_dependencies with realistic data."""

    async def test_list_dependencies_kernel_function_integration(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test with actual kernel function that has dependencies."""
        payload = {"function_name": "ext4_create", "max_depth": 2}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # ext4_create should call other functions
            if data["total_dependencies"] > 0:
                # Verify realistic kernel dependency patterns
                dependency_names = [dep["callee_name"] for dep in data["dependencies"]]

                # Should have some reasonable kernel function names
                assert any(name for name in dependency_names if name), (
                    "Should have non-empty dependency names"
                )

                # Check call site details
                for dependency in data["dependencies"]:
                    assert dependency["callee_name"], (
                        "Dependency name should not be empty"
                    )

                    for site in dependency["call_sites"]:
                        # Should point to kernel source files
                        assert site["file_path"].endswith((".c", ".h")), (
                            "Should point to C/header files"
                        )
                        assert not site["file_path"].startswith("/"), (
                            "Path should be relative to kernel root"
                        )
                        assert site["line_number"] > 0, "Line number should be positive"

    async def test_list_dependencies_depth_chain_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that depth chains are logically consistent."""
        payload = {"function_name": "ext4_create", "max_depth": 5}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            if data["total_dependencies"] > 0:
                # Group call sites by depth
                depth_counts = {}
                for dependency in data["dependencies"]:
                    for site in dependency["call_sites"]:
                        depth = site["depth"]
                        depth_counts[depth] = depth_counts.get(depth, 0) + 1

                # Should have depth 1 calls (direct dependencies)
                assert 1 in depth_counts, "Should have at least depth 1 dependencies"

                # Deeper levels should exist only if shallower levels exist
                max_depth_found = max(depth_counts.keys()) if depth_counts else 0
                for depth in range(1, max_depth_found + 1):
                    assert depth in depth_counts, (
                        f"Should have depth {depth} calls if depth {max_depth_found} exists"
                    )

    async def test_list_dependencies_call_site_accuracy(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that call sites point to actual call locations."""
        payload = {"function_name": "kmalloc", "max_depth": 1}

        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            for dependency in data["dependencies"]:
                for site in dependency["call_sites"]:
                    # File path should be reasonable
                    assert not site["file_path"].startswith("/tmp"), (
                        "Should not point to temporary files"
                    )

                    # Line numbers should be reasonable
                    assert site["line_number"] < 100000, (
                        "Line number should be reasonable for source files"
                    )

                    # Depth should match request constraint
                    assert site["depth"] <= 1, (
                        "Depth should not exceed requested max_depth"
                    )
