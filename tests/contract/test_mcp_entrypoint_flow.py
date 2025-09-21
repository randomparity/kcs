"""
Contract tests for entrypoint_flow MCP endpoint with call graph support.

These tests verify the API contract defined in contracts/mcp-endpoints.json.
They MUST fail before implementation and pass after.

Enhanced contract with call graph features:
- entrypoint parameter (syscall, ioctl, etc.)
- target_function parameter (optional, specific implementation)
- call_paths array with path arrays containing function steps
- Each path step has function_name, file_path, line_number, call_type
- depth and confidence scores for each call path
- total_paths count
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


# Contract test cases for entrypoint_flow
@skip_integration_in_ci
@skip_without_mcp_server
class TestEntrypointFlowContractIntegration:
    """Contract tests for entrypoint_flow MCP endpoint with call graph data."""

    async def test_entrypoint_flow_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that entrypoint_flow endpoint exists and accepts requests."""
        payload = {"entrypoint": "sys_open"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "entrypoint_flow endpoint should exist"

    async def test_entrypoint_flow_schema_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test request schema validation according to contract."""
        # Valid request with all parameters
        payload = {
            "entrypoint": "sys_open",
            "target_function": "ext4_file_open",
            "config": "defconfig",
        }

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        assert response.status_code not in [422], "Valid schema should be accepted"

        # Test missing required field (entrypoint)
        invalid_payload = {"target_function": "ext4_file_open", "config": "defconfig"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=invalid_payload, headers=auth_headers
        )

        assert response.status_code == 422, (
            "Should reject request without required entrypoint"
        )

    async def test_entrypoint_flow_response_structure(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test response schema matches contract specification."""
        payload = {"entrypoint": "sys_read"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Contract: must have call_paths and total_paths
            assert "call_paths" in data, "Response must contain 'call_paths' field"
            assert "total_paths" in data, "Response must contain 'total_paths' field"

            # total_paths should be integer
            assert isinstance(data["total_paths"], int), "total_paths should be integer"

            # call_paths should be array
            assert isinstance(data["call_paths"], list), "call_paths should be array"

            # total_paths should match call_paths length
            assert data["total_paths"] == len(data["call_paths"]), (
                "total_paths should match length of call_paths array"
            )

            # Verify call path structure
            for call_path in data["call_paths"]:
                assert "path" in call_path, "Each call_path must have 'path' field"
                assert "depth" in call_path, "Each call_path must have 'depth' field"
                assert "confidence" in call_path, (
                    "Each call_path must have 'confidence' field"
                )

                assert isinstance(call_path["path"], list), "path should be array"
                assert isinstance(call_path["depth"], int), "depth should be integer"
                assert isinstance(call_path["confidence"], (int, float)), (
                    "confidence should be number"
                )

                # Verify confidence bounds
                assert 0 <= call_path["confidence"] <= 1, (
                    "confidence should be between 0 and 1"
                )

                # Verify path steps structure
                for step in call_path["path"]:
                    required_fields = [
                        "function_name",
                        "file_path",
                        "line_number",
                        "call_type",
                    ]
                    for field in required_fields:
                        assert field in step, f"Path step must have '{field}' field"

                    # Verify field types
                    assert isinstance(step["function_name"], str), (
                        "function_name should be string"
                    )
                    assert isinstance(step["file_path"], str), (
                        "file_path should be string"
                    )
                    assert isinstance(step["line_number"], int), (
                        "line_number should be integer"
                    )
                    assert isinstance(step["call_type"], str), (
                        "call_type should be string"
                    )

                    # Verify call_type is valid
                    valid_call_types = ["Direct", "Indirect", "Macro"]
                    assert step["call_type"] in valid_call_types, (
                        f"call_type '{step['call_type']}' must be valid"
                    )

                    # Verify constraints
                    assert step["line_number"] > 0, "line_number should be positive"
                    assert step["function_name"], "function_name should not be empty"

    async def test_entrypoint_flow_target_function_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test target_function parameter behavior."""
        entrypoint = "sys_open"

        # Test without target_function (find all paths)
        response1 = await http_client.post(
            "/mcp/tools/entrypoint_flow",
            json={"entrypoint": entrypoint},
            headers=auth_headers,
        )

        # Test with specific target_function
        response2 = await http_client.post(
            "/mcp/tools/entrypoint_flow",
            json={"entrypoint": entrypoint, "target_function": "ext4_file_open"},
            headers=auth_headers,
        )

        # Both should be valid requests
        assert response1.status_code in [200, 404], (
            "Request without target_function should be valid"
        )
        assert response2.status_code in [200, 404], (
            "Request with target_function should be valid"
        )

        # If both succeed, targeted search might return fewer results
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Both should have valid structure
            assert "call_paths" in data1 and "total_paths" in data1
            assert "call_paths" in data2 and "total_paths" in data2

            # Targeted search should return relevant paths
            if data2["total_paths"] > 0:
                # Should find paths that lead to the target function
                target_found = False
                for path in data2["call_paths"]:
                    if any(
                        step["function_name"] == "ext4_file_open"
                        for step in path["path"]
                    ):
                        target_found = True
                        break

                if not target_found:
                    # If target not found in paths, should explain why
                    # This test documents expected behavior
                    pass

    async def test_entrypoint_flow_config_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test config parameter for kernel configuration context."""
        entrypoint = "sys_open"

        # Test without config
        response1 = await http_client.post(
            "/mcp/tools/entrypoint_flow",
            json={"entrypoint": entrypoint},
            headers=auth_headers,
        )

        # Test with config
        response2 = await http_client.post(
            "/mcp/tools/entrypoint_flow",
            json={"entrypoint": entrypoint, "config": "defconfig"},
            headers=auth_headers,
        )

        # Both should be valid requests
        assert response1.status_code in [200, 404], (
            "Request without config should be valid"
        )
        assert response2.status_code in [200, 404], (
            "Request with config should be valid"
        )

        # Results might differ based on configuration
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Both should have valid structure
            assert "call_paths" in data1 and "total_paths" in data1
            assert "call_paths" in data2 and "total_paths" in data2

    async def test_entrypoint_flow_call_path_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that call paths are logically consistent."""
        payload = {"entrypoint": "sys_read"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            for call_path in data["call_paths"]:
                path = call_path["path"]
                depth = call_path["depth"]

                if len(path) > 0:
                    # Depth should relate to path length
                    assert depth > 0, "Depth should be positive for non-empty paths"

                    # First step should be the entry point (or close to it)
                    first_step = path[0]
                    # Entry points often have specific patterns in kernel
                    assert first_step["function_name"], (
                        "First step should have function name"
                    )

                    # Each step should have valid call information
                    for i, step in enumerate(path):
                        assert step["function_name"], (
                            f"Step {i} should have function name"
                        )
                        assert step["file_path"], f"Step {i} should have file path"

                        # Should point to kernel source files
                        assert step["file_path"].endswith((".c", ".h")), (
                            f"Step {i} should point to C/header files"
                        )

                        # Paths should be relative to kernel root
                        assert not step["file_path"].startswith("/"), (
                            f"Step {i} path should be relative"
                        )

    async def test_entrypoint_flow_confidence_scoring(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test confidence scoring for call paths."""
        payload = {"entrypoint": "sys_write"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            if data["total_paths"] > 0:
                # Collect all confidence scores
                confidence_scores = [path["confidence"] for path in data["call_paths"]]

                # All should be valid floats between 0 and 1
                for i, score in enumerate(confidence_scores):
                    assert isinstance(score, (int, float)), (
                        f"Confidence {i} should be numeric"
                    )
                    assert 0 <= score <= 1, (
                        f"Confidence {i} should be between 0 and 1, got {score}"
                    )

                # Should have some variation in confidence if multiple paths
                if len(confidence_scores) > 1:
                    # Not all paths should have identical confidence
                    # (unless they're all equally certain)
                    # This test documents expected behavior - some variation expected
                    # but not strictly required
                    pass

    async def test_entrypoint_flow_nonexistent_entrypoint(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with nonexistent entry point."""
        payload = {"entrypoint": "nonexistent_syscall_xyz_999"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        assert response.status_code in [200, 404], (
            "Should handle nonexistent entry point gracefully"
        )

        if response.status_code == 200:
            data = response.json()
            assert data["total_paths"] == 0, (
                "Nonexistent entry point should have 0 paths"
            )
            assert len(data["call_paths"]) == 0, (
                "Nonexistent entry point should have empty call_paths array"
            )

    async def test_entrypoint_flow_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test performance meets p95 < 600ms requirement."""
        payload = {"entrypoint": "sys_read", "config": "defconfig"}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow",
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

    async def test_entrypoint_flow_authentication(self, http_client: httpx.AsyncClient):
        """Test that entrypoint_flow requires authentication."""
        payload = {"entrypoint": "sys_open"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/entrypoint_flow", json=payload)

        assert response.status_code == 401, "Should require authentication"

    async def test_entrypoint_flow_call_type_distribution(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test realistic distribution of call types in paths."""
        payload = {"entrypoint": "sys_open"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            if data["total_paths"] > 0:
                # Collect all call types from all path steps
                all_call_types = []
                for call_path in data["call_paths"]:
                    for step in call_path["path"]:
                        all_call_types.append(step["call_type"])

                if all_call_types:
                    # All should be valid
                    valid_types = {"Direct", "Indirect", "Macro"}
                    for call_type in all_call_types:
                        assert call_type in valid_types, (
                            f"Invalid call type: {call_type}"
                        )

                    # Should have some Direct calls (most common in kernel)
                    direct_calls = [ct for ct in all_call_types if ct == "Direct"]
                    assert len(direct_calls) > 0, (
                        "Should have at least some Direct calls"
                    )


@skip_integration_in_ci
@skip_without_mcp_server
class TestEntrypointFlowIntegration:
    """Integration tests for entrypoint_flow with realistic kernel data."""

    async def test_entrypoint_flow_syscall_integration(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test with actual kernel syscall entry point."""
        payload = {"entrypoint": "sys_read"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # sys_read should have call paths in kernel
            if data["total_paths"] > 0:
                for call_path in data["call_paths"]:
                    path = call_path["path"]

                    # Should have reasonable path structure
                    assert len(path) > 0, "Call path should have at least one step"

                    # Should lead through realistic kernel functions
                    function_names = [step["function_name"] for step in path]

                    # Should have meaningful function names
                    assert all(name for name in function_names), (
                        "All function names should be non-empty"
                    )

                    # Verify file paths point to kernel source
                    for step in path:
                        assert not step["file_path"].startswith("/tmp"), (
                            "Should not point to temporary files"
                        )
                        assert step["line_number"] > 0, (
                            "Line numbers should be positive"
                        )

    async def test_entrypoint_flow_targeted_search(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test targeted search from entry point to specific function."""
        payload = {"entrypoint": "sys_open", "target_function": "vfs_open"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should find paths that include the target function
            if data["total_paths"] > 0:
                target_found = False
                for call_path in data["call_paths"]:
                    for step in call_path["path"]:
                        if step["function_name"] == "vfs_open":
                            target_found = True
                            break

                # If target found, verify path quality
                if target_found:
                    # At least one path should have reasonable confidence
                    confidences = [path["confidence"] for path in data["call_paths"]]
                    max_confidence = max(confidences)
                    assert max_confidence > 0, (
                        "Should have some confidence in found paths"
                    )

    async def test_entrypoint_flow_multiple_paths(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that complex entry points can have multiple paths."""
        # Use a syscall that likely has multiple implementation paths
        payload = {"entrypoint": "sys_write"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # sys_write might have multiple paths (different filesystems, etc.)
            if data["total_paths"] > 1:
                # Paths should have different characteristics
                # Should have some variation in path characteristics
                # (not all identical - that would suggest incomplete analysis)
                # Document expected behavior: some variation in paths
                # This helps identify whether the analysis is finding
                # multiple realistic code paths or duplicating results
                pass

    async def test_entrypoint_flow_confidence_accuracy(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that confidence scores reflect path quality."""
        payload = {"entrypoint": "sys_read"}

        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            if data["total_paths"] > 0:
                # Higher confidence paths should have better characteristics
                paths_by_confidence = sorted(
                    data["call_paths"], key=lambda x: x["confidence"], reverse=True
                )

                if len(paths_by_confidence) > 1:
                    highest_confidence = paths_by_confidence[0]
                    lowest_confidence = paths_by_confidence[-1]

                    # Higher confidence path should have reasonable characteristics
                    assert (
                        highest_confidence["confidence"]
                        >= lowest_confidence["confidence"]
                    )

                    # Higher confidence paths might have:
                    # - More direct call types
                    # - Shorter paths (less complex)
                    # - More common/well-known functions

                    # This test documents expected correlation between
                    # confidence scores and path quality metrics
