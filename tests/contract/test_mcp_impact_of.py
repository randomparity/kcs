"""
Contract tests for impact_of MCP endpoint with call graph support.

These tests verify the API contract defined in contracts/mcp-endpoints.json.
They MUST fail before implementation and pass after.

Enhanced contract with call graph features:
- function_name parameter for function to analyze impact
- analysis_depth parameter (default 3, range 1-10)
- direct_callers count (immediate callers)
- total_affected count (all transitively affected functions)
- affected_subsystems array (kernel subsystems impacted)
- critical_paths array with entry_point, path_length, subsystem
- risk_level enum (Low, Medium, High, Critical)
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


# Contract test cases for impact_of
@skip_integration_in_ci
@skip_without_mcp_server
class TestImpactOfContract:
    """Contract tests for impact_of MCP endpoint with call graph data."""

    async def test_impact_of_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that impact_of endpoint exists and accepts requests."""
        payload = {"function_name": "ext4_writepage"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "impact_of endpoint should exist"

    async def test_impact_of_schema_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test request schema validation according to contract."""
        # Valid request with all parameters
        payload = {
            "function_name": "ext4_writepage",
            "analysis_depth": 5,
            "config": "defconfig",
        }

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        assert response.status_code not in [422], "Valid schema should be accepted"

        # Test missing required field (function_name)
        invalid_payload = {"analysis_depth": 3, "config": "defconfig"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=invalid_payload, headers=auth_headers
        )

        assert response.status_code == 422, (
            "Should reject request without required function_name"
        )

    async def test_impact_of_analysis_depth_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test analysis_depth parameter validation."""
        # Test minimum boundary (1)
        response = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": "test", "analysis_depth": 1},
            headers=auth_headers,
        )
        assert response.status_code != 422, "analysis_depth=1 should be valid"

        # Test maximum boundary (10)
        response = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": "test", "analysis_depth": 10},
            headers=auth_headers,
        )
        assert response.status_code != 422, "analysis_depth=10 should be valid"

        # Test below minimum (0)
        response = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": "test", "analysis_depth": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422, "analysis_depth=0 should be rejected"

        # Test above maximum (11)
        response = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": "test", "analysis_depth": 11},
            headers=auth_headers,
        )
        assert response.status_code == 422, "analysis_depth=11 should be rejected"

        # Test invalid type
        response = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": "test", "analysis_depth": "invalid"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "analysis_depth should be integer"

    async def test_impact_of_response_structure(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test response schema matches contract specification."""
        payload = {"function_name": "kmalloc"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Contract: must have all required fields
            required_fields = [
                "direct_callers",
                "total_affected",
                "affected_subsystems",
                "critical_paths",
                "risk_level",
            ]
            for field in required_fields:
                assert field in data, f"Response must contain '{field}' field"

            # Verify field types
            assert isinstance(data["direct_callers"], int), (
                "direct_callers should be integer"
            )
            assert isinstance(data["total_affected"], int), (
                "total_affected should be integer"
            )
            assert isinstance(data["affected_subsystems"], list), (
                "affected_subsystems should be array"
            )
            assert isinstance(data["critical_paths"], list), (
                "critical_paths should be array"
            )
            assert isinstance(data["risk_level"], str), "risk_level should be string"

            # Verify risk_level is valid enum value
            valid_risk_levels = ["Low", "Medium", "High", "Critical"]
            assert data["risk_level"] in valid_risk_levels, (
                f"risk_level '{data['risk_level']}' must be one of {valid_risk_levels}"
            )

            # Verify counts are non-negative
            assert data["direct_callers"] >= 0, "direct_callers should be non-negative"
            assert data["total_affected"] >= 0, "total_affected should be non-negative"

            # direct_callers should be <= total_affected
            assert data["direct_callers"] <= data["total_affected"], (
                "direct_callers should be <= total_affected"
            )

            # Verify affected_subsystems structure
            for subsystem in data["affected_subsystems"]:
                assert isinstance(subsystem, str), (
                    "Each affected subsystem should be string"
                )
                assert subsystem, "Subsystem name should not be empty"

            # Verify critical_paths structure
            for critical_path in data["critical_paths"]:
                required_path_fields = ["entry_point", "path_length", "subsystem"]
                for field in required_path_fields:
                    assert field in critical_path, (
                        f"Critical path must have '{field}' field"
                    )

                assert isinstance(critical_path["entry_point"], str), (
                    "entry_point should be string"
                )
                assert isinstance(critical_path["path_length"], int), (
                    "path_length should be integer"
                )
                assert isinstance(critical_path["subsystem"], str), (
                    "subsystem should be string"
                )

                # Verify constraints
                assert critical_path["entry_point"], "entry_point should not be empty"
                assert critical_path["path_length"] > 0, (
                    "path_length should be positive"
                )
                assert critical_path["subsystem"], "subsystem should not be empty"

    async def test_impact_of_analysis_depth_default(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that analysis_depth defaults to 3 when not specified."""
        payload = {"function_name": "kmalloc"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should use default depth of 3
            # This affects how deep the impact analysis goes
            # We can't directly verify the depth was 3, but we can check
            # that analysis was performed with reasonable results
            assert data["total_affected"] >= data["direct_callers"], (
                "Should analyze beyond direct callers with default depth"
            )

    async def test_impact_of_analysis_depth_effect(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that analysis_depth parameter affects results depth."""
        function_name = "kmalloc"

        # Test shallow depth
        response1 = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": function_name, "analysis_depth": 1},
            headers=auth_headers,
        )

        # Test deeper depth
        response3 = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": function_name, "analysis_depth": 5},
            headers=auth_headers,
        )

        if response1.status_code == 200 and response3.status_code == 200:
            data1 = response1.json()
            data3 = response3.json()

            # Deeper analysis should potentially find more affected functions
            assert data3["total_affected"] >= data1["total_affected"], (
                "Deeper analysis should find same or more affected functions"
            )

            # Direct callers should be the same (not affected by analysis depth)
            assert data1["direct_callers"] == data3["direct_callers"], (
                "Direct callers count should be consistent across depths"
            )

    async def test_impact_of_config_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test config parameter for kernel configuration context."""
        function_name = "ext4_writepage"

        # Test without config
        response1 = await http_client.post(
            "/mcp/tools/impact_of",
            json={"function_name": function_name},
            headers=auth_headers,
        )

        # Test with config
        response2 = await http_client.post(
            "/mcp/tools/impact_of",
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
            assert all(
                field in data1
                for field in ["direct_callers", "total_affected", "risk_level"]
            )
            assert all(
                field in data2
                for field in ["direct_callers", "total_affected", "risk_level"]
            )

    async def test_impact_of_risk_level_assessment(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that risk levels are assessed appropriately."""
        # Test with different types of functions that might have different risk levels
        functions_to_test = [
            "kmalloc",  # Core memory function - likely high impact
            "ext4_writepage",  # Filesystem function - medium impact
            "helper_func_xyz",  # Non-existent - should be low/not found
        ]

        for function_name in functions_to_test:
            payload = {"function_name": function_name, "analysis_depth": 3}

            response = await http_client.post(
                "/mcp/tools/impact_of", json=payload, headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()

                # Risk level should correlate with impact metrics
                risk_level = data["risk_level"]
                total_affected = data["total_affected"]

                # Higher affected counts should generally correlate with higher risk
                if total_affected > 100:
                    # Many affected functions should indicate higher risk
                    assert risk_level in ["Medium", "High", "Critical"], (
                        f"High impact ({total_affected} affected) should have higher risk level"
                    )
                elif total_affected == 0:
                    # No impact should be low risk
                    assert risk_level == "Low", (
                        "No impact should result in Low risk level"
                    )

    async def test_impact_of_critical_paths_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that critical paths are meaningful and well-formed."""
        payload = {"function_name": "kmalloc"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            for critical_path in data["critical_paths"]:
                # Entry points should be realistic kernel entry points
                entry_point = critical_path["entry_point"]
                assert entry_point, "Entry point should not be empty"

                # Common entry point patterns in kernel
                # (syscalls, interrupts, etc.)
                # This test documents expected patterns but doesn't strictly enforce

                # Path length should be reasonable (not 0, not extremely long)
                path_length = critical_path["path_length"]
                assert 1 <= path_length <= 50, (
                    f"Path length {path_length} should be reasonable"
                )

                # Subsystem should be a meaningful kernel subsystem
                subsystem = critical_path["subsystem"]
                assert subsystem, "Subsystem should not be empty"

                # Common kernel subsystems include:
                # mm, fs, net, kernel, drivers, arch, security, crypto, block, sound
                # This test documents expected subsystems but doesn't strictly enforce

    async def test_impact_of_subsystem_analysis(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that affected subsystems are identified correctly."""
        payload = {"function_name": "kmalloc"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            affected_subsystems = data["affected_subsystems"]

            if len(affected_subsystems) > 0:
                # Should have unique subsystems (no duplicates)
                assert len(affected_subsystems) == len(set(affected_subsystems)), (
                    "Affected subsystems should be unique"
                )

                # Subsystems should be reasonable kernel subsystem names
                for subsystem in affected_subsystems:
                    # Should be non-empty strings
                    assert subsystem and isinstance(subsystem, str), (
                        f"Subsystem '{subsystem}' should be non-empty string"
                    )

                    # Should not be too long (reasonable subsystem names)
                    assert len(subsystem) <= 20, (
                        f"Subsystem name '{subsystem}' should be reasonably short"
                    )

    async def test_impact_of_nonexistent_function(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with nonexistent function."""
        payload = {"function_name": "nonexistent_function_abc_999"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        assert response.status_code in [200, 404], (
            "Should handle nonexistent function gracefully"
        )

        if response.status_code == 200:
            data = response.json()

            # Nonexistent function should have minimal impact
            assert data["direct_callers"] == 0, (
                "Nonexistent function should have 0 direct callers"
            )
            assert data["total_affected"] == 0, (
                "Nonexistent function should have 0 total affected"
            )
            assert data["risk_level"] == "Low", (
                "Nonexistent function should have Low risk level"
            )
            assert len(data["affected_subsystems"]) == 0, (
                "Nonexistent function should affect no subsystems"
            )
            assert len(data["critical_paths"]) == 0, (
                "Nonexistent function should have no critical paths"
            )

    async def test_impact_of_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test performance meets p95 < 600ms requirement."""
        payload = {"function_name": "kmalloc", "analysis_depth": 5}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers, timeout=1.0
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [200, 404]:
            # Constitutional requirement: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_impact_of_authentication(self, http_client: httpx.AsyncClient):
        """Test that impact_of requires authentication."""
        payload = {"function_name": "test"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/impact_of", json=payload)

        assert response.status_code == 401, "Should require authentication"


@skip_integration_in_ci
@skip_without_mcp_server
class TestImpactOfIntegration:
    """Integration tests for impact_of with realistic kernel data."""

    async def test_impact_of_core_kernel_function(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test impact analysis of core kernel function."""
        payload = {"function_name": "kmalloc", "analysis_depth": 3}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # kmalloc is a core function, should have significant impact
            assert data["direct_callers"] > 0, "kmalloc should have direct callers"
            assert data["total_affected"] >= data["direct_callers"], (
                "Total affected should be >= direct callers"
            )

            # Should affect multiple subsystems
            assert len(data["affected_subsystems"]) > 0, (
                "kmalloc should affect multiple subsystems"
            )

            # Should have high risk level due to widespread usage
            assert data["risk_level"] in ["Medium", "High", "Critical"], (
                "Core memory function should have elevated risk level"
            )

            # Should have critical paths through system calls
            if len(data["critical_paths"]) > 0:
                # At least some paths should be reasonably short
                path_lengths = [path["path_length"] for path in data["critical_paths"]]
                min_path_length = min(path_lengths)
                assert min_path_length <= 10, (
                    "Should have some relatively direct paths to kmalloc"
                )

    async def test_impact_of_filesystem_function(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test impact analysis of filesystem-specific function."""
        payload = {"function_name": "ext4_create", "analysis_depth": 3}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should be more contained than core functions
            if data["total_affected"] > 0:
                # Should primarily affect filesystem subsystems
                # This test documents expected behavior
                # Filesystem functions should primarily impact filesystem subsystems

                # Risk level should be appropriate for subsystem-specific function
                # (generally lower than core kernel functions)
                assert data["risk_level"] in ["Low", "Medium", "High"], (
                    "Filesystem function risk should be reasonable"
                )

    async def test_impact_of_depth_scaling(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that impact analysis scales appropriately with depth."""
        function_name = "kmalloc"

        depths_to_test = [1, 3, 5]
        results = {}

        for depth in depths_to_test:
            payload = {"function_name": function_name, "analysis_depth": depth}

            response = await http_client.post(
                "/mcp/tools/impact_of", json=payload, headers=auth_headers
            )

            if response.status_code == 200:
                results[depth] = response.json()

        if len(results) > 1:
            # Compare results across depths
            depths = sorted(results.keys())

            for i in range(len(depths) - 1):
                shallow_depth = depths[i]
                deep_depth = depths[i + 1]

                shallow_data = results[shallow_depth]
                deep_data = results[deep_depth]

                # Deeper analysis should find same or more affected functions
                assert deep_data["total_affected"] >= shallow_data["total_affected"], (
                    f"Depth {deep_depth} should find >= functions than depth {shallow_depth}"
                )

                # Direct callers should remain constant (not affected by depth)
                assert shallow_data["direct_callers"] == deep_data["direct_callers"], (
                    "Direct callers should be consistent across analysis depths"
                )

    async def test_impact_of_risk_correlation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that risk levels correlate with impact metrics."""
        functions_to_test = [
            "kmalloc",  # Core function - should be high impact
            "ext4_create",  # Subsystem function - should be medium impact
        ]

        results = {}
        for function_name in functions_to_test:
            payload = {"function_name": function_name, "analysis_depth": 3}

            response = await http_client.post(
                "/mcp/tools/impact_of", json=payload, headers=auth_headers
            )

            if response.status_code == 200:
                results[function_name] = response.json()

        if len(results) >= 2:
            # Compare risk assessments
            for function_name, data in results.items():
                risk_level = data["risk_level"]
                total_affected = data["total_affected"]
                num_subsystems = len(data["affected_subsystems"])

                # Document expected correlation between metrics and risk
                # Higher impact metrics should generally correlate with higher risk
                if total_affected > 50 and num_subsystems > 3:
                    # High impact should indicate elevated risk
                    assert risk_level in ["Medium", "High", "Critical"], (
                        f"High impact function {function_name} should have elevated risk"
                    )
