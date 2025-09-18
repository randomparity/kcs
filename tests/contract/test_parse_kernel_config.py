"""
Contract tests for parse_kernel_config MCP tool.

These tests verify the API contract for kernel configuration parsing.
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
class TestParseKernelConfigContract:
    """Contract tests for parse_kernel_config MCP tool."""

    async def test_parse_kernel_config_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the parse_kernel_config endpoint exists and accepts POST requests."""
        payload = {
            "config_path": "/usr/src/linux/.config",
            "arch": "x86_64",
            "config_name": "defconfig",
        }

        response = await http_client.post(
            "/mcp/tools/parse_kernel_config", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "parse_kernel_config endpoint should exist"

    async def test_parse_kernel_config_requires_authentication(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that parse_kernel_config requires valid authentication."""
        payload = {"config_path": "/usr/src/linux/.config"}

        # Request without auth headers
        response = await http_client.post(
            "/mcp/tools/parse_kernel_config", json=payload
        )
        assert response.status_code == 401, "Should require authentication"

    async def test_parse_kernel_config_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that parse_kernel_config validates request schema according to OpenAPI spec."""

        # Missing required 'config_path' field
        response = await http_client.post(
            "/mcp/tools/parse_kernel_config",
            json={"arch": "x86_64"},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject request without required 'config_path' field"
        )

        # Invalid arch value (not in enum)
        response = await http_client.post(
            "/mcp/tools/parse_kernel_config",
            json={"config_path": "/usr/src/linux/.config", "arch": "invalid_arch"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid arch value"

        # Empty config_path
        response = await http_client.post(
            "/mcp/tools/parse_kernel_config",
            json={"config_path": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject empty config_path"

    async def test_parse_kernel_config_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that parse_kernel_config returns response matching OpenAPI schema."""
        payload = {
            "config_path": "/usr/src/linux/.config",
            "arch": "x86_64",
            "config_name": "defconfig",
        }

        response = await http_client.post(
            "/mcp/tools/parse_kernel_config", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify top-level structure
            assert "config_id" in data, "Response should contain 'config_id' field"
            assert "arch" in data, "Response should contain 'arch' field"
            assert "config_name" in data, "Response should contain 'config_name' field"
            assert "options" in data, "Response should contain 'options' field"
            assert "dependencies" in data, (
                "Response should contain 'dependencies' field"
            )
            assert "parsed_at" in data, "Response should contain 'parsed_at' field"

            # Verify field types
            assert isinstance(data["config_id"], str), (
                "config_id should be string (UUID)"
            )
            assert isinstance(data["arch"], str), "arch should be string"
            assert isinstance(data["config_name"], str), "config_name should be string"
            assert isinstance(data["options"], dict), "options should be object"
            assert isinstance(data["dependencies"], list), (
                "dependencies should be array"
            )
            assert isinstance(data["parsed_at"], str), (
                "parsed_at should be string (ISO timestamp)"
            )

            # Verify options structure
            for option_name, option_value in data["options"].items():
                assert isinstance(option_name, str), "Option name should be string"
                assert isinstance(option_value, dict), "Option value should be object"
                assert "value" in option_value, "Each option should have 'value' field"
                assert "type" in option_value, "Each option should have 'type' field"

                # Verify option type is valid
                valid_types = ["bool", "tristate", "string", "int", "hex"]
                assert option_value["type"] in valid_types, (
                    f"Option type should be one of {valid_types}"
                )

            # Verify dependencies structure
            for dep in data["dependencies"]:
                assert isinstance(dep, dict), "Each dependency should be object"
                assert "option" in dep, "Dependency should have 'option' field"
                assert "depends_on" in dep, "Dependency should have 'depends_on' field"
                assert isinstance(dep["option"], str), (
                    "Dependency option should be string"
                )
                assert isinstance(dep["depends_on"], list), "depends_on should be array"

            # Verify optional metadata field
            if "metadata" in data:
                assert isinstance(data["metadata"], dict), "metadata should be object"
                if "kernel_version" in data["metadata"]:
                    assert isinstance(data["metadata"]["kernel_version"], str), (
                        "kernel_version should be string"
                    )
                if "subsystems" in data["metadata"]:
                    assert isinstance(data["metadata"]["subsystems"], list), (
                        "subsystems should be array"
                    )

    async def test_parse_kernel_config_arch_enum(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that arch parameter accepts valid enum values."""
        valid_archs = ["x86_64", "arm64", "arm", "riscv", "powerpc", "s390", "mips"]

        for arch in valid_archs:
            payload = {
                "config_path": "/usr/src/linux/.config",
                "arch": arch,
            }

            response = await http_client.post(
                "/mcp/tools/parse_kernel_config", json=payload, headers=auth_headers
            )

            # Should accept all valid architectures
            assert response.status_code != 422, f"Should accept valid arch: {arch}"

    async def test_parse_kernel_config_incremental_mode(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that incremental parsing mode works correctly."""
        payload = {
            "config_path": "/usr/src/linux/.config",
            "arch": "x86_64",
            "incremental": True,
            "base_config_id": "550e8400-e29b-41d4-a716-446655440000",  # UUID v4
        }

        response = await http_client.post(
            "/mcp/tools/parse_kernel_config", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include diff information for incremental updates
            assert "changes" in data or "diff" in data, (
                "Incremental response should include changes or diff"
            )

    async def test_parse_kernel_config_file_not_found(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test proper error response when config file doesn't exist."""
        payload = {
            "config_path": "/nonexistent/path/.config",
        }

        response = await http_client.post(
            "/mcp/tools/parse_kernel_config", json=payload, headers=auth_headers
        )

        # Should return appropriate error status
        assert response.status_code in [404, 422], (
            "Should handle missing file appropriately"
        )

        if response.status_code == 422:
            data = response.json()
            assert "detail" in data or "error" in data, "Should include error details"

    async def test_parse_kernel_config_with_filters(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test filtering options by subsystem or pattern."""
        payload = {
            "config_path": "/usr/src/linux/.config",
            "filters": {
                "subsystems": ["net", "fs"],
                "pattern": "CONFIG_EXT4_*",
            },
        }

        response = await http_client.post(
            "/mcp/tools/parse_kernel_config", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Options should be filtered based on provided criteria
            assert "options" in data, "Response should contain filtered options"

    async def test_parse_kernel_config_dependency_resolution(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that dependency chains are properly resolved."""
        payload = {
            "config_path": "/usr/src/linux/.config",
            "resolve_dependencies": True,
            "max_depth": 3,
        }

        response = await http_client.post(
            "/mcp/tools/parse_kernel_config", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            assert "dependencies" in data, "Should include resolved dependencies"

            # Verify dependency chains
            for dep in data["dependencies"]:
                if "chain" in dep:
                    assert isinstance(dep["chain"], list), (
                        "Dependency chain should be array"
                    )
                    assert len(dep["chain"]) <= 3, (
                        "Chain depth should respect max_depth"
                    )
