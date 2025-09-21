"""
Contract tests for root endpoint.

These tests verify the API contract for the root endpoint (GET /).
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


class TestRootEndpoint:
    """Test cases for root endpoint API discovery."""

    @skip_without_mcp_server
    @skip_integration_in_ci
    async def test_root_endpoint_exists(self):
        """Test that root endpoint exists and returns 200."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{MCP_BASE_URL}/")

        # Should not return 404 (endpoint exists)
        assert response.status_code == 200, "Root endpoint should exist and return 200"

    @skip_without_mcp_server
    @skip_integration_in_ci
    async def test_root_endpoint_schema(self):
        """Test that root endpoint returns correct schema."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{MCP_BASE_URL}/")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        required_fields = [
            "service",
            "title",
            "version",
            "description",
            "mcp",
            "endpoints",
            "constitutional_requirements",
        ]

        for field in required_fields:
            assert field in data, f"Root response should contain '{field}' field"

        # Verify specific values
        assert data["service"] == "kcs", "Service should be 'kcs'"
        assert data["title"] == "Kernel Context Server MCP API", (
            "Title should match expected value"
        )
        assert data["version"] == "1.0.0", "Version should be '1.0.0'"
        assert "Model Context Protocol" in data["description"], (
            "Description should mention MCP"
        )

        # Verify MCP section
        assert "protocol_version" in data["mcp"], "MCP should have protocol_version"
        assert "capabilities" in data["mcp"], "MCP should have capabilities"
        assert isinstance(data["mcp"]["capabilities"], list), (
            "MCP capabilities should be a list"
        )

        # Verify endpoints section
        expected_endpoints = ["health", "metrics", "mcp_tools", "mcp_resources", "docs"]
        for endpoint in expected_endpoints:
            assert endpoint in data["endpoints"], (
                f"Endpoints should contain '{endpoint}'"
            )

        # Verify constitutional requirements
        constitutional = data["constitutional_requirements"]
        assert "read_only" in constitutional, (
            "Constitutional requirements should specify read_only"
        )
        assert "citations_required" in constitutional, (
            "Constitutional requirements should specify citations_required"
        )
        assert "performance_target" in constitutional, (
            "Constitutional requirements should specify performance_target"
        )

    @skip_without_mcp_server
    @skip_integration_in_ci
    async def test_root_endpoint_no_auth_required(self):
        """Test that root endpoint doesn't require authentication."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Request without authentication
            response = await client.get(f"{MCP_BASE_URL}/")

        # Should return 200, not 401
        assert response.status_code == 200, (
            "Root endpoint should not require authentication"
        )

    @skip_without_mcp_server
    @skip_integration_in_ci
    async def test_root_endpoint_content_type(self):
        """Test that root endpoint returns JSON content type."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{MCP_BASE_URL}/")

        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", ""), (
            "Root endpoint should return JSON"
        )
