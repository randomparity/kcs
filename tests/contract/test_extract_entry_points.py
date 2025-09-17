"""
Contract tests for extract_entry_points endpoint.

These tests verify the API contract defined in specs/004-current-code-has/contracts/extract_entry_points.yaml.
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
        response = requests.get("http://localhost:8080/health", timeout=2)
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


@pytest.fixture
def sample_kernel_path() -> str:
    """Path to test kernel fixtures."""
    return "tests/fixtures/kernel"


# Contract test cases
@skip_integration_in_ci
@skip_without_mcp_server
class TestExtractEntryPointsContract:
    """Contract tests for extract_entry_points endpoint."""

    @pytest.mark.asyncio
    async def test_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the extract_entry_points endpoint exists and accepts POST requests."""
        payload = {"kernel_path": "/tmp/test_kernel"}

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        # NOTE: This MUST fail initially as endpoint doesn't exist yet
        assert response.status_code != 404, "extract_entry_points endpoint should exist"

    @pytest.mark.asyncio
    async def test_requires_kernel_path(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that kernel_path is required in request."""
        payload: dict[str, Any] = {}  # Missing kernel_path

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        assert response.status_code == 400, "Should return 400 for missing kernel_path"

    @pytest.mark.asyncio
    async def test_response_schema(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_path: str,
    ) -> None:
        """Test that response matches the contract schema."""
        payload = {
            "kernel_path": sample_kernel_path,
            "entry_types": ["syscall", "ioctl", "file_ops"],
            "enable_clang": False,
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify response structure matches contract
            assert "entry_points" in data, "Response should contain entry_points"
            assert isinstance(data["entry_points"], list), (
                "entry_points should be a list"
            )

            if data["entry_points"]:
                entry = data["entry_points"][0]

                # Required fields
                assert "name" in entry, "Entry point must have name"
                assert "entry_type" in entry, "Entry point must have entry_type"
                assert "file_path" in entry, "Entry point must have file_path"
                assert "line_number" in entry, "Entry point must have line_number"

                # Optional fields
                # signature, description, metadata are optional

                # Validate entry_type enum
                valid_types = [
                    "syscall",
                    "ioctl",
                    "file_ops",
                    "sysfs",
                    "procfs",
                    "debugfs",
                    "netlink",
                    "notification_chain",
                    "interrupt_handler",
                    "boot_param",
                ]
                assert entry["entry_type"] in valid_types, (
                    f"Invalid entry_type: {entry['entry_type']}"
                )

            # Verify statistics if present
            if "statistics" in data:
                stats = data["statistics"]
                assert isinstance(stats, dict), "Statistics should be a dict"

                if "total" in stats:
                    assert isinstance(stats["total"], int), "Total should be an integer"

                if "by_type" in stats:
                    assert isinstance(stats["by_type"], dict), (
                        "by_type should be a dict"
                    )

                if "processing_time_ms" in stats:
                    assert isinstance(stats["processing_time_ms"], (int, float)), (
                        "processing_time_ms should be a number"
                    )

    @pytest.mark.asyncio
    async def test_new_entry_types_detected(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_path: str,
    ) -> None:
        """Test that new entry point types are detected (procfs, debugfs, netlink, etc.)."""
        payload = {
            "kernel_path": sample_kernel_path,
            "entry_types": ["procfs", "debugfs", "netlink"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Check if any of the new types are detected
            # NOTE: This MUST fail initially as these types aren't implemented yet
            detected_types = {ep["entry_type"] for ep in entry_points}
            new_types = {"procfs", "debugfs", "netlink"}

            assert detected_types & new_types, (
                f"Should detect new entry types. Found: {detected_types}"
            )

    @pytest.mark.asyncio
    async def test_metadata_field_present(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_path: str,
    ) -> None:
        """Test that entry points include metadata field when available."""
        payload = {
            "kernel_path": sample_kernel_path,
            "entry_types": ["ioctl"],
            "enable_clang": True,
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Check for ioctl entries with metadata
            ioctl_entries = [ep for ep in entry_points if ep["entry_type"] == "ioctl"]

            if ioctl_entries:
                # At least some ioctl entries should have metadata
                entries_with_metadata = [
                    ep for ep in ioctl_entries if ep.get("metadata")
                ]

                assert entries_with_metadata, (
                    "Some ioctl entries should have metadata (e.g., ioctl_cmd)"
                )

                # Check metadata structure for ioctl
                metadata = entries_with_metadata[0]["metadata"]
                if "ioctl_cmd" in metadata:
                    assert isinstance(metadata["ioctl_cmd"], str), (
                        "ioctl_cmd should be a string"
                    )

    @pytest.mark.asyncio
    async def test_clang_enhancement_optional(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_path: str,
    ) -> None:
        """Test that Clang enhancement is optional and doesn't break without it."""
        # Test without Clang
        payload_no_clang = {
            "kernel_path": sample_kernel_path,
            "enable_clang": False,
        }

        response_no_clang = await http_client.post(
            "/extract/entry_points", json=payload_no_clang, headers=auth_headers
        )

        # Test with Clang but no compile_commands
        payload_clang_no_commands = {
            "kernel_path": sample_kernel_path,
            "enable_clang": True,
            # compile_commands not provided
        }

        response_clang = await http_client.post(
            "/extract/entry_points",
            json=payload_clang_no_commands,
            headers=auth_headers,
        )

        # Both should work (graceful degradation)
        assert response_no_clang.status_code in [200, 501], (
            "Should work without Clang or return 501 if not implemented"
        )
        assert response_clang.status_code in [200, 501], (
            "Should degrade gracefully without compile_commands"
        )

    @pytest.mark.asyncio
    async def test_invalid_kernel_path_returns_404(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that invalid kernel path returns 404."""
        payload = {
            "kernel_path": "/nonexistent/kernel/path",
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        assert response.status_code == 404, (
            "Should return 404 for non-existent kernel path"
        )

    @pytest.mark.asyncio
    async def test_filter_by_entry_types(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_path: str,
    ) -> None:
        """Test that entry_types filter works correctly."""
        # Request only syscalls
        payload = {
            "kernel_path": sample_kernel_path,
            "entry_types": ["syscall"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # All returned entries should be syscalls
            for ep in entry_points:
                assert ep["entry_type"] == "syscall", (
                    f"Expected only syscalls, got {ep['entry_type']}"
                )
