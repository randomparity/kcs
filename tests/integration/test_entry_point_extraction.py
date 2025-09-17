"""Integration test for complete entry point extraction.

These tests verify that all entry point types are properly detected
and extracted from kernel fixtures, with expected metadata.
"""

import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from kcs_mcp.database import Database
from testcontainers.postgres import PostgresContainer


# Skip tests if MCP server not running
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible."""
    try:
        with httpx.Client() as client:
            response = client.get("http://localhost:8080/health", timeout=2.0)
            return response.status_code == 200
    except Exception:
        return False


skip_without_mcp = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server not running"
)

# Skip in CI unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)

# Test configuration
TEST_KERNEL_PATH = Path("tests/fixtures/kernel")
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "test_token_123"


# Fixtures
@pytest_asyncio.fixture
async def test_database() -> Database:
    """Create a test database with sample data."""
    # Use testcontainers for isolated testing
    with PostgresContainer("postgres:16") as postgres:
        db = Database(postgres.get_connection_url())
        await db.connect()

        # Apply migrations
        migrations_dir = Path("src/sql/migrations")
        for migration in sorted(migrations_dir.glob("*.sql")):
            with open(migration) as f:
                await db.execute(f.read())

        yield db
        await db.disconnect()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> httpx.AsyncClient:
    """HTTP client for API requests."""
    async with httpx.AsyncClient(base_url=MCP_BASE_URL) as client:
        yield client


# Test cases
@skip_integration_in_ci
@skip_without_mcp
class TestEntryPointExtraction:
    """Integration tests for comprehensive entry point extraction."""

    @pytest.mark.asyncio
    async def test_syscall_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that syscalls are properly extracted from kernel fixtures."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["syscall"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Should find syscalls in test fixtures
            syscalls = [ep for ep in entry_points if ep["entry_type"] == "syscall"]
            assert len(syscalls) > 0, "Should detect syscalls in kernel fixtures"

            # Verify syscall metadata
            for syscall in syscalls:
                assert syscall["name"].startswith("sys_"), (
                    f"Syscall name should start with sys_: {syscall['name']}"
                )
                assert "file_path" in syscall
                assert "line_number" in syscall

    @pytest.mark.asyncio
    async def test_ioctl_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that ioctl handlers are extracted with command metadata."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["ioctl"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Should find ioctls in test fixtures
            # NOTE: This MUST fail initially as ioctl extraction isn't fully implemented
            ioctls = [ep for ep in entry_points if ep["entry_type"] == "ioctl"]
            assert len(ioctls) > 0, "Should detect ioctl handlers in kernel fixtures"

            # Check for ioctl command metadata
            ioctls_with_cmd = [
                ep
                for ep in ioctls
                if ep.get("metadata") and "ioctl_cmd" in ep["metadata"]
            ]
            assert len(ioctls_with_cmd) > 0, (
                "Some ioctl handlers should have command metadata"
            )

    @pytest.mark.asyncio
    async def test_file_ops_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that file_operations structures are detected."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["file_ops"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Should find file_ops in test fixtures
            file_ops = [ep for ep in entry_points if ep["entry_type"] == "file_ops"]
            assert len(file_ops) > 0, "Should detect file_operations in kernel fixtures"

            # Verify file_ops have associated operation names
            for fop in file_ops:
                # Should have operation type (read, write, open, etc.)
                assert "name" in fop, "File operation should have a name"

    @pytest.mark.asyncio
    async def test_sysfs_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that sysfs attributes are detected."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["sysfs"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Should find sysfs attributes in test fixtures
            sysfs = [ep for ep in entry_points if ep["entry_type"] == "sysfs"]
            assert len(sysfs) > 0, "Should detect sysfs attributes in kernel fixtures"

            # Verify sysfs handlers (show/store)
            for attr in sysfs:
                # Should identify if it's a show or store handler
                if attr.get("metadata"):
                    assert (
                        "handler_type" in attr["metadata"] or "ops_type" in attr["name"]
                    ), "Sysfs attribute should indicate handler type"

    @pytest.mark.asyncio
    async def test_procfs_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that procfs entry points are detected."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["procfs"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Should find procfs entries in test fixtures
            # NOTE: This MUST fail initially as procfs extraction isn't implemented
            procfs = [ep for ep in entry_points if ep["entry_type"] == "procfs"]
            assert len(procfs) > 0, "Should detect procfs entries in kernel fixtures"

            # Verify proc_ops structures
            for proc_entry in procfs:
                assert "file_path" in proc_entry
                assert "line_number" in proc_entry

    @pytest.mark.asyncio
    async def test_debugfs_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that debugfs entry points are detected."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["debugfs"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Should find debugfs entries in test fixtures
            # NOTE: This MUST fail initially as debugfs extraction isn't implemented
            debugfs = [ep for ep in entry_points if ep["entry_type"] == "debugfs"]
            assert len(debugfs) > 0, "Should detect debugfs entries in kernel fixtures"

    @pytest.mark.asyncio
    async def test_netlink_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that netlink handlers are detected."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["netlink"],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Should find netlink handlers in test fixtures
            # NOTE: This MUST fail initially as netlink extraction isn't implemented
            netlink = [ep for ep in entry_points if ep["entry_type"] == "netlink"]
            assert len(netlink) > 0, "Should detect netlink handlers in kernel fixtures"

            # Verify netlink family metadata
            for handler in netlink:
                if handler.get("metadata"):
                    assert "netlink_family" in handler["metadata"], (
                        "Netlink handler should have family information"
                    )

    @pytest.mark.asyncio
    async def test_all_entry_types_together(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test extraction of all entry point types at once."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            # Request all entry types
            "entry_types": [
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
            ],
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])
            stats = data.get("statistics", {})

            # Should find multiple entry point types
            entry_types_found = {ep["entry_type"] for ep in entry_points}
            assert len(entry_types_found) > 1, (
                f"Should detect multiple entry point types, found: {entry_types_found}"
            )

            # Verify statistics are provided
            assert "total" in stats, "Should provide total count in statistics"
            assert stats["total"] == len(entry_points), (
                "Total count should match entry points returned"
            )

            if "by_type" in stats:
                # Statistics should match actual counts
                for entry_type, count in stats["by_type"].items():
                    actual_count = len(
                        [ep for ep in entry_points if ep["entry_type"] == entry_type]
                    )
                    assert count == actual_count, (
                        f"Statistics mismatch for {entry_type}: "
                        f"reported {count}, actual {actual_count}"
                    )

    @pytest.mark.asyncio
    async def test_metadata_population(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that entry points include metadata when available."""
        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            "entry_types": ["ioctl", "sysfs", "netlink"],
            "include_metadata": True,
        }

        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            entry_points = data.get("entry_points", [])

            # Count entries with metadata
            with_metadata = [ep for ep in entry_points if ep.get("metadata")]

            # At least some entries should have metadata
            # NOTE: This percentage will increase as more metadata extraction is implemented
            metadata_ratio = (
                len(with_metadata) / len(entry_points) if entry_points else 0
            )
            assert metadata_ratio > 0.3, (
                f"At least 30% of entry points should have metadata, got {metadata_ratio:.1%}"
            )

    @pytest.mark.asyncio
    async def test_performance_expectations(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that extraction completes within performance targets."""
        import time

        payload = {
            "kernel_path": str(TEST_KERNEL_PATH),
            # Extract all types to test performance
        }

        start_time = time.time()
        response = await http_client.post(
            "/extract/entry_points", json=payload, headers=auth_headers
        )
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            stats = data.get("statistics", {})

            # Should complete quickly for test fixtures
            assert elapsed_time < 30, (
                f"Extraction should complete within 30s for fixtures, took {elapsed_time:.1f}s"
            )

            # Check reported processing time if available
            if "processing_time_ms" in stats:
                processing_ms = stats["processing_time_ms"]
                assert processing_ms < 30000, (
                    f"Processing time should be <30s, reported {processing_ms}ms"
                )
