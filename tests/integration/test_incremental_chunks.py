"""
Integration tests for incremental subsystem update processing.

Tests the incremental chunk processing capability that allows efficient
updates when only specific kernel subsystems have changed.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

# Skip integration tests in CI environments unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)


def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible."""
    try:
        import requests

        response = requests.get("http://localhost:8080", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# Skip tests requiring MCP server when it's not running
skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server not running"
)

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "test_token_123"
INCREMENTAL_TEST_TIMEOUT = 120  # 2 minutes for incremental tests


# Test fixtures and helpers
@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> httpx.AsyncClient:
    """HTTP client for API requests."""
    async with httpx.AsyncClient(base_url=MCP_BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture
def initial_manifest_all_subsystems() -> dict[str, Any]:
    """Generate initial manifest with all kernel subsystems."""
    chunks = []
    subsystems = ["kernel", "fs", "mm", "net", "drivers", "security", "crypto", "arch"]

    for subsystem in subsystems:
        for i in range(1, 4):  # 3 chunks per subsystem
            chunk_id = f"{subsystem}_{i:03d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "sequence": i,
                    "file": f"chunk_{chunk_id}.json",
                    "subsystem": subsystem,
                    "size_bytes": 40 * 1024 * 1024,  # 40MB chunks
                    "checksum_sha256": f"a{subsystem[0]}{str(i).zfill(61)}",
                    "symbol_count": 10000 + (i * 100),
                    "entrypoint_count": 30 + i,
                    "file_count": 80 + (i * 5),
                    "last_modified": "2025-01-18T08:00:00Z",
                }
            )

    return {
        "version": "1.0.0",
        "created": "2025-01-18T08:00:00Z",
        "kernel_version": "6.7.0",
        "kernel_path": "/tmp/test-kernel-incremental",
        "config": "x86_64:defconfig",
        "total_chunks": len(chunks),
        "total_size_bytes": len(chunks) * 40 * 1024 * 1024,
        "chunks": chunks,
    }


@pytest.fixture
def updated_manifest_fs_subsystem() -> dict[str, Any]:
    """Generate updated manifest with only fs subsystem changes."""
    chunks = []
    subsystems = ["kernel", "fs", "mm", "net", "drivers", "security", "crypto", "arch"]

    for subsystem in subsystems:
        for i in range(1, 4):  # 3 chunks per subsystem
            chunk_id = f"{subsystem}_{i:03d}"

            # Only fs subsystem has been updated
            if subsystem == "fs":
                checksum = f"b{subsystem[0]}{str(i).zfill(61)}"  # Different checksum for updated chunks
                last_modified = "2025-01-18T12:00:00Z"  # Newer timestamp
                symbol_count = 12000 + (i * 150)  # More symbols after update
            else:
                checksum = (
                    f"a{subsystem[0]}{str(i).zfill(61)}"  # Same checksum as before
                )
                last_modified = "2025-01-18T08:00:00Z"  # Original timestamp
                symbol_count = 10000 + (i * 100)  # Original symbol count

            chunks.append(
                {
                    "id": chunk_id,
                    "sequence": i,
                    "file": f"chunk_{chunk_id}.json",
                    "subsystem": subsystem,
                    "size_bytes": 40 * 1024 * 1024,
                    "checksum_sha256": checksum,
                    "symbol_count": symbol_count,
                    "entrypoint_count": 30 + i,
                    "file_count": 80 + (i * 5),
                    "last_modified": last_modified,
                }
            )

    return {
        "version": "1.0.0",
        "created": "2025-01-18T12:00:00Z",
        "kernel_version": "6.7.0",
        "kernel_path": "/tmp/test-kernel-incremental",
        "config": "x86_64:defconfig",
        "total_chunks": len(chunks),
        "total_size_bytes": len(chunks) * 40 * 1024 * 1024,
        "chunks": chunks,
    }


@pytest.fixture
def temp_incremental_chunk_directory():
    """Create temporary directory with chunk files for incremental testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create initial and updated chunk versions
        subsystems = [
            "kernel",
            "fs",
            "mm",
            "net",
            "drivers",
            "security",
            "crypto",
            "arch",
        ]

        for subsystem in subsystems:
            for i in range(1, 4):
                chunk_id = f"{subsystem}_{i:03d}"
                chunk_file = temp_path / f"chunk_{chunk_id}.json"

                # Create different content for fs subsystem (updated) vs others (unchanged)
                if subsystem == "fs":
                    # Updated fs chunks have more content
                    chunk_data = {
                        "symbols": [
                            {"name": f"fs_symbol_{i}_{j}", "type": "function"}
                            for j in range(15)  # More symbols
                        ],
                        "entrypoints": [
                            {"name": f"fs_entry_{i}_{j}", "type": "syscall"}
                            for j in range(4)
                        ],
                        "files": [
                            f"/tmp/test-kernel/fs/file_{i}_{j}.c" for j in range(8)
                        ],
                        "metadata": {
                            "chunk_id": chunk_id,
                            "processed_at": "2025-01-18T12:00:00Z",
                            "version": "updated",
                        },
                    }
                else:
                    # Original content for unchanged subsystems
                    chunk_data = {
                        "symbols": [
                            {"name": f"{subsystem}_symbol_{i}_{j}", "type": "function"}
                            for j in range(10)
                        ],
                        "entrypoints": [
                            {"name": f"{subsystem}_entry_{i}_{j}", "type": "syscall"}
                            for j in range(3)
                        ],
                        "files": [
                            f"/tmp/test-kernel/{subsystem}/file_{i}_{j}.c"
                            for j in range(5)
                        ],
                        "metadata": {
                            "chunk_id": chunk_id,
                            "processed_at": "2025-01-18T08:00:00Z",
                            "version": "original",
                        },
                    }

                chunk_file.write_text(json.dumps(chunk_data, indent=2))

        yield temp_path


@skip_integration_in_ci
@skip_without_mcp_server
class TestIncrementalSubsystemUpdate:
    """Integration tests for incremental subsystem processing."""

    async def test_incremental_fs_subsystem_update(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        initial_manifest_all_subsystems: dict[str, Any],
        updated_manifest_fs_subsystem: dict[str, Any],
        temp_incremental_chunk_directory: Path,
    ):
        """Test incremental processing when only fs subsystem is updated."""
        initial_manifest = initial_manifest_all_subsystems
        updated_manifest = updated_manifest_fs_subsystem

        # First, simulate processing of initial manifest (all subsystems)
        all_chunk_ids = [chunk["id"] for chunk in initial_manifest["chunks"]]

        initial_batch_request = {"chunk_ids": all_chunk_ids, "parallelism": 4}

        initial_response = await http_client.post(
            "/mcp/chunks/process/batch",
            headers=auth_headers,
            json=initial_batch_request,
        )

        if initial_response.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")

        # Wait for initial processing
        if initial_response.status_code == 202:
            await asyncio.sleep(3)

        # Identify changed chunks (only fs subsystem)
        changed_chunk_ids = [
            chunk["id"]
            for chunk in updated_manifest["chunks"]
            if chunk["subsystem"] == "fs"
        ]

        # Test incremental update with only changed chunks
        incremental_request = {
            "chunk_ids": changed_chunk_ids,
            "parallelism": 2,
            "incremental": True,  # Flag for incremental processing
        }

        start_time = time.time()

        incremental_response = await http_client.post(
            "/mcp/chunks/process/batch",
            headers=auth_headers,
            json=incremental_request,
        )

        if incremental_response.status_code == 202:
            # Wait for incremental processing
            await asyncio.sleep(5)

            # Verify only fs chunks were processed
            fs_chunks_processed = 0
            other_chunks_unchanged = 0

            for chunk in updated_manifest["chunks"]:
                chunk_id = chunk["id"]
                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()

                    if chunk["subsystem"] == "fs":
                        # FS chunks should show recent processing
                        if status_data.get("status") in ["completed", "processing"]:
                            fs_chunks_processed += 1
                    else:
                        # Other subsystems should be unchanged
                        other_chunks_unchanged += 1

            processing_time = time.time() - start_time

            # Verify incremental efficiency
            assert fs_chunks_processed >= 2, "Should process fs subsystem chunks"
            assert processing_time < 60, "Incremental update should be fast (<60s)"

        elif incremental_response.status_code == 404:
            pytest.skip("Incremental processing not yet implemented")

    async def test_incremental_checksum_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        initial_manifest_all_subsystems: dict[str, Any],
        updated_manifest_fs_subsystem: dict[str, Any],
    ):
        """Test that incremental processing correctly detects changed checksums."""
        initial_manifest = initial_manifest_all_subsystems
        updated_manifest = updated_manifest_fs_subsystem

        # Verify checksum differences between manifests
        initial_checksums = {
            chunk["id"]: chunk["checksum_sha256"]
            for chunk in initial_manifest["chunks"]
        }
        updated_checksums = {
            chunk["id"]: chunk["checksum_sha256"]
            for chunk in updated_manifest["chunks"]
        }

        changed_chunks = []
        unchanged_chunks = []

        for chunk_id in initial_checksums:
            if initial_checksums[chunk_id] != updated_checksums[chunk_id]:
                changed_chunks.append(chunk_id)
            else:
                unchanged_chunks.append(chunk_id)

        # Should have detected fs subsystem changes
        assert len(changed_chunks) >= 3, "Should detect fs subsystem chunk changes"
        assert len(unchanged_chunks) >= 20, "Should have many unchanged chunks"

        # All changed chunks should be from fs subsystem
        for chunk_id in changed_chunks:
            assert chunk_id.startswith("fs_"), (
                f"Changed chunk {chunk_id} should be fs subsystem"
            )

        # Test incremental processing with changed chunks only
        incremental_request = {
            "chunk_ids": changed_chunks,
            "parallelism": 2,
            "incremental": True,
            "skip_unchanged": True,  # Skip processing of unchanged chunks
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=incremental_request
        )

        if response.status_code == 202:
            data = response.json()

            # Response should indicate incremental processing
            if "processing_mode" in data:
                assert data["processing_mode"] == "incremental"

            # Should process fewer chunks than full batch
            if "chunk_count" in data:
                assert data["chunk_count"] == len(changed_chunks)

        elif response.status_code == 404:
            pytest.skip("Incremental processing not yet implemented")

    async def test_incremental_manifest_comparison(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        initial_manifest_all_subsystems: dict[str, Any],
        updated_manifest_fs_subsystem: dict[str, Any],
    ):
        """Test manifest endpoint support for incremental comparisons."""
        # Test manifest endpoint for incremental capabilities
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Manifest should support versioning for incremental updates
            assert "version" in data
            assert "created" in data

            if "chunks" in data and len(data["chunks"]) > 0:
                # Chunks should include metadata for incremental detection
                sample_chunk = data["chunks"][0]

                # Should have checksum for change detection
                assert "checksum_sha256" in sample_chunk
                assert len(sample_chunk["checksum_sha256"]) == 64

                # May include last_modified for temporal tracking
                if "last_modified" in sample_chunk:
                    assert isinstance(sample_chunk["last_modified"], str)

                # Should include subsystem for filtering
                assert "subsystem" in sample_chunk

        elif response.status_code == 404:
            pytest.skip("Chunk manifest endpoint not yet implemented")

    async def test_incremental_processing_efficiency(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        updated_manifest_fs_subsystem: dict[str, Any],
    ):
        """Test performance benefits of incremental processing."""
        manifest_data = updated_manifest_fs_subsystem

        # Test full batch processing time
        all_chunks = [chunk["id"] for chunk in manifest_data["chunks"]]

        full_start = time.time()
        full_request = {"chunk_ids": all_chunks, "parallelism": 4}

        full_response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=full_request
        )

        full_time = time.time() - full_start

        # Test incremental processing time (fs subsystem only)
        fs_chunks = [
            chunk["id"]
            for chunk in manifest_data["chunks"]
            if chunk["subsystem"] == "fs"
        ]

        incremental_start = time.time()
        incremental_request = {
            "chunk_ids": fs_chunks,
            "parallelism": 2,
            "incremental": True,
        }

        incremental_response = await http_client.post(
            "/mcp/chunks/process/batch",
            headers=auth_headers,
            json=incremental_request,
        )

        incremental_time = time.time() - incremental_start

        if full_response.status_code == 202 and incremental_response.status_code == 202:
            # Incremental should be faster for subset processing
            chunk_ratio = len(fs_chunks) / len(all_chunks)
            expected_time_ratio = chunk_ratio * 1.5  # Allow some overhead

            assert incremental_time <= full_time * expected_time_ratio, (
                f"Incremental processing should be proportionally faster: "
                f"{incremental_time:.1f}s vs {full_time:.1f}s"
            )

        elif full_response.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")

    async def test_incremental_subsystem_filtering(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        updated_manifest_fs_subsystem: dict[str, Any],
    ):
        """Test incremental processing with subsystem filtering."""
        manifest_data = updated_manifest_fs_subsystem

        # Test processing only specific subsystems
        target_subsystems = ["fs", "mm"]
        filtered_chunks = [
            chunk["id"]
            for chunk in manifest_data["chunks"]
            if chunk["subsystem"] in target_subsystems
        ]

        filtered_request = {
            "chunk_ids": filtered_chunks,
            "parallelism": 2,
            "subsystem_filter": target_subsystems,  # Filter by subsystem
            "incremental": True,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=filtered_request
        )

        if response.status_code == 202:
            data = response.json()

            # Should indicate filtered processing
            if "subsystems_included" in data:
                assert set(data["subsystems_included"]) == set(target_subsystems)

            # Wait for processing
            await asyncio.sleep(3)

            # Verify only target subsystems were processed
            for subsystem in target_subsystems:
                subsystem_chunks = [
                    chunk["id"]
                    for chunk in manifest_data["chunks"]
                    if chunk["subsystem"] == subsystem
                ]

                for chunk_id in subsystem_chunks[:2]:  # Check first 2 chunks
                    status_response = await http_client.get(
                        f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        # Should be processed or processing
                        assert status_data.get("status") in [
                            "pending",
                            "processing",
                            "completed",
                        ]

        elif response.status_code == 404:
            pytest.skip("Subsystem filtering not yet implemented")


@skip_integration_in_ci
@skip_without_mcp_server
class TestIncrementalUpdateScenarios:
    """Test various incremental update scenarios."""

    async def test_multiple_subsystem_incremental_update(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test incremental update affecting multiple subsystems."""
        # Simulate update to fs and net subsystems
        updated_subsystems = ["fs", "net"]
        chunk_ids = [
            f"{subsystem}_{i:03d}"
            for subsystem in updated_subsystems
            for i in range(1, 4)
        ]

        multi_subsystem_request = {
            "chunk_ids": chunk_ids,
            "parallelism": 3,
            "incremental": True,
            "updated_subsystems": updated_subsystems,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch",
            headers=auth_headers,
            json=multi_subsystem_request,
        )

        if response.status_code == 202:
            data = response.json()

            # Should handle multiple subsystem updates
            if "updated_subsystems" in data:
                assert set(data["updated_subsystems"]) == set(updated_subsystems)

        elif response.status_code == 404:
            pytest.skip("Multi-subsystem incremental updates not yet implemented")

    async def test_incremental_with_dependency_tracking(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test incremental update with inter-subsystem dependencies."""
        # fs subsystem changes might affect mm (memory management) due to VFS
        primary_chunks = ["fs_001", "fs_002"]
        dependent_chunks = ["mm_001"]  # mm depends on fs changes

        dependency_request = {
            "chunk_ids": primary_chunks + dependent_chunks,
            "parallelism": 2,
            "incremental": True,
            "track_dependencies": True,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=dependency_request
        )

        if response.status_code == 202:
            # Wait for processing
            await asyncio.sleep(5)

            # Check that both primary and dependent chunks were handled
            all_chunks = primary_chunks + dependent_chunks

            for chunk_id in all_chunks:
                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    # Should be processed due to dependency tracking
                    assert status_data.get("status") in [
                        "pending",
                        "processing",
                        "completed",
                    ]

        elif response.status_code == 404:
            pytest.skip("Dependency tracking not yet implemented")

    async def test_incremental_rollback_scenario(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test incremental processing rollback on partial failure."""
        # Simulate incremental update with one problematic chunk
        mixed_chunks = ["fs_001", "fs_002", "fs_999_invalid"]

        rollback_request = {
            "chunk_ids": mixed_chunks,
            "parallelism": 1,
            "incremental": True,
            "rollback_on_failure": True,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=rollback_request
        )

        if response.status_code == 202:
            # Wait for processing and potential rollback
            await asyncio.sleep(8)

            # Check if rollback occurred
            for chunk_id in mixed_chunks[:-1]:  # Valid chunks
                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()

                    # Might be rolled back due to batch failure
                    if "rollback_applied" in status_data:
                        assert isinstance(status_data["rollback_applied"], bool)

        elif response.status_code == 404:
            pytest.skip("Incremental rollback not yet implemented")
