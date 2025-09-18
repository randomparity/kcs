"""
Integration tests for resuming chunk processing after failures.

Tests the chunk processing resume capability that allows recovery
from partial failures during large-scale kernel indexing.
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
RESUME_TEST_TIMEOUT = 180  # 3 minutes for resume tests


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
def sample_30_chunk_manifest_with_failure() -> dict[str, Any]:
    """Generate manifest designed to test resume from chunk 15 failure."""
    chunks = []
    subsystems = ["kernel", "fs", "mm", "net", "drivers"]

    for i in range(1, 31):  # 30 chunks total, failure at chunk 15
        subsystem = subsystems[i % len(subsystems)]
        chunk_id = f"{subsystem}_{i:03d}"

        # Mark chunk 15 as problematic for testing resume functionality
        if i == 15:
            chunk_id = "fs_015_corrupt"  # Intentionally problematic ID

        chunks.append(
            {
                "id": chunk_id,
                "sequence": i,
                "file": f"chunk_{chunk_id}.json",
                "subsystem": subsystem,
                "size_bytes": 50 * 1024 * 1024,  # 50MB chunks
                "checksum_sha256": f"b{str(i).zfill(63)}",
                "symbol_count": 12000 + (i * 50),
                "entry_point_count": 40 + i,
                "file_count": 100 + (i * 2),
            }
        )

    return {
        "version": "1.0.0",
        "created": "2025-01-18T12:00:00Z",
        "kernel_version": "6.7.0",
        "kernel_path": "/tmp/test-kernel-resume",
        "config": "x86_64:defconfig",
        "total_chunks": 30,
        "total_size_bytes": 30 * 50 * 1024 * 1024,
        "chunks": chunks,
    }


@pytest.fixture
def temp_chunk_directory_with_failure():
    """Create temporary directory with chunk files including a problematic one."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i in range(1, 31):
            subsystems = ["kernel", "fs", "mm", "net", "drivers"]
            subsystem = subsystems[i % len(subsystems)]

            if i == 15:
                chunk_id = "fs_015_corrupt"
                # Create a corrupt/incomplete chunk file
                chunk_file = temp_path / f"chunk_{chunk_id}.json"
                chunk_file.write_text('{"incomplete": "data"')  # Invalid JSON
            else:
                chunk_id = f"{subsystem}_{i:03d}"
                chunk_file = temp_path / f"chunk_{chunk_id}.json"

                # Create valid chunk content
                chunk_data = {
                    "symbols": [
                        {"name": f"symbol_{i}_{j}", "type": "function"}
                        for j in range(8)
                    ],
                    "entry_points": [
                        {"name": f"entry_{i}_{j}", "type": "syscall"} for j in range(2)
                    ],
                    "files": [f"/tmp/test-kernel/file_{i}_{j}.c" for j in range(4)],
                    "metadata": {
                        "chunk_id": chunk_id,
                        "processed_at": "2025-01-18T12:00:00Z",
                    },
                }

                chunk_file.write_text(json.dumps(chunk_data, indent=2))

        yield temp_path


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkProcessingResume:
    """Integration tests for chunk processing resume functionality."""

    async def test_resume_from_chunk_15_failure(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_30_chunk_manifest_with_failure: dict[str, Any],
        temp_chunk_directory_with_failure: Path,
    ):
        """Test resuming chunk processing after failure at chunk 15."""
        manifest_data = sample_30_chunk_manifest_with_failure

        # Simulate initial processing that fails at chunk 15
        initial_batch = [
            chunk["id"] for chunk in manifest_data["chunks"][:20]
        ]  # Process first 20 chunks

        batch_request = {"chunk_ids": initial_batch, "parallelism": 3}

        batch_response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=batch_request
        )

        if batch_response.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")

        # Track which chunks completed before failure
        completed_chunks = []
        failed_chunks = []

        if batch_response.status_code == 202:
            # Wait for processing and expect failure at chunk 15
            await asyncio.sleep(5)  # Allow some processing time

            # Check status of all chunks in batch
            for chunk_id in initial_batch:
                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    chunk_status = status_data.get("status", "unknown")

                    if chunk_status == "completed":
                        completed_chunks.append(chunk_id)
                    elif chunk_status == "failed":
                        failed_chunks.append(chunk_id)

        # Verify that some chunks completed and chunk 15 (or related) failed
        assert len(completed_chunks) >= 10, (
            "Should have some successful chunks before failure"
        )

        # Now test resume functionality - process remaining chunks
        remaining_chunks = [
            chunk["id"] for chunk in manifest_data["chunks"][20:]
        ]  # Chunks 21-30

        resume_request = {
            "chunk_ids": remaining_chunks,
            "parallelism": 2,
            "resume": True,  # Indicate this is a resume operation
        }

        resume_response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=resume_request
        )

        if resume_response.status_code == 202:
            # Wait for resume processing to complete
            await asyncio.sleep(10)

            # Check final status of all chunks
            final_completed = []
            final_failed = []

            for chunk in manifest_data["chunks"]:
                chunk_id = chunk["id"]
                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    chunk_status = status_data.get("status", "unknown")

                    if chunk_status == "completed":
                        final_completed.append(chunk_id)
                    elif chunk_status == "failed":
                        final_failed.append(chunk_id)

            # Verify resume worked - should have most chunks completed
            total_processed = len(final_completed) + len(final_failed)
            if total_processed > 0:
                success_rate = len(final_completed) / total_processed
                assert success_rate >= 0.85, (
                    f"Resume success rate {success_rate:.1%} below 85%"
                )

            # Should have at least one known failure (the corrupt chunk)
            assert len(final_failed) >= 1, "Should have failed chunks from corruption"

    async def test_resume_status_tracking(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_30_chunk_manifest_with_failure: dict[str, Any],
    ):
        """Test that resume operations are properly tracked in status."""
        # Test status endpoint for chunk that should fail
        corrupt_chunk_id = "fs_015_corrupt"

        status_response = await http_client.get(
            f"/mcp/chunks/{corrupt_chunk_id}/status", headers=auth_headers
        )

        if status_response.status_code == 200:
            status_data = status_response.json()

            # Verify status response includes resume-related fields
            assert "chunk_id" in status_data
            assert status_data["chunk_id"] == corrupt_chunk_id

            # Check for retry tracking
            if "retry_count" in status_data:
                assert isinstance(status_data["retry_count"], int)
                assert 0 <= status_data["retry_count"] <= 3

            # Check for failure details if chunk failed
            if status_data.get("status") == "failed":
                assert "error_message" in status_data
                assert len(status_data["error_message"]) > 0

        elif status_response.status_code == 404:
            pytest.skip("Chunk status endpoint not yet implemented")

    async def test_resume_idempotency(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_30_chunk_manifest_with_failure: dict[str, Any],
    ):
        """Test that resume operations are idempotent."""
        manifest_data = sample_30_chunk_manifest_with_failure

        # Get subset of chunks for testing
        test_chunks = [chunk["id"] for chunk in manifest_data["chunks"][:5]]

        # First resume request
        resume_request_1 = {
            "chunk_ids": test_chunks,
            "parallelism": 2,
            "resume": True,
        }

        response_1 = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=resume_request_1
        )

        # Second identical resume request (should be idempotent)
        resume_request_2 = {
            "chunk_ids": test_chunks,
            "parallelism": 2,
            "resume": True,
        }

        response_2 = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=resume_request_2
        )

        if response_1.status_code == 202 and response_2.status_code == 202:
            # Both requests should be accepted
            data_1 = response_1.json()
            data_2 = response_2.json()

            # Should have similar structure (idempotent behavior)
            assert "batch_id" in data_1 or "request_id" in data_1
            assert "batch_id" in data_2 or "request_id" in data_2

        elif response_1.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")

    async def test_resume_progress_reporting(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_30_chunk_manifest_with_failure: dict[str, Any],
    ):
        """Test progress reporting during resume operations."""
        manifest_data = sample_30_chunk_manifest_with_failure

        # Start resume operation
        resume_chunks = [chunk["id"] for chunk in manifest_data["chunks"][5:15]]

        resume_request = {
            "chunk_ids": resume_chunks,
            "parallelism": 1,  # Slow processing for progress tracking
            "resume": True,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=resume_request
        )

        if response.status_code == 202:
            # Monitor progress over time
            progress_snapshots = []

            for _ in range(5):  # Take 5 progress snapshots
                await asyncio.sleep(2)

                # Check status of chunks in resume batch
                completed_count = 0
                processing_count = 0
                failed_count = 0

                for chunk_id in resume_chunks:
                    status_response = await http_client.get(
                        f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        chunk_status = status_data.get("status", "unknown")

                        if chunk_status == "completed":
                            completed_count += 1
                        elif chunk_status == "processing":
                            processing_count += 1
                        elif chunk_status == "failed":
                            failed_count += 1

                progress_snapshots.append(
                    {
                        "completed": completed_count,
                        "processing": processing_count,
                        "failed": failed_count,
                        "timestamp": time.time(),
                    }
                )

            # Verify progress is trackable
            if len(progress_snapshots) > 1:
                # Progress should be monotonically increasing (completed + failed)
                first_total = (
                    progress_snapshots[0]["completed"] + progress_snapshots[0]["failed"]
                )
                last_total = (
                    progress_snapshots[-1]["completed"]
                    + progress_snapshots[-1]["failed"]
                )

                assert last_total >= first_total, "Progress should increase over time"

        elif response.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")

    async def test_resume_with_manifest_endpoint(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_30_chunk_manifest_with_failure: dict[str, Any],
    ):
        """Test resume functionality with manifest endpoint integration."""
        # Check manifest endpoint for resume-related information
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Verify manifest includes processing status information
            if "chunks" in data and len(data["chunks"]) > 0:
                # Check for status-related fields in chunk metadata
                sample_chunk = data["chunks"][0]

                # Manifest might include processing status
                if "status" in sample_chunk:
                    assert sample_chunk["status"] in [
                        "pending",
                        "processing",
                        "completed",
                        "failed",
                    ]

                # Verify chunk IDs are consistent with resume operations
                chunk_ids = [chunk["id"] for chunk in data["chunks"]]
                assert len(chunk_ids) == len(set(chunk_ids)), (
                    "Chunk IDs should be unique"
                )

        elif response.status_code == 404:
            pytest.skip("Chunk manifest endpoint not yet implemented")


@skip_integration_in_ci
@skip_without_mcp_server
class TestResumeErrorHandling:
    """Test error handling during resume operations."""

    async def test_resume_invalid_chunk_ids(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test resume behavior with invalid chunk IDs."""
        # Try to resume with non-existent chunk IDs
        invalid_request = {
            "chunk_ids": ["nonexistent_001", "invalid_chunk_999"],
            "parallelism": 1,
            "resume": True,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=invalid_request
        )

        # Should handle invalid chunk IDs gracefully
        if response.status_code == 400:
            # Expected validation error
            error_data = response.json()
            assert "error" in error_data or "message" in error_data
        elif response.status_code == 202:
            # Might accept but track failures
            pass
        elif response.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")

    async def test_resume_chunk_already_completed(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test resume behavior when chunk is already completed."""
        # Test with chunk ID that should be already processed
        already_processed_request = {
            "chunk_ids": ["kernel_001"],  # Assume this was processed earlier
            "parallelism": 1,
            "resume": True,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch",
            headers=auth_headers,
            json=already_processed_request,
        )

        if response.status_code == 202:
            # Should handle gracefully (idempotent)
            data = response.json()
            # Response should indicate no new work needed
            assert isinstance(data, dict)

        elif response.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")

    async def test_resume_concurrent_operations(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test resume behavior with concurrent operations."""
        # Start two concurrent resume operations
        chunk_set_1 = ["kernel_010", "kernel_011", "kernel_012"]
        chunk_set_2 = ["fs_010", "fs_011", "fs_012"]

        request_1 = {"chunk_ids": chunk_set_1, "parallelism": 1, "resume": True}
        request_2 = {"chunk_ids": chunk_set_2, "parallelism": 1, "resume": True}

        # Send both requests concurrently
        task_1 = http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=request_1
        )
        task_2 = http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=request_2
        )

        response_1, response_2 = await asyncio.gather(task_1, task_2)

        # Both should be handled appropriately
        if response_1.status_code == 202 and response_2.status_code == 202:
            # Both accepted - system handles concurrent resumes
            data_1 = response_1.json()
            data_2 = response_2.json()

            # Should have different batch/request identifiers
            if "batch_id" in data_1 and "batch_id" in data_2:
                assert data_1["batch_id"] != data_2["batch_id"]

        elif response_1.status_code == 404:
            pytest.skip("Batch processing endpoint not yet implemented")
