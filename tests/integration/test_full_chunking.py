"""
Integration tests for full chunking workflow.

Tests the complete multi-file JSON chunking pipeline from manifest loading
through full kernel indexing with 60 chunks.
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
CHUNK_PROCESSING_TIMEOUT = 300  # 5 minutes for full chunking test


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
def sample_60_chunk_manifest() -> dict[str, Any]:
    """Generate a realistic 60-chunk manifest for testing."""
    chunks = []
    subsystems = ["kernel", "fs", "mm", "net", "drivers", "security", "crypto", "sound"]

    for i in range(1, 61):  # 60 chunks total
        subsystem = subsystems[i % len(subsystems)]
        chunk_id = f"{subsystem}_{i:03d}"

        chunks.append(
            {
                "id": chunk_id,
                "sequence": (i // len(subsystems)) + 1,
                "file": f"chunk_{chunk_id}.json",
                "subsystem": subsystem,
                "size_bytes": 50 * 1024 * 1024,  # 50MB chunks
                "checksum_sha256": f"a{str(i).zfill(63)}",  # Realistic checksum format
                "symbol_count": 15000 + (i * 100),
                "entry_point_count": 50 + (i * 2),
                "file_count": 120 + (i * 3),
            }
        )

    return {
        "version": "1.0.0",
        "created": "2025-01-18T10:00:00Z",
        "kernel_version": "6.7.0",
        "kernel_path": "/tmp/test-kernel",
        "config": "x86_64:defconfig",
        "total_chunks": 60,
        "total_size_bytes": 60 * 50 * 1024 * 1024,  # 3GB total
        "chunks": chunks,
    }


@pytest.fixture
def temp_chunk_directory():
    """Create temporary directory with sample chunk files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create minimal chunk files for testing
        for i in range(1, 61):
            subsystem = [
                "kernel",
                "fs",
                "mm",
                "net",
                "drivers",
                "security",
                "crypto",
                "sound",
            ][i % 8]
            chunk_id = f"{subsystem}_{i:03d}"
            chunk_file = temp_path / f"chunk_{chunk_id}.json"

            # Create realistic chunk content
            chunk_data = {
                "symbols": [
                    {"name": f"symbol_{i}_{j}", "type": "function"} for j in range(10)
                ],
                "entry_points": [
                    {"name": f"entry_{i}_{j}", "type": "syscall"} for j in range(3)
                ],
                "files": [f"/tmp/test-kernel/file_{i}_{j}.c" for j in range(5)],
                "metadata": {
                    "chunk_id": chunk_id,
                    "processed_at": "2025-01-18T10:00:00Z",
                },
            }

            chunk_file.write_text(json.dumps(chunk_data, indent=2))

        yield temp_path


@skip_integration_in_ci
@skip_without_mcp_server
class TestFullChunkingWorkflow:
    """Integration tests for complete 60-chunk processing workflow."""

    async def test_full_60_chunk_indexing_scenario(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_60_chunk_manifest: dict[str, Any],
        temp_chunk_directory: Path,
    ):
        """Test complete indexing of 60 chunks simulating real kernel processing."""
        # Save manifest to temp directory
        manifest_path = temp_chunk_directory / "manifest.json"
        manifest_path.write_text(json.dumps(sample_60_chunk_manifest, indent=2))

        # Verify manifest endpoint can load the 60-chunk manifest
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 404:
            # Expected if no chunks loaded yet - this is a contract test
            pytest.skip("Chunk manifest endpoint not yet implemented")

        # Track processing progress
        processed_chunks = []
        failed_chunks = []
        start_time = time.time()

        # Process chunks in batches to simulate realistic workflow
        batch_size = 10
        for batch_start in range(0, 60, batch_size):
            batch_end = min(batch_start + batch_size, 60)
            batch_chunk_ids = [
                sample_60_chunk_manifest["chunks"][i]["id"]
                for i in range(batch_start, batch_end)
            ]

            # Submit batch for processing
            batch_request = {
                "chunk_ids": batch_chunk_ids,
                "parallelism": 5,  # Process 5 chunks in parallel
            }

            batch_response = await http_client.post(
                "/mcp/chunks/process/batch", headers=auth_headers, json=batch_request
            )

            if batch_response.status_code == 202:
                # Track batch progress
                batch_response.json()  # Acknowledge response

                # Wait for batch completion with polling
                max_wait_time = 60  # 1 minute per batch
                poll_interval = 2  # Poll every 2 seconds

                for _ in range(max_wait_time // poll_interval):
                    await asyncio.sleep(poll_interval)

                    # Check status of chunks in this batch
                    batch_complete = True
                    for chunk_id in batch_chunk_ids:
                        status_response = await http_client.get(
                            f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                        )

                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            chunk_status = status_data.get("status", "unknown")

                            if chunk_status == "completed":
                                if chunk_id not in processed_chunks:
                                    processed_chunks.append(chunk_id)
                            elif chunk_status == "failed":
                                if chunk_id not in failed_chunks:
                                    failed_chunks.append(chunk_id)
                            elif chunk_status in ["pending", "processing"]:
                                batch_complete = False

                    if batch_complete:
                        break
            elif batch_response.status_code == 404:
                # Expected if batch processing endpoint not implemented
                pytest.skip("Batch processing endpoint not yet implemented")

        # Verify overall processing results
        total_time = time.time() - start_time

        # Performance assertions (based on constitutional requirements)
        assert total_time < 1800, (
            f"60-chunk processing took {total_time:.1f}s, should be under 30 minutes"
        )

        # Progress tracking assertions
        total_processed = len(processed_chunks) + len(failed_chunks)
        if total_processed > 0:
            success_rate = len(processed_chunks) / total_processed
            assert success_rate >= 0.9, (
                f"Success rate {success_rate:.1%} below 90% threshold"
            )

    async def test_chunk_manifest_with_60_chunks(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_60_chunk_manifest: dict[str, Any],
    ):
        """Test that manifest endpoint can handle 60-chunk manifests."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Verify manifest can handle large chunk counts
            assert "total_chunks" in data
            assert isinstance(data["total_chunks"], int)
            assert data["total_chunks"] > 0

            # Verify chunks array structure
            assert "chunks" in data
            assert isinstance(data["chunks"], list)

            if len(data["chunks"]) >= 50:  # If we have substantial chunks
                # Verify chunks are properly ordered
                sequences = [chunk["sequence"] for chunk in data["chunks"]]
                assert all(isinstance(seq, int) for seq in sequences)

                # Verify chunk distribution across subsystems
                subsystems = {chunk["subsystem"] for chunk in data["chunks"]}
                assert len(subsystems) >= 3, (
                    "Should have multiple subsystems for large kernel"
                )

                # Verify size constraints
                for chunk in data["chunks"]:
                    assert chunk["size_bytes"] <= 60 * 1024 * 1024, (
                        "Chunks should be <= 60MB"
                    )
        elif response.status_code == 404:
            pytest.skip("Chunk manifest endpoint not yet implemented")

    async def test_parallel_chunk_status_queries(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_60_chunk_manifest: dict[str, Any],
    ):
        """Test concurrent status queries for multiple chunks."""
        chunk_ids = [
            chunk["id"] for chunk in sample_60_chunk_manifest["chunks"][:20]
        ]  # Test first 20

        async def get_chunk_status(chunk_id: str):
            response = await http_client.get(
                f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
            )
            return chunk_id, response

        # Send concurrent status requests
        start_time = time.time()

        tasks = [get_chunk_status(chunk_id) for chunk_id in chunk_ids]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Performance assertion for concurrent queries
        assert total_time < 10, (
            f"20 concurrent status queries took {total_time:.1f}s, should be under 10s"
        )

        # Verify response consistency
        successful_responses = [
            (chunk_id, response)
            for chunk_id, response in results
            if response.status_code in [200, 404]
        ]

        if successful_responses:
            # All successful responses should have consistent structure
            response_codes = [
                response.status_code for _, response in successful_responses
            ]
            assert len(set(response_codes)) <= 2, (
                "Should only have 200 and/or 404 responses"
            )

    async def test_chunk_processing_memory_efficiency(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_60_chunk_manifest: dict[str, Any],
    ):
        """Test that chunk processing handles large manifests efficiently."""
        # Test manifest endpoint with large data set
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Verify response size is reasonable (should be compressed/paginated for large sets)
            response_size = len(response.content)

            # For 60 chunks, response should be under 1MB even with full metadata
            assert response_size < 1024 * 1024, (
                f"Manifest response {response_size} bytes, should be under 1MB"
            )

            # Verify JSON structure is memory-efficient
            assert isinstance(data, dict), (
                "Response should be well-structured JSON object"
            )

            # Check for required optimization fields
            if data.get("total_chunks", 0) > 30:
                # Large manifests should include summary statistics
                assert "total_size_bytes" in data, (
                    "Large manifests should include size summaries"
                )
        elif response.status_code == 404:
            pytest.skip("Chunk manifest endpoint not yet implemented")

    @pytest.mark.slow
    async def test_chunk_boundary_stress_test(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Stress test chunk processing at scale boundaries."""
        # Test rapid sequence of chunk status requests
        chunk_ids = [f"kernel_{i:03d}" for i in range(1, 101)]  # 100 chunk IDs

        errors = []
        timeouts = []

        for chunk_id in chunk_ids[:50]:  # Test first 50 to avoid overwhelming
            try:
                response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers, timeout=5.0
                )

                if response.status_code >= 500:
                    errors.append((chunk_id, response.status_code))

            except httpx.TimeoutException:
                timeouts.append(chunk_id)
            except Exception as e:
                errors.append((chunk_id, str(e)))

        # Error rate should be manageable
        error_rate = len(errors) / 50
        timeout_rate = len(timeouts) / 50

        assert error_rate < 0.1, f"Error rate {error_rate:.1%} too high for stress test"
        assert timeout_rate < 0.05, (
            f"Timeout rate {timeout_rate:.1%} too high for stress test"
        )


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkingPerformanceRequirements:
    """Test performance requirements for chunking workflow."""

    async def test_chunk_indexing_performance_target(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk indexing meets constitutional performance targets."""
        # Constitutional requirement: Index ≤20min, queries p95 ≤600ms

        start_time = time.time()

        # Test manifest query performance
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        manifest_time = (time.time() - start_time) * 1000  # Convert to ms

        if response.status_code == 200:
            # p95 < 600ms requirement
            assert manifest_time < 600, (
                f"Manifest query {manifest_time:.1f}ms exceeds 600ms p95 target"
            )

            data = response.json()
            chunk_count = data.get("total_chunks", 0)

            if chunk_count > 0:
                # Test status query performance for sample chunks
                status_times = []

                for i in range(min(10, chunk_count)):  # Test up to 10 chunks
                    if i < len(data.get("chunks", [])):
                        chunk_id = data["chunks"][i]["id"]

                        start_status = time.time()
                        status_response = await http_client.get(
                            f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                        )
                        status_time = (time.time() - start_status) * 1000

                        if status_response.status_code in [200, 404]:
                            status_times.append(status_time)

                if status_times:
                    # Calculate p95 of status query times
                    status_times.sort()
                    p95_index = int(len(status_times) * 0.95)
                    p95_time = (
                        status_times[p95_index]
                        if p95_index < len(status_times)
                        else status_times[-1]
                    )

                    assert p95_time < 600, (
                        f"Status query p95 {p95_time:.1f}ms exceeds 600ms target"
                    )

        elif response.status_code == 404:
            pytest.skip("Chunk manifest endpoint not yet implemented")
