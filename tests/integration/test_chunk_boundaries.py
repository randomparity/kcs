"""
Integration tests for chunk boundary validation.

Tests the chunk boundary validation system that ensures proper chunk
segmentation, size limits, and boundary handling during kernel indexing.
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
BOUNDARY_TEST_TIMEOUT = 90  # 1.5 minutes for boundary tests

# Chunk size limits for testing
MAX_CHUNK_SIZE = 60 * 1024 * 1024  # 60MB constitutional limit
TARGET_CHUNK_SIZE = 50 * 1024 * 1024  # 50MB target size
MIN_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB minimum size


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
def boundary_test_manifest() -> dict[str, Any]:
    """Generate manifest with chunks testing various boundary conditions."""
    chunks = []

    # Test different chunk sizes around boundaries
    test_cases = [
        # (chunk_id, size_bytes, expected_valid)
        ("kernel_tiny_001", 5 * 1024 * 1024, False),  # Too small (5MB)
        ("kernel_small_002", 15 * 1024 * 1024, True),  # Valid small (15MB)
        ("kernel_target_003", 50 * 1024 * 1024, True),  # Target size (50MB)
        ("kernel_large_004", 58 * 1024 * 1024, True),  # Near limit (58MB)
        ("kernel_limit_005", 60 * 1024 * 1024, True),  # At limit (60MB)
        ("kernel_over_006", 65 * 1024 * 1024, False),  # Over limit (65MB)
        ("fs_boundary_007", 49 * 1024 * 1024, True),  # Just under target
        ("fs_boundary_008", 51 * 1024 * 1024, True),  # Just over target
        ("mm_edge_009", 59 * 1024 * 1024, True),  # Edge case near limit
        ("net_invalid_010", 100 * 1024 * 1024, False),  # Way over limit
    ]

    for i, (chunk_id, size_bytes, expected_valid) in enumerate(test_cases, 1):
        # Calculate symbol count proportional to size
        symbols_per_mb = 200
        symbol_count = (size_bytes // (1024 * 1024)) * symbols_per_mb

        chunks.append(
            {
                "id": chunk_id,
                "sequence": i,
                "file": f"chunk_{chunk_id}.json",
                "subsystem": chunk_id.split("_")[0],
                "size_bytes": size_bytes,
                "checksum_sha256": f"c{str(i).zfill(63)}",
                "symbol_count": symbol_count,
                "entrypoint_count": max(10, size_bytes // (5 * 1024 * 1024)),
                "file_count": max(20, size_bytes // (2 * 1024 * 1024)),
                "expected_valid": expected_valid,  # Test metadata
            }
        )

    return {
        "version": "1.0.0",
        "created": "2025-01-18T14:00:00Z",
        "kernel_version": "6.7.0",
        "kernel_path": "/tmp/test-kernel-boundaries",
        "config": "x86_64:defconfig",
        "total_chunks": len(chunks),
        "total_size_bytes": sum(chunk["size_bytes"] for chunk in chunks),
        "chunks": chunks,
    }


@pytest.fixture
def oversized_chunk_manifest() -> dict[str, Any]:
    """Generate manifest with intentionally oversized chunks for validation testing."""
    chunks = [
        {
            "id": "kernel_oversized_001",
            "sequence": 1,
            "file": "chunk_kernel_oversized_001.json",
            "subsystem": "kernel",
            "size_bytes": 80 * 1024 * 1024,  # 80MB - exceeds 60MB limit
            "checksum_sha256": "d" + "0" * 63,
            "symbol_count": 16000,
            "entrypoint_count": 80,
            "file_count": 200,
        },
        {
            "id": "fs_massive_002",
            "sequence": 1,
            "file": "chunk_fs_massive_002.json",
            "subsystem": "fs",
            "size_bytes": 120 * 1024 * 1024,  # 120MB - way over limit
            "checksum_sha256": "e" + "1" * 63,
            "symbol_count": 24000,
            "entrypoint_count": 120,
            "file_count": 400,
        },
    ]

    return {
        "version": "1.0.0",
        "created": "2025-01-18T14:00:00Z",
        "kernel_version": "6.7.0",
        "kernel_path": "/tmp/test-kernel-oversized",
        "config": "x86_64:defconfig",
        "total_chunks": len(chunks),
        "total_size_bytes": sum(chunk["size_bytes"] for chunk in chunks),
        "chunks": chunks,
    }


@pytest.fixture
def temp_boundary_chunk_directory():
    """Create temporary directory with chunk files of various sizes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create chunk files with different content sizes
        chunk_sizes = [
            ("kernel_tiny_001", 100),  # Small content
            ("kernel_small_002", 1000),  # Medium content
            ("kernel_target_003", 5000),  # Large content
            ("kernel_large_004", 8000),  # Very large content
            ("kernel_limit_005", 10000),  # Maximum content
            ("kernel_over_006", 15000),  # Oversized content
            ("fs_boundary_007", 4900),  # Just under target
            ("fs_boundary_008", 5100),  # Just over target
            ("mm_edge_009", 9500),  # Near limit
            ("net_invalid_010", 20000),  # Way over limit
        ]

        for chunk_id, content_items in chunk_sizes:
            chunk_file = temp_path / f"chunk_{chunk_id}.json"

            # Generate proportional content
            chunk_data = {
                "symbols": [
                    {"name": f"symbol_{chunk_id}_{i}", "type": "function"}
                    for i in range(content_items)
                ],
                "entrypoints": [
                    {"name": f"entry_{chunk_id}_{i}", "type": "syscall"}
                    for i in range(content_items // 10)
                ],
                "files": [
                    f"/tmp/test-kernel/{chunk_id}/file_{i}.c"
                    for i in range(content_items // 5)
                ],
                "metadata": {
                    "chunk_id": chunk_id,
                    "content_size": content_items,
                    "processed_at": "2025-01-18T14:00:00Z",
                },
            }

            chunk_file.write_text(json.dumps(chunk_data, indent=2))

        yield temp_path


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkBoundaryValidation:
    """Integration tests for chunk boundary validation."""

    async def test_chunk_size_boundary_validation(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        boundary_test_manifest: dict[str, Any],
        temp_boundary_chunk_directory: Path,
    ):
        """Test validation of chunk size boundaries."""
        manifest_data = boundary_test_manifest

        # Test processing chunks with various sizes
        valid_chunks = [
            chunk["id"]
            for chunk in manifest_data["chunks"]
            if chunk.get("expected_valid", True)
        ]
        invalid_chunks = [
            chunk["id"]
            for chunk in manifest_data["chunks"]
            if not chunk.get("expected_valid", True)
        ]

        # Test valid chunks first
        if valid_chunks:
            valid_request = {"chunk_ids": valid_chunks, "parallelism": 2}

            valid_response = await http_client.post(
                "/mcp/chunks/process/batch", headers=auth_headers, json=valid_request
            )

            if valid_response.status_code == 202:
                # Valid chunks should be accepted
                data = valid_response.json()

                if "validation_errors" in data:
                    assert len(data["validation_errors"]) == 0, (
                        "Valid chunks should have no validation errors"
                    )

            elif valid_response.status_code == 404:
                pytest.skip("Batch processing endpoint not yet implemented")

        # Test invalid chunks
        if invalid_chunks:
            invalid_request = {"chunk_ids": invalid_chunks, "parallelism": 1}

            invalid_response = await http_client.post(
                "/mcp/chunks/process/batch", headers=auth_headers, json=invalid_request
            )

            if invalid_response.status_code == 400:
                # Expected validation error for oversized chunks
                error_data = invalid_response.json()
                assert "error" in error_data or "validation_errors" in error_data

            elif invalid_response.status_code == 202:
                # Might accept but mark as failed during processing
                await asyncio.sleep(3)

                # Check status of invalid chunks
                for chunk_id in invalid_chunks:
                    status_response = await http_client.get(
                        f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get("status") == "failed":
                            # Should have size-related error message
                            error_msg = status_data.get("error_message", "")
                            assert (
                                "size" in error_msg.lower()
                                or "limit" in error_msg.lower()
                            )

    async def test_chunk_manifest_size_validation(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        oversized_chunk_manifest: dict[str, Any],
    ):
        """Test manifest endpoint validation of chunk sizes."""
        # Test manifest endpoint with oversized chunks
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            if "chunks" in data and len(data["chunks"]) > 0:
                # Verify size constraints are enforced
                for chunk in data["chunks"]:
                    chunk_size = chunk.get("size_bytes", 0)

                    # Constitutional limit: chunks should not exceed 60MB
                    assert chunk_size <= MAX_CHUNK_SIZE, (
                        f"Chunk {chunk.get('id')} size {chunk_size} exceeds {MAX_CHUNK_SIZE} byte limit"
                    )

                    # Practical minimum: chunks should be substantial
                    if chunk_size > 0:
                        assert chunk_size >= MIN_CHUNK_SIZE, (
                            f"Chunk {chunk.get('id')} size {chunk_size} below {MIN_CHUNK_SIZE} byte minimum"
                        )

        elif response.status_code == 404:
            pytest.skip("Chunk manifest endpoint not yet implemented")

    async def test_chunk_boundary_splitting_logic(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        boundary_test_manifest: dict[str, Any],
    ):
        """Test logic for splitting oversized content into proper chunks."""
        manifest_data = boundary_test_manifest

        # Find chunks that are near or over the boundary
        boundary_chunks = [
            chunk["id"]
            for chunk in manifest_data["chunks"]
            if chunk["size_bytes"] >= TARGET_CHUNK_SIZE
        ]

        if boundary_chunks:
            boundary_request = {"chunk_ids": boundary_chunks, "parallelism": 1}

            response = await http_client.post(
                "/mcp/chunks/process/batch", headers=auth_headers, json=boundary_request
            )

            if response.status_code == 202:
                # Wait for processing
                await asyncio.sleep(5)

                # Check if oversized chunks were split or rejected
                for chunk_id in boundary_chunks:
                    status_response = await http_client.get(
                        f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        chunk_status = status_data.get("status", "unknown")

                        if chunk_status == "failed":
                            # Should fail with size-related error
                            error_msg = status_data.get("error_message", "")
                            assert any(
                                keyword in error_msg.lower()
                                for keyword in ["size", "limit", "boundary", "split"]
                            )

                        elif chunk_status == "completed":
                            # If completed, verify it was processed within limits
                            processed_size = status_data.get("processed_size_bytes")
                            if processed_size:
                                assert processed_size <= MAX_CHUNK_SIZE

            elif response.status_code == 404:
                pytest.skip("Chunk boundary processing not yet implemented")

    async def test_concurrent_boundary_validation(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        boundary_test_manifest: dict[str, Any],
    ):
        """Test boundary validation under concurrent processing load."""
        manifest_data = boundary_test_manifest

        # Create multiple concurrent requests with different chunk sizes
        small_chunks = [chunk["id"] for chunk in manifest_data["chunks"][:3]]
        medium_chunks = [chunk["id"] for chunk in manifest_data["chunks"][3:6]]
        large_chunks = [chunk["id"] for chunk in manifest_data["chunks"][6:9]]

        requests = [
            {"chunk_ids": small_chunks, "parallelism": 1},
            {"chunk_ids": medium_chunks, "parallelism": 1},
            {"chunk_ids": large_chunks, "parallelism": 1},
        ]

        # Send concurrent requests
        tasks = [
            http_client.post(
                "/mcp/chunks/process/batch", headers=auth_headers, json=req
            )
            for req in requests
        ]

        responses = await asyncio.gather(*tasks)

        # Verify all requests were handled appropriately
        successful_requests = sum(1 for r in responses if r.status_code in [202, 400])

        if successful_requests > 0:
            # Should handle concurrent boundary validation consistently
            assert successful_requests >= 2, (
                "Should handle most concurrent boundary requests"
            )

            # Wait for processing
            await asyncio.sleep(3)

            # Check for consistency in boundary enforcement
            all_chunks = small_chunks + medium_chunks + large_chunks
            validation_results = []

            for chunk_id in all_chunks:
                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    validation_results.append(
                        {
                            "chunk_id": chunk_id,
                            "status": status_data.get("status"),
                            "has_error": "error_message" in status_data,
                        }
                    )

            # Boundary validation should be consistent across concurrent requests
            if validation_results:
                error_count = sum(1 for r in validation_results if r["has_error"])
                total_count = len(validation_results)

                # Some chunks may fail due to size limits
                error_rate = error_count / total_count if total_count > 0 else 0
                assert error_rate <= 0.5, (
                    f"Error rate {error_rate:.1%} too high for boundary validation"
                )

        elif responses[0].status_code == 404:
            pytest.skip("Concurrent boundary validation not yet implemented")

    async def test_chunk_size_optimization_hints(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        boundary_test_manifest: dict[str, Any],
    ):
        """Test chunk size optimization and hints for better boundaries."""
        manifest_data = boundary_test_manifest

        # Test with chunks that could be optimized
        optimization_candidates = [
            chunk["id"]
            for chunk in manifest_data["chunks"]
            if 45 * 1024 * 1024
            <= chunk["size_bytes"]
            <= 55 * 1024 * 1024  # Near target size
        ]

        if optimization_candidates:
            optimization_request = {
                "chunk_ids": optimization_candidates,
                "parallelism": 1,
                "optimize_boundaries": True,  # Request boundary optimization
            }

            response = await http_client.post(
                "/mcp/chunks/process/batch",
                headers=auth_headers,
                json=optimization_request,
            )

            if response.status_code == 202:
                data = response.json()

                # Should provide optimization feedback
                if "optimization_hints" in data:
                    hints = data["optimization_hints"]
                    assert isinstance(hints, (list, dict)), (
                        "Optimization hints should be structured"
                    )

                # Wait for processing
                await asyncio.sleep(3)

                # Check for optimization results
                for chunk_id in optimization_candidates:
                    status_response = await http_client.get(
                        f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()

                        # May include optimization metadata
                        if "optimization_applied" in status_data:
                            assert isinstance(status_data["optimization_applied"], bool)

                        if "optimal_size_bytes" in status_data:
                            optimal_size = status_data["optimal_size_bytes"]
                            assert MIN_CHUNK_SIZE <= optimal_size <= MAX_CHUNK_SIZE

            elif response.status_code == 404:
                pytest.skip("Chunk optimization not yet implemented")


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkBoundaryErrorHandling:
    """Test error handling for chunk boundary violations."""

    async def test_extreme_chunk_size_handling(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test handling of extremely large or small chunk sizes."""
        # Test with extreme sizes
        extreme_chunks = [
            {
                "id": "tiny_chunk_001",
                "size_bytes": 1024,  # 1KB - extremely small
            },
            {
                "id": "massive_chunk_002",
                "size_bytes": 500 * 1024 * 1024,  # 500MB - extremely large
            },
        ]

        for chunk_data in extreme_chunks:
            chunk_id = chunk_data["id"]

            extreme_request = {
                "chunk_ids": [chunk_id],
                "parallelism": 1,
            }

            response = await http_client.post(
                "/mcp/chunks/process/batch", headers=auth_headers, json=extreme_request
            )

            # Should handle extreme sizes gracefully
            if response.status_code == 400:
                # Expected validation error
                error_data = response.json()
                assert "error" in error_data

            elif response.status_code == 202:
                # Might accept but fail during processing
                await asyncio.sleep(2)

                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "failed":
                        # Should have descriptive error message
                        error_msg = status_data.get("error_message", "")
                        assert len(error_msg) > 0

            elif response.status_code == 404:
                pytest.skip("Extreme size handling not yet implemented")

    async def test_boundary_validation_error_messages(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test quality of boundary validation error messages."""
        # Test with known oversized chunk
        oversized_request = {
            "chunk_ids": ["test_oversized_chunk"],
            "parallelism": 1,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=oversized_request
        )

        if response.status_code == 400:
            error_data = response.json()

            # Error message should be descriptive
            error_msg = error_data.get("message", "")
            assert any(
                keyword in error_msg.lower()
                for keyword in ["size", "limit", "boundary", "exceeded"]
            ), f"Error message should describe size issue: {error_msg}"

        elif response.status_code == 404:
            pytest.skip("Boundary validation error messages not yet implemented")

    async def test_chunk_boundary_recovery(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test recovery from boundary validation failures."""
        # Test processing valid chunks after boundary failures
        mixed_chunks = [
            "valid_chunk_001",  # Should work
            "oversized_chunk_002",  # Should fail
            "valid_chunk_003",  # Should work
        ]

        mixed_request = {
            "chunk_ids": mixed_chunks,
            "parallelism": 1,
            "continue_on_failure": True,  # Continue processing despite failures
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", headers=auth_headers, json=mixed_request
        )

        if response.status_code == 202:
            # Wait for processing
            await asyncio.sleep(5)

            # Check final status
            valid_count = 0
            failed_count = 0

            for chunk_id in mixed_chunks:
                status_response = await http_client.get(
                    f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    chunk_status = status_data.get("status", "unknown")

                    if chunk_status == "completed":
                        valid_count += 1
                    elif chunk_status == "failed":
                        failed_count += 1

            # Should have processed valid chunks despite boundary failures
            if valid_count + failed_count > 0:
                success_rate = valid_count / (valid_count + failed_count)
                assert success_rate >= 0.5, "Should recover and process valid chunks"

        elif response.status_code == 404:
            pytest.skip("Boundary recovery not yet implemented")
