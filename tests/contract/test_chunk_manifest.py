"""
Contract tests for chunk manifest MCP tool.

These tests verify the API contract defined in contracts/chunk-api.yaml.
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


# Contract test cases
@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkManifestContract:
    """Contract tests for chunk manifest MCP tool."""

    async def test_chunk_manifest_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the chunk manifest endpoint exists and accepts GET requests."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "chunk manifest endpoint should exist"

    async def test_chunk_manifest_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that chunk manifest requires valid authentication."""
        # Request without auth headers
        response = await http_client.get("/mcp/chunks/manifest")
        assert response.status_code == 401, "Should require authentication"

    async def test_chunk_manifest_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk manifest returns response matching OpenAPI schema."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Verify required top-level structure
            assert "version" in data, "Response should contain 'version' field"
            assert "created" in data, "Response should contain 'created' field"
            assert "total_chunks" in data, (
                "Response should contain 'total_chunks' field"
            )
            assert "chunks" in data, "Response should contain 'chunks' field"

            # Verify field types and constraints
            assert isinstance(data["version"], str), "Version should be string"
            assert len(data["version"].split(".")) == 3, (
                "Version should be semver format"
            )

            assert isinstance(data["created"], str), "Created should be ISO string"

            assert isinstance(data["total_chunks"], int), (
                "Total chunks should be integer"
            )
            assert data["total_chunks"] >= 1, "Total chunks should be >= 1"

            assert isinstance(data["chunks"], list), "Chunks should be array"

            # Verify optional fields if present
            if "kernel_version" in data:
                assert isinstance(data["kernel_version"], str), (
                    "Kernel version should be string"
                )

            if "kernel_path" in data:
                assert isinstance(data["kernel_path"], str), (
                    "Kernel path should be string"
                )

            if "config" in data:
                assert isinstance(data["config"], str), "Config should be string"

            if "total_size_bytes" in data:
                assert isinstance(data["total_size_bytes"], int), (
                    "Total size should be integer"
                )
                assert data["total_size_bytes"] > 0, "Total size should be positive"

            # Verify chunks array structure
            for chunk in data["chunks"]:
                assert isinstance(chunk, dict), "Each chunk should be object"

                # Required fields
                assert "id" in chunk, "Chunk should have 'id' field"
                assert "sequence" in chunk, "Chunk should have 'sequence' field"
                assert "file" in chunk, "Chunk should have 'file' field"
                assert "subsystem" in chunk, "Chunk should have 'subsystem' field"
                assert "size_bytes" in chunk, "Chunk should have 'size_bytes' field"
                assert "checksum_sha256" in chunk, (
                    "Chunk should have 'checksum_sha256' field"
                )

                # Field types and constraints
                assert isinstance(chunk["id"], str), "Chunk ID should be string"
                assert isinstance(chunk["sequence"], int), "Sequence should be integer"
                assert chunk["sequence"] >= 1, "Sequence should be >= 1"
                assert isinstance(chunk["file"], str), "File should be string"
                assert isinstance(chunk["subsystem"], str), "Subsystem should be string"
                assert isinstance(chunk["size_bytes"], int), "Size should be integer"
                assert chunk["size_bytes"] > 0, "Size should be positive"
                assert isinstance(chunk["checksum_sha256"], str), (
                    "Checksum should be string"
                )
                assert len(chunk["checksum_sha256"]) == 64, (
                    "Checksum should be 64 hex chars"
                )
                assert all(c in "0123456789abcdef" for c in chunk["checksum_sha256"]), (
                    "Checksum should be hex"
                )

                # Optional fields
                if "symbol_count" in chunk:
                    assert isinstance(chunk["symbol_count"], int), (
                        "Symbol count should be integer"
                    )
                    assert chunk["symbol_count"] >= 0, "Symbol count should be >= 0"

                if "entry_point_count" in chunk:
                    assert isinstance(chunk["entry_point_count"], int), (
                        "Entry point count should be integer"
                    )
                    assert chunk["entry_point_count"] >= 0, (
                        "Entry point count should be >= 0"
                    )

                if "file_count" in chunk:
                    assert isinstance(chunk["file_count"], int), (
                        "File count should be integer"
                    )
                    assert chunk["file_count"] >= 0, "File count should be >= 0"

    async def test_chunk_manifest_no_manifest_404(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that 404 is returned when no manifest exists."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 404:
            data = response.json()
            # Should follow error schema
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_manifest_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk manifest meets p95 < 600ms performance requirement."""
        import time

        start_time = time.time()

        response = await http_client.get(
            "/mcp/chunks/manifest",
            headers=auth_headers,
            timeout=1.0,  # 1 second timeout
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [200, 404]:
            # Performance requirement from constitution: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_chunk_manifest_chunk_ordering(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunks are ordered by sequence number."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()
            chunks = data["chunks"]

            if len(chunks) > 1:
                # Verify chunks are sorted by sequence
                sequences = [chunk["sequence"] for chunk in chunks]
                assert sequences == sorted(sequences), (
                    "Chunks should be ordered by sequence"
                )

                # Verify sequences are consecutive starting from 1
                expected_sequences = list(range(1, len(chunks) + 1))
                assert sequences == expected_sequences, (
                    "Sequences should be consecutive from 1"
                )

    async def test_chunk_manifest_consistency(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test internal consistency of manifest data."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Verify total_chunks matches actual chunk count
            assert data["total_chunks"] == len(data["chunks"]), (
                "total_chunks should match chunks array length"
            )

            # Verify total_size_bytes if present
            if "total_size_bytes" in data:
                calculated_size = sum(chunk["size_bytes"] for chunk in data["chunks"])
                assert data["total_size_bytes"] == calculated_size, (
                    "total_size_bytes should match sum of chunk sizes"
                )

    @pytest.mark.integration
    async def test_chunk_manifest_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample chunk data (if available)."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()

            # Should have reasonable structure for kernel chunks
            assert len(data["chunks"]) > 0, "Should have at least one chunk"

            # Verify chunk IDs follow expected pattern
            chunk_ids = [chunk["id"] for chunk in data["chunks"]]
            assert all(chunk_id.startswith("kernel_") for chunk_id in chunk_ids), (
                "Chunk IDs should follow kernel_XXX pattern"
            )

            # Verify subsystems are reasonable
            subsystems = {chunk["subsystem"] for chunk in data["chunks"]}
            expected_subsystems = {"kernel", "fs", "mm", "net", "drivers"}
            assert subsystems.intersection(expected_subsystems), (
                "Should contain common kernel subsystems"
            )


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkManifestErrorHandling:
    """Test error handling for chunk manifest tool."""

    async def test_chunk_manifest_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        response = await http_client.get("/mcp/chunks/manifest", headers=auth_headers)

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_manifest_invalid_accept_header(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with invalid Accept header."""
        headers = {**auth_headers, "Accept": "text/plain"}

        response = await http_client.get("/mcp/chunks/manifest", headers=headers)

        # Should either handle gracefully or return 406
        assert response.status_code in [200, 406], (
            "Should handle Accept header gracefully"
        )
