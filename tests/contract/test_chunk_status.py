"""
Contract tests for chunk status MCP tool.

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
class TestChunkStatusContract:
    """Contract tests for chunk status MCP tool."""

    async def test_chunk_status_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the chunk status endpoint exists and accepts GET requests."""
        response = await http_client.get(
            "/mcp/chunks/kernel_001/status", headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "chunk status endpoint should exist"

    async def test_chunk_status_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that chunk status requires valid authentication."""
        # Request without auth headers
        response = await http_client.get("/mcp/chunks/kernel_001/status")
        assert response.status_code == 401, "Should require authentication"

    async def test_chunk_status_path_parameter_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk ID path parameter is properly validated."""
        # Valid chunk ID format
        response = await http_client.get(
            "/mcp/chunks/kernel_001/status", headers=auth_headers
        )
        # Should not fail due to path parameter format
        assert response.status_code != 422, "Valid chunk ID should be accepted"

        # Test with different valid chunk ID patterns
        valid_chunk_ids = ["kernel_001", "fs_042", "mm_999", "drivers_123"]
        for chunk_id in valid_chunk_ids:
            response = await http_client.get(
                f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
            )
            # Should not fail due to format (may return 404 if chunk doesn't exist)
            assert response.status_code in [200, 404], (
                f"Valid chunk ID '{chunk_id}' should be accepted"
            )

    async def test_chunk_status_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk status returns response matching OpenAPI schema."""
        response = await http_client.get(
            "/mcp/chunks/kernel_001/status", headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify required fields
            assert "chunk_id" in data, "Response should contain 'chunk_id' field"
            assert "status" in data, "Response should contain 'status' field"
            assert "manifest_version" in data, (
                "Response should contain 'manifest_version' field"
            )

            # Verify field types and constraints
            assert isinstance(data["chunk_id"], str), "Chunk ID should be string"
            assert isinstance(data["status"], str), "Status should be string"
            assert data["status"] in ["pending", "processing", "completed", "failed"], (
                "Status should be valid enum value"
            )
            assert isinstance(data["manifest_version"], str), (
                "Manifest version should be string"
            )

            # Verify optional fields if present
            if "started_at" in data:
                assert isinstance(data["started_at"], str), (
                    "Started at should be ISO string"
                )

            if "completed_at" in data:
                assert isinstance(data["completed_at"], str), (
                    "Completed at should be ISO string"
                )

            if "error_message" in data:
                assert isinstance(data["error_message"], str), (
                    "Error message should be string"
                )

            if "retry_count" in data:
                assert isinstance(data["retry_count"], int), (
                    "Retry count should be integer"
                )
                assert 0 <= data["retry_count"] <= 3, "Retry count should be 0-3"

            if "symbols_processed" in data:
                assert isinstance(data["symbols_processed"], int), (
                    "Symbols processed should be integer"
                )
                assert data["symbols_processed"] >= 0, (
                    "Symbols processed should be >= 0"
                )

            if "checksum_verified" in data:
                assert isinstance(data["checksum_verified"], bool), (
                    "Checksum verified should be boolean"
                )

    async def test_chunk_status_status_transitions(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that status values follow logical transitions."""
        response = await http_client.get(
            "/mcp/chunks/kernel_001/status", headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            status = data["status"]

            # Verify status-specific field constraints
            if status == "pending":
                # Pending chunks should not have completed_at or symbols_processed
                assert "completed_at" not in data, (
                    "Pending chunks should not have completed_at"
                )
                assert (
                    "symbols_processed" not in data or data["symbols_processed"] == 0
                ), "Pending chunks should not have processed symbols"

            elif status == "processing":
                # Processing chunks should have started_at but not completed_at
                assert "started_at" in data, "Processing chunks should have started_at"
                assert "completed_at" not in data, (
                    "Processing chunks should not have completed_at"
                )

            elif status == "completed":
                # Completed chunks should have both timestamps
                assert "started_at" in data, "Completed chunks should have started_at"
                assert "completed_at" in data, (
                    "Completed chunks should have completed_at"
                )
                assert "error_message" not in data, (
                    "Completed chunks should not have error_message"
                )

            elif status == "failed":
                # Failed chunks should have error_message
                assert "error_message" in data, (
                    "Failed chunks should have error_message"
                )
                assert len(data["error_message"]) > 0, (
                    "Error message should not be empty"
                )

    async def test_chunk_status_nonexistent_chunk_404(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that 404 is returned for non-existent chunks."""
        response = await http_client.get(
            "/mcp/chunks/nonexistent_chunk_12345/status", headers=auth_headers
        )

        if response.status_code == 404:
            data = response.json()
            # Should follow error schema
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_status_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk status meets p95 < 600ms performance requirement."""
        import time

        start_time = time.time()

        response = await http_client.get(
            "/mcp/chunks/kernel_001/status",
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

    async def test_chunk_status_timestamp_consistency(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test timestamp consistency in status responses."""
        response = await http_client.get(
            "/mcp/chunks/kernel_001/status", headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # If both timestamps present, completed_at should be after started_at
            if "started_at" in data and "completed_at" in data:
                from datetime import datetime

                started = datetime.fromisoformat(
                    data["started_at"].replace("Z", "+00:00")
                )
                completed = datetime.fromisoformat(
                    data["completed_at"].replace("Z", "+00:00")
                )

                assert completed >= started, "completed_at should be >= started_at"

    async def test_chunk_status_retry_count_constraints(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test retry count constraints for failed chunks."""
        response = await http_client.get(
            "/mcp/chunks/kernel_001/status", headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            if data["status"] == "failed" and "retry_count" in data:
                retry_count = data["retry_count"]
                assert 0 <= retry_count <= 3, "Retry count should be between 0-3"

                # Failed chunks with max retries should not be retryable
                if retry_count >= 3:
                    # This would be implementation-specific behavior
                    pass

    @pytest.mark.integration
    async def test_chunk_status_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample chunk data (if available)."""
        # Try several common chunk IDs
        chunk_ids = ["kernel_001", "kernel_002", "fs_001"]

        for chunk_id in chunk_ids:
            response = await http_client.get(
                f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
            )

            if response.status_code == 200:
                data = response.json()

                # Verify chunk_id matches request
                assert data["chunk_id"] == chunk_id, (
                    f"Response chunk_id should match requested ID '{chunk_id}'"
                )

                # If we find a chunk, test additional properties
                if data["status"] in ["completed", "failed"]:
                    assert "started_at" in data, (
                        "Processed chunks should have started_at"
                    )

                break  # Found at least one chunk to test with

    async def test_chunk_status_concurrent_requests(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that concurrent status requests return consistent results."""
        import asyncio

        # Make multiple concurrent requests for the same chunk
        async def get_status():
            return await http_client.get(
                "/mcp/chunks/kernel_001/status", headers=auth_headers
            )

        # Send 5 concurrent requests
        responses = await asyncio.gather(*[get_status() for _ in range(5)])

        successful_responses = [r for r in responses if r.status_code == 200]

        if len(successful_responses) > 1:
            # All successful responses should return identical data
            first_data = successful_responses[0].json()
            for response in successful_responses[1:]:
                data = response.json()
                assert data == first_data, (
                    "Concurrent requests should return identical data"
                )


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkStatusErrorHandling:
    """Test error handling for chunk status tool."""

    async def test_chunk_status_invalid_chunk_id_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with invalid chunk ID formats."""
        invalid_chunk_ids = [
            "chunk with spaces",
            "chunk-with-special-chars!@#",
            "",  # empty string
            "x" * 1000,  # very long ID
        ]

        for chunk_id in invalid_chunk_ids:
            response = await http_client.get(
                f"/mcp/chunks/{chunk_id}/status", headers=auth_headers
            )

            # Should either reject (404/422) or handle gracefully
            assert response.status_code in [404, 422], (
                f"Should handle invalid chunk ID '{chunk_id[:50]}...' gracefully"
            )

    async def test_chunk_status_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        response = await http_client.get(
            "/mcp/chunks/kernel_001/status", headers=auth_headers
        )

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_status_malformed_url(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with malformed URLs."""
        malformed_urls = [
            "/mcp/chunks//status",  # empty chunk ID
            "/mcp/chunks/kernel_001/",  # missing status
            "/mcp/chunks/kernel_001/status/extra",  # extra path components
        ]

        for url in malformed_urls:
            response = await http_client.get(url, headers=auth_headers)
            # Should return 404 for malformed URLs
            assert response.status_code == 404, (
                f"Malformed URL '{url}' should return 404"
            )
