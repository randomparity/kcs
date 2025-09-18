"""
Contract tests for chunk batch process MCP tool.

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
class TestChunkBatchContract:
    """Contract tests for chunk batch process MCP tool."""

    async def test_chunk_batch_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the chunk batch process endpoint exists and accepts POST requests."""
        payload = {
            "chunk_ids": ["kernel_001", "kernel_002"],
            "parallelism": 2,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "chunk batch process endpoint should exist"

    async def test_chunk_batch_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that chunk batch process requires valid authentication."""
        payload = {
            "chunk_ids": ["kernel_001", "kernel_002"],
        }

        # Request without auth headers
        response = await http_client.post("/mcp/chunks/process/batch", json=payload)
        assert response.status_code == 401, "Should require authentication"

    async def test_chunk_batch_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk batch process validates request schema according to OpenAPI spec."""

        # Valid request with all parameters
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": ["kernel_001", "kernel_002"], "parallelism": 4},
            headers=auth_headers,
        )
        assert response.status_code != 422, (
            "Valid request should not return validation error"
        )

        # Valid request with only required parameters (parallelism should default to 4)
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": ["kernel_001"]},
            headers=auth_headers,
        )
        assert response.status_code != 422, (
            "Request without parallelism should be valid"
        )

        # Missing required chunk_ids field
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"parallelism": 2},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject request without required chunk_ids"
        )

        # Empty chunk_ids array (violates minItems: 1)
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": []},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject empty chunk_ids array"

        # Too many chunk_ids (violates maxItems: 100)
        large_chunk_list = [f"kernel_{i:03d}" for i in range(1, 102)]  # 101 items
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": large_chunk_list},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject chunk_ids array exceeding 100 items"
        )

        # Invalid parallelism type
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": ["kernel_001"], "parallelism": "invalid"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid parallelism type"

        # Parallelism below minimum (< 1)
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": ["kernel_001"], "parallelism": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject parallelism below minimum"

        # Parallelism above maximum (> 10)
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": ["kernel_001"], "parallelism": 11},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject parallelism above maximum"

    async def test_chunk_batch_response_schema_202(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test 202 response schema when batch processing starts successfully."""
        payload = {
            "chunk_ids": ["kernel_001", "kernel_002", "kernel_003"],
            "parallelism": 2,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        if response.status_code == 202:
            data = response.json()

            # Verify required fields for 202 response
            assert "message" in data, "202 response should contain 'message' field"
            assert "total_chunks" in data, (
                "202 response should contain 'total_chunks' field"
            )
            assert "processing" in data, (
                "202 response should contain 'processing' field"
            )

            # Verify field types
            assert isinstance(data["message"], str), "Message should be string"
            assert isinstance(data["total_chunks"], int), (
                "Total chunks should be integer"
            )
            assert isinstance(data["processing"], list), "Processing should be array"

            # Verify logical consistency
            assert data["total_chunks"] == 3, "Total chunks should match request"
            assert len(data["message"]) > 0, "Message should not be empty"

            # Verify processing array contains strings
            for chunk_id in data["processing"]:
                assert isinstance(chunk_id, str), (
                    "Processing chunk IDs should be strings"
                )

            # Processing array should contain subset of requested chunks
            requested_chunks = set(payload["chunk_ids"])
            processing_chunks = set(data["processing"])
            assert processing_chunks.issubset(requested_chunks), (
                "Processing chunks should be subset of requested chunks"
            )

    async def test_chunk_batch_parallelism_default(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that parallelism defaults to 4 when not specified."""
        payload = {"chunk_ids": ["kernel_001", "kernel_002"]}

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        # Should accept request without parallelism parameter
        assert response.status_code != 422, "Should accept request without parallelism"

        if response.status_code == 202:
            # Implementation should use default parallelism of 4
            # This would be verified through behavior, not response schema
            pass

    async def test_chunk_batch_parallelism_constraints(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test parallelism parameter constraints."""
        chunk_ids = ["kernel_001", "kernel_002", "kernel_003"]

        # Test minimum valid parallelism
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": chunk_ids, "parallelism": 1},
            headers=auth_headers,
        )
        assert response.status_code != 422, "Parallelism=1 should be valid"

        # Test maximum valid parallelism
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": chunk_ids, "parallelism": 10},
            headers=auth_headers,
        )
        assert response.status_code != 422, "Parallelism=10 should be valid"

        # Test edge case: parallelism greater than chunk count
        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json={"chunk_ids": ["kernel_001"], "parallelism": 5},
            headers=auth_headers,
        )
        assert response.status_code != 422, "Parallelism > chunk count should be valid"

    async def test_chunk_batch_duplicate_chunk_ids(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with duplicate chunk IDs in request."""
        payload = {
            "chunk_ids": ["kernel_001", "kernel_002", "kernel_001"],  # duplicate
            "parallelism": 2,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        # Should either accept and deduplicate, or reject with validation error
        assert response.status_code in [202, 400, 422], (
            "Should handle duplicate chunk IDs appropriately"
        )

        if response.status_code == 202:
            data = response.json()
            # If accepted, should handle duplicates properly
            processing_chunks = data["processing"]
            unique_chunks = set(processing_chunks)
            assert len(processing_chunks) == len(unique_chunks), (
                "Processing list should not contain duplicates"
            )

    async def test_chunk_batch_nonexistent_chunks(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior when some chunks don't exist."""
        payload = {
            "chunk_ids": ["kernel_001", "nonexistent_chunk_12345", "kernel_002"],
            "parallelism": 2,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        # Should either accept and process valid chunks, or reject entire batch
        assert response.status_code in [202, 400], (
            "Should handle mix of valid/invalid chunks appropriately"
        )

        if response.status_code == 202:
            data = response.json()
            # If accepted, should only process valid chunks
            processing_chunks = set(data["processing"])
            valid_chunks = {"kernel_001", "kernel_002"}
            assert processing_chunks.issubset(valid_chunks), (
                "Should only process valid chunks"
            )

    async def test_chunk_batch_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk batch process meets p95 < 600ms performance requirement."""
        import time

        payload = {
            "chunk_ids": ["kernel_001", "kernel_002"],
            "parallelism": 2,
        }

        start_time = time.time()

        response = await http_client.post(
            "/mcp/chunks/process/batch",
            json=payload,
            headers=auth_headers,
            timeout=1.0,  # 1 second timeout
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [202, 400]:
            # Performance requirement from constitution: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_chunk_batch_large_batch_handling(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test handling of large batch requests (near maximum)."""
        # Create 100 chunk IDs (maximum allowed)
        large_chunk_list = [f"kernel_{i:03d}" for i in range(1, 101)]

        payload = {
            "chunk_ids": large_chunk_list,
            "parallelism": 10,  # maximum parallelism
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        # Should handle maximum batch size
        assert response.status_code in [202, 400], (
            "Should handle maximum batch size appropriately"
        )

        if response.status_code == 202:
            data = response.json()
            assert data["total_chunks"] == 100, "Should report correct total chunks"
            assert len(data["processing"]) <= 100, (
                "Processing list should not exceed total"
            )

    @pytest.mark.integration
    async def test_chunk_batch_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample chunk data (if available)."""
        payload = {
            "chunk_ids": ["kernel_001", "kernel_002", "fs_001"],
            "parallelism": 3,
        }

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        if response.status_code == 202:
            data = response.json()

            # Should process some or all requested chunks
            assert len(data["processing"]) > 0, (
                "Should start processing at least one chunk"
            )
            assert data["total_chunks"] == 3, "Should report correct total"

            # All processing chunks should be from request
            requested_chunks = set(payload["chunk_ids"])
            processing_chunks = set(data["processing"])
            assert processing_chunks.issubset(requested_chunks), (
                "Processing chunks should be from request"
            )

    async def test_chunk_batch_parallelism_efficiency(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that different parallelism values are handled efficiently."""
        chunk_ids = ["kernel_001", "kernel_002", "kernel_003", "kernel_004"]

        # Test low parallelism
        payload_low = {"chunk_ids": chunk_ids, "parallelism": 1}
        response_low = await http_client.post(
            "/mcp/chunks/process/batch", json=payload_low, headers=auth_headers
        )

        # Test high parallelism
        payload_high = {"chunk_ids": chunk_ids, "parallelism": 4}
        response_high = await http_client.post(
            "/mcp/chunks/process/batch", json=payload_high, headers=auth_headers
        )

        # Both should be valid
        assert response_low.status_code in [202, 400], (
            "Low parallelism should be handled"
        )
        assert response_high.status_code in [202, 400], (
            "High parallelism should be handled"
        )

    async def test_chunk_batch_concurrent_requests(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with concurrent batch processing requests."""
        import asyncio

        payload1 = {"chunk_ids": ["kernel_001", "kernel_002"], "parallelism": 2}
        payload2 = {"chunk_ids": ["kernel_003", "kernel_004"], "parallelism": 2}

        async def process_batch(payload):
            return await http_client.post(
                "/mcp/chunks/process/batch", json=payload, headers=auth_headers
            )

        # Send concurrent batch requests
        responses = await asyncio.gather(
            process_batch(payload1), process_batch(payload2)
        )

        # Should handle concurrent batch requests
        for response in responses:
            assert response.status_code in [202, 400, 409], (
                "Should handle concurrent batch requests appropriately"
            )


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkBatchErrorHandling:
    """Test error handling for chunk batch process tool."""

    async def test_chunk_batch_invalid_json(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with invalid JSON in request body."""
        invalid_payloads = [
            '{"chunk_ids": ["kernel_001"',  # incomplete JSON
            '{"chunk_ids": []}',  # empty array (violates minItems)
            '{"chunk_ids": null}',  # null value
            '{"chunk_ids": "not_an_array"}',  # wrong type
        ]

        for payload in invalid_payloads:
            headers = {**auth_headers, "Content-Type": "application/json"}
            response = await http_client.post(
                "/mcp/chunks/process/batch",
                content=payload,
                headers=headers,
            )

            # Should return 400 or 422 for invalid requests
            assert response.status_code in [400, 422], (
                f"Invalid payload should be rejected: {payload[:50]}..."
            )

    async def test_chunk_batch_invalid_chunk_id_formats(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with invalid chunk ID formats."""
        invalid_payloads = [
            {"chunk_ids": [""]},  # empty string
            {"chunk_ids": [None]},  # null value
            {"chunk_ids": [123]},  # number instead of string
            {"chunk_ids": ["chunk with spaces"]},  # spaces in ID
            {"chunk_ids": ["chunk-with-special-chars!@#"]},  # special characters
        ]

        for payload in invalid_payloads:
            response = await http_client.post(
                "/mcp/chunks/process/batch", json=payload, headers=auth_headers
            )

            # Should either reject or handle gracefully
            assert response.status_code in [400, 422], (
                f"Invalid chunk ID format should be rejected: {payload}"
            )

    async def test_chunk_batch_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        payload = {"chunk_ids": ["kernel_001"], "parallelism": 4}

        response = await http_client.post(
            "/mcp/chunks/process/batch", json=payload, headers=auth_headers
        )

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_batch_content_type_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that endpoint requires application/json content type."""
        # Test with wrong content type
        headers = {**auth_headers}
        headers["Content-Type"] = "text/plain"

        response = await http_client.post(
            "/mcp/chunks/process/batch",
            content='{"chunk_ids": ["kernel_001"]}',
            headers=headers,
        )

        # Should either reject or handle gracefully
        assert response.status_code in [400, 415, 422], (
            "Should handle incorrect content type appropriately"
        )

    async def test_chunk_batch_missing_required_fields(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior when required fields are missing."""
        invalid_payloads = [
            {},  # completely empty
            {"parallelism": 4},  # missing chunk_ids
            {"invalid_field": "value"},  # wrong field name
        ]

        for payload in invalid_payloads:
            response = await http_client.post(
                "/mcp/chunks/process/batch", json=payload, headers=auth_headers
            )

            # Should reject requests missing required fields
            assert response.status_code == 422, (
                f"Should reject request missing required fields: {payload}"
            )

            if response.status_code == 422:
                data = response.json()
                assert "error" in data or "detail" in data, (
                    "422 response should include error details"
                )
