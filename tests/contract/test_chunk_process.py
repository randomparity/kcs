"""
Contract tests for chunk process MCP tool.

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
class TestChunkProcessContract:
    """Contract tests for chunk process MCP tool."""

    async def test_chunk_process_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the chunk process endpoint exists and accepts POST requests."""
        payload = {"force": False}

        response = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "chunk process endpoint should exist"

    async def test_chunk_process_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that chunk process requires valid authentication."""
        payload = {"force": False}

        # Request without auth headers
        response = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload
        )
        assert response.status_code == 401, "Should require authentication"

    async def test_chunk_process_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk process validates request schema according to OpenAPI spec."""

        # Valid request with force parameter
        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json={"force": True},
            headers=auth_headers,
        )
        assert response.status_code != 422, (
            "Valid request should not return validation error"
        )

        # Valid request without force parameter (should default to false)
        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json={},
            headers=auth_headers,
        )
        assert response.status_code != 422, "Request without force should be valid"

        # Invalid force type
        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json={"force": "invalid"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid force type"

        # Extra invalid fields
        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json={"force": True, "invalid_field": "value"},
            headers=auth_headers,
        )
        # Should either ignore or reject extra fields
        assert response.status_code in [202, 400, 422], (
            "Should handle extra fields appropriately"
        )

    async def test_chunk_process_path_parameter_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk ID path parameter is properly validated."""
        payload = {"force": False}

        # Valid chunk ID format
        response = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload, headers=auth_headers
        )
        # Should not fail due to path parameter format
        assert response.status_code != 422, "Valid chunk ID should be accepted"

        # Test with different valid chunk ID patterns
        valid_chunk_ids = ["kernel_001", "fs_042", "mm_999", "drivers_123"]
        for chunk_id in valid_chunk_ids:
            response = await http_client.post(
                f"/mcp/chunks/{chunk_id}/process", json=payload, headers=auth_headers
            )
            # Should not fail due to format (may return other codes based on chunk state)
            assert response.status_code not in [422, 404], (
                f"Valid chunk ID '{chunk_id}' should be accepted"
            )

    async def test_chunk_process_response_schema_202(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test 202 response schema when processing starts successfully."""
        payload = {"force": False}

        response = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload, headers=auth_headers
        )

        if response.status_code == 202:
            data = response.json()

            # Verify required fields for 202 response
            assert "message" in data, "202 response should contain 'message' field"
            assert "chunk_id" in data, "202 response should contain 'chunk_id' field"
            assert "status" in data, "202 response should contain 'status' field"

            # Verify field types
            assert isinstance(data["message"], str), "Message should be string"
            assert isinstance(data["chunk_id"], str), "Chunk ID should be string"
            assert isinstance(data["status"], str), "Status should be string"

            # Verify logical consistency
            assert data["chunk_id"] == "kernel_001", "Chunk ID should match request"
            assert len(data["message"]) > 0, "Message should not be empty"

    async def test_chunk_process_force_parameter_behavior(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test force parameter behavior."""
        # Test with force=false (default behavior)
        payload_no_force = {"force": False}
        response_no_force = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json=payload_no_force,
            headers=auth_headers,
        )

        # Test with force=true (should force reprocessing)
        payload_force = {"force": True}
        response_force = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload_force, headers=auth_headers
        )

        # Both should be valid requests (though behavior may differ)
        assert response_no_force.status_code in [202, 400, 409], (
            "Request with force=false should be handled"
        )
        assert response_force.status_code in [202, 400, 409], (
            "Request with force=true should be handled"
        )

        # If chunk is already completed, force=false might return 409, force=true should allow reprocessing
        if response_no_force.status_code == 409:
            # force=true should potentially allow reprocessing
            assert response_force.status_code in [202, 400], (
                "force=true should allow reprocessing of completed chunk"
            )

    async def test_chunk_process_default_force_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that force parameter defaults to false when not specified."""
        # Request without force parameter
        response_no_param = await http_client.post(
            "/mcp/chunks/kernel_001/process", json={}, headers=auth_headers
        )

        # Request with explicit force=false
        response_explicit_false = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json={"force": False},
            headers=auth_headers,
        )

        # Both should behave identically (force defaults to false)
        assert response_no_param.status_code == response_explicit_false.status_code, (
            "Missing force parameter should default to false"
        )

    async def test_chunk_process_already_processing_409(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test 409 response when chunk is already being processed."""
        payload = {"force": False}

        response = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload, headers=auth_headers
        )

        if response.status_code == 409:
            data = response.json()

            # Should follow error schema
            assert "error" in data, "409 response should have 'error' field"
            assert "message" in data, "409 response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_process_bad_request_400(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test 400 response for invalid requests."""
        # Test with completely invalid JSON
        headers = {**auth_headers, "Content-Type": "application/json"}
        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            content="invalid json{",
            headers=headers,
        )

        if response.status_code == 400:
            data = response.json()

            # Should follow error schema
            assert "error" in data, "400 response should have 'error' field"
            assert "message" in data, "400 response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_process_nonexistent_chunk(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior when chunk doesn't exist."""
        payload = {"force": False}

        response = await http_client.post(
            "/mcp/chunks/nonexistent_chunk_12345/process",
            json=payload,
            headers=auth_headers,
        )

        # Should return 400 or 404 for non-existent chunks
        assert response.status_code in [400, 404], (
            "Should handle non-existent chunks appropriately"
        )

        if response.status_code in [400, 404]:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"

    async def test_chunk_process_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that chunk process meets p95 < 600ms performance requirement."""
        import time

        payload = {"force": False}

        start_time = time.time()

        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json=payload,
            headers=auth_headers,
            timeout=1.0,  # 1 second timeout
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code in [202, 400, 409]:
            # Performance requirement from constitution: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_chunk_process_content_type_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that endpoint requires application/json content type."""
        # Test with wrong content type
        headers = {**auth_headers}
        headers["Content-Type"] = "text/plain"

        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            content='{"force": false}',
            headers=headers,
        )

        # Should either reject or handle gracefully
        assert response.status_code in [400, 415, 422], (
            "Should handle incorrect content type appropriately"
        )

    @pytest.mark.integration
    async def test_chunk_process_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample chunk data (if available)."""
        payload = {"force": True}  # Use force to ensure processing can start

        # Try several common chunk IDs
        chunk_ids = ["kernel_001", "kernel_002", "fs_001"]

        for chunk_id in chunk_ids:
            response = await http_client.post(
                f"/mcp/chunks/{chunk_id}/process", json=payload, headers=auth_headers
            )

            if response.status_code == 202:
                data = response.json()

                # Verify response matches request
                assert data["chunk_id"] == chunk_id, (
                    f"Response chunk_id should match requested ID '{chunk_id}'"
                )

                # Processing should have started
                assert "status" in data, "Should include status in response"
                break  # Found at least one chunk to test with

    async def test_chunk_process_concurrent_requests(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with concurrent processing requests for same chunk."""
        import asyncio

        payload = {"force": False}

        async def process_chunk():
            return await http_client.post(
                "/mcp/chunks/kernel_001/process", json=payload, headers=auth_headers
            )

        # Send 3 concurrent requests for same chunk
        responses = await asyncio.gather(*[process_chunk() for _ in range(3)])

        # Should handle concurrent requests gracefully
        success_count = sum(1 for r in responses if r.status_code == 202)
        conflict_count = sum(1 for r in responses if r.status_code == 409)

        # Either one succeeds and others conflict, or all handle gracefully
        assert success_count <= 1, "At most one concurrent request should succeed"
        assert success_count + conflict_count >= 1, (
            "Should have at least one success or conflict response"
        )

    async def test_chunk_process_idempotency_with_force(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test idempotency behavior with force parameter."""
        payload_force = {"force": True}

        # Make two identical requests with force=true
        response1 = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload_force, headers=auth_headers
        )

        # Wait a brief moment to avoid race conditions
        import asyncio

        await asyncio.sleep(0.1)

        response2 = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload_force, headers=auth_headers
        )

        # Both force requests should be accepted or handled consistently
        if response1.status_code == 202:
            # Second request should either succeed (if first completed) or conflict
            assert response2.status_code in [202, 409], (
                "Second force request should be handled consistently"
            )


@skip_integration_in_ci
@skip_without_mcp_server
class TestChunkProcessErrorHandling:
    """Test error handling for chunk process tool."""

    async def test_chunk_process_invalid_chunk_id_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with invalid chunk ID formats."""
        payload = {"force": False}

        invalid_chunk_ids = [
            "chunk with spaces",
            "chunk-with-special-chars!@#",
            "",  # empty string would be caught by URL routing
            "x" * 1000,  # very long ID
        ]

        for chunk_id in invalid_chunk_ids:
            response = await http_client.post(
                f"/mcp/chunks/{chunk_id}/process", json=payload, headers=auth_headers
            )

            # Should either reject (400/404/422) or handle gracefully
            assert response.status_code in [400, 404, 422], (
                f"Should handle invalid chunk ID '{chunk_id[:50]}...' gracefully"
            )

    async def test_chunk_process_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        payload = {"force": False}

        response = await http_client.post(
            "/mcp/chunks/kernel_001/process", json=payload, headers=auth_headers
        )

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
            assert isinstance(data["error"], str), "Error should be string"
            assert isinstance(data["message"], str), "Message should be string"

    async def test_chunk_process_malformed_request_body(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with malformed request bodies."""
        malformed_payloads = [
            '{"force": ',  # incomplete JSON
            '{"force": null}',  # null value
            '{"force": []}',  # array instead of boolean
            '{"force": {}}',  # object instead of boolean
        ]

        for payload in malformed_payloads:
            headers = {**auth_headers, "Content-Type": "application/json"}
            response = await http_client.post(
                "/mcp/chunks/kernel_001/process",
                content=payload,
                headers=headers,
            )

            # Should return 400 or 422 for malformed requests
            assert response.status_code in [400, 422], (
                f"Malformed payload should be rejected: {payload[:50]}..."
            )

    async def test_chunk_process_missing_content_type(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior when Content-Type header is missing."""
        headers = {k: v for k, v in auth_headers.items() if k != "Content-Type"}

        response = await http_client.post(
            "/mcp/chunks/kernel_001/process",
            json={"force": False},
            headers=headers,
        )

        # Should either handle gracefully or require explicit content type
        assert response.status_code in [202, 400, 415, 422], (
            "Should handle missing Content-Type appropriately"
        )
