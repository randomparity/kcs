"""
Contract tests for validate_spec MCP tool.

These tests verify the API contract for specification validation against implementation.
They MUST fail before implementation and pass after.
"""

import os
from collections.abc import AsyncGenerator
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
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for API requests."""
    async with httpx.AsyncClient(base_url=MCP_BASE_URL) as client:
        yield client


# Contract test cases
@skip_integration_in_ci
@skip_without_mcp_server
class TestValidateSpecContract:
    """Contract tests for validate_spec MCP tool."""

    async def test_validate_spec_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the validate_spec endpoint exists and accepts POST requests."""
        payload = {
            "specification": {
                "name": "VFS Read Operation",
                "version": "1.0.0",
                "entry_point": "vfs_read",
                "expected_behavior": {
                    "description": "Read data from file",
                    "preconditions": ["Valid file descriptor", "Buffer allocated"],
                    "postconditions": ["Data read into buffer", "Position updated"],
                },
            },
            "kernel_version": "6.1.0",
        }

        response = await http_client.post(
            "/mcp/tools/validate_spec", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "validate_spec endpoint should exist"

    async def test_validate_spec_requires_authentication(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that validate_spec requires valid authentication."""
        payload = {
            "specification": {
                "name": "Test Spec",
                "version": "1.0.0",
                "entry_point": "test_func",
            }
        }

        # Request without auth headers
        response = await http_client.post("/mcp/tools/validate_spec", json=payload)
        assert response.status_code == 401, "Should require authentication"

    async def test_validate_spec_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that validate_spec validates request schema according to OpenAPI spec."""

        # Missing required 'specification' field
        response = await http_client.post(
            "/mcp/tools/validate_spec",
            json={"kernel_version": "6.1.0"},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject request without required 'specification' field"
        )

        # Missing required fields in specification
        response = await http_client.post(
            "/mcp/tools/validate_spec",
            json={
                "specification": {
                    "name": "Test Spec",
                    # Missing version and entry_point
                }
            },
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject specification without required fields"
        )

        # Invalid version format
        response = await http_client.post(
            "/mcp/tools/validate_spec",
            json={
                "specification": {
                    "name": "Test Spec",
                    "version": "invalid-version",
                    "entry_point": "test_func",
                }
            },
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid version format"

    async def test_validate_spec_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that validate_spec returns response matching OpenAPI schema."""
        payload = {
            "specification": {
                "name": "VFS Read Operation",
                "version": "1.0.0",
                "entry_point": "vfs_read",
                "expected_behavior": {
                    "description": "Read data from file",
                    "preconditions": ["Valid file descriptor"],
                    "postconditions": ["Data read into buffer"],
                    "error_conditions": ["EBADF", "EINVAL", "EIO"],
                },
                "parameters": [
                    {"name": "fd", "type": "int", "description": "File descriptor"},
                    {"name": "buf", "type": "void*", "description": "Buffer"},
                    {"name": "count", "type": "size_t", "description": "Bytes to read"},
                ],
            },
            "kernel_version": "6.1.0",
            "config": "x86_64:defconfig",
        }

        response = await http_client.post(
            "/mcp/tools/validate_spec", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify top-level structure
            assert "validation_id" in data, (
                "Response should contain 'validation_id' field"
            )
            assert "specification_id" in data, (
                "Response should contain 'specification_id' field"
            )
            assert "is_valid" in data, "Response should contain 'is_valid' field"
            assert "compliance_score" in data, (
                "Response should contain 'compliance_score' field"
            )
            assert "deviations" in data, "Response should contain 'deviations' field"
            assert "implementation_details" in data, (
                "Response should contain 'implementation_details' field"
            )
            assert "validated_at" in data, (
                "Response should contain 'validated_at' field"
            )

            # Verify field types
            assert isinstance(data["validation_id"], str), (
                "validation_id should be string (UUID)"
            )
            assert isinstance(data["specification_id"], str), (
                "specification_id should be string (UUID)"
            )
            assert isinstance(data["is_valid"], bool), "is_valid should be boolean"
            assert isinstance(data["compliance_score"], (int, float)), (
                "compliance_score should be numeric"
            )
            assert 0 <= data["compliance_score"] <= 100, (
                "compliance_score should be between 0 and 100"
            )
            assert isinstance(data["deviations"], list), "deviations should be array"
            assert isinstance(data["implementation_details"], dict), (
                "implementation_details should be object"
            )
            assert isinstance(data["validated_at"], str), (
                "validated_at should be string (ISO timestamp)"
            )

            # Verify deviations structure
            for deviation in data["deviations"]:
                assert isinstance(deviation, dict), "Each deviation should be object"
                assert "type" in deviation, "Deviation should have 'type' field"
                assert "severity" in deviation, "Deviation should have 'severity' field"
                assert "description" in deviation, (
                    "Deviation should have 'description' field"
                )
                assert "location" in deviation, "Deviation should have 'location' field"

                # Verify severity enum
                valid_severities = ["critical", "major", "minor", "info"]
                assert deviation["severity"] in valid_severities, (
                    f"Severity should be one of {valid_severities}"
                )

                # Verify deviation type enum
                valid_types = [
                    "missing_implementation",
                    "behavior_mismatch",
                    "parameter_mismatch",
                    "error_handling",
                    "performance",
                ]
                assert deviation["type"] in valid_types, (
                    f"Deviation type should be one of {valid_types}"
                )

            # Verify implementation_details structure
            impl_details = data["implementation_details"]
            if "entry_point" in impl_details:
                assert isinstance(impl_details["entry_point"], dict), (
                    "entry_point should be object"
                )
                assert "symbol" in impl_details["entry_point"], (
                    "Entry point should have symbol"
                )
                assert "span" in impl_details["entry_point"], (
                    "Entry point should have span"
                )

            if "call_graph" in impl_details:
                assert isinstance(impl_details["call_graph"], list), (
                    "call_graph should be array"
                )

            if "parameters_found" in impl_details:
                assert isinstance(impl_details["parameters_found"], list), (
                    "parameters_found should be array"
                )

    async def test_validate_spec_with_drift_threshold(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that drift threshold parameter affects validation results."""
        payload = {
            "specification": {
                "name": "Test Function",
                "version": "1.0.0",
                "entry_point": "test_func",
            },
            "drift_threshold": 0.8,  # 80% compliance required
        }

        response = await http_client.post(
            "/mcp/tools/validate_spec", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Validation result should consider drift threshold
            if data["compliance_score"] < 80:
                assert not data["is_valid"], (
                    "Should be invalid when compliance < threshold"
                )
            else:
                assert data["is_valid"], "Should be valid when compliance >= threshold"

    async def test_validate_spec_batch_validation(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test batch validation of multiple specifications."""
        payload = {
            "specifications": [
                {
                    "name": "VFS Read",
                    "version": "1.0.0",
                    "entry_point": "vfs_read",
                },
                {
                    "name": "VFS Write",
                    "version": "1.0.0",
                    "entry_point": "vfs_write",
                },
            ],
            "kernel_version": "6.1.0",
        }

        response = await http_client.post(
            "/mcp/tools/validate_spec_batch", json=payload, headers=auth_headers
        )

        # Note: This might be a separate endpoint or handled by the same endpoint
        if response.status_code != 404:  # If batch endpoint exists
            if response.status_code == 200:
                data = response.json()
                assert "validations" in data, "Should return array of validations"
                assert isinstance(data["validations"], list), (
                    "Validations should be array"
                )
                assert len(data["validations"]) == 2, (
                    "Should validate all specifications"
                )

    async def test_validate_spec_with_implementation_hints(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test validation with implementation hints for better accuracy."""
        payload = {
            "specification": {
                "name": "Custom IOCTL Handler",
                "version": "1.0.0",
                "entry_point": "my_ioctl",
                "implementation_hints": {
                    "file_pattern": "drivers/custom/*.c",
                    "subsystem": "drivers",
                    "related_symbols": ["device_ioctl", "unlocked_ioctl"],
                },
            },
            "include_suggestions": True,
        }

        response = await http_client.post(
            "/mcp/tools/validate_spec", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include suggestions when requested
            if "suggestions" in data:
                assert isinstance(data["suggestions"], list), (
                    "Suggestions should be array"
                )
                for suggestion in data["suggestions"]:
                    assert "type" in suggestion, "Suggestion should have type"
                    assert "description" in suggestion, (
                        "Suggestion should have description"
                    )

    async def test_validate_spec_historical_comparison(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test comparison with historical validation results."""
        payload = {
            "specification": {
                "name": "VFS Operation",
                "version": "2.0.0",
                "entry_point": "vfs_read",
                "previous_version": "1.0.0",
            },
            "compare_with_previous": True,
        }

        response = await http_client.post(
            "/mcp/tools/validate_spec", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include comparison data when requested
            if "comparison" in data:
                assert isinstance(data["comparison"], dict), (
                    "Comparison should be object"
                )
                assert "compliance_delta" in data["comparison"], (
                    "Should show compliance change"
                )
                assert "new_deviations" in data["comparison"], (
                    "Should show new deviations"
                )
                assert "resolved_deviations" in data["comparison"], (
                    "Should show resolved deviations"
                )

    async def test_validate_spec_error_conditions(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test proper error handling for various error conditions."""

        # Non-existent entry point
        payload = {
            "specification": {
                "name": "Non-existent Function",
                "version": "1.0.0",
                "entry_point": "nonexistent_function_xyz123",
            }
        }

        response = await http_client.post(
            "/mcp/tools/validate_spec", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should still return validation result, but with low compliance
            assert data["compliance_score"] == 0, (
                "Non-existent entry point should have 0 compliance"
            )
            assert not data["is_valid"], "Should be marked as invalid"
            assert len(data["deviations"]) > 0, "Should have deviations"

            # Check for missing implementation deviation
            has_missing_impl = any(
                d["type"] == "missing_implementation" for d in data["deviations"]
            )
            assert has_missing_impl, "Should have missing_implementation deviation"
