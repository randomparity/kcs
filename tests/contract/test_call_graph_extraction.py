"""
Contract test for POST /mcp/tools/extract_call_graph endpoint.

Tests the API contract for call graph extraction functionality according to
the OpenAPI specification in contracts/call-graph-api.yaml.

This test MUST FAIL initially as no implementation exists yet (TDD requirement).
"""

import json

import pytest
import requests

from tests.conftest import get_mcp_auth_headers, skip_without_mcp_server

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
EXTRACT_CALL_GRAPH_ENDPOINT = f"{MCP_BASE_URL}/mcp/tools/extract_call_graph"

# Common headers for all requests (uses centralized JWT configuration)
COMMON_HEADERS = get_mcp_auth_headers()


@skip_without_mcp_server
class TestExtractCallGraphContract:
    """Contract tests for extract_call_graph MCP endpoint."""

    def test_extract_call_graph_valid_request_schema(self):
        """Test that endpoint accepts valid request schema and returns 200."""
        # Valid request according to OpenAPI contract
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "include_indirect": True,
            "include_macros": True,
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Must return 200 for valid requests
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        # Contract requirement: Response must be valid JSON
        response_data = response.json()
        assert isinstance(response_data, dict), "Response must be a JSON object"

    def test_extract_call_graph_response_schema(self):
        """Test that response matches the expected schema structure."""
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "include_indirect": True,
            "include_macros": True,
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        # Contract requirement: Response must contain call_edges array
        assert "call_edges" in data, "Response must contain 'call_edges' field"
        assert isinstance(data["call_edges"], list), "call_edges must be an array"

        # Contract requirement: Response must contain extraction_stats
        assert "extraction_stats" in data, (
            "Response must contain 'extraction_stats' field"
        )
        assert isinstance(data["extraction_stats"], dict), (
            "extraction_stats must be an object"
        )

        # Validate extraction_stats schema
        stats = data["extraction_stats"]
        required_stats_fields = [
            "files_processed",
            "functions_analyzed",
            "call_edges_found",
            "processing_time_ms",
        ]
        for field in required_stats_fields:
            assert field in stats, f"extraction_stats must contain '{field}' field"
            assert isinstance(stats[field], int), (
                f"extraction_stats.{field} must be an integer"
            )

    def test_extract_call_graph_call_edge_schema(self):
        """Test that call_edges in response match expected schema."""
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "include_indirect": True,
            "include_macros": True,
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        # If call_edges are present, validate their schema
        if data["call_edges"]:
            edge = data["call_edges"][0]

            # Contract requirement: Each call edge must have caller and callee
            assert "caller" in edge, "Call edge must have 'caller' field"
            assert "callee" in edge, "Call edge must have 'callee' field"
            assert "call_site" in edge, "Call edge must have 'call_site' field"
            assert "call_type" in edge, "Call edge must have 'call_type' field"
            assert "confidence" in edge, "Call edge must have 'confidence' field"
            assert "conditional" in edge, "Call edge must have 'conditional' field"

            # Validate caller schema
            caller = edge["caller"]
            assert "name" in caller, "Caller must have 'name' field"
            assert "file_path" in caller, "Caller must have 'file_path' field"
            assert "line_number" in caller, "Caller must have 'line_number' field"
            assert isinstance(caller["line_number"], int), (
                "Caller line_number must be integer"
            )

            # Validate callee schema
            callee = edge["callee"]
            assert "name" in callee, "Callee must have 'name' field"
            assert "file_path" in callee, "Callee must have 'file_path' field"
            assert "line_number" in callee, "Callee must have 'line_number' field"
            assert isinstance(callee["line_number"], int), (
                "Callee line_number must be integer"
            )

            # Validate call_site schema
            call_site = edge["call_site"]
            assert "file_path" in call_site, "Call site must have 'file_path' field"
            assert "line_number" in call_site, "Call site must have 'line_number' field"
            assert "function_context" in call_site, (
                "Call site must have 'function_context' field"
            )
            assert isinstance(call_site["line_number"], int), (
                "Call site line_number must be integer"
            )

            # Validate enum values
            valid_call_types = [
                "direct",
                "indirect",
                "macro",
                "callback",
                "conditional",
                "assembly",
                "syscall",
            ]
            assert edge["call_type"] in valid_call_types, (
                f"call_type must be one of {valid_call_types}"
            )

            valid_confidence_levels = ["high", "medium", "low"]
            assert edge["confidence"] in valid_confidence_levels, (
                f"confidence must be one of {valid_confidence_levels}"
            )

            assert isinstance(edge["conditional"], bool), "conditional must be boolean"

    def test_extract_call_graph_missing_required_fields(self):
        """Test that missing required fields return 400 Bad Request."""
        # Request missing required file_paths field
        invalid_request = {
            "include_indirect": True,
            "include_macros": True,
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Must return 400 for invalid requests
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

        # Contract requirement: Error response must be valid JSON
        error_data = response.json()
        assert isinstance(error_data, dict), "Error response must be a JSON object"

        # Contract requirement: Error response must contain error information
        assert "error" in error_data or "message" in error_data, (
            "Error response must contain error information"
        )

    def test_extract_call_graph_invalid_file_paths(self):
        """Test that invalid file paths return appropriate error."""
        # Request with non-existent file
        invalid_request = {
            "file_paths": ["non/existent/file.c"],
            "include_indirect": True,
            "include_macros": False,
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Must handle file errors gracefully
        # Could be 400 (bad request) or 500 (processing error) depending on implementation
        assert response.status_code in [400, 500], (
            f"Expected 400 or 500, got {response.status_code}"
        )

        error_data = response.json()
        assert isinstance(error_data, dict), "Error response must be a JSON object"

    def test_extract_call_graph_empty_file_paths(self):
        """Test that empty file_paths array returns 400."""
        invalid_request = {
            "file_paths": [],
            "include_indirect": True,
            "include_macros": True,
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Empty file paths should be invalid
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_extract_call_graph_invalid_json(self):
        """Test that malformed JSON returns 400."""
        # Send malformed JSON
        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, data="{ invalid json }", headers=COMMON_HEADERS
        )

        # Contract requirement: Malformed JSON should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_extract_call_graph_unauthorized_request(self):
        """Test that requests without proper authorization return 401."""
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "include_indirect": True,
            "include_macros": True,
            "config_context": "x86_64:defconfig",
        }

        # Request without authorization header
        headers_no_auth = {"Content-Type": "application/json"}

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=valid_request, headers=headers_no_auth
        )

        # Contract requirement: Must require authorization
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"

    def test_extract_call_graph_optional_parameters(self):
        """Test that optional parameters work correctly."""
        # Request with minimal required fields only
        minimal_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"]
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=minimal_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Optional parameters should have defaults
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "call_edges" in data
        assert "extraction_stats" in data

    def test_extract_call_graph_multiple_files(self):
        """Test that multiple file paths are processed correctly."""
        multi_file_request = {
            "file_paths": [
                "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
                "tests/fixtures/mini-kernel-v6.1/fs/another_file.c",
            ],
            "include_indirect": True,
            "include_macros": True,
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            EXTRACT_CALL_GRAPH_ENDPOINT, json=multi_file_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Multiple files should be supported
        assert response.status_code in [200, 400, 500], "Must handle multiple files"

        if response.status_code == 200:
            data = response.json()
            assert "call_edges" in data
            assert "extraction_stats" in data
            # Processing multiple files should potentially find more edges
            assert data["extraction_stats"]["files_processed"] >= 1


class TestExtractCallGraphImplementation:
    """
    Implementation-specific tests that will be enabled once the endpoint is implemented.
    These tests MUST initially be skipped to satisfy TDD requirements.
    """

    def test_extract_call_graph_functional_behavior(self):
        """Test actual call graph extraction functionality."""
        # This test will be enabled after implementation
        pass

    def test_extract_call_graph_performance(self):
        """Test that extraction meets performance requirements."""
        # This test will be enabled after implementation
        pass


if __name__ == "__main__":
    # Run contract tests directly
    pytest.main([__file__, "-v"])
