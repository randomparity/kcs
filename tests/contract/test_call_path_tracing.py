"""
Contract test for POST /mcp/tools/trace_call_path endpoint.

Tests the API contract for call path tracing functionality according to
the OpenAPI specification in contracts/call-graph-api.yaml.

This test MUST FAIL initially as no implementation exists yet (TDD requirement).
"""

import json

import pytest
import requests

from tests.conftest import get_mcp_auth_headers, skip_without_mcp_server

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TRACE_CALL_PATH_ENDPOINT = f"{MCP_BASE_URL}/mcp/tools/trace_call_path"

# Common headers for all requests (uses centralized JWT configuration)
COMMON_HEADERS = get_mcp_auth_headers()


@skip_without_mcp_server
class TestTraceCallPathContract:
    """Contract tests for trace_call_path MCP endpoint."""

    def test_trace_call_path_valid_request_schema(self):
        """Test that endpoint accepts valid request schema and returns 200."""
        # Valid request according to OpenAPI contract
        valid_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Must return 200 for valid requests
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        # Contract requirement: Response must be valid JSON
        response_data = response.json()
        assert isinstance(response_data, dict), "Response must be a JSON object"

    def test_trace_call_path_response_schema(self):
        """Test that response matches the expected schema structure."""
        valid_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        # Contract requirement: Response must contain from_function and to_function
        assert "from_function" in data, "Response must contain 'from_function' field"
        assert "to_function" in data, "Response must contain 'to_function' field"
        assert isinstance(data["from_function"], str), "from_function must be a string"
        assert isinstance(data["to_function"], str), "to_function must be a string"
        assert data["from_function"] == "sys_open", "from_function must match request"
        assert data["to_function"] == "generic_file_open", (
            "to_function must match request"
        )

        # Contract requirement: Response must contain paths array
        assert "paths" in data, "Response must contain 'paths' field"
        assert isinstance(data["paths"], list), "paths must be an array"

    def test_trace_call_path_paths_schema(self):
        """Test that paths in response match expected schema."""
        valid_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        # If paths are present, validate their schema
        if data["paths"]:
            path = data["paths"][0]

            # Contract requirement: Each path must have required fields
            assert "path_edges" in path, "Path must have 'path_edges' field"
            assert "path_length" in path, "Path must have 'path_length' field"
            assert "total_confidence" in path, "Path must have 'total_confidence' field"
            assert "config_context" in path, "Path must have 'config_context' field"

            # Validate path_edges schema
            assert isinstance(path["path_edges"], list), "path_edges must be an array"
            assert len(path["path_edges"]) >= 1, (
                "path_edges must contain at least one edge"
            )

            # Validate individual edge schema
            if path["path_edges"]:
                edge = path["path_edges"][0]

                assert "caller" in edge, "Path edge must have 'caller' field"
                assert "callee" in edge, "Path edge must have 'callee' field"
                assert "call_type" in edge, "Path edge must have 'call_type' field"
                assert "confidence" in edge, "Path edge must have 'confidence' field"

                # Validate caller/callee schema
                for func_role in ["caller", "callee"]:
                    func = edge[func_role]
                    assert "name" in func, f"{func_role} must have 'name' field"
                    assert isinstance(func["name"], str), (
                        f"{func_role} name must be string"
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

            # Validate path metrics
            assert isinstance(path["path_length"], int), "path_length must be integer"
            assert path["path_length"] >= 1, "path_length must be at least 1"
            assert path["path_length"] <= 5, "path_length must not exceed max_depth"
            assert path["path_length"] == len(path["path_edges"]), (
                "path_length must match path_edges count"
            )

            assert isinstance(path["total_confidence"], (int, float)), (
                "total_confidence must be numeric"
            )
            assert 0.0 <= path["total_confidence"] <= 1.0, (
                "total_confidence must be between 0.0 and 1.0"
            )

            assert isinstance(path["config_context"], str), (
                "config_context must be string"
            )

    def test_trace_call_path_missing_required_fields(self):
        """Test that missing required fields return 400 Bad Request."""
        # Request missing required from_function field
        invalid_request = {
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
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

    def test_trace_call_path_same_function(self):
        """Test that tracing from a function to itself is handled correctly."""
        same_function_request = {
            "from_function": "generic_file_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 1,
            "max_depth": 1,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=same_function_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Same function should be handled (could be 200 with empty paths or 400)
        assert response.status_code in [200, 400], (
            f"Expected 200 or 400, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "paths" in data
            # Should either be empty paths or a single path with length 0
            if data["paths"]:
                assert data["paths"][0]["path_length"] == 0 or len(data["paths"]) == 0

    def test_trace_call_path_invalid_max_values(self):
        """Test that invalid max_paths and max_depth values are handled correctly."""
        # Test negative max_paths
        invalid_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": -1,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Invalid max_paths should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

        # Test zero max_paths
        invalid_request["max_paths"] = 0
        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

        # Test negative max_depth
        invalid_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": -1,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_trace_call_path_nonexistent_functions(self):
        """Test that nonexistent functions return appropriate response."""
        # Request with nonexistent from_function
        invalid_request = {
            "from_function": "nonexistent_function_12345",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Nonexistent function should return empty results (200) or 404
        assert response.status_code in [200, 404], (
            f"Expected 200 or 404, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "paths" in data
            assert data["paths"] == [], (
                "Should return empty paths for nonexistent function"
            )

        # Request with nonexistent to_function
        invalid_request = {
            "from_function": "sys_open",
            "to_function": "nonexistent_function_54321",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=invalid_request, headers=COMMON_HEADERS
        )

        assert response.status_code in [200, 404], (
            f"Expected 200 or 404, got {response.status_code}"
        )

    def test_trace_call_path_unauthorized_request(self):
        """Test that requests without proper authorization return 401."""
        valid_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        # Request without authorization header
        headers_no_auth = {"Content-Type": "application/json"}

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=valid_request, headers=headers_no_auth
        )

        # Contract requirement: Must require authorization
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"

    def test_trace_call_path_optional_parameters(self):
        """Test that optional parameters work correctly."""
        # Request with minimal required fields only
        minimal_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=minimal_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Optional parameters should have defaults
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "from_function" in data
        assert "to_function" in data
        assert "paths" in data

    def test_trace_call_path_max_paths_limit(self):
        """Test that max_paths parameter limits the number of returned paths."""
        limited_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 1,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=limited_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        # Contract requirement: Should not return more paths than max_paths
        assert len(data["paths"]) <= 1, "Should not return more than max_paths"

    def test_trace_call_path_config_context_filtering(self):
        """Test that config_context parameter affects path finding."""
        # Test with specific config context
        config_specific_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "arm64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT,
            json=config_specific_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Different config contexts should be supported
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "paths" in data

    def test_trace_call_path_malformed_json(self):
        """Test that malformed JSON returns 400."""
        # Send malformed JSON
        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, data="{ invalid json }", headers=COMMON_HEADERS
        )

        # Contract requirement: Malformed JSON should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_trace_call_path_large_limits(self):
        """Test that very large limit values are handled appropriately."""
        large_limits_request = {
            "from_function": "sys_open",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 1000,
            "max_depth": 100,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=large_limits_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Large limits should be handled (possibly with internal limits)
        assert response.status_code in [200, 400], (
            f"Expected 200 or 400, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "paths" in data
            # Should apply reasonable internal limits
            assert len(data["paths"]) <= 100, "Should apply reasonable internal limits"

    def test_trace_call_path_empty_function_names(self):
        """Test that empty function names return 400."""
        empty_name_request = {
            "from_function": "",
            "to_function": "generic_file_open",
            "config_context": "x86_64:defconfig",
            "max_paths": 3,
            "max_depth": 5,
        }

        response = requests.post(
            TRACE_CALL_PATH_ENDPOINT, json=empty_name_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Empty function names should be invalid
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"


class TestTraceCallPathImplementation:
    """
    Implementation-specific tests that will be enabled once the endpoint is implemented.
    These tests MUST initially be skipped to satisfy TDD requirements.
    """

    def test_trace_call_path_functional_behavior(self):
        """Test actual call path tracing functionality."""
        # This test will be enabled after implementation
        pass

    def test_trace_call_path_performance(self):
        """Test that path tracing meets performance requirements."""
        # This test will be enabled after implementation
        pass


if __name__ == "__main__":
    # Run contract tests directly
    pytest.main([__file__, "-v"])
