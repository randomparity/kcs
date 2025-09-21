"""
Contract test for POST /mcp/tools/analyze_function_pointers endpoint.

Tests the API contract for function pointer analysis functionality according to
the OpenAPI specification in contracts/call-graph-api.yaml.

This test MUST FAIL initially as no implementation exists yet (TDD requirement).
"""

import json

import pytest
import requests

from tests.conftest import get_mcp_auth_headers, skip_without_mcp_server

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
ANALYZE_FUNCTION_POINTERS_ENDPOINT = (
    f"{MCP_BASE_URL}/mcp/tools/analyze_function_pointers"
)

# Common headers for all requests (uses centralized JWT configuration)
COMMON_HEADERS = get_mcp_auth_headers()


@skip_without_mcp_server
class TestAnalyzeFunctionPointersContract:
    """Contract tests for analyze_function_pointers MCP endpoint."""

    def test_analyze_function_pointers_valid_request_schema(self):
        """Test that endpoint accepts valid request schema and returns 200."""
        # Valid request according to OpenAPI contract
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=valid_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Must return 200 for valid requests
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        # Contract requirement: Response must be valid JSON
        response_data = response.json()
        assert isinstance(response_data, dict), "Response must be a JSON object"

    def test_analyze_function_pointers_response_schema(self):
        """Test that response matches the expected schema structure."""
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=valid_request,
            headers=COMMON_HEADERS,
        )

        assert response.status_code == 200
        data = response.json()

        # Contract requirement: Response must contain function_pointers array
        assert "function_pointers" in data, (
            "Response must contain 'function_pointers' field"
        )
        assert isinstance(data["function_pointers"], list), (
            "function_pointers must be an array"
        )

        # Contract requirement: Response must contain analysis_stats
        assert "analysis_stats" in data, "Response must contain 'analysis_stats' field"
        assert isinstance(data["analysis_stats"], dict), (
            "analysis_stats must be an object"
        )

        # Validate analysis_stats schema
        stats = data["analysis_stats"]
        required_stats_fields = [
            "pointers_analyzed",
            "assignments_found",
            "usage_sites_found",
            "callback_patterns_matched",
        ]
        for field in required_stats_fields:
            assert field in stats, f"analysis_stats must contain '{field}' field"
            assert isinstance(stats[field], int), (
                f"analysis_stats.{field} must be an integer"
            )

    def test_analyze_function_pointers_pointer_schema(self):
        """Test that function_pointers in response match expected schema."""
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=valid_request,
            headers=COMMON_HEADERS,
        )

        assert response.status_code == 200
        data = response.json()

        # If function_pointers are present, validate their schema
        if data["function_pointers"]:
            pointer = data["function_pointers"][0]

            # Contract requirement: Each function pointer must have required fields
            assert "pointer_name" in pointer, (
                "Function pointer must have 'pointer_name' field"
            )
            assert "assignment_site" in pointer, (
                "Function pointer must have 'assignment_site' field"
            )
            assert "assigned_function" in pointer, (
                "Function pointer must have 'assigned_function' field"
            )
            assert "struct_context" in pointer, (
                "Function pointer must have 'struct_context' field"
            )

            # Validate pointer_name
            assert isinstance(pointer["pointer_name"], str), (
                "pointer_name must be string"
            )
            assert len(pointer["pointer_name"]) > 0, "pointer_name must not be empty"

            # Validate assignment_site schema
            assignment_site = pointer["assignment_site"]
            assert "file_path" in assignment_site, (
                "Assignment site must have 'file_path' field"
            )
            assert "line_number" in assignment_site, (
                "Assignment site must have 'line_number' field"
            )
            assert isinstance(assignment_site["file_path"], str), (
                "Assignment site file_path must be string"
            )
            assert isinstance(assignment_site["line_number"], int), (
                "Assignment site line_number must be integer"
            )
            assert assignment_site["line_number"] > 0, (
                "Assignment site line_number must be positive"
            )

            # Validate assigned_function schema
            assigned_function = pointer["assigned_function"]
            assert "name" in assigned_function, (
                "Assigned function must have 'name' field"
            )
            assert "file_path" in assigned_function, (
                "Assigned function must have 'file_path' field"
            )
            assert "line_number" in assigned_function, (
                "Assigned function must have 'line_number' field"
            )
            assert isinstance(assigned_function["name"], str), (
                "Assigned function name must be string"
            )
            assert isinstance(assigned_function["file_path"], str), (
                "Assigned function file_path must be string"
            )
            assert isinstance(assigned_function["line_number"], int), (
                "Assigned function line_number must be integer"
            )
            assert assigned_function["line_number"] > 0, (
                "Assigned function line_number must be positive"
            )

            # Validate struct_context
            assert isinstance(pointer["struct_context"], str), (
                "struct_context must be string"
            )

    def test_analyze_function_pointers_missing_required_fields(self):
        """Test that missing required fields return 400 Bad Request."""
        # Request missing required file_paths field
        invalid_request = {
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=invalid_request,
            headers=COMMON_HEADERS,
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

    def test_analyze_function_pointers_empty_file_paths(self):
        """Test that empty file_paths array returns 400."""
        invalid_request = {
            "file_paths": [],
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=invalid_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Empty file paths should be invalid
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_analyze_function_pointers_invalid_file_paths(self):
        """Test that invalid file paths return appropriate error."""
        # Request with non-existent file
        invalid_request = {
            "file_paths": ["non/existent/file.c"],
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=invalid_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Must handle file errors gracefully
        # Could be 400 (bad request) or 500 (processing error) depending on implementation
        assert response.status_code in [400, 500], (
            f"Expected 400 or 500, got {response.status_code}"
        )

        error_data = response.json()
        assert isinstance(error_data, dict), "Error response must be a JSON object"

    def test_analyze_function_pointers_multiple_patterns(self):
        """Test that multiple pointer patterns are supported."""
        multi_pattern_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": [
                "file_operations",
                "inode_operations",
                "super_operations",
            ],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=multi_pattern_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Multiple patterns should be supported
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "function_pointers" in data
        assert "analysis_stats" in data

    def test_analyze_function_pointers_empty_patterns(self):
        """Test that empty pointer_patterns array is handled correctly."""
        empty_patterns_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": [],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=empty_patterns_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Empty patterns could be valid (analyze all) or invalid
        assert response.status_code in [200, 400], (
            f"Expected 200 or 400, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "function_pointers" in data
            assert "analysis_stats" in data

    def test_analyze_function_pointers_unauthorized_request(self):
        """Test that requests without proper authorization return 401."""
        valid_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        # Request without authorization header
        headers_no_auth = {"Content-Type": "application/json"}

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=valid_request,
            headers=headers_no_auth,
        )

        # Contract requirement: Must require authorization
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"

    def test_analyze_function_pointers_optional_parameters(self):
        """Test that optional parameters work correctly."""
        # Request with minimal required fields only
        minimal_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"]
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=minimal_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Optional parameters should have defaults
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "function_pointers" in data
        assert "analysis_stats" in data

    def test_analyze_function_pointers_config_context_filtering(self):
        """Test that config_context parameter affects analysis."""
        # Test with specific config context
        config_specific_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["file_operations"],
            "config_context": "arm64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=config_specific_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Different config contexts should be supported
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "function_pointers" in data

    def test_analyze_function_pointers_malformed_json(self):
        """Test that malformed JSON returns 400."""
        # Send malformed JSON
        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            data="{ invalid json }",
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Malformed JSON should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_analyze_function_pointers_multiple_files(self):
        """Test that multiple file paths are processed correctly."""
        multi_file_request = {
            "file_paths": [
                "tests/fixtures/mini-kernel-v6.1/fs/test_file.c",
                "tests/fixtures/mini-kernel-v6.1/fs/another_file.c",
            ],
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=multi_file_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Multiple files should be supported
        assert response.status_code in [200, 400, 500], "Must handle multiple files"

        if response.status_code == 200:
            data = response.json()
            assert "function_pointers" in data
            assert "analysis_stats" in data

    def test_analyze_function_pointers_custom_patterns(self):
        """Test that custom pointer patterns are accepted."""
        custom_pattern_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["custom_operations", "my_callback_struct"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=custom_pattern_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Custom patterns should be accepted
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "function_pointers" in data
        assert "analysis_stats" in data

    def test_analyze_function_pointers_wildcard_patterns(self):
        """Test that wildcard patterns are handled correctly."""
        wildcard_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["*_operations", "*_ops"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=wildcard_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Wildcard patterns should be supported
        assert response.status_code in [200, 400], (
            f"Expected 200 or 400, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "function_pointers" in data

    def test_analyze_function_pointers_invalid_pattern_format(self):
        """Test that invalid pattern formats are handled appropriately."""
        invalid_pattern_request = {
            "file_paths": ["tests/fixtures/mini-kernel-v6.1/fs/test_file.c"],
            "pointer_patterns": ["", None, 123],  # Invalid pattern types
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=invalid_pattern_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Invalid patterns should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_analyze_function_pointers_large_file_set(self):
        """Test that large sets of files are handled correctly."""
        # Generate a large list of files (some may not exist)
        large_file_list = [f"tests/fixtures/file_{i}.c" for i in range(100)]

        large_request = {
            "file_paths": large_file_list,
            "pointer_patterns": ["file_operations"],
            "config_context": "x86_64:defconfig",
        }

        response = requests.post(
            ANALYZE_FUNCTION_POINTERS_ENDPOINT,
            json=large_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Large file sets should be handled (possibly with limits)
        assert response.status_code in [200, 400, 413], (
            f"Expected 200, 400, or 413, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "function_pointers" in data
            assert "analysis_stats" in data


class TestAnalyzeFunctionPointersImplementation:
    """
    Implementation-specific tests that will be enabled once the endpoint is implemented.
    These tests MUST initially be skipped to satisfy TDD requirements.
    """

    def test_analyze_function_pointers_functional_behavior(self):
        """Test actual function pointer analysis functionality."""
        # This test will be enabled after implementation
        pass

    def test_analyze_function_pointers_performance(self):
        """Test that analysis meets performance requirements."""
        # This test will be enabled after implementation
        pass

    def test_analyze_function_pointers_callback_detection(self):
        """Test that callback patterns are detected correctly."""
        # This test will be enabled after implementation
        pass


if __name__ == "__main__":
    # Run contract tests directly
    pytest.main([__file__, "-v"])
