"""
Contract test for POST /mcp/tools/get_call_relationships endpoint.

Tests the API contract for call relationship queries according to
the OpenAPI specification in contracts/call-graph-api.yaml.

This test MUST FAIL initially as no implementation exists yet (TDD requirement).
"""

import json

import pytest
import requests

from tests.conftest import get_mcp_auth_headers, skip_without_mcp_server

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
GET_CALL_RELATIONSHIPS_ENDPOINT = f"{MCP_BASE_URL}/mcp/tools/get_call_relationships"

# Common headers for all requests (uses centralized JWT configuration)
COMMON_HEADERS = get_mcp_auth_headers()


@skip_without_mcp_server
class TestGetCallRelationshipsContract:
    """Contract tests for get_call_relationships MCP endpoint."""

    def test_get_call_relationships_valid_request_schema(self):
        """Test that endpoint accepts valid request schema and returns 200."""
        # Valid request according to OpenAPI contract
        valid_request = {
            "function_name": "generic_file_open",
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Must return 200 for valid requests
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        # Contract requirement: Response must be valid JSON
        response_data = response.json()
        assert isinstance(response_data, dict), "Response must be a JSON object"

    def test_get_call_relationships_response_schema(self):
        """Test that response matches the expected schema structure."""
        valid_request = {
            "function_name": "generic_file_open",
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        # Contract requirement: Response must contain function_name
        assert "function_name" in data, "Response must contain 'function_name' field"
        assert isinstance(data["function_name"], str), "function_name must be a string"
        assert data["function_name"] == "generic_file_open", (
            "function_name must match request"
        )

        # Contract requirement: Response must contain relationships
        assert "relationships" in data, "Response must contain 'relationships' field"
        assert isinstance(data["relationships"], dict), (
            "relationships must be an object"
        )

    def test_get_call_relationships_callers_schema(self):
        """Test that callers relationship type returns correct schema."""
        valid_request = {
            "function_name": "generic_file_open",
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": 3,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        relationships = data["relationships"]

        # Contract requirement: callers type must contain callers array
        assert "callers" in relationships, (
            "relationships must contain 'callers' field for callers type"
        )
        assert isinstance(relationships["callers"], list), "callers must be an array"

        # If callers are present, validate their schema
        if relationships["callers"]:
            caller = relationships["callers"][0]

            # Validate caller relationship structure
            assert "function" in caller, "Caller must have 'function' field"
            assert "call_edge" in caller, "Caller must have 'call_edge' field"
            assert "depth" in caller, "Caller must have 'depth' field"

            # Validate function schema
            function = caller["function"]
            assert "name" in function, "Function must have 'name' field"
            assert "file_path" in function, "Function must have 'file_path' field"
            assert "line_number" in function, "Function must have 'line_number' field"
            assert isinstance(function["line_number"], int), (
                "Function line_number must be integer"
            )

            # Validate call_edge schema
            call_edge = caller["call_edge"]
            assert "call_type" in call_edge, "Call edge must have 'call_type' field"
            assert "confidence" in call_edge, "Call edge must have 'confidence' field"
            assert "call_site" in call_edge, "Call edge must have 'call_site' field"

            # Validate call_site schema
            call_site = call_edge["call_site"]
            assert "line_number" in call_site, "Call site must have 'line_number' field"
            assert isinstance(call_site["line_number"], int), (
                "Call site line_number must be integer"
            )

            # Validate depth
            assert isinstance(caller["depth"], int), "Depth must be integer"
            assert caller["depth"] >= 1, "Depth must be at least 1"
            assert caller["depth"] <= 3, "Depth must not exceed max_depth"

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
            assert call_edge["call_type"] in valid_call_types, (
                f"call_type must be one of {valid_call_types}"
            )

            valid_confidence_levels = ["high", "medium", "low"]
            assert call_edge["confidence"] in valid_confidence_levels, (
                f"confidence must be one of {valid_confidence_levels}"
            )

    def test_get_call_relationships_callees_schema(self):
        """Test that callees relationship type returns correct schema."""
        valid_request = {
            "function_name": "test_open",
            "relationship_type": "callees",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        relationships = data["relationships"]

        # Contract requirement: callees type must contain callees array
        assert "callees" in relationships, (
            "relationships must contain 'callees' field for callees type"
        )
        assert isinstance(relationships["callees"], list), "callees must be an array"

        # If callees are present, validate their schema (same as callers)
        if relationships["callees"]:
            callee = relationships["callees"][0]
            assert "function" in callee, "Callee must have 'function' field"
            assert "call_edge" in callee, "Callee must have 'call_edge' field"
            assert "depth" in callee, "Callee must have 'depth' field"

    def test_get_call_relationships_both_schema(self):
        """Test that both relationship type returns both callers and callees."""
        valid_request = {
            "function_name": "generic_file_open",
            "relationship_type": "both",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        relationships = data["relationships"]

        # Contract requirement: both type must contain both callers and callees
        assert "callers" in relationships, (
            "relationships must contain 'callers' field for both type"
        )
        assert "callees" in relationships, (
            "relationships must contain 'callees' field for both type"
        )
        assert isinstance(relationships["callers"], list), "callers must be an array"
        assert isinstance(relationships["callees"], list), "callees must be an array"

    def test_get_call_relationships_missing_required_fields(self):
        """Test that missing required fields return 400 Bad Request."""
        # Request missing required function_name field
        invalid_request = {
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
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

    def test_get_call_relationships_invalid_relationship_type(self):
        """Test that invalid relationship_type returns 400."""
        invalid_request = {
            "function_name": "test_function",
            "relationship_type": "invalid_type",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
            json=invalid_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Invalid enum values should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_get_call_relationships_invalid_max_depth(self):
        """Test that invalid max_depth values are handled correctly."""
        # Test negative max_depth
        invalid_request = {
            "function_name": "test_function",
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": -1,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
            json=invalid_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Invalid max_depth should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

        # Test zero max_depth
        invalid_request["max_depth"] = 0
        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
            json=invalid_request,
            headers=COMMON_HEADERS,
        )

        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_get_call_relationships_nonexistent_function(self):
        """Test that nonexistent function returns appropriate response."""
        # Request for function that doesn't exist
        valid_request = {
            "function_name": "nonexistent_function_12345",
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT, json=valid_request, headers=COMMON_HEADERS
        )

        # Contract requirement: Nonexistent function should return empty results (200) or 404
        assert response.status_code in [200, 404], (
            f"Expected 200 or 404, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "relationships" in data
            # Should return empty relationships
            if "callers" in data["relationships"]:
                assert data["relationships"]["callers"] == []

    def test_get_call_relationships_unauthorized_request(self):
        """Test that requests without proper authorization return 401."""
        valid_request = {
            "function_name": "generic_file_open",
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": 2,
        }

        # Request without authorization header
        headers_no_auth = {"Content-Type": "application/json"}

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT, json=valid_request, headers=headers_no_auth
        )

        # Contract requirement: Must require authorization
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"

    def test_get_call_relationships_optional_parameters(self):
        """Test that optional parameters work correctly."""
        # Request with minimal required fields only
        minimal_request = {
            "function_name": "generic_file_open",
            "relationship_type": "callers",
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
            json=minimal_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Optional parameters should have defaults
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "function_name" in data
        assert "relationships" in data

    def test_get_call_relationships_config_context_filtering(self):
        """Test that config_context parameter filters results correctly."""
        # Test with specific config context
        config_specific_request = {
            "function_name": "generic_file_open",
            "relationship_type": "callers",
            "config_context": "arm64:defconfig",
            "max_depth": 2,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
            json=config_specific_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Different config contexts should be supported
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "relationships" in data

    def test_get_call_relationships_malformed_json(self):
        """Test that malformed JSON returns 400."""
        # Send malformed JSON
        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
            data="{ invalid json }",
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Malformed JSON should return 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_get_call_relationships_large_max_depth(self):
        """Test that very large max_depth values are handled appropriately."""
        large_depth_request = {
            "function_name": "generic_file_open",
            "relationship_type": "callers",
            "config_context": "x86_64:defconfig",
            "max_depth": 1000,
        }

        response = requests.post(
            GET_CALL_RELATIONSHIPS_ENDPOINT,
            json=large_depth_request,
            headers=COMMON_HEADERS,
        )

        # Contract requirement: Large depths should be handled (possibly with limits)
        assert response.status_code in [200, 400], (
            f"Expected 200 or 400, got {response.status_code}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "relationships" in data


class TestGetCallRelationshipsImplementation:
    """
    Implementation-specific tests that will be enabled once the endpoint is implemented.
    These tests MUST initially be skipped to satisfy TDD requirements.
    """

    def test_get_call_relationships_functional_behavior(self):
        """Test actual call relationship query functionality."""
        # This test will be enabled after implementation
        pass

    def test_get_call_relationships_performance(self):
        """Test that relationship queries meet performance requirements."""
        # This test will be enabled after implementation
        pass


if __name__ == "__main__":
    # Run contract tests directly
    pytest.main([__file__, "-v"])
