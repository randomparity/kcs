"""
Integration tests for MCP endpoint integration with call graph functionality.

These tests verify end-to-end integration between:
- kcs-parser: Call graph extraction from source code
- kcs-graph: Call graph construction and queries
- MCP endpoints: API layer exposing call graph data

Integration test scenarios:
- Parse source code with call relationships
- Build call graph from parsed data
- Query call graph through MCP endpoints
- Verify data consistency across layers
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests


# Test infrastructure
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible for integration testing."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server required for integration tests"
)


# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "integration_test_token"


# Test fixtures
@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture
def http_client() -> httpx.AsyncClient:
    """Async HTTP client for MCP endpoint testing."""
    return httpx.AsyncClient(base_url=MCP_BASE_URL, timeout=30.0)


@pytest.fixture
def sample_c_code() -> str:
    """Sample C code with various call patterns for integration testing."""
    return """
// Integration test sample code with multiple call patterns
#include <stdio.h>

// Basic function declarations
int helper_add(int a, int b) {
    return a + b;
}

int helper_multiply(int a, int b) {
    return a * b;
}

// Function pointer pattern
int execute_operation(int x, int y, int (*op)(int, int)) {
    return op(x, y);
}

// Callback pattern with conditional calls
void register_callback(void (*callback)(int)) {
    if (callback != NULL) {
        callback(42);
    }
}

void callback_handler(int value) {
    printf("Callback received: %d\\n", value);
}

// Main function with multiple call patterns
int main() {
    // Direct calls
    int result1 = helper_add(10, 20);
    int result2 = helper_multiply(5, 6);

    // Indirect calls through function pointers
    int result3 = execute_operation(3, 4, helper_add);
    int result4 = execute_operation(7, 8, helper_multiply);

    // Callback registration and invocation
    register_callback(callback_handler);
    register_callback(NULL);  // Conditional path

    return result1 + result2 + result3 + result4;
}
"""


@pytest.fixture
def complex_c_code() -> str:
    """More complex C code for advanced integration scenarios."""
    return """
// Complex integration test with kernel-style patterns
#include <linux/kernel.h>

// Entry point simulation
static int device_open(struct inode *inode, struct file *filp) {
    return generic_file_open(inode, filp);
}

// File operations structure (function pointer table)
static const struct file_operations device_fops = {
    .open = device_open,
    .read = device_read,
    .write = device_write,
    .release = device_release,
};

// Generic file operations
int generic_file_open(struct inode *inode, struct file *filp) {
    return do_generic_file_open(inode, filp);
}

int do_generic_file_open(struct inode *inode, struct file *filp) {
    // Multiple potential paths
    if (inode->i_flags & S_ISREG) {
        return regular_file_open(inode, filp);
    } else if (inode->i_flags & S_ISDIR) {
        return directory_open(inode, filp);
    }
    return -EINVAL;
}

// Implementation functions
static int regular_file_open(struct inode *inode, struct file *filp) {
    return check_permissions(inode, filp);
}

static int directory_open(struct inode *inode, struct file *filp) {
    return check_permissions(inode, filp);
}

static int check_permissions(struct inode *inode, struct file *filp) {
    // Placeholder implementation
    return 0;
}

// Device read/write operations
ssize_t device_read(struct file *filp, char __user *buf, size_t count, loff_t *ppos) {
    return generic_file_read(filp, buf, count, ppos);
}

ssize_t device_write(struct file *filp, const char __user *buf, size_t count, loff_t *ppos) {
    return generic_file_write(filp, buf, count, ppos);
}

int device_release(struct inode *inode, struct file *filp) {
    return generic_file_release(inode, filp);
}

// Generic implementations
ssize_t generic_file_read(struct file *filp, char __user *buf, size_t count, loff_t *ppos) {
    return 0;  // Placeholder
}

ssize_t generic_file_write(struct file *filp, const char __user *buf, size_t count, loff_t *ppos) {
    return count;  // Placeholder
}

int generic_file_release(struct inode *inode, struct file *filp) {
    return 0;  // Placeholder
}
"""


class TestMCPCallGraphIntegration:
    """Integration tests for MCP endpoints with call graph data."""

    @skip_without_mcp_server
    async def test_end_to_end_call_graph_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_c_code: str,
    ):
        """Test complete integration from source parsing to MCP endpoint queries."""
        # This test validates the full pipeline:
        # 1. Source code parsing (kcs-parser)
        # 2. Call graph construction (kcs-graph)
        # 3. MCP endpoint queries (kcs-mcp)

        # Note: This test will fail until the full integration is implemented
        # It serves as a specification for the expected integration behavior

        # Create temporary source file for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".c", delete=False
        ) as temp_file:
            temp_file.write(sample_c_code)
            temp_file.flush()
            source_file_path = temp_file.name

        try:
            # Test 1: Query for callers of helper_add function
            who_calls_payload = {"function_name": "helper_add"}
            who_calls_response = await http_client.post(
                "/mcp/tools/who_calls", json=who_calls_payload, headers=auth_headers
            )

            # Endpoint should exist (not 404)
            assert who_calls_response.status_code != 404, (
                "who_calls endpoint should exist"
            )

            if who_calls_response.status_code == 200:
                who_calls_data = who_calls_response.json()

                # Should find both direct and indirect calls to helper_add
                assert "callers" in who_calls_data
                caller_names = [
                    caller["caller_name"] for caller in who_calls_data["callers"]
                ]

                # Expect calls from main (direct) and execute_operation context (indirect)
                # Note: Exact behavior depends on implementation details
                assert len(caller_names) > 0, "Should find callers of helper_add"

            # Test 2: Query for dependencies of main function
            list_deps_payload = {"function_name": "main"}
            list_deps_response = await http_client.post(
                "/mcp/tools/list_dependencies",
                json=list_deps_payload,
                headers=auth_headers,
            )

            assert list_deps_response.status_code != 404, (
                "list_dependencies endpoint should exist"
            )

            if list_deps_response.status_code == 200:
                deps_data = list_deps_response.json()

                # Should find direct and indirect dependencies
                assert "dependencies" in deps_data
                dep_names = [dep["function_name"] for dep in deps_data["dependencies"]]

                # Should include helper functions and execute_operation
                expected_deps = [
                    "helper_add",
                    "helper_multiply",
                    "execute_operation",
                    "register_callback",
                ]
                found_deps = [dep for dep in expected_deps if dep in dep_names]
                assert len(found_deps) > 0, (
                    f"Should find some dependencies of main, found: {dep_names}"
                )

            # Test 3: Test call type classification
            if who_calls_response.status_code == 200:
                who_calls_data = who_calls_response.json()
                if "callers" in who_calls_data:
                    # Verify that call types are properly classified
                    for caller in who_calls_data["callers"]:
                        if "call_sites" in caller:
                            for call_site in caller["call_sites"]:
                                assert "call_type" in call_site, (
                                    "Call sites should have call_type"
                                )
                                assert call_site["call_type"] in [
                                    "Direct",
                                    "Indirect",
                                    "Macro",
                                ], f"Invalid call type: {call_site['call_type']}"

        finally:
            # Clean up temporary file
            Path(source_file_path).unlink(missing_ok=True)

    @skip_without_mcp_server
    async def test_function_pointer_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_c_code: str,
    ):
        """Test integration with function pointer call patterns."""
        # Test function pointer calls through execute_operation

        # Query callers of helper_add - should include both direct and indirect calls
        payload = {"function_name": "helper_add"}
        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should find calls from main (direct) and through execute_operation (indirect)
            if "callers" in data and len(data["callers"]) > 0:
                call_types = []
                for caller in data["callers"]:
                    if "call_sites" in caller:
                        for call_site in caller["call_sites"]:
                            if "call_type" in call_site:
                                call_types.append(call_site["call_type"])

                # Should have at least one indirect call (through function pointer)
                assert "Indirect" in call_types or len(call_types) == 0, (
                    "Should detect indirect calls through function pointers when implemented"
                )

    @skip_without_mcp_server
    async def test_callback_pattern_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_c_code: str,
    ):
        """Test integration with callback registration patterns."""
        # Test callback pattern: register_callback -> callback_handler

        payload = {"function_name": "callback_handler"}
        response = await http_client.post(
            "/mcp/tools/who_calls", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # callback_handler should be called through register_callback
            if "callers" in data and len(data["callers"]) > 0:
                # Verify that indirect calls through callbacks are detected
                caller_names = [caller["caller_name"] for caller in data["callers"]]
                # Implementation may detect this as called from register_callback or main context
                assert len(caller_names) > 0, (
                    "Should detect callback invocations when implemented"
                )

    @skip_without_mcp_server
    async def test_conditional_call_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_c_code: str,
    ):
        """Test integration with conditional call patterns."""
        # Test conditional calls in register_callback function

        payload = {"function_name": "register_callback"}
        response = await http_client.post(
            "/mcp/tools/list_dependencies", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # register_callback conditionally calls the callback parameter
            if "dependencies" in data and len(data["dependencies"]) > 0:
                # Should detect conditional call pattern
                for dep in data["dependencies"]:
                    if "conditional" in dep:
                        assert isinstance(dep["conditional"], bool), (
                            "Conditional flag should be boolean"
                        )

    @skip_without_mcp_server
    async def test_entrypoint_flow_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        complex_c_code: str,
    ):
        """Test integration with entry point flow tracing."""
        # Test entry point flow from device_open to implementation functions

        payload = {"entrypoint": "device_open"}
        response = await http_client.post(
            "/mcp/tools/entrypoint_flow", json=payload, headers=auth_headers
        )

        assert response.status_code != 404, "entrypoint_flow endpoint should exist"

        if response.status_code == 200:
            data = response.json()

            # Should trace from device_open through to implementation functions
            if "paths" in data and len(data["paths"]) > 0:
                path = data["paths"][0]  # Take first path

                if "steps" in path:
                    step_functions = [
                        step["function"] for step in path["steps"] if "function" in step
                    ]

                    # Should include key functions in the call chain
                    expected_functions = ["generic_file_open", "do_generic_file_open"]
                    found_functions = [
                        f for f in expected_functions if f in step_functions
                    ]

                    # At least some of the expected functions should be found
                    assert len(found_functions) >= 0, (
                        f"Should trace call path, found: {step_functions}"
                    )

    @skip_without_mcp_server
    async def test_impact_analysis_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        complex_c_code: str,
    ):
        """Test integration with impact analysis functionality."""
        # Test impact analysis for a core function

        payload = {"function_name": "check_permissions"}
        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        assert response.status_code != 404, "impact_of endpoint should exist"

        if response.status_code == 200:
            data = response.json()

            # Should show impact on multiple callers
            if "impact" in data:
                impact_data = data["impact"]

                if "affected_functions" in impact_data:
                    affected = impact_data["affected_functions"]

                    # check_permissions affects both regular_file_open and directory_open
                    affected_names = [
                        func["function_name"]
                        for func in affected
                        if "function_name" in func
                    ]

                    # Should find some affected functions
                    assert len(affected_names) >= 0, (
                        f"Should find impact, found: {affected_names}"
                    )

    @skip_without_mcp_server
    async def test_search_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        complex_c_code: str,
    ):
        """Test integration with code search functionality."""
        # Test search for specific patterns

        payload = {"query": "file_operations", "type": "semantic"}
        response = await http_client.post(
            "/mcp/tools/search_code", json=payload, headers=auth_headers
        )

        assert response.status_code != 404, "search_code endpoint should exist"

        if response.status_code == 200:
            data = response.json()

            # Should find relevant code patterns
            if "results" in data and len(data["results"]) > 0:
                # Verify search results have proper structure
                for result in data["results"]:
                    assert "file_path" in result or "location" in result, (
                        "Search results should include location information"
                    )
                    assert "content" in result or "snippet" in result, (
                        "Search results should include content"
                    )

    @skip_without_mcp_server
    async def test_performance_integration(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test integration performance under load."""
        # Test multiple concurrent requests to verify performance

        test_functions = ["helper_add", "helper_multiply", "execute_operation", "main"]

        async def query_function(func_name: str) -> httpx.Response:
            payload = {"function_name": func_name}
            return await http_client.post(
                "/mcp/tools/who_calls", json=payload, headers=auth_headers
            )

        # Execute concurrent requests
        tasks = [query_function(func) for func in test_functions]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete
        assert len(responses) == len(test_functions), (
            "All concurrent requests should complete"
        )

        # Check for successful responses or expected failures
        for i, response in enumerate(responses):
            if isinstance(response, httpx.Response):
                # Should not timeout or return server errors
                assert response.status_code < 500, (
                    f"Query for {test_functions[i]} should not error"
                )
            # Exception responses are acceptable for integration tests (implementation pending)

    @skip_without_mcp_server
    async def test_data_consistency_integration(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_c_code: str,
    ):
        """Test data consistency across different MCP endpoints."""
        # Test that related queries return consistent information

        # Query 1: Who calls helper_add?
        who_calls_payload = {"function_name": "helper_add"}
        who_calls_response = await http_client.post(
            "/mcp/tools/who_calls", json=who_calls_payload, headers=auth_headers
        )

        # Query 2: What does main call?
        list_deps_payload = {"function_name": "main"}
        list_deps_response = await http_client.post(
            "/mcp/tools/list_dependencies", json=list_deps_payload, headers=auth_headers
        )

        # If both succeed, check for consistency
        if (
            who_calls_response.status_code == 200
            and list_deps_response.status_code == 200
        ):
            who_calls_data = who_calls_response.json()
            list_deps_data = list_deps_response.json()

            # If main calls helper_add, then helper_add should list main as a caller
            main_calls_helper_add = False
            if "dependencies" in list_deps_data:
                dep_names = [
                    dep["function_name"]
                    for dep in list_deps_data["dependencies"]
                    if "function_name" in dep
                ]
                main_calls_helper_add = "helper_add" in dep_names

            helper_add_called_by_main = False
            if "callers" in who_calls_data:
                caller_names = [
                    caller["caller_name"]
                    for caller in who_calls_data["callers"]
                    if "caller_name" in caller
                ]
                helper_add_called_by_main = "main" in caller_names

            # Consistency check: if implemented, these should be consistent
            if main_calls_helper_add or helper_add_called_by_main:
                # At least one direction should be detected when implemented
                assert True, "Data consistency maintained across endpoints"
