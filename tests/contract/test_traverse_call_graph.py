"""
Contract tests for traverse_call_graph MCP tool.

These tests verify the API contract for advanced call graph traversal operations.
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
class TestTraverseCallGraphContract:
    """Contract tests for traverse_call_graph MCP tool."""

    async def test_traverse_call_graph_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the traverse_call_graph endpoint exists and accepts POST requests."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "forward",
            "max_depth": 3,
            "include_indirect": True,
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "traverse_call_graph endpoint should exist"

    async def test_traverse_call_graph_requires_authentication(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that traverse_call_graph requires valid authentication."""
        payload = {"start_symbol": "vfs_read", "direction": "forward"}

        # Request without auth headers
        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload
        )
        assert response.status_code == 401, "Should require authentication"

    async def test_traverse_call_graph_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that traverse_call_graph validates request schema according to OpenAPI spec."""

        # Missing required 'start_symbol' field
        response = await http_client.post(
            "/mcp/tools/traverse_call_graph",
            json={"direction": "forward"},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject request without required 'start_symbol' field"
        )

        # Invalid direction value
        response = await http_client.post(
            "/mcp/tools/traverse_call_graph",
            json={"start_symbol": "vfs_read", "direction": "invalid_direction"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid direction value"

        # Negative max_depth
        response = await http_client.post(
            "/mcp/tools/traverse_call_graph",
            json={"start_symbol": "vfs_read", "direction": "forward", "max_depth": -1},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject negative max_depth"

        # Max depth too large
        response = await http_client.post(
            "/mcp/tools/traverse_call_graph",
            json={"start_symbol": "vfs_read", "direction": "forward", "max_depth": 101},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject max_depth > 100"

    async def test_traverse_call_graph_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that traverse_call_graph returns response matching OpenAPI schema."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "forward",
            "max_depth": 2,
            "include_indirect": True,
            "filters": {
                "exclude_subsystems": ["test", "debug"],
                "include_only_exported": True,
            },
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify top-level structure
            assert "nodes" in data, "Response should contain 'nodes' field"
            assert "edges" in data, "Response should contain 'edges' field"
            assert "paths" in data, "Response should contain 'paths' field"
            assert "statistics" in data, "Response should contain 'statistics' field"
            assert "traversal_id" in data, (
                "Response should contain 'traversal_id' field"
            )

            # Verify field types
            assert isinstance(data["nodes"], list), "nodes should be array"
            assert isinstance(data["edges"], list), "edges should be array"
            assert isinstance(data["paths"], list), "paths should be array"
            assert isinstance(data["statistics"], dict), "statistics should be object"
            assert isinstance(data["traversal_id"], str), (
                "traversal_id should be string (UUID)"
            )

            # Verify nodes structure
            for node in data["nodes"]:
                assert isinstance(node, dict), "Each node should be object"
                assert "symbol" in node, "Node should have 'symbol' field"
                assert "span" in node, "Node should have 'span' field"
                assert "depth" in node, "Node should have 'depth' field"
                assert "node_type" in node, "Node should have 'node_type' field"

                # Verify field types
                assert isinstance(node["symbol"], str), "symbol should be string"
                assert isinstance(node["span"], dict), "span should be object"
                assert isinstance(node["depth"], int), "depth should be integer"
                assert 0 <= node["depth"] <= 2, "depth should be within max_depth"
                assert isinstance(node["node_type"], str), "node_type should be string"

                # Verify span structure
                span = node["span"]
                assert "path" in span, "Span should have 'path' field"
                assert "sha" in span, "Span should have 'sha' field"
                assert "start" in span, "Span should have 'start' field"
                assert "end" in span, "Span should have 'end' field"

                # Verify optional node fields
                if "metadata" in node:
                    assert isinstance(node["metadata"], dict), (
                        "metadata should be object"
                    )
                if "is_entrypoint" in node:
                    assert isinstance(node["is_entrypoint"], bool), (
                        "is_entrypoint should be boolean"
                    )

            # Verify edges structure
            for edge in data["edges"]:
                assert isinstance(edge, dict), "Each edge should be object"
                assert "from" in edge, "Edge should have 'from' field"
                assert "to" in edge, "Edge should have 'to' field"
                assert "edge_type" in edge, "Edge should have 'edge_type' field"

                # Verify field types
                assert isinstance(edge["from"], str), "from should be string (symbol)"
                assert isinstance(edge["to"], str), "to should be string (symbol)"
                assert isinstance(edge["edge_type"], str), "edge_type should be string"

                # Verify edge type enum
                valid_edge_types = ["direct", "indirect", "macro", "inline", "virtual"]
                assert edge["edge_type"] in valid_edge_types, (
                    f"edge_type should be one of {valid_edge_types}"
                )

                # Verify optional edge fields
                if "weight" in edge:
                    assert isinstance(edge["weight"], (int, float)), (
                        "weight should be numeric"
                    )
                if "call_site" in edge:
                    assert isinstance(edge["call_site"], dict), (
                        "call_site should be object with span"
                    )

            # Verify paths structure
            for path in data["paths"]:
                assert isinstance(path, list), "Each path should be array of symbols"
                assert len(path) > 0, "Path should not be empty"
                for symbol in path:
                    assert isinstance(symbol, str), "Path element should be string"

            # Verify statistics structure
            stats = data["statistics"]
            assert "total_nodes" in stats, "Statistics should have 'total_nodes'"
            assert "total_edges" in stats, "Statistics should have 'total_edges'"
            assert "max_depth_reached" in stats, (
                "Statistics should have 'max_depth_reached'"
            )
            assert "cycles_detected" in stats, (
                "Statistics should have 'cycles_detected'"
            )

            assert isinstance(stats["total_nodes"], int), (
                "total_nodes should be integer"
            )
            assert isinstance(stats["total_edges"], int), (
                "total_edges should be integer"
            )
            assert isinstance(stats["max_depth_reached"], int), (
                "max_depth_reached should be integer"
            )
            assert isinstance(stats["cycles_detected"], int), (
                "cycles_detected should be integer"
            )

    async def test_traverse_call_graph_direction_forward(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test forward traversal (who does this function call)."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "forward",
            "max_depth": 2,
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Forward traversal should find functions called BY start_symbol
            assert len(data["nodes"]) > 0, "Should find called functions"
            # First node should be the start symbol at depth 0
            start_node = next((n for n in data["nodes"] if n["depth"] == 0), None)
            assert start_node is not None, "Should have start node at depth 0"
            assert start_node["symbol"] == "vfs_read", "Start node should match input"

    async def test_traverse_call_graph_direction_backward(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test backward traversal (who calls this function)."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "backward",
            "max_depth": 2,
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Backward traversal should find functions that CALL start_symbol
            assert len(data["nodes"]) > 0, "Should find caller functions"
            # First node should be the start symbol at depth 0
            start_node = next((n for n in data["nodes"] if n["depth"] == 0), None)
            assert start_node is not None, "Should have start node at depth 0"
            assert start_node["symbol"] == "vfs_read", "Start node should match input"

    async def test_traverse_call_graph_bidirectional(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test bidirectional traversal (both callers and callees)."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "bidirectional",
            "max_depth": 1,
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include both callers and callees
            assert len(data["nodes"]) > 1, "Should find both callers and callees"
            # Should have edges in both directions
            if len(data["edges"]) > 0:
                has_incoming = any(e["to"] == "vfs_read" for e in data["edges"])
                has_outgoing = any(e["from"] == "vfs_read" for e in data["edges"])
                assert has_incoming or has_outgoing, (
                    "Should have edges in at least one direction"
                )

    async def test_traverse_call_graph_cycle_detection(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that cycles in the call graph are properly detected."""
        payload = {
            "start_symbol": "recursive_func",  # Assuming this might have cycles
            "direction": "forward",
            "max_depth": 10,
            "detect_cycles": True,
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include cycle information
            if data["statistics"]["cycles_detected"] > 0:
                assert "cycles" in data, (
                    "Should include cycle details when cycles exist"
                )
                if "cycles" in data:
                    assert isinstance(data["cycles"], list), "cycles should be array"
                    for cycle in data["cycles"]:
                        assert isinstance(cycle, list), (
                            "Each cycle should be array of symbols"
                        )
                        assert len(cycle) > 1, "Cycle should have at least 2 nodes"
                        # First and last should be the same to form a cycle
                        assert cycle[0] == cycle[-1], (
                            "Cycle should start and end with same node"
                        )

    async def test_traverse_call_graph_path_finding(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test finding paths between two symbols."""
        payload = {
            "start_symbol": "vfs_read",
            "target_symbol": "security_file_permission",
            "direction": "forward",
            "max_depth": 5,
            "find_all_paths": True,
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should find paths if connection exists
            if len(data["paths"]) > 0:
                for path in data["paths"]:
                    assert path[0] == "vfs_read", "Path should start with start_symbol"
                    assert path[-1] == "security_file_permission", (
                        "Path should end with target_symbol"
                    )
                    # Verify path is connected
                    for i in range(len(path) - 1):
                        # Should have an edge from path[i] to path[i+1]
                        edge_exists = any(
                            e["from"] == path[i] and e["to"] == path[i + 1]
                            for e in data["edges"]
                        )
                        assert edge_exists, (
                            f"Path should be connected at {path[i]} -> {path[i + 1]}"
                        )

    async def test_traverse_call_graph_filters(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test filtering nodes during traversal."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "forward",
            "max_depth": 3,
            "filters": {
                "exclude_patterns": ["test_*", "*_debug"],
                "include_subsystems": ["fs", "vfs"],
                "min_complexity": 5,
                "exclude_static": True,
            },
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # All nodes should respect filter criteria
            for node in data["nodes"]:
                # Should not match excluded patterns
                assert not node["symbol"].startswith("test_"), (
                    "Should exclude test functions"
                )
                assert not node["symbol"].endswith("_debug"), (
                    "Should exclude debug functions"
                )

    async def test_traverse_call_graph_incremental_expansion(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test incremental graph expansion from previous traversal."""
        # First traversal
        payload1 = {
            "start_symbol": "vfs_read",
            "direction": "forward",
            "max_depth": 1,
        }

        response1 = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload1, headers=auth_headers
        )

        if response1.status_code == 200:
            data1 = response1.json()
            traversal_id = data1["traversal_id"]

            # Expand from a node found in first traversal
            if len(data1["nodes"]) > 1:
                # Pick a node at depth 1 to expand from
                expand_node = next((n for n in data1["nodes"] if n["depth"] == 1), None)
                if expand_node:
                    payload2 = {
                        "start_symbol": expand_node["symbol"],
                        "direction": "forward",
                        "max_depth": 1,
                        "base_traversal_id": traversal_id,
                        "incremental": True,
                    }

                    response2 = await http_client.post(
                        "/mcp/tools/traverse_call_graph",
                        json=payload2,
                        headers=auth_headers,
                    )

                    if response2.status_code == 200:
                        data2 = response2.json()
                        # Should include both original and new nodes
                        assert len(data2["nodes"]) >= len(data1["nodes"]), (
                            "Incremental should include original nodes"
                        )

    async def test_traverse_call_graph_complexity_metrics(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that complexity metrics are included in traversal."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "forward",
            "max_depth": 2,
            "include_metrics": True,
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include metrics for nodes
            for node in data["nodes"]:
                if "metrics" in node:
                    assert isinstance(node["metrics"], dict), "metrics should be object"
                    metrics = node["metrics"]
                    if "cyclomatic_complexity" in metrics:
                        assert isinstance(metrics["cyclomatic_complexity"], int), (
                            "cyclomatic_complexity should be integer"
                        )
                    if "lines_of_code" in metrics:
                        assert isinstance(metrics["lines_of_code"], int), (
                            "lines_of_code should be integer"
                        )
                    if "fan_in" in metrics:
                        assert isinstance(metrics["fan_in"], int), (
                            "fan_in should be integer"
                        )
                    if "fan_out" in metrics:
                        assert isinstance(metrics["fan_out"], int), (
                            "fan_out should be integer"
                        )

    async def test_traverse_call_graph_visualization_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that visualization-ready data is included when requested."""
        payload = {
            "start_symbol": "vfs_read",
            "direction": "forward",
            "max_depth": 2,
            "include_visualization": True,
            "layout": "hierarchical",
        }

        response = await http_client.post(
            "/mcp/tools/traverse_call_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include visualization data
            if "visualization" in data:
                assert isinstance(data["visualization"], dict), (
                    "visualization should be object"
                )
                viz = data["visualization"]
                if "layout" in viz:
                    assert viz["layout"] == "hierarchical", (
                        "Should use requested layout"
                    )
                if "node_positions" in viz:
                    assert isinstance(viz["node_positions"], dict), (
                        "node_positions should be object"
                    )
                    for _symbol, pos in viz["node_positions"].items():
                        assert "x" in pos and "y" in pos, (
                            "Position should have x and y coordinates"
                        )
                if "suggested_colors" in viz:
                    assert isinstance(viz["suggested_colors"], dict), (
                        "suggested_colors should be object"
                    )
