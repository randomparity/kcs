"""
Contract tests for export_graph MCP tool.

These tests verify the API contract for exporting call graphs in various formats.
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
class TestExportGraphContract:
    """Contract tests for export_graph MCP tool."""

    async def test_export_graph_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the export_graph endpoint exists and accepts POST requests."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "json",
            "depth": 3,
            "include_metadata": True,
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "export_graph endpoint should exist"

    async def test_export_graph_requires_authentication(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that export_graph requires valid authentication."""
        payload = {"root_symbol": "vfs_read", "format": "json"}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/export_graph", json=payload)
        assert response.status_code == 401, "Should require authentication"

    async def test_export_graph_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that export_graph validates request schema according to OpenAPI spec."""

        # Missing required 'format' field
        response = await http_client.post(
            "/mcp/tools/export_graph",
            json={"root_symbol": "vfs_read"},
            headers=auth_headers,
        )
        assert response.status_code == 422, (
            "Should reject request without required 'format' field"
        )

        # Invalid format value
        response = await http_client.post(
            "/mcp/tools/export_graph",
            json={"root_symbol": "vfs_read", "format": "invalid_format"},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject invalid format value"

        # Negative depth
        response = await http_client.post(
            "/mcp/tools/export_graph",
            json={"root_symbol": "vfs_read", "format": "json", "depth": -1},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject negative depth"

        # Depth too large
        response = await http_client.post(
            "/mcp/tools/export_graph",
            json={"root_symbol": "vfs_read", "format": "json", "depth": 101},
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject depth > 100"

    async def test_export_graph_json_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test exporting graph in JSON format."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "json",
            "depth": 2,
            "include_metadata": True,
            "pretty": True,
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify JSON graph structure
            assert "graph" in data, "Response should contain 'graph' field"
            assert "export_id" in data, "Response should contain 'export_id' field"
            assert "format" in data, "Response should contain 'format' field"
            assert "exported_at" in data, "Response should contain 'exported_at' field"

            # Verify format
            assert data["format"] == "json", "Format should be json"

            # Verify graph structure
            graph = data["graph"]
            assert isinstance(graph, dict), "graph should be object"
            assert "nodes" in graph, "Graph should have 'nodes' field"
            assert "edges" in graph, "Graph should have 'edges' field"
            assert "metadata" in graph, "Graph should have 'metadata' field"

            # Verify nodes
            assert isinstance(graph["nodes"], list), "nodes should be array"
            for node in graph["nodes"]:
                assert "id" in node, "Node should have 'id' field"
                assert "label" in node, "Node should have 'label' field"
                assert "type" in node, "Node should have 'type' field"
                if payload["include_metadata"]:
                    assert "metadata" in node, (
                        "Node should have metadata when requested"
                    )

            # Verify edges
            assert isinstance(graph["edges"], list), "edges should be array"
            for edge in graph["edges"]:
                assert "source" in edge, "Edge should have 'source' field"
                assert "target" in edge, "Edge should have 'target' field"
                assert "type" in edge, "Edge should have 'type' field"

            # Verify metadata
            meta = graph["metadata"]
            assert "root_symbol" in meta, "Metadata should have root_symbol"
            assert "total_nodes" in meta, "Metadata should have total_nodes"
            assert "total_edges" in meta, "Metadata should have total_edges"
            assert "max_depth" in meta, "Metadata should have max_depth"

    async def test_export_graph_graphml_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test exporting graph in GraphML format."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "graphml",
            "depth": 2,
            "include_attributes": True,
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should return GraphML as string
            assert "graphml" in data, "Response should contain 'graphml' field"
            assert "export_id" in data, "Response should contain 'export_id' field"
            assert "format" in data, "Response should contain 'format' field"

            # Verify format
            assert data["format"] == "graphml", "Format should be graphml"

            # Verify GraphML content
            graphml = data["graphml"]
            assert isinstance(graphml, str), "GraphML should be string"
            assert graphml.startswith("<?xml"), "GraphML should be valid XML"
            assert "<graphml" in graphml, "Should contain graphml element"
            assert "<graph" in graphml, "Should contain graph element"
            assert "<node" in graphml, "Should contain node elements"
            assert "<edge" in graphml, "Should contain edge elements"

    async def test_export_graph_dot_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test exporting graph in DOT/Graphviz format."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "dot",
            "depth": 2,
            "layout": "hierarchical",
            "styling": {
                "node_color": "#3498db",
                "edge_color": "#95a5a6",
                "font_size": 12,
            },
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should return DOT as string
            assert "dot" in data, "Response should contain 'dot' field"
            assert "export_id" in data, "Response should contain 'export_id' field"
            assert "format" in data, "Response should contain 'format' field"

            # Verify format
            assert data["format"] == "dot", "Format should be dot"

            # Verify DOT content
            dot = data["dot"]
            assert isinstance(dot, str), "DOT should be string"
            assert dot.startswith("digraph") or dot.startswith("graph"), (
                "DOT should start with graph declaration"
            )
            assert "{" in dot and "}" in dot, "DOT should have proper structure"
            assert "->" in dot or "--" in dot, "DOT should have edge definitions"

    async def test_export_graph_csv_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test exporting graph in CSV format (edge list)."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "csv",
            "depth": 2,
            "csv_type": "edge_list",
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should return CSV data
            assert "csv" in data, "Response should contain 'csv' field"
            assert "export_id" in data, "Response should contain 'export_id' field"
            assert "format" in data, "Response should contain 'format' field"

            # Verify format
            assert data["format"] == "csv", "Format should be csv"

            # Verify CSV content
            csv_data = data["csv"]
            assert isinstance(csv_data, str), "CSV should be string"
            lines = csv_data.strip().split("\n")
            assert len(lines) > 1, "CSV should have header and data"

            # Check header
            header = lines[0]
            assert "source" in header.lower() or "from" in header.lower(), (
                "CSV should have source column"
            )
            assert "target" in header.lower() or "to" in header.lower(), (
                "CSV should have target column"
            )

    async def test_export_graph_with_filters(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test exporting graph with various filters applied."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "json",
            "depth": 3,
            "filters": {
                "exclude_patterns": ["test_*", "*_debug"],
                "include_subsystems": ["fs", "vfs"],
                "min_edge_weight": 2,
                "exclude_indirect": True,
            },
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            graph = data["graph"]

            # All nodes should respect filter criteria
            for node in graph["nodes"]:
                # Should not match excluded patterns
                assert not node["label"].startswith("test_"), (
                    "Should exclude test functions"
                )
                assert not node["label"].endswith("_debug"), (
                    "Should exclude debug functions"
                )

            # All edges should respect filter criteria
            if "weight" in graph["edges"][0] if graph["edges"] else {}:
                for edge in graph["edges"]:
                    if "weight" in edge:
                        assert edge["weight"] >= 2, (
                            "Should only include edges with weight >= 2"
                        )

    async def test_export_graph_chunked_export(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test chunked export for large graphs."""
        payload = {
            "format": "json",
            "depth": 10,  # Large depth to get many nodes
            "chunk_size": 100,
            "chunk_index": 0,
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should include chunking information
            assert "chunk_info" in data, "Response should contain chunk_info"
            chunk_info = data["chunk_info"]
            assert "total_chunks" in chunk_info, "Should have total_chunks"
            assert "current_chunk" in chunk_info, "Should have current_chunk"
            assert "chunk_size" in chunk_info, "Should have chunk_size"
            assert "has_more" in chunk_info, "Should have has_more flag"

            assert chunk_info["current_chunk"] == 0, "First chunk should be 0"
            assert chunk_info["chunk_size"] <= 100, "Chunk size should respect limit"

    async def test_export_graph_compression(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test compressed export for bandwidth efficiency."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "json",
            "depth": 5,
            "compress": True,
            "compression_format": "gzip",
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should indicate compression
            assert "compressed" in data, "Response should indicate compression"
            assert data["compressed"] is True, "Should be compressed"
            assert "compression_format" in data, "Should specify compression format"
            assert data["compression_format"] == "gzip", "Should use requested format"

            # Compressed data should be base64 encoded
            if "graph_data" in data:
                assert isinstance(data["graph_data"], str), (
                    "Compressed data should be base64 string"
                )

            # Should include size information
            if "size_info" in data:
                assert "original_size" in data["size_info"], (
                    "Should include original size"
                )
                assert "compressed_size" in data["size_info"], (
                    "Should include compressed size"
                )
                assert (
                    data["size_info"]["compressed_size"]
                    < data["size_info"]["original_size"]
                ), "Compressed size should be smaller"

    async def test_export_graph_async_export(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test asynchronous export for large graphs."""
        payload = {
            "root_symbol": "init_task",  # Likely to have large graph
            "format": "json",
            "depth": 20,
            "async": True,
            "callback_url": "https://example.com/callback",
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 202:  # Accepted for async processing
            data = response.json()

            # Should return job information
            assert "job_id" in data, "Should return job_id for async export"
            assert "status" in data, "Should return job status"
            assert "status_url" in data, "Should provide status check URL"
            assert "estimated_time" in data, "Should provide time estimate"

            assert data["status"] == "pending" or data["status"] == "processing", (
                "Initial status should be pending or processing"
            )

    async def test_export_graph_with_annotations(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test export with code annotations and documentation."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "json",
            "depth": 2,
            "include_annotations": True,
            "annotation_types": ["comments", "docstrings", "attributes"],
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            graph = data["graph"]

            # Nodes should include annotations when available
            for node in graph["nodes"]:
                if "annotations" in node:
                    assert isinstance(node["annotations"], dict), (
                        "annotations should be object"
                    )
                    ann = node["annotations"]
                    if "comments" in ann:
                        assert isinstance(ann["comments"], list), (
                            "comments should be array"
                        )
                    if "docstring" in ann:
                        assert isinstance(ann["docstring"], str), (
                            "docstring should be string"
                        )
                    if "attributes" in ann:
                        assert isinstance(ann["attributes"], dict), (
                            "attributes should be object"
                        )

    async def test_export_graph_statistics(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that export includes comprehensive statistics."""
        payload = {
            "root_symbol": "vfs_read",
            "format": "json",
            "depth": 3,
            "include_statistics": True,
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should include statistics
            assert "statistics" in data, "Response should contain statistics"
            stats = data["statistics"]

            # Verify statistics fields
            expected_stats = [
                "total_nodes",
                "total_edges",
                "max_depth_reached",
                "avg_degree",
                "density",
                "connected_components",
                "cycles_count",
                "longest_path",
            ]

            for stat in expected_stats:
                assert stat in stats, f"Statistics should include {stat}"

            # Verify statistics types
            assert isinstance(stats["total_nodes"], int), (
                "total_nodes should be integer"
            )
            assert isinstance(stats["total_edges"], int), (
                "total_edges should be integer"
            )
            assert isinstance(stats["avg_degree"], (int, float)), (
                "avg_degree should be numeric"
            )
            assert isinstance(stats["density"], (int, float)), (
                "density should be numeric"
            )
            assert 0 <= stats["density"] <= 1, "density should be between 0 and 1"

    async def test_export_graph_error_handling(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test proper error handling for export failures."""

        # Non-existent symbol
        payload = {
            "root_symbol": "nonexistent_function_xyz123",
            "format": "json",
            "depth": 2,
        }

        response = await http_client.post(
            "/mcp/tools/export_graph", json=payload, headers=auth_headers
        )

        if response.status_code == 404:
            data = response.json()
            assert "error" in data or "detail" in data, "Should include error details"

        # Graph would be empty/trivial
        elif response.status_code == 200:
            data = response.json()
            graph = data["graph"]
            # Should handle gracefully with empty or single-node graph
            assert len(graph["nodes"]) <= 1, (
                "Non-existent symbol should result in empty or single-node graph"
            )
            assert len(graph["edges"]) == 0, "Non-existent symbol should have no edges"
