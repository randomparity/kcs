"""
Contract tests for infrastructure MCP endpoints.
These tests MUST fail initially (RED phase of TDD).
"""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock
import json


class TestParseKernelConfig:
    """Contract tests for /mcp/tools/parse_kernel_config endpoint."""

    @pytest.mark.asyncio
    async def test_parse_kernel_config_success(self, test_client: AsyncClient):
        """Test successful kernel config parsing."""
        request = {
            "config_path": "/kernel/arch/x86/configs/x86_64_defconfig",
            "kernel_root": "/kernel",
            "architecture": "x86_64",
            "config_type": "defconfig"
        }

        response = await test_client.post(
            "/mcp/tools/parse_kernel_config",
            json=request
        )

        assert response.status_code == 200
        data = response.json()

        # Contract assertions
        assert "config_name" in data
        assert data["config_name"] == "x86_64:defconfig"
        assert "features_enabled" in data
        assert isinstance(data["features_enabled"], int)
        assert data["features_enabled"] > 0
        assert "features_disabled" in data
        assert "features_module" in data
        assert "dependencies_resolved" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    @pytest.mark.asyncio
    async def test_parse_kernel_config_invalid_arch(self, test_client: AsyncClient):
        """Test config parsing with invalid architecture."""
        request = {
            "config_path": "/kernel/.config",
            "kernel_root": "/kernel",
            "architecture": "invalid_arch"
        }

        response = await test_client.post(
            "/mcp/tools/parse_kernel_config",
            json=request
        )

        assert response.status_code == 400
        assert "error" in response.json()


class TestValidateSpec:
    """Contract tests for /mcp/tools/validate_spec endpoint."""

    @pytest.mark.asyncio
    async def test_validate_spec_success(self, test_client: AsyncClient):
        """Test successful spec validation."""
        request = {
            "spec_content": """
            # VFS Specification
            - System MUST provide open() syscall
            - System MUST validate file permissions
            """,
            "spec_format": "markdown",
            "kernel_commit": "abc123def456",
            "config_name": "x86_64:defconfig"
        }

        response = await test_client.post(
            "/mcp/tools/validate_spec",
            json=request
        )

        assert response.status_code == 200
        data = response.json()

        # Contract assertions
        assert "report_id" in data
        assert "total_requirements" in data
        assert data["total_requirements"] == 2
        assert "passed" in data
        assert "failed" in data
        assert "unknown" in data
        assert data["passed"] + data["failed"] + data["unknown"] == data["total_requirements"]
        assert "severity" in data
        assert data["severity"] in ["critical", "major", "minor", "info"]
        assert "violations" in data
        assert isinstance(data["violations"], list)

        if data["violations"]:
            violation = data["violations"][0]
            assert "requirement" in violation
            assert "expected" in violation
            assert "actual" in violation
            assert "location" in violation
            assert "severity" in violation


class TestSemanticSearch:
    """Contract tests for /mcp/tools/semantic_search endpoint."""

    @pytest.mark.asyncio
    async def test_semantic_search_concept(self, test_client: AsyncClient):
        """Test semantic search for concepts."""
        request = {
            "query": "memory allocation in page cache",
            "query_type": "concept",
            "context": "mm",
            "limit": 5
        }

        response = await test_client.post(
            "/mcp/tools/semantic_search",
            json=request
        )

        assert response.status_code == 200
        data = response.json()

        # Contract assertions
        assert "query_id" in data
        assert "total_results" in data
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= 5

        if data["results"]:
            result = data["results"][0]
            assert "symbol" in result
            assert "file_path" in result
            assert "line" in result
            assert "similarity_score" in result
            assert 0.0 <= result["similarity_score"] <= 1.0
            assert "context" in result
            assert "span" in result
            assert "file" in result["span"]
            assert "line" in result["span"]
            assert "sha" in result["span"]

    @pytest.mark.asyncio
    async def test_semantic_search_similar_to(self, test_client: AsyncClient):
        """Test finding similar functions."""
        request = {
            "query": "Functions similar to kmalloc",
            "query_type": "similar_to",
            "limit": 10
        }

        response = await test_client.post(
            "/mcp/tools/semantic_search",
            json=request
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] > 0
        # kmalloc variants should score high
        symbols = [r["symbol"] for r in data["results"]]
        assert any("alloc" in s.lower() for s in symbols)


class TestTraverseCallGraph:
    """Contract tests for /mcp/tools/traverse_call_graph endpoint."""

    @pytest.mark.asyncio
    async def test_traverse_forward(self, test_client: AsyncClient):
        """Test forward call graph traversal."""
        request = {
            "source": "sys_open",
            "target": "do_filp_open",
            "direction": "forward",
            "max_depth": 5,
            "detect_cycles": True
        }

        response = await test_client.post(
            "/mcp/tools/traverse_call_graph",
            json=request
        )

        assert response.status_code == 200
        data = response.json()

        # Contract assertions
        assert "source_node" in data
        assert data["source_node"] == "sys_open"
        assert "target_node" in data
        assert "paths_found" in data
        assert "shortest_path_length" in data
        assert "has_cycles" in data
        assert "paths" in data
        assert isinstance(data["paths"], list)

        if data["paths"]:
            path = data["paths"][0]
            assert "depth" in path
            assert "nodes" in path
            assert "edges" in path
            assert len(path["edges"]) == len(path["nodes"]) - 1

            node = path["nodes"][0]
            assert "function" in node
            assert "file" in node
            assert "line" in node

    @pytest.mark.asyncio
    async def test_traverse_with_cycles(self, test_client: AsyncClient):
        """Test cycle detection in call graph."""
        request = {
            "source": "recursive_function",
            "direction": "forward",
            "max_depth": 10,
            "detect_cycles": True
        }

        response = await test_client.post(
            "/mcp/tools/traverse_call_graph",
            json=request
        )

        assert response.status_code == 200
        data = response.json()

        if data["has_cycles"]:
            assert "cycle_nodes" in data
            assert isinstance(data["cycle_nodes"], list)
            assert len(data["cycle_nodes"]) > 0


class TestExportGraph:
    """Contract tests for /mcp/tools/export_graph endpoint."""

    @pytest.mark.asyncio
    async def test_export_json_graph(self, test_client: AsyncClient):
        """Test graph export in JSON format."""
        request = {
            "format": "json_graph",
            "scope": "subgraph",
            "root_function": "vfs_read",
            "max_depth": 3,
            "chunk_size_mb": 100
        }

        response = await test_client.post(
            "/mcp/tools/export_graph",
            json=request
        )

        assert response.status_code == 200
        data = response.json()

        # Contract assertions
        assert "export_id" in data
        assert "format" in data
        assert data["format"] == "json_graph"
        assert "node_count" in data
        assert data["node_count"] > 0
        assert "edge_count" in data
        assert "file_size_bytes" in data
        assert "chunk_count" in data
        assert data["chunk_count"] >= 1
        assert "chunks" in data
        assert len(data["chunks"]) == data["chunk_count"]

        chunk = data["chunks"][0]
        assert "chunk_index" in chunk
        assert "chunk_url" in chunk
        assert "size_bytes" in chunk
        assert chunk["size_bytes"] <= 100 * 1024 * 1024  # Under 100MB

    @pytest.mark.asyncio
    async def test_export_graphml(self, test_client: AsyncClient):
        """Test graph export in GraphML format."""
        request = {
            "format": "graphml",
            "scope": "full",
            "compression": "gzip"
        }

        response = await test_client.post(
            "/mcp/tools/export_graph",
            json=request
        )

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "graphml"
        assert "metadata" in data
        assert "export_time_ms" in data["metadata"]
        assert "memory_used_mb" in data["metadata"]


@pytest.fixture
async def test_client():
    """Create test client with mocked dependencies."""
    from kcs_mcp.server import app
    from httpx import AsyncClient

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client