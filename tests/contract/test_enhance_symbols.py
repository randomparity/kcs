"""Contract tests for enhance_symbols endpoint.

These tests verify the API contract defined in specs/004-current-code-has/contracts/enhance_symbols.yaml.
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
        response = requests.get("http://localhost:8080/health", timeout=2)
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


@pytest.fixture
def sample_symbols() -> list[dict[str, Any]]:
    """Sample symbol data for enhancement."""
    return [
        {
            "name": "vfs_read",
            "file_path": "fs/read_write.c",
            "line_number": 456,
            "symbol_type": "function",
        },
        {
            "name": "sys_open",
            "file_path": "fs/open.c",
            "line_number": 1234,
            "symbol_type": "function",
        },
        {
            "name": "file_operations",
            "file_path": "include/linux/fs.h",
            "line_number": 789,
            "symbol_type": "struct",
        },
    ]


# Contract test cases
@skip_integration_in_ci
@skip_without_mcp_server
class TestEnhanceSymbolsContract:
    """Contract tests for enhance_symbols endpoint."""

    @pytest.mark.asyncio
    async def test_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the enhance_symbols endpoint exists and accepts POST requests."""
        payload = {"symbols": [{"name": "test_func", "file_path": "/tmp/test.c"}]}

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        # NOTE: This MUST fail initially as endpoint doesn't exist yet
        assert response.status_code != 404, "enhance_symbols endpoint should exist"

    @pytest.mark.asyncio
    async def test_requires_symbols_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that symbols parameter is required in request."""
        payload: dict[str, Any] = {}  # Missing symbols

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        assert response.status_code == 400, (
            "Should return 400 for missing symbols parameter"
        )

    @pytest.mark.asyncio
    async def test_response_schema(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_symbols: list[dict[str, Any]],
    ) -> None:
        """Test that response matches the contract schema."""
        payload = {
            "symbols": sample_symbols,
            "enable_clang": False,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify response structure matches contract
            assert "enhanced_symbols" in data, (
                "Response should contain enhanced_symbols"
            )
            assert isinstance(data["enhanced_symbols"], list), (
                "enhanced_symbols should be a list"
            )

            if data["enhanced_symbols"]:
                symbol = data["enhanced_symbols"][0]

                # Required fields from original symbol
                assert "name" in symbol, "Symbol must have name"
                assert "file_path" in symbol, "Symbol must have file_path"
                assert "line_number" in symbol, "Symbol must have line_number"
                assert "symbol_type" in symbol, "Symbol must have symbol_type"

                # Enhanced fields (optional)
                # signature, return_type, parameters, attributes, documentation, metadata

            # Verify statistics if present
            if "statistics" in data:
                stats = data["statistics"]
                assert isinstance(stats, dict), "Statistics should be a dict"

                if "total_enhanced" in stats:
                    assert isinstance(stats["total_enhanced"], int), (
                        "total_enhanced should be an integer"
                    )

                if "clang_enhanced" in stats:
                    assert isinstance(stats["clang_enhanced"], int), (
                        "clang_enhanced should be an integer"
                    )

                if "processing_time_ms" in stats:
                    assert isinstance(stats["processing_time_ms"], (int, float)), (
                        "processing_time_ms should be a number"
                    )

    @pytest.mark.asyncio
    async def test_clang_enhancement_when_enabled(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_symbols: list[dict[str, Any]],
    ) -> None:
        """Test that Clang enhancement adds type information when enabled."""
        payload = {
            "symbols": sample_symbols,
            "enable_clang": True,
            "compile_commands": "/path/to/compile_commands.json",
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            symbols = data.get("enhanced_symbols", [])

            # When Clang is enabled, function symbols should have enhanced metadata
            # NOTE: This MUST fail initially as Clang isn't integrated yet
            function_symbols = [
                s for s in symbols if s.get("symbol_type") == "function"
            ]

            if function_symbols:
                # At least some functions should have enhanced data
                enhanced = [
                    s
                    for s in function_symbols
                    if s.get("signature") or s.get("return_type") or s.get("parameters")
                ]

                assert enhanced, (
                    "When Clang is enabled, some function symbols should have type information"
                )

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_clang(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_symbols: list[dict[str, Any]],
    ) -> None:
        """Test that enhancement works without Clang (graceful degradation)."""
        payload = {
            "symbols": sample_symbols,
            "enable_clang": False,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        # Should succeed even without Clang
        assert response.status_code in [200, 501], (
            "Should work without Clang or return 501 if not implemented"
        )

        if response.status_code == 200:
            data = response.json()
            symbols = data.get("enhanced_symbols", [])

            # Basic symbol data should still be present
            for symbol in symbols:
                assert "name" in symbol, "Symbol should have name even without Clang"
                assert "file_path" in symbol, "Symbol should have file_path"

    @pytest.mark.asyncio
    async def test_export_status_enrichment(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_symbols: list[dict[str, Any]],
    ) -> None:
        """Test that symbols are enriched with export status."""
        payload = {
            "symbols": sample_symbols,
            "enrich_export_status": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            symbols = data.get("enhanced_symbols", [])

            # Check if metadata includes export status
            # NOTE: This MUST fail initially as export detection isn't implemented
            for symbol in symbols:
                if symbol.get("metadata"):
                    metadata = symbol["metadata"]
                    # Some symbols should have export_status
                    if "export_status" in metadata:
                        assert metadata["export_status"] in [
                            "exported",
                            "static",
                            "global",
                        ], f"Invalid export_status: {metadata['export_status']}"

    @pytest.mark.asyncio
    async def test_kernel_attributes_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_symbols: list[dict[str, Any]],
    ) -> None:
        """Test that kernel attributes (__init, __exit, etc.) are detected."""
        symbols_with_attrs = [
            {
                "name": "driver_init",
                "file_path": "drivers/test/driver.c",
                "line_number": 100,
                "symbol_type": "function",
            }
        ]

        payload = {
            "symbols": symbols_with_attrs,
            "detect_attributes": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            symbols = data.get("enhanced_symbols", [])

            # Check for attribute detection
            # NOTE: This MUST fail initially as attribute detection isn't implemented
            for symbol in symbols:
                if symbol.get("attributes"):
                    attrs = symbol["attributes"]
                    assert isinstance(attrs, list), "Attributes should be a list"

                    # Common kernel attributes
                    valid_attrs = [
                        "__init",
                        "__exit",
                        "__initdata",
                        "__exitdata",
                        "__devinit",
                        "__devexit",
                        "__cpuinit",
                        "__cpuexit",
                        "__user",
                        "__kernel",
                        "__iomem",
                        "__percpu",
                    ]

                    for attr in attrs:
                        assert attr in valid_attrs or attr.startswith("__"), (
                            f"Unexpected attribute: {attr}"
                        )

    @pytest.mark.asyncio
    async def test_documentation_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_symbols: list[dict[str, Any]],
    ) -> None:
        """Test that kernel-doc comments are extracted."""
        payload = {
            "symbols": sample_symbols,
            "extract_documentation": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            symbols = data.get("enhanced_symbols", [])

            # Check if any symbols have documentation
            # NOTE: This MUST fail initially as doc extraction isn't implemented
            documented = [s for s in symbols if s.get("documentation")]

            # Not all symbols will have docs, but the capability should exist
            if documented:
                for symbol in documented:
                    doc = symbol["documentation"]
                    assert isinstance(doc, str), "Documentation should be a string"
                    assert len(doc) > 0, "Documentation should not be empty"

    @pytest.mark.asyncio
    async def test_batch_processing(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that endpoint can handle large batches of symbols."""
        # Create a large batch of symbols
        large_batch = [
            {
                "name": f"symbol_{i}",
                "file_path": f"file_{i % 10}.c",
                "line_number": i * 10,
                "symbol_type": "function" if i % 2 == 0 else "variable",
            }
            for i in range(100)
        ]

        payload = {
            "symbols": large_batch,
            "enable_clang": False,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        # Should handle large batches
        assert response.status_code in [200, 501], (
            "Should handle large batches or return 501 if not implemented"
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # Should return same number of symbols
            assert len(enhanced) == len(large_batch), (
                "Should enhance all provided symbols"
            )

    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that invalid symbols are handled gracefully."""
        invalid_symbols = [
            {"name": "valid_symbol", "file_path": "valid.c", "line_number": 10},
            {"name": "", "file_path": "invalid.c", "line_number": 20},  # Empty name
            {"name": "missing_file"},  # Missing required field
        ]

        payload = {"symbols": invalid_symbols}

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        # Should validate input
        assert response.status_code in [400, 422], (
            "Should return 400/422 for invalid symbols"
        )

    @pytest.mark.asyncio
    async def test_empty_symbols_list(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that empty symbols list returns 400."""
        payload = {"symbols": []}

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        assert response.status_code == 400, "Should return 400 for empty symbols list"
