"""Contract tests for detect_patterns endpoint.

These tests verify the API contract defined in specs/004-current-code-has/contracts/detect_patterns.yaml.
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
def sample_kernel_files() -> list[str]:
    """List of sample kernel files for testing."""
    return [
        "tests/fixtures/kernel/fs/file.c",
        "tests/fixtures/kernel/drivers/char/mem.c",
        "tests/fixtures/kernel/net/socket.c",
    ]


# Contract test cases
@skip_integration_in_ci
@skip_without_mcp_server
class TestDetectPatternsContract:
    """Contract tests for detect_patterns endpoint."""

    @pytest.mark.asyncio
    async def test_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that the detect_patterns endpoint exists and accepts POST requests."""
        payload = {"files": ["/tmp/test.c"]}

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        # NOTE: This MUST fail initially as endpoint doesn't exist yet
        assert response.status_code != 404, "detect_patterns endpoint should exist"

    @pytest.mark.asyncio
    async def test_requires_files_parameter(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test that files parameter is required in request."""
        payload: dict[str, Any] = {}  # Missing files

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        assert response.status_code == 400, (
            "Should return 400 for missing files parameter"
        )

    @pytest.mark.asyncio
    async def test_response_schema(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_files: list[str],
    ) -> None:
        """Test that response matches the contract schema."""
        payload = {
            "files": sample_kernel_files,
            "pattern_types": ["ExportSymbol", "ModuleParam"],
        }

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify response structure matches contract
            assert "patterns" in data, "Response should contain patterns"
            assert isinstance(data["patterns"], list), "patterns should be a list"

            if data["patterns"]:
                pattern = data["patterns"][0]

                # Required fields
                assert "pattern_type" in pattern, "Pattern must have pattern_type"
                assert "file_path" in pattern, "Pattern must have file_path"
                assert "line_number" in pattern, "Pattern must have line_number"
                assert "raw_text" in pattern, "Pattern must have raw_text"

                # Optional fields
                # symbol_name, metadata are optional

                # Validate pattern_type enum
                valid_types = [
                    "ExportSymbol",
                    "ExportSymbolGPL",
                    "ExportSymbolNS",
                    "ModuleParam",
                    "ModuleParamArray",
                    "ModuleParmDesc",
                    "EarlyParam",
                    "CoreParam",
                    "SetupParam",
                ]
                assert pattern["pattern_type"] in valid_types, (
                    f"Invalid pattern_type: {pattern['pattern_type']}"
                )

            # Verify statistics if present
            if "statistics" in data:
                stats = data["statistics"]
                assert isinstance(stats, dict), "Statistics should be a dict"

                if "total_patterns" in stats:
                    assert isinstance(stats["total_patterns"], int), (
                        "total_patterns should be an integer"
                    )

                if "by_type" in stats:
                    assert isinstance(stats["by_type"], dict), (
                        "by_type should be a dict"
                    )

                if "files_processed" in stats:
                    assert isinstance(stats["files_processed"], int), (
                        "files_processed should be an integer"
                    )

                if "processing_time_ms" in stats:
                    assert isinstance(stats["processing_time_ms"], (int, float)), (
                        "processing_time_ms should be a number"
                    )

    @pytest.mark.asyncio
    async def test_export_symbol_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_files: list[str],
    ) -> None:
        """Test that EXPORT_SYMBOL patterns are detected."""
        payload = {
            "files": sample_kernel_files,
            "pattern_types": ["ExportSymbol", "ExportSymbolGPL"],
        }

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            patterns = data.get("patterns", [])

            # Check if EXPORT_SYMBOL patterns are detected
            # NOTE: This MUST fail initially as patterns aren't implemented yet
            export_patterns = [
                p
                for p in patterns
                if p["pattern_type"] in ["ExportSymbol", "ExportSymbolGPL"]
            ]

            assert export_patterns, (
                "Should detect EXPORT_SYMBOL patterns in kernel files"
            )

            # Verify pattern has symbol name in metadata
            for pattern in export_patterns:
                if pattern.get("metadata"):
                    assert "symbol_name" in pattern["metadata"], (
                        "EXPORT_SYMBOL pattern should include symbol name"
                    )

    @pytest.mark.asyncio
    async def test_module_param_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_files: list[str],
    ) -> None:
        """Test that module_param patterns are detected."""
        payload = {
            "files": sample_kernel_files,
            "pattern_types": ["ModuleParam", "ModuleParamArray", "ModuleParmDesc"],
        }

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            patterns = data.get("patterns", [])

            # Check if module_param patterns are detected
            # NOTE: This MUST fail initially as patterns aren't implemented yet
            param_patterns = [
                p
                for p in patterns
                if p["pattern_type"] in ["ModuleParam", "ModuleParamArray"]
            ]

            assert param_patterns, "Should detect module_param patterns in kernel files"

            # Verify pattern has parameter metadata
            for pattern in param_patterns:
                if pattern.get("metadata"):
                    # Should have parameter name and type
                    assert (
                        "param_name" in pattern["metadata"]
                        or "name" in pattern["metadata"]
                    ), "module_param pattern should include parameter name"

    @pytest.mark.asyncio
    async def test_module_parm_desc_association(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_files: list[str],
    ) -> None:
        """Test that MODULE_PARM_DESC is associated with module_param."""
        payload = {
            "files": sample_kernel_files,
            "pattern_types": ["ModuleParam", "ModuleParmDesc"],
            "associate_descriptions": True,
        }

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            patterns = data.get("patterns", [])

            # Find module params with descriptions
            param_patterns = [p for p in patterns if p["pattern_type"] == "ModuleParam"]
            desc_patterns = [
                p for p in patterns if p["pattern_type"] == "ModuleParmDesc"
            ]

            # If we have both params and descriptions, some should be associated
            if param_patterns and desc_patterns:
                params_with_desc = [
                    p
                    for p in param_patterns
                    if "metadata" in p
                    and p["metadata"]
                    and "description" in p["metadata"]
                ]

                assert params_with_desc, (
                    "Some module_param patterns should have associated descriptions"
                )

    @pytest.mark.asyncio
    async def test_boot_parameter_patterns(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_files: list[str],
    ) -> None:
        """Test that boot parameter patterns are detected."""
        payload = {
            "files": sample_kernel_files,
            "pattern_types": ["EarlyParam", "CoreParam", "SetupParam"],
        }

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            patterns = data.get("patterns", [])

            # Check if any boot parameter patterns are detected
            # NOTE: This MUST fail initially as patterns aren't implemented yet
            boot_patterns = [
                p
                for p in patterns
                if p["pattern_type"] in ["EarlyParam", "CoreParam", "SetupParam"]
            ]

            # Boot parameters may not be in all files, so check statistics instead
            if "statistics" in data and "by_type" in data["statistics"]:
                boot_types = [
                    t
                    for t in ["EarlyParam", "CoreParam", "SetupParam"]
                    if t in data["statistics"]["by_type"]
                ]
                # At least one type should be supported
                assert boot_types or boot_patterns, (
                    "Should support boot parameter pattern detection"
                )

    @pytest.mark.asyncio
    async def test_filter_by_pattern_types(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        sample_kernel_files: list[str],
    ) -> None:
        """Test that pattern_types filter works correctly."""
        # Request only ExportSymbol patterns
        payload = {"files": sample_kernel_files, "pattern_types": ["ExportSymbol"]}

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            patterns = data.get("patterns", [])

            # All returned patterns should be ExportSymbol type
            for pattern in patterns:
                assert pattern["pattern_type"] == "ExportSymbol", (
                    f"Expected only ExportSymbol, got {pattern['pattern_type']}"
                )

    @pytest.mark.asyncio
    async def test_invalid_file_path_returns_404(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that invalid file paths return 404."""
        payload = {"files": ["/nonexistent/file.c"]}

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        assert response.status_code == 404, "Should return 404 for non-existent files"

    @pytest.mark.asyncio
    async def test_empty_files_list_returns_400(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that empty files list returns 400."""
        payload = {"files": []}

        response = await http_client.post(
            "/detect/patterns", json=payload, headers=auth_headers
        )

        assert response.status_code == 400, "Should return 400 for empty files list"
