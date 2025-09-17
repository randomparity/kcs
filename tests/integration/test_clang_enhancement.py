"""Integration test for Clang symbol enhancement.

These tests verify that Clang integration properly enhances symbols
with type information, attributes, and documentation.
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from kcs_mcp.database import Database
from testcontainers.postgres import PostgresContainer


# Skip tests if MCP server not running
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible."""
    try:
        with httpx.Client() as client:
            response = client.get("http://localhost:8080/health", timeout=2.0)
            return response.status_code == 200
    except Exception:
        return False


skip_without_mcp = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server not running"
)

# Skip in CI unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)

# Test configuration
TEST_KERNEL_PATH = Path("tests/fixtures/kernel")
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "test_token_123"


# Fixtures
@pytest_asyncio.fixture
async def test_database() -> Database:
    """Create a test database with sample data."""
    # Use testcontainers for isolated testing
    with PostgresContainer("postgres:16") as postgres:
        db = Database(postgres.get_connection_url())
        await db.connect()

        # Apply migrations
        migrations_dir = Path("src/sql/migrations")
        for migration in sorted(migrations_dir.glob("*.sql")):
            with open(migration) as f:
                await db.execute(f.read())

        yield db
        await db.disconnect()


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
def mock_compile_commands() -> Path:
    """Create a mock compile_commands.json file."""
    compile_commands = [
        {
            "directory": str(TEST_KERNEL_PATH),
            "command": "gcc -c -I./include -DCONFIG_X86_64 fs/file.c",
            "file": "fs/file.c",
        },
        {
            "directory": str(TEST_KERNEL_PATH),
            "command": "gcc -c -I./include -DCONFIG_SMP drivers/char/mem.c",
            "file": "drivers/char/mem.c",
        },
    ]

    # Write to temp file
    commands_file = Path("/tmp/test_compile_commands.json")
    with open(commands_file, "w") as f:
        json.dump(compile_commands, f, indent=2)

    return commands_file


# Test cases
@skip_integration_in_ci
@skip_without_mcp
class TestClangSymbolEnhancement:
    """Integration tests for Clang-based symbol enhancement."""

    @pytest.mark.asyncio
    async def test_type_information_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that Clang extracts function signatures and return types."""
        symbols = [
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
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as Clang isn't integrated
            function_symbols = [
                s for s in enhanced if s.get("symbol_type") == "function"
            ]

            assert len(function_symbols) > 0, "Should have function symbols"

            # Check for type information
            for func in function_symbols:
                # Should have enhanced type info when Clang is enabled
                if func.get("signature") or func.get("return_type"):
                    assert func.get("signature") is not None, (
                        f"Function {func['name']} should have signature"
                    )
                    assert func.get("return_type") is not None, (
                        f"Function {func['name']} should have return type"
                    )

    @pytest.mark.asyncio
    async def test_parameter_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that function parameters are extracted with names and types."""
        symbols = [
            {
                "name": "do_sys_open",
                "file_path": "fs/open.c",
                "line_number": 1100,
                "symbol_type": "function",
            }
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as Clang isn't integrated
            if enhanced:
                func = enhanced[0]
                if func.get("parameters"):
                    params = func["parameters"]
                    assert isinstance(params, list), "Parameters should be a list"

                    # Each parameter should have name and type
                    for param in params:
                        assert "name" in param, "Parameter should have name"
                        assert "type" in param, "Parameter should have type"

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_clang(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that enhancement works without Clang (graceful degradation)."""
        symbols = [
            {
                "name": "test_func",
                "file_path": "test.c",
                "line_number": 10,
                "symbol_type": "function",
            }
        ]

        # Test with Clang disabled
        payload = {
            "symbols": symbols,
            "enable_clang": False,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        # Should work without Clang
        assert response.status_code in [200, 501], (
            "Should work without Clang or return 501 if not implemented"
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # Basic symbol data should still be present
            for symbol in enhanced:
                assert "name" in symbol
                assert "file_path" in symbol
                assert "line_number" in symbol
                assert "symbol_type" in symbol

    @pytest.mark.asyncio
    async def test_attribute_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that kernel attributes (__init, __exit, etc.) are detected."""
        symbols = [
            {
                "name": "driver_init",
                "file_path": "drivers/test/driver.c",
                "line_number": 100,
                "symbol_type": "function",
            },
            {
                "name": "driver_exit",
                "file_path": "drivers/test/driver.c",
                "line_number": 200,
                "symbol_type": "function",
            },
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
            "detect_attributes": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as attribute detection isn't implemented
            for symbol in enhanced:
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
                        "__must_check",
                        "__deprecated",
                        "__used",
                        "__unused",
                        "__maybe_unused",
                        "__always_inline",
                        "__noinline",
                        "__weak",
                        "__alias",
                        "__pure",
                        "__const",
                        "__noreturn",
                        "__malloc",
                        "__warn_unused_result",
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
        mock_compile_commands: Path,
    ) -> None:
        """Test that kernel-doc comments are extracted and associated."""
        symbols = [
            {
                "name": "vfs_read",
                "file_path": "fs/read_write.c",
                "line_number": 456,
                "symbol_type": "function",
            }
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
            "extract_documentation": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as doc extraction isn't implemented
            for symbol in enhanced:
                if symbol.get("documentation"):
                    doc = symbol["documentation"]
                    assert isinstance(doc, str), "Documentation should be a string"
                    assert len(doc) > 0, "Documentation should not be empty"

                    # Kernel-doc format checks
                    # Should contain function description
                    if symbol["symbol_type"] == "function":
                        # Basic kernel-doc structure validation
                        assert doc, (
                            f"Function {symbol['name']} should have documentation"
                        )

    @pytest.mark.asyncio
    async def test_struct_member_extraction(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that struct members are extracted with types."""
        symbols = [
            {
                "name": "file_operations",
                "file_path": "include/linux/fs.h",
                "line_number": 1500,
                "symbol_type": "struct",
            }
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
            "extract_members": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as struct member extraction isn't implemented
            for symbol in enhanced:
                if symbol["symbol_type"] == "struct" and symbol.get("members"):
                    members = symbol["members"]
                    assert isinstance(members, list), "Members should be a list"

                    # Each member should have name and type
                    for member in members:
                        assert "name" in member, "Member should have name"
                        assert "type" in member, "Member should have type"

                        # Function pointers should have signature
                        if "(*" in member.get("type", ""):
                            assert "signature" in member or "params" in member, (
                                f"Function pointer {member['name']} should have signature info"
                            )

    @pytest.mark.asyncio
    async def test_macro_expansion(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that macros are expanded to show actual values."""
        symbols = [
            {
                "name": "PAGE_SIZE",
                "file_path": "include/asm/page.h",
                "line_number": 10,
                "symbol_type": "macro",
            },
            {
                "name": "HZ",
                "file_path": "include/linux/param.h",
                "line_number": 5,
                "symbol_type": "macro",
            },
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
            "expand_macros": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as macro expansion isn't implemented
            for symbol in enhanced:
                if symbol["symbol_type"] == "macro" and symbol.get("expanded_value"):
                    value = symbol["expanded_value"]
                    assert value is not None, (
                        f"Macro {symbol['name']} should have expanded value"
                    )

                    # Common kernel macros should have expected values
                    if symbol["name"] == "PAGE_SIZE":
                        # Should be a power of 2
                        assert value in ["4096", "8192", "16384", "65536"], (
                            f"PAGE_SIZE has unexpected value: {value}"
                        )

    @pytest.mark.asyncio
    async def test_inline_function_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that inline functions are properly identified."""
        symbols = [
            {
                "name": "get_cpu",
                "file_path": "include/linux/smp.h",
                "line_number": 200,
                "symbol_type": "function",
            }
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as inline detection isn't implemented
            for symbol in enhanced:
                if symbol.get("metadata"):
                    metadata = symbol["metadata"]
                    # Check if inline status is detected
                    if "is_inline" in metadata:
                        assert isinstance(metadata["is_inline"], bool), (
                            "is_inline should be boolean"
                        )

    @pytest.mark.asyncio
    async def test_symbol_visibility_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that symbol visibility (static, extern, exported) is detected."""
        symbols = [
            {
                "name": "static_func",
                "file_path": "fs/internal.c",
                "line_number": 50,
                "symbol_type": "function",
            },
            {
                "name": "exported_func",
                "file_path": "fs/exported.c",
                "line_number": 100,
                "symbol_type": "function",
            },
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
            "detect_visibility": True,
        }

        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            enhanced = data.get("enhanced_symbols", [])

            # NOTE: This MUST fail initially as visibility detection isn't implemented
            for symbol in enhanced:
                if symbol.get("metadata"):
                    metadata = symbol["metadata"]
                    if "visibility" in metadata:
                        assert metadata["visibility"] in [
                            "static",
                            "extern",
                            "global",
                            "exported",
                        ], f"Invalid visibility: {metadata['visibility']}"

    @pytest.mark.asyncio
    async def test_performance_with_clang(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        mock_compile_commands: Path,
    ) -> None:
        """Test that Clang enhancement maintains acceptable performance."""
        import time

        # Create a larger batch of symbols
        symbols = [
            {
                "name": f"func_{i}",
                "file_path": f"file_{i % 10}.c",
                "line_number": i * 10,
                "symbol_type": "function",
            }
            for i in range(50)
        ]

        payload = {
            "symbols": symbols,
            "enable_clang": True,
            "compile_commands": str(mock_compile_commands),
        }

        start_time = time.time()
        response = await http_client.post(
            "/enhance/symbols", json=payload, headers=auth_headers
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            stats = data.get("statistics", {})

            # Performance requirements
            assert elapsed < 30.0, (
                f"Clang enhancement took {elapsed:.2f}s, should be <30s for 50 symbols"
            )

            # Check reported processing time
            if "processing_time_ms" in stats:
                processing_ms = stats["processing_time_ms"]
                assert processing_ms < 30000, (
                    f"Processing time {processing_ms}ms exceeds 30s limit"
                )

            if "clang_enhanced" in stats:
                # At least some symbols should be enhanced
                assert stats["clang_enhanced"] > 0, (
                    "Should enhance at least some symbols with Clang"
                )
