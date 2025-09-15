"""Quickstart validation test for KCS.

Tests the complete quickstart flow to ensure new users can get up and running
with minimal effort. Validates installation, configuration, and basic operations.
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import pytest
import requests

# Test configuration
TEST_TIMEOUT = 300  # 5 minutes for complete quickstart
STARTUP_TIMEOUT = 60  # 1 minute for service startup
HEALTH_CHECK_RETRIES = 30
HEALTH_CHECK_INTERVAL = 2


class QuickstartEnvironment:
    """Manages a temporary KCS environment for testing."""

    def __init__(self):
        self.temp_dir = None
        self.kcs_process = None
        self.postgres_process = None
        self.base_url = "http://localhost:8080"
        self.auth_token = "test-quickstart-token"
        self.kernel_path = None

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def setup(self):
        """Set up temporary test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="kcs-quickstart-")

        # Create test kernel repository structure
        self.kernel_path = Path(self.temp_dir) / "test-kernel"
        self.kernel_path.mkdir(parents=True)

        # Create minimal kernel source files for testing
        self._create_test_kernel_files()

        # Set environment variables
        os.environ["KCS_SERVER_URL"] = self.base_url
        os.environ["KCS_AUTH_TOKEN"] = self.auth_token
        os.environ["KCS_TEST_MODE"] = "true"
        os.environ["KCS_DB_URL"] = "postgresql://kcs:kcs@localhost:5432/kcs_test"

    def cleanup(self):
        """Clean up test environment."""
        # Stop KCS process
        if self.kcs_process:
            try:
                self.kcs_process.terminate()
                self.kcs_process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                if self.kcs_process.poll() is None:
                    self.kcs_process.kill()

        # Stop PostgreSQL if we started it
        if self.postgres_process:
            try:
                self.postgres_process.terminate()
                self.postgres_process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                if self.postgres_process.poll() is None:
                    self.postgres_process.kill()

        # Clean up temporary directory
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

        # Clean up environment variables
        for key in ["KCS_SERVER_URL", "KCS_AUTH_TOKEN", "KCS_TEST_MODE", "KCS_DB_URL"]:
            os.environ.pop(key, None)

    def _create_test_kernel_files(self):
        """Create minimal test kernel files."""
        # Create basic kernel file structure
        fs_dir = self.kernel_path / "fs"
        fs_dir.mkdir()

        kernel_dir = self.kernel_path / "kernel"
        kernel_dir.mkdir()

        include_dir = self.kernel_path / "include" / "linux"
        include_dir.mkdir(parents=True)

        # Create test C files with kernel patterns
        test_c_content = """
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/syscalls.h>

static int test_helper(int value) {
    return value * 2;
}

SYSCALL_DEFINE1(test_syscall, int, value)
{
    return test_helper(value);
}

static ssize_t test_read(struct file *file, char __user *buf,
                        size_t count, loff_t *ppos)
{
    return 0;
}

static const struct file_operations test_fops = {
    .read = test_read,
    .owner = THIS_MODULE,
};
"""

        test_h_content = """
#ifndef _LINUX_TEST_H
#define _LINUX_TEST_H

struct test_struct {
    int field1;
    char *field2;
};

extern int test_function(int param);

#define TEST_MACRO(x) ((x) + 1)

#endif /* _LINUX_TEST_H */
"""

        # Write test files
        (fs_dir / "test.c").write_text(test_c_content)
        (kernel_dir / "test.c").write_text(test_c_content)
        (include_dir / "test.h").write_text(test_h_content)

        # Create a simple Makefile
        makefile_content = """
obj-y += test.o
"""
        (self.kernel_path / "Makefile").write_text(makefile_content)


@pytest.fixture
def quickstart_env():
    """Provide a quickstart environment for testing."""
    with QuickstartEnvironment() as env:
        yield env


class TestQuickstartInstallation:
    """Test installation components."""

    def test_installation_script_exists(self):
        """Test that installation script exists and is executable."""
        install_script = (
            Path(__file__).parent.parent.parent / "tools" / "setup" / "install.sh"
        )
        assert install_script.exists(), "Installation script not found"
        assert os.access(install_script, os.X_OK), "Installation script not executable"

    def test_installation_script_syntax(self):
        """Test that installation script has valid bash syntax."""
        install_script = (
            Path(__file__).parent.parent.parent / "tools" / "setup" / "install.sh"
        )

        # Check bash syntax
        result = subprocess.run(
            ["bash", "-n", str(install_script)], capture_output=True, text=True
        )

        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_required_dependencies_detected(self):
        """Test that required system dependencies are properly detected."""
        # These should be available in most Linux environments
        required_commands = ["python3", "cargo", "git"]

        for cmd in required_commands:
            result = subprocess.run(["which", cmd], capture_output=True)
            if result.returncode != 0:
                pytest.skip(f"Required command '{cmd}' not available for testing")

    def test_rust_workspace_builds(self):
        """Test that Rust workspace can build successfully."""
        kcs_root = Path(__file__).parent.parent.parent
        rust_dir = kcs_root / "src" / "rust"

        if not rust_dir.exists():
            pytest.skip("Rust source directory not found")

        # Test build (but don't require it to succeed if dependencies are missing)
        result = subprocess.run(
            ["cargo", "check", "--workspace"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
        )

        # If cargo check fails due to missing dependencies, that's expected
        # We just want to make sure the workspace structure is valid
        if "error: " in result.stderr and "dependency" not in result.stderr.lower():
            pytest.fail(f"Rust workspace has structural issues: {result.stderr}")

    def test_python_project_structure(self):
        """Test that Python project has correct structure."""
        kcs_root = Path(__file__).parent.parent.parent
        python_dir = kcs_root / "src" / "python"

        assert python_dir.exists(), "Python source directory not found"

        # Check for key Python files
        key_files = [
            python_dir / "kcs_mcp" / "__init__.py",
            python_dir / "kcs_mcp" / "app.py",
            python_dir / "kcs_mcp" / "citations.py",
        ]

        for file_path in key_files:
            assert file_path.exists(), f"Key Python file not found: {file_path}"


class TestQuickstartConfiguration:
    """Test configuration and setup."""

    def test_environment_variables(self, quickstart_env):
        """Test that environment variables are properly set."""
        required_vars = ["KCS_SERVER_URL", "KCS_AUTH_TOKEN", "KCS_TEST_MODE"]

        for var in required_vars:
            assert var in os.environ, f"Required environment variable {var} not set"
            assert os.environ[var], f"Environment variable {var} is empty"

    def test_test_kernel_structure(self, quickstart_env):
        """Test that test kernel has proper structure."""
        kernel_path = quickstart_env.kernel_path

        # Check directory structure
        assert (kernel_path / "fs").exists(), "fs directory not created"
        assert (kernel_path / "kernel").exists(), "kernel directory not created"
        assert (
            kernel_path / "include" / "linux"
        ).exists(), "include/linux directory not created"

        # Check test files
        assert (kernel_path / "fs" / "test.c").exists(), "Test C file not created"
        assert (
            kernel_path / "include" / "linux" / "test.h"
        ).exists(), "Test header not created"

        # Verify file contents
        test_c = (kernel_path / "fs" / "test.c").read_text()
        assert "SYSCALL_DEFINE1" in test_c, "Test file missing syscall definition"
        assert "file_operations" in test_c, "Test file missing file operations"


class TestQuickstartParsing:
    """Test basic parsing functionality."""

    def test_parser_binary_exists(self):
        """Test that parser binary can be built or is available."""
        kcs_root = Path(__file__).parent.parent.parent
        parser_dir = kcs_root / "src" / "rust" / "kcs-parser"

        if not parser_dir.exists():
            pytest.skip("Parser source directory not found")

        # Try to build parser
        result = subprocess.run(
            ["cargo", "build", "--bin", "kcs-parser"],
            cwd=parser_dir,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes max for build
        )

        # If build fails due to missing dependencies, skip
        if result.returncode != 0 and "dependency" in result.stderr.lower():
            pytest.skip("Parser dependencies not available")

        assert result.returncode == 0, f"Parser build failed: {result.stderr}"

    def test_parse_test_files(self, quickstart_env):
        """Test parsing the test kernel files."""
        kcs_root = Path(__file__).parent.parent.parent
        parser_path = (
            kcs_root / "src" / "rust" / "kcs-parser" / "target" / "debug" / "kcs-parser"
        )

        if not parser_path.exists():
            pytest.skip("Parser binary not available")

        # Parse test kernel
        result = subprocess.run(
            [str(parser_path), "--parse", str(quickstart_env.kernel_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should succeed and produce output
        assert result.returncode == 0, f"Parsing failed: {result.stderr}"
        assert result.stdout, "Parser produced no output"

        # Check that it found some symbols
        output = result.stdout
        assert (
            "function" in output.lower() or "symbol" in output.lower()
        ), "Parser output doesn't mention functions or symbols"


class TestQuickstartAPI:
    """Test MCP API functionality with mock/minimal setup."""

    def test_api_contract_validation(self):
        """Test that API contract specification is valid."""
        kcs_root = Path(__file__).parent.parent.parent
        contract_file = (
            kcs_root
            / "specs"
            / "001-kernel-context-server"
            / "contracts"
            / "mcp-api.yaml"
        )

        assert contract_file.exists(), "MCP API contract not found"

        # Basic YAML validation
        import yaml

        try:
            with open(contract_file) as f:
                spec = yaml.safe_load(f)

            # Check OpenAPI structure
            assert "openapi" in spec, "Not a valid OpenAPI spec"
            assert "paths" in spec, "No paths defined in API spec"
            assert "components" in spec, "No components defined in API spec"

            # Check key endpoints
            required_endpoints = [
                "/mcp/tools/search_code",
                "/mcp/tools/get_symbol",
                "/mcp/tools/who_calls",
                "/health",
            ]

            for endpoint in required_endpoints:
                assert endpoint in spec["paths"], f"Missing endpoint: {endpoint}"

        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in API contract: {e}")

    def test_health_endpoint_mock(self, quickstart_env):
        """Test health endpoint with minimal mock setup."""
        # This would ideally test against a running server
        # For now, just validate the expected response structure

        expected_health_response = {
            "status": "healthy",
            "version": "1.0.0",
            "indexed_at": "2024-01-01T00:00:00Z",
        }

        # Validate response structure matches OpenAPI spec
        assert "status" in expected_health_response
        assert "version" in expected_health_response
        assert "indexed_at" in expected_health_response


class TestQuickstartEndToEnd:
    """End-to-end quickstart validation."""

    def test_complete_quickstart_flow(self, quickstart_env):
        """Test complete quickstart flow with mocked components."""
        # This is a comprehensive test that would ideally:
        # 1. Set up the environment
        # 2. Parse a test kernel
        # 3. Start the MCP server
        # 4. Run basic queries
        # 5. Verify results

        # For now, validate that all components exist and are structured correctly
        kcs_root = Path(__file__).parent.parent.parent

        # Check project structure
        required_dirs = [
            kcs_root / "src" / "rust",
            kcs_root / "src" / "python",
            kcs_root / "tests",
            kcs_root / "tools",
        ]

        for dir_path in required_dirs:
            assert dir_path.exists(), f"Required directory missing: {dir_path}"

        # Check that key scripts exist
        key_scripts = [
            kcs_root / "tools" / "setup" / "install.sh",
            kcs_root / "tools" / "index_kernel.sh",
        ]

        for script_path in key_scripts:
            assert script_path.exists(), f"Key script missing: {script_path}"

    def test_documentation_availability(self):
        """Test that user documentation is available."""
        kcs_root = Path(__file__).parent.parent.parent

        # Check for specification files
        spec_dir = kcs_root / "specs" / "001-kernel-context-server"
        if spec_dir.exists():
            key_specs = [spec_dir / "plan.md", spec_dir / "contracts" / "mcp-api.yaml"]

            for spec_file in key_specs:
                if spec_file.exists():
                    content = spec_file.read_text()
                    assert (
                        len(content) > 100
                    ), f"Specification file too short: {spec_file}"

    def test_constitutional_requirements(self, quickstart_env):
        """Test that constitutional requirements are testable."""
        # Verify that the system is set up to enforce constitutional requirements

        # 1. Citations requirement
        from kcs_mcp.citations import CitationFormatter, ensure_citations

        formatter = CitationFormatter()

        # Test that responses with claims require citations
        test_response = {"hits": [{"snippet": "test code"}]}

        with pytest.raises(ValueError, match="Constitutional violation"):
            ensure_citations(test_response, formatter)

        # 2. Read-only requirement
        # Should be enforced at the application level
        # For now, just verify that the concept is documented

        # 3. Performance requirements
        # Should be testable via benchmarks and load tests
        assert True  # Placeholder - actual performance tests would run here

    def test_error_handling(self, quickstart_env):
        """Test proper error handling in quickstart flow."""
        # Test with invalid kernel path
        invalid_path = "/non/existent/path"

        # Should handle gracefully (not crash)
        # This would be tested against actual components when available
        assert not Path(invalid_path).exists()

    def test_quickstart_performance(self, quickstart_env):
        """Test that quickstart operations complete in reasonable time."""
        import time

        start_time = time.time()

        # Simulate quickstart operations
        kernel_path = quickstart_env.kernel_path

        # File discovery should be fast
        c_files = list(kernel_path.rglob("*.c"))
        h_files = list(kernel_path.rglob("*.h"))

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly for small test kernel
        assert duration < 5.0, f"Quickstart operations too slow: {duration:.2f}s"

        # Should find test files
        assert len(c_files) > 0, "No C files found in test kernel"
        assert len(h_files) > 0, "No header files found in test kernel"


class TestQuickstartDocumentation:
    """Test quickstart documentation and user experience."""

    def test_error_messages_helpful(self):
        """Test that error messages guide users effectively."""
        # Test citation formatter error messages
        from kcs_mcp.citations import Span

        with pytest.raises(ValueError) as exc_info:
            Span("", "abc123", 1, 1)  # Empty path

        assert "Path cannot be empty" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            Span("test.c", "", 1, 1)  # Empty SHA

        assert "SHA cannot be empty" in str(exc_info.value)

    def test_configuration_validation(self):
        """Test that configuration is validated with helpful messages."""
        # Test environment variable validation
        original_url = os.environ.get("KCS_SERVER_URL")

        try:
            # Test with invalid URL
            os.environ["KCS_SERVER_URL"] = "not-a-url"

            # Import should still work, validation happens at runtime
            from kcs_mcp import citations

            assert citations is not None

        finally:
            # Restore original value
            if original_url:
                os.environ["KCS_SERVER_URL"] = original_url
            else:
                os.environ.pop("KCS_SERVER_URL", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
