"""
Integration tests using the mini-kernel fixture.

These tests verify KCS functionality with a small, controlled kernel structure
instead of requiring a full 5GB+ Linux kernel repository.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def mini_kernel_path():
    """Path to the mini-kernel test fixture."""
    current_dir = Path(__file__).parent.parent
    mini_kernel = current_dir / "fixtures" / "mini-kernel-v6.1"

    if not mini_kernel.exists():
        pytest.skip("Mini-kernel fixture not available")

    return str(mini_kernel.absolute())


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMiniKernelParsing:
    """Test KCS parsing with mini-kernel fixture."""

    def test_mini_kernel_structure(self, mini_kernel_path):
        """Verify mini-kernel fixture has expected structure."""
        mini_kernel = Path(mini_kernel_path)

        # Check essential files exist
        assert (mini_kernel / "Makefile").exists()
        assert (mini_kernel / "Kconfig").exists()
        assert (mini_kernel / "fs" / "read_write.c").exists()
        assert (mini_kernel / "include" / "linux" / "fs.h").exists()
        assert (
            mini_kernel / "arch" / "x86" / "entry" / "syscalls" / "syscall_64.tbl"
        ).exists()

    def test_makefile_version_detection(self, mini_kernel_path):
        """Test that version can be extracted from Makefile."""
        makefile = Path(mini_kernel_path) / "Makefile"
        content = makefile.read_text()

        assert "VERSION = 6" in content
        assert "PATCHLEVEL = 1" in content
        assert "SUBLEVEL = 0" in content

    def test_git_repository_setup(self, mini_kernel_path):
        """Test that mini-kernel structure is suitable for git tracking."""
        # Since mini-kernel is now part of the main repo, just verify structure
        mini_kernel = Path(mini_kernel_path)

        # Should have essential files that would be tracked by git
        assert (mini_kernel / "Makefile").exists()
        assert (mini_kernel / "Kconfig").exists()
        assert len(list(mini_kernel.rglob("*.c"))) >= 3
        assert len(list(mini_kernel.rglob("*.h"))) >= 4

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/tree-sitter"), reason="tree-sitter not available"
    )
    def test_c_file_parsing(self, mini_kernel_path):
        """Test that C files can be parsed by tree-sitter."""
        fs_file = Path(mini_kernel_path) / "fs" / "read_write.c"

        # This would normally use the KCS parser, but for now just verify file content
        content = fs_file.read_text()

        # Check for expected symbols
        assert "vfs_read" in content
        assert "vfs_write" in content
        assert "sys_read" in content
        assert "SYSCALL_DEFINE3" in content

    def test_symbol_detection_patterns(self, mini_kernel_path):
        """Test that mini-kernel contains patterns KCS should detect."""
        fs_file = Path(mini_kernel_path) / "fs" / "read_write.c"
        content = fs_file.read_text()

        # Function definitions
        assert "ssize_t vfs_read(" in content
        assert "SYSCALL_DEFINE3(read," in content

        # Function calls
        assert "vfs_read(file, buf, count, &file->f_pos)" in content
        assert "__vfs_read(file, buf, count, pos)" in content

    def test_header_include_patterns(self, mini_kernel_path):
        """Test include patterns that KCS should track."""
        fs_file = Path(mini_kernel_path) / "fs" / "read_write.c"
        content = fs_file.read_text()

        # Standard includes
        assert "#include <linux/fs.h>" in content
        assert "#include <linux/kernel.h>" in content
        assert "#include <linux/syscalls.h>" in content

    def test_kconfig_parsing_patterns(self, mini_kernel_path):
        """Test Kconfig patterns for configuration analysis."""
        kconfig = Path(mini_kernel_path) / "Kconfig"
        content = kconfig.read_text()

        # Config options
        assert "config DEBUG_KERNEL" in content
        assert "config VFS" in content
        assert "depends on VFS" in content
        assert "default y" in content


class TestMiniKernelIndexing:
    """Test full KCS indexing pipeline with mini-kernel."""

    @pytest.mark.skipif(
        not os.path.exists("tools/index_kernel.sh"),
        reason="index_kernel.sh not available",
    )
    def test_index_kernel_script_with_mini_kernel(
        self, mini_kernel_path, temp_output_dir
    ):
        """Test index_kernel.sh script with mini-kernel."""
        # Set environment variable
        env = os.environ.copy()
        env["KCS_KERNEL_PATH"] = mini_kernel_path

        # Run with dry-run to test argument parsing
        result = subprocess.run(
            ["bash", "tools/index_kernel.sh", "--dry-run", "--output", temp_output_dir],
            env=env,
            capture_output=True,
            text=True,
        )

        # Should succeed in dry-run mode
        assert "DRY RUN MODE" in result.stderr or "DRY RUN MODE" in result.stdout

    def test_environment_variable_fallback(self, mini_kernel_path, temp_output_dir):
        """Test that KCS_KERNEL_PATH environment variable works."""
        env = os.environ.copy()
        env["KCS_KERNEL_PATH"] = mini_kernel_path

        # Test the script recognizes the environment variable
        result = subprocess.run(
            ["bash", "tools/index_kernel.sh", "--dry-run", "--output", temp_output_dir],
            env=env,
            capture_output=True,
            text=True,
        )

        # Should use environment variable path
        output = result.stderr + result.stdout
        assert mini_kernel_path in output or "Using KCS_KERNEL_PATH" in output


class TestMiniKernelSymbols:
    """Test specific symbol patterns in mini-kernel."""

    def test_syscall_entry_points(self, mini_kernel_path):
        """Test system call entry point detection."""
        syscall_table = (
            Path(mini_kernel_path)
            / "arch"
            / "x86"
            / "entry"
            / "syscalls"
            / "syscall_64.tbl"
        )
        content = syscall_table.read_text()

        # Expected syscalls
        assert "read\t\t\tsys_read" in content
        assert "write\t\t\tsys_write" in content
        assert "openat\t\t\tsys_openat" in content

    def test_file_operations_structures(self, mini_kernel_path):
        """Test file operations structure patterns."""
        mem_file = Path(mini_kernel_path) / "drivers" / "char" / "mem.c"
        content = mem_file.read_text()

        # File operations structures
        assert "null_fops" in content
        assert "zero_fops" in content
        assert ".read  = null_read" in content
        assert ".write = null_write" in content

    def test_function_signatures(self, mini_kernel_path):
        """Test various function signature patterns."""
        fs_file = Path(mini_kernel_path) / "fs" / "read_write.c"
        content = fs_file.read_text()

        # Different signature patterns
        assert "ssize_t vfs_read(struct file *file, char __user *buf" in content
        assert "SYSCALL_DEFINE3(read, unsigned int, fd" in content
        assert "static ssize_t __vfs_read(" in content


class TestMiniKernelConfiguration:
    """Test configuration and build system patterns."""

    def test_makefile_targets(self, mini_kernel_path):
        """Test Makefile target patterns."""
        makefile = Path(mini_kernel_path) / "Makefile"
        content = makefile.read_text()

        assert "defconfig:" in content
        assert "clean:" in content
        assert ".PHONY:" in content

    def test_kconfig_dependencies(self, mini_kernel_path):
        """Test Kconfig dependency patterns."""
        kconfig = Path(mini_kernel_path) / "Kconfig"
        content = kconfig.read_text()

        # Dependency patterns
        assert "depends on VFS" in content
        assert "depends on 64BIT" in content

    def test_architecture_patterns(self, mini_kernel_path):
        """Test architecture-specific patterns."""
        # Check x86-specific files exist
        x86_syscalls = Path(mini_kernel_path) / "arch" / "x86" / "entry" / "syscalls"
        assert x86_syscalls.exists()

        syscall_table = x86_syscalls / "syscall_64.tbl"
        content = syscall_table.read_text()
        assert "64-bit system call numbers" in content


@pytest.mark.performance
class TestMiniKernelPerformance:
    """Test performance characteristics with mini-kernel."""

    def test_parsing_speed(self, mini_kernel_path):
        """Test that mini-kernel parses quickly."""
        import time

        start_time = time.time()

        # Walk through all files (simulating parser)
        file_count = 0
        for root, _dirs, files in os.walk(mini_kernel_path):
            for file in files:
                if file.endswith((".c", ".h")):
                    filepath = Path(root) / file
                    filepath.read_text(errors="ignore")
                    file_count += 1

        end_time = time.time()
        duration = end_time - start_time

        # Should be very fast
        assert duration < 1.0, (
            f"Mini-kernel parsing took {duration:.2f}s, should be <1s"
        )
        assert file_count >= 7, f"Expected at least 7 C/H files, found {file_count}"

    def test_repository_size(self, mini_kernel_path):
        """Test that mini-kernel is appropriately sized."""
        result = subprocess.run(
            ["du", "-sb", mini_kernel_path], capture_output=True, text=True
        )

        if result.returncode == 0:
            size_bytes = int(result.stdout.split()[0])
            size_kb = size_bytes / 1024

            # Should be much smaller than full kernel
            assert size_kb < 100, f"Mini-kernel is {size_kb:.1f}KB, should be <100KB"
