"""Integration test for ioctl command extraction.

These tests verify that ioctl magic numbers and commands are properly
detected and extracted from kernel code, including handler identification.
"""

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

# Test configuration
TEST_KERNEL_PATH = Path("tests/fixtures/mini-kernel-v6.1")
TEST_IOCTL_FILE = TEST_KERNEL_PATH / "drivers/char/test_ioctl.c"

# Skip in CI unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)


def run_ioctl_extraction(kernel_path: Path) -> dict[str, Any]:
    """Run the ioctl extraction tool and return parsed results.

    This simulates what the full pipeline would do when processing
    kernel source files for ioctl detection.
    """
    # For now, use the extract_entrypoints_streaming tool if available
    extract_tool = Path("tools/extract_entrypoints_streaming.py")

    if not extract_tool.exists():
        # If tool doesn't exist yet, return empty results
        # This ensures the test fails initially (TDD)
        return {"ioctls": [], "commands": [], "handlers": []}

    try:
        # Run the extraction tool with ioctl type filter
        result = subprocess.run(
            ["python", str(extract_tool), "--entry-types", "ioctl", str(kernel_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {"ioctls": [], "commands": [], "handlers": []}

        # Parse the JSON output (newline-delimited)
        ioctls = []
        for line in result.stdout.strip().split("\n"):
            if line:
                import json

                entry = json.loads(line)
                if entry.get("entry_type") == "ioctl":
                    ioctls.append(entry)

        # Extract commands and handlers from results
        commands = []
        handlers = []
        for ioctl in ioctls:
            if ioctl.get("metadata"):
                if "commands" in ioctl["metadata"]:
                    commands.extend(ioctl["metadata"]["commands"])
                if "handler" in ioctl["metadata"]:
                    handlers.append(ioctl["metadata"]["handler"])

        return {
            "ioctls": ioctls,
            "commands": commands,
            "handlers": handlers,
        }

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return {"ioctls": [], "commands": [], "handlers": []}


@skip_integration_in_ci
class TestIoctlDetection:
    """Integration tests for ioctl command extraction."""

    def test_ioctl_magic_number_detection(self) -> None:
        """Test that ioctl magic numbers are correctly extracted."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Should detect ioctl handlers
        assert len(results["ioctls"]) > 0, (
            "Should detect ioctl handlers in test fixtures"
        )

        # Should extract magic numbers
        magic_numbers = set()
        for ioctl in results["ioctls"]:
            if ioctl.get("metadata") and "magic" in ioctl["metadata"]:
                magic_numbers.add(ioctl["metadata"]["magic"])

        # We defined TEST_IOC_MAGIC 'T' and ALT_IOC_MAGIC 0xAB
        assert "'T'" in magic_numbers or "0x54" in magic_numbers, (
            f"Should detect TEST_IOC_MAGIC, found: {magic_numbers}"
        )
        assert "0xAB" in magic_numbers or "171" in magic_numbers, (
            f"Should detect ALT_IOC_MAGIC, found: {magic_numbers}"
        )

    def test_ioctl_command_extraction(self) -> None:
        """Test that individual ioctl commands are extracted."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Should extract command definitions
        assert len(results["commands"]) > 0, "Should extract ioctl command definitions"

        # Check for expected commands
        command_names = {
            cmd.get("name") for cmd in results["commands"] if isinstance(cmd, dict)
        }
        expected_commands = {
            "TEST_IOCRESET",
            "TEST_IOCGETVAL",
            "TEST_IOCSETVAL",
            "TEST_IOCGSTATUS",
            "TEST_IOCSSTATUS",
            "TEST_IOCXCHANGE",
            "ALT_IOCCOMMAND",
        }

        found_commands = command_names & expected_commands
        assert len(found_commands) >= 5, (
            f"Should find at least 5 test commands, found: {found_commands}"
        )

    def test_ioctl_direction_detection(self) -> None:
        """Test that ioctl command directions (_IO, _IOR, _IOW, _IOWR) are identified."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Look for command direction information
        directions_found = set()
        for cmd in results["commands"]:
            if isinstance(cmd, dict) and "direction" in cmd:
                directions_found.add(cmd["direction"])

        # Should identify different direction types
        expected_directions = {"none", "read", "write", "read_write"}
        assert len(directions_found & expected_directions) >= 2, (
            f"Should detect multiple ioctl directions, found: {directions_found}"
        )

    def test_ioctl_handler_identification(self) -> None:
        """Test that ioctl handler functions are correctly identified."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Should identify handler functions
        assert len(results["handlers"]) > 0, "Should identify ioctl handler functions"

        # Check for expected handlers
        handler_names = {
            h.get("name") for h in results["handlers"] if isinstance(h, dict)
        }
        expected_handlers = {
            "test_ioctl",
            "test_compat_ioctl",
            "another_device_ioctl",
        }

        found_handlers = handler_names & expected_handlers
        assert len(found_handlers) >= 2, (
            f"Should find at least 2 ioctl handlers, found: {found_handlers}"
        )

    def test_ioctl_file_operations_association(self) -> None:
        """Test that ioctl handlers are associated with file_operations structures."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Look for file_operations associations
        fops_associations = []
        for ioctl in results["ioctls"]:
            if ioctl.get("metadata") and "file_operations" in ioctl["metadata"]:
                fops_associations.append(ioctl["metadata"]["file_operations"])

        # Should find associations with test_fops and another_fops
        assert len(fops_associations) > 0, (
            "Should associate ioctl handlers with file_operations structures"
        )

        fops_names = {f.get("name") for f in fops_associations if isinstance(f, dict)}
        assert "test_fops" in fops_names or "another_fops" in fops_names, (
            f"Should find test file_operations structures, found: {fops_names}"
        )

    def test_ioctl_command_metadata_storage(self) -> None:
        """Test that ioctl command metadata is properly stored."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Check that metadata is populated for commands
        commands_with_metadata = [
            cmd
            for cmd in results["commands"]
            if isinstance(cmd, dict) and len(cmd) > 1  # More than just name
        ]

        assert len(commands_with_metadata) > 0, (
            "Ioctl commands should have metadata (magic, number, direction, etc.)"
        )

        # Verify metadata structure
        for cmd in commands_with_metadata[:3]:  # Check first few
            # Should have at least some of these fields
            expected_fields = {
                "name",
                "magic",
                "number",
                "direction",
                "type",
                "file_path",
                "line_number",
            }
            actual_fields = set(cmd.keys())

            assert len(actual_fields & expected_fields) >= 3, (
                f"Command should have metadata fields, found: {actual_fields}"
            )

    def test_ioctl_handler_with_switch_statement(self) -> None:
        """Test that ioctl handlers with switch statements are analyzed."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Find the test_ioctl handler
        test_handler = None
        for ioctl in results["ioctls"]:
            if ioctl.get("name") == "test_ioctl":
                test_handler = ioctl
                break

        assert test_handler is not None, "Should find test_ioctl handler"

        # Check if switch cases were detected
        if (
            test_handler.get("metadata")
            and "handled_commands" in test_handler["metadata"]
        ):
            handled = test_handler["metadata"]["handled_commands"]

            # Should detect commands handled in switch statement
            assert len(handled) >= 5, (
                f"Should detect commands in switch statement, found: {len(handled)}"
            )

    def test_ioctl_compat_handler_detection(self) -> None:
        """Test that compat_ioctl handlers are detected for 32-bit compatibility."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Look for compat_ioctl handlers
        compat_handlers = [
            h
            for h in results["handlers"]
            if isinstance(h, dict) and "compat" in h.get("name", "").lower()
        ]

        assert len(compat_handlers) > 0, "Should detect compat_ioctl handlers"

        # Check that test_compat_ioctl was found
        compat_names = {h.get("name") for h in compat_handlers}
        assert "test_compat_ioctl" in compat_names, (
            f"Should find test_compat_ioctl, found: {compat_names}"
        )

    def test_ioctl_extraction_performance(self) -> None:
        """Test that ioctl extraction completes within performance targets."""
        import time

        # Time the extraction
        start_time = time.time()
        results = run_ioctl_extraction(TEST_KERNEL_PATH)
        elapsed_time = time.time() - start_time

        # Should complete quickly for test fixtures
        assert elapsed_time < 5.0, (
            f"Ioctl extraction should complete within 5s for test fixtures, took {elapsed_time:.2f}s"
        )

        # If results were returned, check they're reasonable
        if results["ioctls"]:
            # Should not have excessive false positives
            assert len(results["ioctls"]) < 100, (
                f"Should not detect excessive ioctls in small test set, found {len(results['ioctls'])}"
            )

    def test_ioctl_macro_type_detection(self) -> None:
        """Test detection of different ioctl macro types (_IO, _IOR, _IOW, _IOWR)."""
        results = run_ioctl_extraction(TEST_KERNEL_PATH)

        # Look for commands by macro type
        macro_types = {}
        for cmd in results["commands"]:
            if isinstance(cmd, dict) and "macro_type" in cmd:
                macro_type = cmd["macro_type"]
                if macro_type not in macro_types:
                    macro_types[macro_type] = []
                macro_types[macro_type].append(cmd.get("name"))

        # Should detect all four macro types from our test file
        expected_types = {"_IO", "_IOR", "_IOW", "_IOWR"}
        found_types = set(macro_types.keys())

        assert len(found_types & expected_types) >= 3, (
            f"Should detect at least 3 ioctl macro types, found: {found_types}"
        )

        # Verify specific commands are categorized correctly
        if "_IO" in macro_types:
            assert "TEST_IOCRESET" in macro_types["_IO"], (
                "TEST_IOCRESET should be detected as _IO macro"
            )

        if "_IOWR" in macro_types:
            assert "TEST_IOCXCHANGE" in macro_types["_IOWR"], (
                "TEST_IOCXCHANGE should be detected as _IOWR macro"
            )
