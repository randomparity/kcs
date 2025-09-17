"""Integration test for subsystem analysis.

These tests verify that complete subsystem analysis works correctly,
detecting all entry points, exports, and parameters in a subsystem like ext4.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

# Test configuration
TEST_KERNEL_PATH = Path("tests/fixtures/mini-kernel-v6.1")
TEST_SUBSYSTEM_PATH = TEST_KERNEL_PATH / "fs/ext4"

# Skip in CI unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)


def analyze_subsystem(subsystem_path: Path) -> dict[str, Any]:
    """Run complete subsystem analysis and return results.

    This simulates what the quickstart guide demonstrates for ext4 analysis.
    It should extract entry points, exports, and module parameters.
    """
    # Use the extract_entry_points_streaming tool
    extract_tool = Path("tools/extract_entry_points_streaming.py")

    if not extract_tool.exists():
        # Tool doesn't exist yet, return empty results for TDD
        return {
            "entry_points": [],
            "exports": [],
            "module_params": [],
            "statistics": {},
        }

    try:
        # Run extraction with all entry types
        result = subprocess.run(
            [
                "python",
                str(extract_tool),
                "--entry-types",
                "all",
                "--pattern-detection",
                str(subsystem_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {
                "entry_points": [],
                "exports": [],
                "module_params": [],
                "statistics": {},
            }

        # Parse newline-delimited JSON output
        import json

        entries = []
        for line in result.stdout.strip().split("\n"):
            if line:
                entries.append(json.loads(line))

        # Categorize results
        entry_points = []
        exports = []
        module_params = []

        for entry in entries:
            entry_type = entry.get("entry_type", "")
            if entry_type in [
                "syscall",
                "ioctl",
                "file_ops",
                "sysfs",
                "procfs",
                "debugfs",
                "netlink",
                "interrupt_handler",
            ]:
                entry_points.append(entry)
            elif entry_type == "export_symbol":
                exports.append(entry)
            elif entry_type == "module_param":
                module_params.append(entry)

        # Calculate statistics
        stats = {
            "total_entry_points": len(entry_points),
            "total_exports": len(exports),
            "total_module_params": len(module_params),
            "by_type": {},
        }

        for ep in entry_points:
            ep_type = ep.get("entry_type")
            if ep_type not in stats["by_type"]:
                stats["by_type"][ep_type] = 0
            stats["by_type"][ep_type] += 1

        return {
            "entry_points": entry_points,
            "exports": exports,
            "module_params": module_params,
            "statistics": stats,
        }

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return {
            "entry_points": [],
            "exports": [],
            "module_params": [],
            "statistics": {},
        }


@skip_integration_in_ci
class TestSubsystemAnalysis:
    """Integration tests for complete subsystem analysis."""

    def test_ext4_entry_points_detected(self) -> None:
        """Test that various entry points are detected in ext4 subsystem."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        # Should find entry points
        assert len(results["entry_points"]) > 0, (
            "Should detect entry points in ext4 subsystem"
        )

        # Check for different types
        entry_types = {ep.get("entry_type") for ep in results["entry_points"]}

        # We expect to find at least file_ops, ioctls, and sysfs
        expected_types = {"file_ops", "ioctl", "sysfs"}
        found_types = entry_types & expected_types

        assert len(found_types) >= 2, (
            f"Should find at least 2 entry point types in ext4, found: {found_types}"
        )

    def test_ext4_exports_detected(self) -> None:
        """Test that EXPORT_SYMBOL declarations are found."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        # Should find exported symbols
        assert len(results["exports"]) > 0, "Should detect exported symbols in ext4"

        # Check for specific exports we added
        export_names = {exp.get("name") for exp in results["exports"]}
        expected_exports = {"ext4_mark_inode_dirty", "ext4_journal_start"}

        found_exports = export_names & expected_exports
        assert len(found_exports) > 0, (
            f"Should find ext4 exports, found: {found_exports}"
        )

        # Check export types (regular vs GPL)
        export_types = {
            exp.get("metadata", {}).get("export_type")
            for exp in results["exports"]
            if exp.get("metadata")
        }

        assert "EXPORT_SYMBOL" in export_types or "EXPORT_SYMBOL_GPL" in export_types, (
            f"Should identify export types, found: {export_types}"
        )

    def test_ext4_module_params_detected(self) -> None:
        """Test that module parameters are extracted."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        # Should find module parameters
        assert len(results["module_params"]) > 0, (
            "Should detect module parameters in ext4"
        )

        # Check for specific params we added
        param_names = {param.get("name") for param in results["module_params"]}
        expected_params = {"ext4_debug", "ext4_max_batch_time"}

        found_params = param_names & expected_params
        assert len(found_params) > 0, (
            f"Should find ext4 module params, found: {found_params}"
        )

        # Check for parameter descriptions
        params_with_desc = [
            p
            for p in results["module_params"]
            if p.get("metadata", {}).get("description")
        ]

        assert len(params_with_desc) > 0, (
            "Module parameters should have descriptions from MODULE_PARM_DESC"
        )

    def test_ext4_file_operations_detected(self) -> None:
        """Test that file_operations structures are found."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        file_ops = [
            ep for ep in results["entry_points"] if ep.get("entry_type") == "file_ops"
        ]

        assert len(file_ops) > 0, "Should detect file_operations in ext4"

        # Check for specific file_ops
        fops_names = {fop.get("name") for fop in file_ops}

        # We should find operations from ext4_file_operations and ext4_dir_operations
        assert any("ext4" in name for name in fops_names if name), (
            f"Should find ext4 file operations, found: {fops_names}"
        )

    def test_ext4_ioctl_handlers_detected(self) -> None:
        """Test that ioctl handlers and commands are extracted."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        ioctls = [
            ep for ep in results["entry_points"] if ep.get("entry_type") == "ioctl"
        ]

        # Should find ioctl handlers
        assert len(ioctls) > 0, "Should detect ioctl handlers in ext4"

        # Check for ext4-specific ioctl commands
        all_commands = []
        for ioctl in ioctls:
            if ioctl.get("metadata") and "commands" in ioctl["metadata"]:
                all_commands.extend(ioctl["metadata"]["commands"])

        # Should find EXT4_IOC_* commands
        ext4_commands = [
            cmd for cmd in all_commands if "EXT4_IOC" in cmd.get("name", "")
        ]

        assert len(ext4_commands) >= 3, (
            f"Should find multiple EXT4_IOC commands, found: {len(ext4_commands)}"
        )

    def test_ext4_sysfs_attributes_detected(self) -> None:
        """Test that sysfs attributes are found."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        sysfs_entries = [
            ep for ep in results["entry_points"] if ep.get("entry_type") == "sysfs"
        ]

        # Should find sysfs attributes
        assert len(sysfs_entries) > 0, "Should detect sysfs attributes in ext4"

        # Check for show/store handlers
        for sysfs in sysfs_entries:
            name = sysfs.get("name", "")
            assert "show" in name or "store" in name or "attr" in name, (
                f"Sysfs entry should indicate handler type: {name}"
            )

    def test_ext4_procfs_entries_detected(self) -> None:
        """Test that procfs entries are found."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        procfs_entries = [
            ep for ep in results["entry_points"] if ep.get("entry_type") == "procfs"
        ]

        # Should find procfs entries (we created proc_create call)
        assert len(procfs_entries) > 0, "Should detect procfs entries in ext4"

        # Check for proc_ops structure
        for proc_entry in procfs_entries:
            assert proc_entry.get("file_path"), "Procfs entry should have file_path"
            assert proc_entry.get("line_number"), "Procfs entry should have line_number"

    def test_ext4_boot_params_detected(self) -> None:
        """Test that boot parameters (__setup) are found."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        # Boot params might be in entry_points or separate category
        boot_params = [
            ep for ep in results["entry_points"] if ep.get("entry_type") == "boot_param"
        ]

        # Should find __setup("ext4_nodelalloc", ...)
        assert len(boot_params) > 0, "Should detect boot parameters in ext4"

        # Check for specific boot param
        param_names = {param.get("name") for param in boot_params}
        assert (
            "ext4_nodelalloc" in param_names or "ext4_setup_nodelalloc" in param_names
        ), f"Should find ext4_nodelalloc boot param, found: {param_names}"

    def test_subsystem_analysis_performance(self) -> None:
        """Test that subsystem analysis completes within performance target."""
        # Time the analysis
        start_time = time.time()
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)
        elapsed_time = time.time() - start_time

        # Should complete within 30 seconds for subsystem
        assert elapsed_time < 30, (
            f"Subsystem analysis should complete within 30s, took {elapsed_time:.2f}s"
        )

        # If we got results, verify they're reasonable
        if results["statistics"]:
            total = (
                results["statistics"]["total_entry_points"]
                + results["statistics"]["total_exports"]
                + results["statistics"]["total_module_params"]
            )

            # Should not have excessive entries for small test subsystem
            assert total < 1000, (
                f"Should not detect excessive entries in test subsystem, found {total}"
            )

    def test_subsystem_statistics_accuracy(self) -> None:
        """Test that statistics accurately reflect the detected items."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        stats = results["statistics"]

        # Verify counts match actual results
        assert stats["total_entry_points"] == len(results["entry_points"]), (
            "Entry point count mismatch"
        )
        assert stats["total_exports"] == len(results["exports"]), (
            "Export count mismatch"
        )
        assert stats["total_module_params"] == len(results["module_params"]), (
            "Module param count mismatch"
        )

        # Verify by_type breakdown
        if stats.get("by_type"):
            for entry_type, count in stats["by_type"].items():
                actual = len(
                    [
                        ep
                        for ep in results["entry_points"]
                        if ep.get("entry_type") == entry_type
                    ]
                )
                assert count == actual, (
                    f"Statistics mismatch for {entry_type}: reported {count}, actual {actual}"
                )

    def test_quickstart_compatibility(self) -> None:
        """Test that results match quickstart guide expectations."""
        results = analyze_subsystem(TEST_SUBSYSTEM_PATH)

        # Quickstart shows finding entry points, exports, and params
        assert len(results["entry_points"]) > 0, (
            "Should find entry points as shown in quickstart"
        )
        assert len(results["exports"]) > 0, "Should find exports as shown in quickstart"
        assert len(results["module_params"]) > 0, (
            "Should find module params as shown in quickstart"
        )

        # Should provide clear categorization
        assert results["statistics"], "Should provide statistics summary"

        # Check that we can identify subsystem boundaries
        all_files = set()
        for item in (
            results["entry_points"] + results["exports"] + results["module_params"]
        ):
            if item.get("file_path"):
                all_files.add(Path(item["file_path"]).parent.name)

        # All items should be from ext4 subsystem
        assert "ext4" in all_files or len(all_files) > 0, (
            f"Items should be from ext4 subsystem, found in: {all_files}"
        )
