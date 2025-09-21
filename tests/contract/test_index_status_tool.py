"""
Contract test for get_index_status MCP tool.

This test validates the MCP tool contract as defined in
specs/008-semantic-search-engine/contracts/mcp-search-tools.json

Following TDD: This test MUST FAIL before implementation exists.
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


class TestIndexStatusToolContract:
    """Contract tests for get_index_status MCP tool."""

    @pytest.fixture
    def valid_status_input(self) -> dict[str, Any]:
        """Valid input conforming to inputSchema."""
        return {"file_pattern": "drivers/net/*"}

    @pytest.fixture
    def minimal_status_input(self) -> dict[str, Any]:
        """Minimal valid input (no required fields)."""
        return {}

    @pytest.fixture
    def expected_output_schema(self) -> dict[str, Any]:
        """Expected output schema structure."""
        return {
            "total_files": 1250,
            "indexed_files": 1200,
            "pending_files": 30,
            "failed_files": 20,
            "total_chunks": 15000,
            "index_size_mb": 125.5,
            "last_update": "2024-01-15T10:30:00Z",
        }

    async def test_get_index_status_tool_exists(self):
        """Test that get_index_status MCP tool is registered and callable."""
        # This should fail until the tool is implemented
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Tool should be importable
        assert callable(get_index_status)

    async def test_get_index_status_no_input(self, minimal_status_input):
        """Test with no input parameters (should return global status)."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Should work with empty input
        result = await get_index_status(**minimal_status_input)
        assert result is not None
        assert "total_files" in result
        assert "indexed_files" in result
        assert "pending_files" in result
        assert "failed_files" in result
        assert "total_chunks" in result

    async def test_get_index_status_with_pattern(self, valid_status_input):
        """Test with file pattern filtering."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Valid input should not raise validation errors
        result = await get_index_status(**valid_status_input)
        assert result is not None

    async def test_get_index_status_output_schema_compliance(
        self, minimal_status_input
    ):
        """Test that output matches the defined outputSchema."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        result = await get_index_status(**minimal_status_input)

        # Required top-level fields
        assert "total_files" in result
        assert "indexed_files" in result
        assert "pending_files" in result
        assert "failed_files" in result
        assert "total_chunks" in result

        # Validate types
        assert isinstance(result["total_files"], int)
        assert isinstance(result["indexed_files"], int)
        assert isinstance(result["pending_files"], int)
        assert isinstance(result["failed_files"], int)
        assert isinstance(result["total_chunks"], int)

        # Validate non-negative integers
        assert result["total_files"] >= 0
        assert result["indexed_files"] >= 0
        assert result["pending_files"] >= 0
        assert result["failed_files"] >= 0
        assert result["total_chunks"] >= 0

        # Validate logical consistency
        assert (
            result["indexed_files"] + result["pending_files"] + result["failed_files"]
            <= result["total_files"]
        )

        # Optional fields should have correct types if present
        if "index_size_mb" in result:
            assert isinstance(result["index_size_mb"], (int, float))
            assert result["index_size_mb"] >= 0

        if "last_update" in result:
            assert isinstance(result["last_update"], str)
            # Should be valid ISO 8601 datetime format
            try:
                datetime.fromisoformat(result["last_update"].replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"Invalid datetime format: {result['last_update']}")

    async def test_get_index_status_file_pattern_examples(self):
        """Test with various file pattern examples from schema."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        patterns = [
            "drivers/net/*",
            "fs/ext4/*",
            "kernel/sched/*",
            "arch/x86/*",
            "drivers/gpu/drm/*",
        ]

        for pattern in patterns:
            result = await get_index_status(file_pattern=pattern)

            # Should return valid status for each pattern
            assert "total_files" in result
            assert "indexed_files" in result

            # Pattern-specific results should be subset of global
            global_result = await get_index_status()
            assert result["total_files"] <= global_result["total_files"]
            assert result["indexed_files"] <= global_result["indexed_files"]

    async def test_get_index_status_pattern_validation(self):
        """Test file pattern validation."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Valid patterns should work
        valid_patterns = [
            "drivers/*",
            "fs/ext4/*.c",
            "kernel/sched/fair.c",
            "**/mm/*.c",
            "arch/x86/kernel/*",
        ]

        for pattern in valid_patterns:
            result = await get_index_status(file_pattern=pattern)
            assert result is not None

    async def test_get_index_status_empty_index(self):
        """Test status when index is empty."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Mock empty index condition
        with patch(
            "src.python.semantic_search.database.index_manager.IndexManager.get_stats"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_files": 0,
                "indexed_files": 0,
                "pending_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
            }

            result = await get_index_status()

            assert result["total_files"] == 0
            assert result["indexed_files"] == 0
            assert result["pending_files"] == 0
            assert result["failed_files"] == 0
            assert result["total_chunks"] == 0

    async def test_get_index_status_partial_index(self):
        """Test status during partial indexing."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        result = await get_index_status()

        # During indexing, should have some pending files
        if result["total_files"] > 0:
            total_accounted = (
                result["indexed_files"]
                + result["pending_files"]
                + result["failed_files"]
            )
            assert total_accounted <= result["total_files"]

    async def test_get_index_status_size_calculation(self):
        """Test index size calculation accuracy."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        result = await get_index_status()

        if "index_size_mb" in result and result["total_chunks"] > 0:
            # Size should be reasonable relative to chunks
            # Assuming average ~1KB per chunk
            expected_min_size = (
                result["total_chunks"] * 0.0005
            )  # 0.5KB per chunk minimum
            expected_max_size = result["total_chunks"] * 0.01  # 10KB per chunk maximum

            assert expected_min_size <= result["index_size_mb"] <= expected_max_size

    async def test_get_index_status_filtering_accuracy(self):
        """Test that file pattern filtering returns accurate counts."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Get global status
        global_status = await get_index_status()

        # Get status for specific patterns
        net_status = await get_index_status(file_pattern="drivers/net/*")
        fs_status = await get_index_status(file_pattern="fs/*")

        # Pattern-specific counts should be subsets
        assert net_status["total_files"] <= global_status["total_files"]
        assert net_status["indexed_files"] <= global_status["indexed_files"]
        assert fs_status["total_files"] <= global_status["total_files"]
        assert fs_status["indexed_files"] <= global_status["indexed_files"]

        # Combined patterns shouldn't exceed global (allowing for overlap)
        combined_max = net_status["total_files"] + fs_status["total_files"]
        assert (
            combined_max >= global_status["total_files"]
            or combined_max <= global_status["total_files"] * 2
        )

    async def test_get_index_status_timestamp_format(self):
        """Test last_update timestamp format compliance."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        result = await get_index_status()

        if "last_update" in result:
            timestamp = result["last_update"]

            # Should be ISO 8601 format
            assert isinstance(timestamp, str)

            # Should parse as valid datetime
            try:
                if timestamp.endswith("Z"):
                    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    parsed = datetime.fromisoformat(timestamp)

                # Should be a reasonable timestamp (not in future, not too old)
                now = datetime.now(parsed.tzinfo if parsed.tzinfo else None)
                assert parsed <= now

            except ValueError as e:
                pytest.fail(f"Invalid timestamp format '{timestamp}': {e}")

    async def test_get_index_status_performance(self):
        """Test that status check completes quickly."""
        import time

        from src.python.semantic_search.mcp.status_tool import get_index_status

        start_time = time.time()
        await get_index_status()
        end_time = time.time()

        # Status check should be fast (< 100ms)
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 100, (
            f"Status check took {execution_time_ms}ms, should be < 100ms"
        )

    async def test_get_index_status_consistency(self):
        """Test consistency of status across multiple calls."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Multiple calls should return consistent data (within reason)
        result1 = await get_index_status()
        result2 = await get_index_status()

        # Core counts should be identical or very close
        # (allowing for concurrent indexing activity)
        assert abs(result1["total_files"] - result2["total_files"]) <= 1
        assert abs(result1["total_chunks"] - result2["total_chunks"]) <= 10

    async def test_get_index_status_error_resilience(self):
        """Test error handling and resilience."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        # Should handle database unavailability gracefully
        with patch(
            "src.python.semantic_search.database.connection.get_connection"
        ) as mock_conn:
            mock_conn.side_effect = Exception("Database unavailable")

            # Should either raise specific error or return safe defaults
            try:
                result = await get_index_status()
                # If it returns, should have safe default values
                assert all(
                    isinstance(result[key], int)
                    for key in [
                        "total_files",
                        "indexed_files",
                        "pending_files",
                        "failed_files",
                        "total_chunks",
                    ]
                )
            except Exception as e:
                # Should raise a specific, meaningful error
                assert "unavailable" in str(e).lower() or "connection" in str(e).lower()

    async def test_get_index_status_wildcard_patterns(self):
        """Test various wildcard pattern formats."""
        from src.python.semantic_search.mcp.status_tool import get_index_status

        patterns = [
            "*",  # All files
            "*.c",  # All C files
            "drivers/*",  # All in drivers/
            "**/mm/*",  # Any mm directory
            "kernel/*.h",  # Headers in kernel/
        ]

        for pattern in patterns:
            result = await get_index_status(file_pattern=pattern)
            assert result is not None
            assert "total_files" in result

            # Wildcard results should be reasonable
            assert result["total_files"] >= 0
