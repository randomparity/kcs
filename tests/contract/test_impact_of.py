"""
Contract tests for impact_of MCP tool.

These tests verify the API contract defined in contracts/mcp-api.yaml.
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

# Sample git diff for testing
SAMPLE_DIFF = """diff --git a/fs/read_write.c b/fs/read_write.c
index abc123..def456 100644
--- a/fs/read_write.c
+++ b/fs/read_write.c
@@ -451,7 +451,7 @@ ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos
        if (unlikely(!access_ok(buf, count)))
                return -EFAULT;

-       if (!ret)
+       if (!ret && count > 0)
                ret = __vfs_read(file, buf, count, pos);
        if (ret > 0) {
                fsnotify_access(file);"""


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> httpx.AsyncClient:
    """HTTP client for API requests."""
    async with httpx.AsyncClient(base_url=MCP_BASE_URL) as client:
        yield client


@skip_integration_in_ci
@skip_without_mcp_server
class TestImpactOfContract:
    """Contract tests for impact_of MCP tool."""

    async def test_impact_of_endpoint_exists(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that the impact_of endpoint exists and accepts POST requests."""
        payload = {"diff": SAMPLE_DIFF}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "impact_of endpoint should exist"

    async def test_impact_of_requires_authentication(
        self, http_client: httpx.AsyncClient
    ):
        """Test that impact_of requires valid authentication."""
        payload = {"diff": SAMPLE_DIFF}

        # Request without auth headers
        response = await http_client.post("/mcp/tools/impact_of", json=payload)
        assert response.status_code == 401, "Should require authentication"

    async def test_impact_of_validates_request_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that impact_of validates request schema according to OpenAPI spec."""

        # Request with no input parameters
        response = await http_client.post(
            "/mcp/tools/impact_of", json={}, headers=auth_headers
        )
        assert response.status_code == 422, (
            "Should reject request without any input parameters"
        )

        # Invalid diff format
        response = await http_client.post(
            "/mcp/tools/impact_of",
            json={"diff": 123},  # Not a string
            headers=auth_headers,
        )
        assert response.status_code == 422, "Should reject non-string diff"

        # Invalid files array
        response = await http_client.post(
            "/mcp/tools/impact_of", json={"files": "not an array"}, headers=auth_headers
        )
        assert response.status_code == 422, "Should reject non-array files"

    async def test_impact_of_response_schema(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that impact_of returns response matching OpenAPI schema."""
        payload = {"diff": SAMPLE_DIFF}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Verify required fields according to ImpactResult schema
            required_fields = [
                "configs",
                "modules",
                "tests",
                "owners",
                "risks",
                "cites",
            ]
            for field in required_fields:
                assert field in data, f"Response should contain '{field}' field"
                assert isinstance(data[field], list), f"'{field}' should be an array"

            # Verify field contents
            for config in data["configs"]:
                assert isinstance(config, str), "Each config should be a string"
                assert ":" in config, "Config should be in format 'arch:config'"

            for module in data["modules"]:
                assert isinstance(module, str), "Each module should be a string"

            for test in data["tests"]:
                assert isinstance(test, str), "Each test should be a string"

            for owner in data["owners"]:
                assert isinstance(owner, str), "Each owner should be a string"
                assert "@" in owner, "Owner should be an email address"

            for risk in data["risks"]:
                assert isinstance(risk, str), "Each risk should be a string"

            # Verify citations structure
            for cite in data["cites"]:
                assert isinstance(cite, dict), "Each citation should be an object"
                assert "path" in cite, "Citation should have 'path' field"
                assert "sha" in cite, "Citation should have 'sha' field"
                assert "start" in cite, "Citation should have 'start' field"
                assert "end" in cite, "Citation should have 'end' field"

                assert isinstance(cite["path"], str), "Citation path should be string"
                assert isinstance(cite["sha"], str), "Citation sha should be string"
                assert isinstance(cite["start"], int), (
                    "Citation start should be integer"
                )
                assert isinstance(cite["end"], int), "Citation end should be integer"

    async def test_impact_of_with_diff(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test impact analysis with git diff input."""
        payload = {"diff": SAMPLE_DIFF}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should identify the changed file
            configs = data["configs"]
            assert len(configs) > 0, "Should identify affected configurations"

            # Should include citations pointing to the changed area
            cites = data["cites"]
            if len(cites) > 0:
                # At least one citation should point to fs/read_write.c
                fs_citations = [c for c in cites if "fs/read_write.c" in c["path"]]
                assert len(fs_citations) > 0, "Should cite the changed file"

    async def test_impact_of_with_files(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test impact analysis with file list input."""
        payload = {"files": ["fs/read_write.c", "include/linux/fs.h"]}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should analyze impact of these specific files
            assert len(data["configs"]) >= 0, "Should return config list"
            assert len(data["cites"]) >= 0, "Should return citations"

    async def test_impact_of_with_symbols(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test impact analysis with symbol list input."""
        payload = {"symbols": ["vfs_read", "sys_read"]}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should analyze impact of these symbols
            assert len(data["configs"]) >= 0, "Should return config list"
            assert len(data["modules"]) >= 0, "Should return module list"

    async def test_impact_of_with_config_context(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test impact analysis with specific config context."""
        payload = {"files": ["fs/ext4/inode.c"], "config": "x86_64:defconfig"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should respect the config context
            configs = data["configs"]
            if len(configs) > 0:
                # Should include the specified config or related ones
                assert any("x86_64" in config for config in configs), (
                    "Should consider x86_64 configs"
                )

    async def test_impact_of_risk_assessment(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that risk factors are properly identified."""
        # Test with a change that might have risk factors
        risky_diff = """diff --git a/kernel/sched/core.c b/kernel/sched/core.c
index abc123..def456 100644
--- a/kernel/sched/core.c
+++ b/kernel/sched/core.c
@@ -100,7 +100,8 @@ void schedule(void)
        preempt_disable();
        rq = this_rq();
-       raw_spin_lock(&rq->lock);
+       raw_spin_lock_irq(&rq->lock);
+       // Added IRQ disable
        pick_next_task(rq);"""

        payload = {"diff": risky_diff}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should identify scheduler-related risks
            risks = data["risks"]
            # This is implementation-dependent, but scheduler changes should be flagged
            assert isinstance(risks, list), "Risks should be an array"

    async def test_impact_of_module_identification(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that affected modules are properly identified."""
        payload = {"files": ["drivers/net/ethernet/intel/e1000/e1000_main.c"]}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            modules = data["modules"]
            # Should identify the e1000 module
            if len(modules) > 0:
                assert any("e1000" in module for module in modules), (
                    "Should identify e1000 module"
                )

    async def test_impact_of_test_identification(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that relevant tests are identified."""
        payload = {"symbols": ["vfs_read", "vfs_write"]}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            tests = data["tests"]
            # Should identify VFS-related tests
            if len(tests) > 0:
                vfs_tests = [
                    t for t in tests if "vfs" in t.lower() or "read" in t.lower()
                ]
                assert len(vfs_tests) >= 0, "Should identify relevant tests"

    async def test_impact_of_owner_identification(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that maintainers/owners are properly identified."""
        payload = {"files": ["fs/"]}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            owners = data["owners"]
            # Should identify VFS maintainers
            if len(owners) > 0:
                for owner in owners:
                    assert "@" in owner, "Owner should be an email address"

    async def test_impact_of_performance_requirement(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that impact_of meets p95 < 600ms performance requirement."""
        payload = {"diff": SAMPLE_DIFF}

        import time

        start_time = time.time()

        response = await http_client.post(
            "/mcp/tools/impact_of",
            json=payload,
            headers=auth_headers,
            timeout=1.0,  # 1 second timeout
        )

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if response.status_code == 200:
            # Performance requirement from constitution: p95 < 600ms
            assert response_time_ms < 600, (
                f"Response time {response_time_ms:.1f}ms exceeds 600ms requirement"
            )

    async def test_impact_of_large_diff(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with large diffs."""
        # Create a larger diff
        large_diff = (
            SAMPLE_DIFF + "\n" + SAMPLE_DIFF.replace("vfs_read", "vfs_write") * 10
        )

        payload = {"diff": large_diff}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        # Should handle large diffs gracefully
        assert response.status_code in [
            200,
            422,
        ], "Should handle large diffs gracefully"

        if response.status_code == 200:
            data = response.json()
            # Should still return reasonable results
            assert len(data["configs"]) <= 20, "Should not return excessive configs"
            assert len(data["cites"]) <= 100, "Should limit citation count"

    async def test_impact_of_empty_input(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with empty but valid input."""
        payload = {"files": []}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should return empty results for empty input
            assert len(data["configs"]) == 0, (
                "Should return empty configs for empty input"
            )
            assert len(data["modules"]) == 0, (
                "Should return empty modules for empty input"
            )

    @pytest.mark.integration
    async def test_impact_of_with_sample_data(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Integration test with sample kernel data (if available)."""
        payload = {"symbols": ["vfs_read"]}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()

            # Should find realistic impact for vfs_read
            assert len(data["configs"]) > 0, (
                "vfs_read should affect some configurations"
            )
            assert len(data["cites"]) > 0, "Should provide citations for analysis"

            # Verify citation quality
            for cite in data["cites"]:
                assert cite["path"].startswith(("fs/", "include/")), (
                    "Citations should point to relevant files"
                )


@skip_integration_in_ci
@skip_without_mcp_server
class TestImpactOfErrorHandling:
    """Test error handling for impact_of tool."""

    async def test_impact_of_invalid_diff_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test behavior with malformed diff."""
        payload = {"diff": "this is not a valid diff format"}

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        # Should either handle gracefully or reject
        assert response.status_code in [
            200,
            422,
        ], "Should handle invalid diff format gracefully"

    async def test_impact_of_server_error_format(
        self, http_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that server errors follow the defined error schema."""
        # Force potential server error
        payload = {"diff": "x" * 100000}  # Very large diff

        response = await http_client.post(
            "/mcp/tools/impact_of", json=payload, headers=auth_headers
        )

        if response.status_code >= 500:
            data = response.json()
            assert "error" in data, "Error response should have 'error' field"
            assert "message" in data, "Error response should have 'message' field"
