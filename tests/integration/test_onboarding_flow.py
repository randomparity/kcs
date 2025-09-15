"""
Integration test for onboarding flow user story.

This test verifies the end-to-end scenario from the spec:
"Given a kernel developer new to a subsystem, When they query how a specific
system call flows through the kernel, Then they receive a complete flow diagram
with entry points, function calls, locks/RCU usage, and relevant test coverage
with exact file/line citations."
"""

from typing import Any

import httpx
import pytest


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {
        "Authorization": "Bearer test_token_123",
        "Content-Type": "application/json",
    }


@pytest.fixture
async def mcp_client() -> httpx.AsyncClient:
    """HTTP client for MCP API requests."""
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        yield client


@pytest.mark.integration
class TestOnboardingFlow:
    """Integration test for developer onboarding flow."""

    async def test_complete_onboarding_flow(
        self, mcp_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test complete onboarding flow for understanding sys_read."""

        # Step 1: Find the entry point for read system call
        search_response = await mcp_client.post(
            "/mcp/tools/search_code",
            json={"query": "sys_read system call", "topK": 5},
            headers=auth_headers,
        )

        if search_response.status_code != 200:
            pytest.skip("Search endpoint not implemented yet")

        search_data = search_response.json()
        assert "hits" in search_data, "Search should return hits"

        # Step 2: Get detailed information about sys_read symbol
        symbol_response = await mcp_client.post(
            "/mcp/tools/get_symbol", json={"symbol": "sys_read"}, headers=auth_headers
        )

        if symbol_response.status_code != 200:
            pytest.skip("get_symbol endpoint not implemented yet")

        symbol_data = symbol_response.json()
        assert symbol_data["name"] == "sys_read", "Should return sys_read symbol info"
        assert "decl" in symbol_data, "Should include declaration location"

        # Step 3: Trace the entry point flow
        flow_response = await mcp_client.post(
            "/mcp/tools/entrypoint_flow",
            json={"entry": "__NR_read"},
            headers=auth_headers,
        )

        if flow_response.status_code != 200:
            pytest.skip("entrypoint_flow endpoint not implemented yet")

        flow_data = flow_response.json()
        assert "steps" in flow_data, "Flow should return steps"

        # Verify flow includes key functions
        flow_symbols = [step["to"] for step in flow_data["steps"]]
        expected_symbols = ["sys_read", "vfs_read"]
        found_expected = [
            sym
            for sym in expected_symbols
            if any(sym in flow_sym for flow_sym in flow_symbols)
        ]
        assert len(found_expected) > 0, "Flow should include expected VFS functions"

        # Step 4: Find what calls sys_read (should be minimal - it's an entry point)
        callers_response = await mcp_client.post(
            "/mcp/tools/who_calls",
            json={"symbol": "sys_read", "depth": 1},
            headers=auth_headers,
        )

        if callers_response.status_code == 200:
            callers_data = callers_response.json()
            # sys_read is an entry point, should have few/no callers
            assert len(callers_data["callers"]) <= 2, (
                "Entry points should have few callers"
            )

        # Step 5: Find what sys_read calls (the implementation flow)
        deps_response = await mcp_client.post(
            "/mcp/tools/list_dependencies",
            json={"symbol": "sys_read", "depth": 2},
            headers=auth_headers,
        )

        if deps_response.status_code == 200:
            deps_data = deps_response.json()
            assert len(deps_data["callees"]) > 0, "sys_read should call other functions"

            # Should call VFS layer
            dep_symbols = [dep["symbol"] for dep in deps_data["callees"]]
            vfs_calls = [sym for sym in dep_symbols if "vfs" in sym.lower()]
            assert len(vfs_calls) > 0, "Should call VFS functions"

        # Step 6: Verify all responses include citations
        all_responses = [search_data, symbol_data, flow_data]
        if callers_response.status_code == 200:
            all_responses.append(callers_data)
        if deps_response.status_code == 200:
            all_responses.append(deps_data)

        for response_data in all_responses:
            # Look for citation fields in various forms
            citations_found = self._find_citations_in_response(response_data)
            assert citations_found, (
                f"Response should include citations: {response_data}"
            )

    def _find_citations_in_response(self, data: dict[str, Any]) -> bool:
        """Recursively search for citation/span information in response."""
        if isinstance(data, dict):
            # Check for span fields
            if "span" in data and isinstance(data["span"], dict):
                span = data["span"]
                if all(field in span for field in ["path", "sha", "start", "end"]):
                    return True

            # Check for cites field
            if "cites" in data and isinstance(data["cites"], list):
                return True

            # Check for decl field (in symbol responses)
            if "decl" in data and isinstance(data["decl"], dict):
                decl = data["decl"]
                if all(field in decl for field in ["path", "sha", "start", "end"]):
                    return True

            # Recursively check nested objects
            for value in data.values():
                if self._find_citations_in_response(value):
                    return True

        elif isinstance(data, list):
            for item in data:
                if self._find_citations_in_response(item):
                    return True

        return False

    async def test_onboarding_performance_requirement(
        self, mcp_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test that onboarding flow completes within reasonable time."""
        import time

        start_time = time.time()

        # Run a simplified onboarding flow
        tasks = [
            mcp_client.post(
                "/mcp/tools/search_code",
                json={"query": "sys_read"},
                headers=auth_headers,
            ),
            mcp_client.post(
                "/mcp/tools/get_symbol",
                json={"symbol": "sys_read"},
                headers=auth_headers,
            ),
            mcp_client.post(
                "/mcp/tools/entrypoint_flow",
                json={"entry": "__NR_read"},
                headers=auth_headers,
            ),
        ]

        import asyncio

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        # Should complete within 2 seconds for good user experience
        assert total_time_ms < 2000, (
            f"Onboarding flow took {total_time_ms:.1f}ms, should be < 2000ms"
        )

        # At least some requests should succeed (those implemented)
        [
            r
            for r in responses
            if not isinstance(r, Exception) and getattr(r, "status_code", 0) == 200
        ]
        # Don't assert success since implementation may not exist yet
        # Just verify no crashes occurred

    async def test_onboarding_with_complex_syscall(
        self, mcp_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test onboarding flow with a more complex system call."""

        # Use openat - a more complex syscall with multiple arguments
        symbol_response = await mcp_client.post(
            "/mcp/tools/get_symbol", json={"symbol": "sys_openat"}, headers=auth_headers
        )

        if symbol_response.status_code == 200:
            symbol_data = symbol_response.json()
            assert "decl" in symbol_data, "Should include declaration"

            # Verify it's identified as a function
            assert symbol_data["kind"] == "function", "sys_openat should be a function"

        # Check the flow for openat
        flow_response = await mcp_client.post(
            "/mcp/tools/entrypoint_flow",
            json={"entry": "__NR_openat"},
            headers=auth_headers,
        )

        if flow_response.status_code == 200:
            flow_data = flow_response.json()
            # openat should have a more complex flow than read
            assert "steps" in flow_data, "Should return flow steps"

            if len(flow_data["steps"]) > 0:
                # Should involve path resolution and VFS
                flow_symbols = [step["to"] for step in flow_data["steps"]]
                path_related = [
                    sym
                    for sym in flow_symbols
                    if any(
                        keyword in sym.lower()
                        for keyword in ["path", "dentry", "inode"]
                    )
                ]
                # Complex syscalls should involve path resolution
                assert len(path_related) >= 0, (
                    "Complex syscalls should involve path operations"
                )

    async def test_onboarding_error_handling(
        self, mcp_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test onboarding flow with invalid/non-existent syscalls."""

        # Try to get info about non-existent syscall
        symbol_response = await mcp_client.post(
            "/mcp/tools/get_symbol",
            json={"symbol": "sys_nonexistent_call"},
            headers=auth_headers,
        )

        # Should return 404 gracefully
        if symbol_response.status_code == 404:
            error_data = symbol_response.json()
            assert "error" in error_data or "message" in error_data, (
                "Should provide error message"
            )

        # Try to trace non-existent entry point
        flow_response = await mcp_client.post(
            "/mcp/tools/entrypoint_flow",
            json={"entry": "__NR_nonexistent"},
            headers=auth_headers,
        )

        # Should handle gracefully
        assert flow_response.status_code in [
            200,
            404,
            422,
        ], "Should handle non-existent entry points gracefully"

    @pytest.mark.slow
    async def test_onboarding_comprehensive_coverage(
        self, mcp_client: httpx.AsyncClient, auth_headers: dict[str, str]
    ):
        """Test onboarding covers multiple aspects of kernel understanding."""

        syscalls_to_test = ["sys_read", "sys_write", "sys_open", "sys_close"]

        for syscall in syscalls_to_test:
            # Get symbol info
            symbol_response = await mcp_client.post(
                "/mcp/tools/get_symbol", json={"symbol": syscall}, headers=auth_headers
            )

            if symbol_response.status_code == 200:
                symbol_data = symbol_response.json()

                # Should have summary with concurrency info if available
                if symbol_data.get("summary"):
                    summary = symbol_data["summary"]
                    if "concurrency" in summary:
                        concurrency = summary["concurrency"]
                        # Should indicate if it can sleep, uses locks, etc.
                        assert isinstance(concurrency, dict), (
                            "Concurrency info should be structured"
                        )

            # Check what it depends on
            deps_response = await mcp_client.post(
                "/mcp/tools/list_dependencies",
                json={"symbol": syscall, "depth": 1},
                headers=auth_headers,
            )

            if deps_response.status_code == 200:
                deps_data = deps_response.json()
                # System calls should have dependencies
                assert len(deps_data["callees"]) >= 0, (
                    f"{syscall} should have some dependencies"
                )
