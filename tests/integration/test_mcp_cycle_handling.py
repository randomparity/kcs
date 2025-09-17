"""
Integration tests for MCP cycle detection and handling.

These tests verify that circular dependencies in the call graph are handled
correctly, preventing infinite loops and ensuring proper visited tracking.
"""

import os
from collections.abc import Generator

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
skip_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI",
)

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8080")
TEST_TOKEN = os.getenv("TEST_TOKEN", "test_token")


@pytest.fixture(scope="module")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Create a test PostgreSQL container."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest_asyncio.fixture
async def test_db(postgres_container: PostgresContainer) -> Database:
    """Create test database with circular call graph for cycle testing."""
    db = Database(postgres_container.get_connection_url())
    await db.initialize()

    # Create schema
    async with db.acquire() as conn:
        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS file (
                id SERIAL PRIMARY KEY,
                path TEXT NOT NULL,
                sha TEXT NOT NULL,
                last_parsed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_id INTEGER REFERENCES file(id),
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                signature TEXT,
                config_bitmap INTEGER DEFAULT 1
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS call_edge (
                id SERIAL PRIMARY KEY,
                caller_id INTEGER REFERENCES symbol(id),
                callee_id INTEGER REFERENCES symbol(id),
                call_type TEXT DEFAULT 'direct',
                line_number INTEGER,
                config_bitmap INTEGER DEFAULT 1
            )
        """)

        # Create test file
        file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('kernel/test_cycles.c', 'sha_cycle_test')
            RETURNING id
        """)

        # Create symbols for circular dependency testing
        symbol_ids = {}

        # Simple 3-function cycle: func_a -> func_b -> func_c -> func_a
        for func_name in ["func_a", "func_b", "func_c"]:
            symbol_ids[func_name] = await conn.fetchval(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
                VALUES ($1, 'function', $2, $3, $4, $5)
                RETURNING id
                """,
                func_name,
                file_id,
                {"func_a": 100, "func_b": 200, "func_c": 300}[func_name],
                {"func_a": 120, "func_b": 220, "func_c": 320}[func_name],
                f"void {func_name}(void)",
            )

        # Create more complex cycle with branching
        # mutex_lock -> mutex_unlock -> might_sleep -> schedule ->
        # __schedule -> preempt_schedule -> mutex_lock (cycle)
        complex_funcs = [
            "mutex_lock",
            "mutex_unlock",
            "might_sleep",
            "schedule",
            "__schedule",
            "preempt_schedule",
        ]

        for i, func_name in enumerate(complex_funcs):
            symbol_ids[func_name] = await conn.fetchval(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
                VALUES ($1, 'function', $2, $3, $4, $5)
                RETURNING id
                """,
                func_name,
                file_id,
                1000 + i * 100,
                1000 + i * 100 + 50,
                f"void {func_name}(void)",
            )

        # Create diamond pattern with cycles
        # diamond_top -> diamond_left -> diamond_bottom -> diamond_right -> diamond_top
        diamond_funcs = [
            "diamond_top",
            "diamond_left",
            "diamond_bottom",
            "diamond_right",
        ]
        for i, func_name in enumerate(diamond_funcs):
            symbol_ids[func_name] = await conn.fetchval(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
                VALUES ($1, 'function', $2, $3, $4, $5)
                RETURNING id
                """,
                func_name,
                file_id,
                2000 + i * 100,
                2000 + i * 100 + 50,
                f"void {func_name}(void)",
            )

        # Self-referencing function
        symbol_ids["recursive_func"] = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('recursive_func', 'function', $1, 3000, 3050, 'void recursive_func(int depth)')
            RETURNING id
            """,
            file_id,
        )

        # Create simple 3-function cycle edges
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 110)
            """,
            symbol_ids["func_a"],
            symbol_ids["func_b"],
        )

        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 210)
            """,
            symbol_ids["func_b"],
            symbol_ids["func_c"],
        )

        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 310)
            """,
            symbol_ids["func_c"],
            symbol_ids["func_a"],
        )

        # Create complex cycle edges
        complex_cycle_edges = [
            ("mutex_lock", "mutex_unlock", 1010),
            ("mutex_unlock", "might_sleep", 1110),
            ("might_sleep", "schedule", 1210),
            ("schedule", "__schedule", 1310),
            ("__schedule", "preempt_schedule", 1410),
            ("preempt_schedule", "mutex_lock", 1510),  # Completes the cycle
        ]

        for caller, callee, line in complex_cycle_edges:
            await conn.execute(
                """
                INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
                VALUES ($1, $2, 'direct', $3)
                """,
                symbol_ids[caller],
                symbol_ids[callee],
                line,
            )

        # Create diamond cycle edges
        diamond_cycle_edges = [
            ("diamond_top", "diamond_left", 2010),
            ("diamond_left", "diamond_bottom", 2110),
            ("diamond_bottom", "diamond_right", 2210),
            ("diamond_right", "diamond_top", 2310),  # Completes the cycle
        ]

        for caller, callee, line in diamond_cycle_edges:
            await conn.execute(
                """
                INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
                VALUES ($1, $2, 'direct', $3)
                """,
                symbol_ids[caller],
                symbol_ids[callee],
                line,
            )

        # Create self-reference edge
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $1, 'direct', 3025)
            """,
            symbol_ids["recursive_func"],
        )

        # Add some non-cyclic functions that connect to cycles
        symbol_ids["entry_func"] = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('entry_func', 'function', $1, 4000, 4050, 'void entry_func(void)')
            RETURNING id
            """,
            file_id,
        )

        symbol_ids["exit_func"] = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('exit_func', 'function', $1, 4100, 4150, 'void exit_func(void)')
            RETURNING id
            """,
            file_id,
        )

        # entry_func calls into the simple cycle
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 4025)
            """,
            symbol_ids["entry_func"],
            symbol_ids["func_a"],
        )

        # Complex cycle calls exit_func
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 1025)
            """,
            symbol_ids["mutex_lock"],
            symbol_ids["exit_func"],
        )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


class TestSimpleCycleDetection:
    """Test basic cycle detection with simple 3-function cycle."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_who_calls_handles_simple_cycle(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that who_calls handles simple cycles without infinite loop."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "func_a", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()

            # Should find all functions in the cycle
            caller_names = {caller["symbol"] for caller in data["callers"]}

            # Should include other cycle members
            assert "func_b" in caller_names or "func_c" in caller_names, (
                "Should find cycle members as callers"
            )

            # Should also find entry_func which calls into the cycle
            assert "entry_func" in caller_names, "Should find external caller"

            # Verify no infinite loop - should complete quickly
            # Check for duplicates
            symbol_counts: dict[str, int] = {}
            for caller in data["callers"]:
                symbol = caller["symbol"]
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

            for symbol, count in symbol_counts.items():
                assert count == 1, (
                    f"Symbol {symbol} appears {count} times, should appear only once"
                )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_list_dependencies_handles_simple_cycle(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that list_dependencies handles simple cycles without infinite loop."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "func_a", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()

            # Should find all functions in the cycle
            callee_names = {callee["symbol"] for callee in data["callees"]}

            # Should include other cycle members
            assert "func_b" in callee_names or "func_c" in callee_names, (
                "Should find cycle members as callees"
            )

            # Check for duplicates - each symbol should appear only once
            symbol_counts: dict[str, int] = {}
            for callee in data["callees"]:
                symbol = callee["symbol"]
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

            for symbol, count in symbol_counts.items():
                assert count == 1, (
                    f"Symbol {symbol} appears {count} times, should appear only once"
                )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_cycle_detection_with_different_depths(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test cycle detection works correctly at different depth limits."""
        with httpx.Client(timeout=10.0) as client:
            results_by_depth = {}

            for depth in [1, 2, 3, 5]:
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/who_calls",
                    headers=auth_headers,
                    json={"symbol": "func_a", "depth": depth},
                )

                assert response.status_code == 200
                data = response.json()
                results_by_depth[depth] = {
                    caller["symbol"] for caller in data["callers"]
                }

            # At depth 1, should find direct callers (func_c and entry_func)
            depth1_results = results_by_depth[1]
            assert "func_c" in depth1_results, (
                "Depth 1 should find direct caller func_c"
            )
            assert "entry_func" in depth1_results, (
                "Depth 1 should find direct caller entry_func"
            )
            assert "func_b" not in depth1_results, (
                "Depth 1 should not find indirect caller func_b"
            )

            # At higher depths, should find cycle members but not duplicate them
            for depth in [2, 3, 5]:
                depth_results = results_by_depth[depth]

                # Should be superset of previous depth
                assert depth_results >= results_by_depth[1], (
                    f"Depth {depth} should include all depth 1 results"
                )

                # Should include cycle members at depth >= 2
                if depth >= 2:
                    assert "func_b" in depth_results, (
                        f"Depth {depth} should find func_b via cycle"
                    )


class TestComplexCycleDetection:
    """Test cycle detection with more complex cycle patterns."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_long_cycle_detection(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test detection of longer cycles (6 functions)."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "mutex_lock", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()

            callee_names = {callee["symbol"] for callee in data["callees"]}

            # Should find members of the complex cycle
            expected_cycle_members = {
                "mutex_unlock",
                "might_sleep",
                "schedule",
                "__schedule",
                "preempt_schedule",
            }

            found_cycle_members = expected_cycle_members.intersection(callee_names)
            assert len(found_cycle_members) > 0, (
                "Should find at least some members of the complex cycle"
            )

            # Should also find exit_func which is called by mutex_lock
            assert "exit_func" in callee_names, "Should find direct callee exit_func"

            # Ensure no duplicates
            symbol_counts: dict[str, int] = {}
            for callee in data["callees"]:
                symbol = callee["symbol"]
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

            for symbol, count in symbol_counts.items():
                assert count == 1, (
                    f"Symbol {symbol} appears {count} times, should appear only once"
                )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_diamond_cycle_detection(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test detection of diamond pattern cycles."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "diamond_top", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()

            caller_names = {caller["symbol"] for caller in data["callers"]}

            # Should find diamond_right as direct caller (completes the cycle)
            assert "diamond_right" in caller_names, (
                "Should find diamond_right as caller in cycle"
            )

            # At sufficient depth, should find other diamond members
            diamond_members = {"diamond_left", "diamond_bottom", "diamond_right"}
            found_diamond_members = diamond_members.intersection(caller_names)
            assert len(found_diamond_members) > 0, "Should find diamond cycle members"

            # Ensure no duplicates
            symbol_counts: dict[str, int] = {}
            for caller in data["callers"]:
                symbol = caller["symbol"]
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

            for symbol, count in symbol_counts.items():
                assert count == 1, (
                    f"Symbol {symbol} appears {count} times, should appear only once"
                )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_self_reference_handling(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test handling of self-referencing functions."""
        with httpx.Client(timeout=10.0) as client:
            # Test who_calls for self-referencing function
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "recursive_func", "depth": 3},
            )

            assert response.status_code == 200
            data = response.json()

            caller_names = {caller["symbol"] for caller in data["callers"]}

            # Should find itself as a caller (self-reference)
            assert "recursive_func" in caller_names, (
                "Should find itself as caller in self-reference"
            )

            # Should appear only once despite self-reference
            recursive_count = sum(
                1 for caller in data["callers"] if caller["symbol"] == "recursive_func"
            )
            assert recursive_count == 1, (
                f"recursive_func appears {recursive_count} times, should appear only once"
            )

            # Test list_dependencies for self-referencing function
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "recursive_func", "depth": 3},
            )

            assert response.status_code == 200
            data = response.json()

            callee_names = {callee["symbol"] for callee in data["callees"]}

            # Should find itself as a callee (self-reference)
            assert "recursive_func" in callee_names, (
                "Should find itself as callee in self-reference"
            )

            # Should appear only once
            recursive_count = sum(
                1 for callee in data["callees"] if callee["symbol"] == "recursive_func"
            )
            assert recursive_count == 1, (
                f"recursive_func appears {recursive_count} times, should appear only once"
            )


class TestCyclePerformance:
    """Test performance characteristics of cycle detection."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_cycle_traversal_performance(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that cycle detection doesn't cause performance degradation."""
        import time

        with httpx.Client(timeout=30.0) as client:
            # Test deep traversal of cyclic graph
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "func_a", "depth": 5},
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 1.0, (
                f"Cycle traversal took {elapsed:.3f}s, should be < 1.0s"
            )

            # Test the complex cycle as well
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "mutex_lock", "depth": 5},
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 1.0, (
                f"Complex cycle traversal took {elapsed:.3f}s, should be < 1.0s"
            )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_multiple_cycle_queries_consistent(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that multiple queries on cyclic data return consistent results."""
        with httpx.Client(timeout=10.0) as client:
            # Run the same query multiple times
            responses = []
            for _ in range(3):
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/who_calls",
                    headers=auth_headers,
                    json={"symbol": "func_b", "depth": 3},
                )
                assert response.status_code == 200
                responses.append(response.json())

            # All responses should be identical
            first_response = responses[0]
            for i, response in enumerate(responses[1:], 1):
                assert response == first_response, (
                    f"Response {i + 1} differs from first response"
                )

            # Each response should have no duplicates
            for response in responses:
                caller_names = [caller["symbol"] for caller in response["callers"]]
                unique_caller_names = set(caller_names)
                assert len(caller_names) == len(unique_caller_names), (
                    "Response contains duplicate symbols"
                )


class TestCycleEdgeCases:
    """Test edge cases in cycle detection."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_nonexistent_symbol_in_cyclic_graph(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test queries for non-existent symbols don't break cycle detection."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "nonexistent_func_xyz", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty results
            assert data == {"callers": []}, (
                "Non-existent symbol should return empty results"
            )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_zero_with_cycles(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test depth=0 behavior with cyclic data."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "func_a", "depth": 0},
            )

            # Should either reject depth=0 or return empty
            if response.status_code == 200:
                data = response.json()
                assert data["callers"] == [], "Depth=0 should return empty list"
            else:
                assert response.status_code == 422, "Should reject invalid depth"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_cycle_with_all_endpoints(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that all MCP endpoints handle cycles correctly."""
        with httpx.Client(timeout=15.0) as client:
            # Test who_calls
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "func_a", "depth": 3},
            )
            assert response.status_code == 200
            who_calls_data = response.json()

            # Test list_dependencies
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "func_a", "depth": 3},
            )
            assert response.status_code == 200
            list_deps_data = response.json()

            # Both should complete without hanging
            assert "callers" in who_calls_data
            assert "callees" in list_deps_data

            # Both should have no duplicates
            who_calls_symbols = [c["symbol"] for c in who_calls_data["callers"]]
            list_deps_symbols = [c["symbol"] for c in list_deps_data["callees"]]

            assert len(who_calls_symbols) == len(set(who_calls_symbols)), (
                "who_calls has duplicate symbols"
            )
            assert len(list_deps_symbols) == len(set(list_deps_symbols)), (
                "list_dependencies has duplicate symbols"
            )

            # Test entrypoint_flow (if func_a were an entry point)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "func_a"},
            )
            # This may return 200 with empty results or appropriate error
            assert response.status_code in [200, 404], (
                "entrypoint_flow should handle gracefully"
            )
            if response.status_code == 200:
                flow_data = response.json()
                assert "steps" in flow_data
