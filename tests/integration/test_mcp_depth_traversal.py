"""
Integration tests for MCP depth-limited traversal.

These tests verify that depth parameters correctly limit graph traversal
in who_calls and list_dependencies endpoints.
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
    """Create test database with deep call graph for depth testing."""
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

        # Create a deep call chain for testing depth limits
        # Chain: level0 -> level1 -> level2 -> level3 -> level4 -> level5
        file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('kernel/test_depth.c', 'sha_depth_test')
            RETURNING id
        """)

        # Create symbols for each level
        symbol_ids = {}
        for i in range(6):
            symbol_ids[f"level{i}"] = await conn.fetchval(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
                VALUES ($1, 'function', $2, $3, $4, $5)
                RETURNING id
                """,
                f"level{i}_func",
                file_id,
                i * 100 + 1,
                i * 100 + 50,
                f"void level{i}_func(void)",
            )

        # Create call edges forming a chain
        for i in range(5):
            await conn.execute(
                """
                INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
                VALUES ($1, $2, 'direct', $3)
                """,
                symbol_ids[f"level{i}"],
                symbol_ids[f"level{i + 1}"],
                i * 100 + 25,
            )

        # Create a branching structure for more complex testing
        # level2 also calls branch_a and branch_b
        branch_a_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('branch_a_func', 'function', $1, 1000, 1050, 'void branch_a_func(void)')
            RETURNING id
            """,
            file_id,
        )

        branch_b_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('branch_b_func', 'function', $1, 1100, 1150, 'void branch_b_func(void)')
            RETURNING id
            """,
            file_id,
        )

        # level2 calls both branches
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 230)
            """,
            symbol_ids["level2"],
            branch_a_id,
        )

        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 240)
            """,
            symbol_ids["level2"],
            branch_b_id,
        )

        # Both branches call a common function
        common_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('common_func', 'function', $1, 1200, 1250, 'void common_func(void)')
            RETURNING id
            """,
            file_id,
        )

        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 1025)
            """,
            branch_a_id,
            common_id,
        )

        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 1125)
            """,
            branch_b_id,
            common_id,
        )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


class TestDepthLimitingWhoCalls:
    """Test depth limiting for who_calls endpoint."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_1_returns_direct_callers_only(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth=1 returns only direct callers."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level3_func", "depth": 1},
            )

            assert response.status_code == 200
            data = response.json()

            # Should only find level2_func (direct caller)
            caller_names = {caller["symbol"] for caller in data["callers"]}
            assert "level2_func" in caller_names, "Should find direct caller"
            assert "level1_func" not in caller_names, "Should not find depth-2 caller"
            assert "level0_func" not in caller_names, "Should not find depth-3 caller"
            assert len(caller_names) == 1, "Should only find 1 direct caller"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_3_traverses_three_levels(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth=3 traverses three levels of callers."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level5_func", "depth": 3},
            )

            assert response.status_code == 200
            data = response.json()

            # Should find level4, level3, and level2 (3 levels up)
            caller_names = {caller["symbol"] for caller in data["callers"]}
            assert "level4_func" in caller_names, "Should find depth-1 caller"
            assert "level3_func" in caller_names, "Should find depth-2 caller"
            assert "level2_func" in caller_names, "Should find depth-3 caller"
            assert "level1_func" not in caller_names, "Should not find depth-4 caller"
            assert "level0_func" not in caller_names, "Should not find depth-5 caller"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_5_maximum_enforced(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth=5 is enforced as maximum."""
        with httpx.Client() as client:
            # First verify what depth=5 returns
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level5_func", "depth": 5},
            )

            assert response.status_code == 200
            data_depth_5 = response.json()

            # Should find all callers up to level0 (5 levels up)
            caller_names = {caller["symbol"] for caller in data_depth_5["callers"]}
            expected_callers = {
                "level4_func",
                "level3_func",
                "level2_func",
                "level1_func",
                "level0_func",
            }
            assert caller_names == expected_callers, "Depth=5 should find all 5 levels"

            # Verify depth > 5 is rejected
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level5_func", "depth": 10},
            )
            assert response.status_code == 422, "Should reject depth > 5"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_increasing_depth_expands_results(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that results expand consistently with increasing depth."""
        with httpx.Client() as client:
            results_by_depth = {}

            for depth in [1, 2, 3, 4, 5]:
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/who_calls",
                    headers=auth_headers,
                    json={"symbol": "level5_func", "depth": depth},
                )
                assert response.status_code == 200
                data = response.json()
                results_by_depth[depth] = {
                    caller["symbol"] for caller in data["callers"]
                }

            # Each depth should be a superset of the previous depth
            for d in range(2, 6):
                assert results_by_depth[d] >= results_by_depth[d - 1], (
                    f"Depth {d} results should include all depth {d - 1} results"
                )

            # Verify expected counts
            assert len(results_by_depth[1]) == 1, "Depth 1 should find 1 caller"
            assert len(results_by_depth[2]) == 2, "Depth 2 should find 2 callers"
            assert len(results_by_depth[3]) == 3, "Depth 3 should find 3 callers"
            assert len(results_by_depth[4]) == 4, "Depth 4 should find 4 callers"
            assert len(results_by_depth[5]) == 5, "Depth 5 should find 5 callers"


class TestDepthLimitingListDependencies:
    """Test depth limiting for list_dependencies endpoint."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_1_returns_direct_callees_only(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth=1 returns only direct callees."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "level2_func", "depth": 1},
            )

            assert response.status_code == 200
            data = response.json()

            # Should find level3_func, branch_a_func, branch_b_func (direct callees)
            callee_names = {callee["symbol"] for callee in data["callees"]}
            assert "level3_func" in callee_names, "Should find direct callee"
            assert "branch_a_func" in callee_names, "Should find branch A"
            assert "branch_b_func" in callee_names, "Should find branch B"
            assert "level4_func" not in callee_names, "Should not find depth-2 callee"
            assert "common_func" not in callee_names, "Should not find depth-2 callee"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_2_includes_indirect_callees(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth=2 includes indirect callees."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "level2_func", "depth": 2},
            )

            assert response.status_code == 200
            data = response.json()

            # Should find direct and depth-2 callees
            callee_names = {callee["symbol"] for callee in data["callees"]}
            # Direct callees
            assert "level3_func" in callee_names
            assert "branch_a_func" in callee_names
            assert "branch_b_func" in callee_names
            # Depth-2 callees
            assert "level4_func" in callee_names, "Should find level3's callee"
            assert "common_func" in callee_names, "Should find branches' callee"
            # Should not find depth-3
            assert "level5_func" not in callee_names, "Should not find depth-3 callee"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_handles_diamond_pattern(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth traversal handles diamond patterns (common_func)."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "level2_func", "depth": 2},
            )

            assert response.status_code == 200
            data = response.json()

            # common_func should appear once despite being reachable via two paths
            common_count = sum(
                1 for callee in data["callees"] if callee["symbol"] == "common_func"
            )
            assert common_count == 1, "Common function should appear only once"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_5_traverses_entire_chain(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth=5 can traverse the entire call chain."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "level0_func", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()

            # Should find all functions in the chain
            callee_names = {callee["symbol"] for callee in data["callees"]}
            for i in range(1, 6):
                assert f"level{i}_func" in callee_names, f"Should find level{i}_func"

            # Should also find branches from level2
            assert "branch_a_func" in callee_names
            assert "branch_b_func" in callee_names
            assert "common_func" in callee_names


class TestDepthConsistency:
    """Test consistency of depth behavior across endpoints."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_default_depth_is_1(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that default depth (when not specified) is 1."""
        with httpx.Client() as client:
            # Test who_calls without depth
            response_no_depth = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level3_func"},
            )

            # Test who_calls with explicit depth=1
            response_depth_1 = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level3_func", "depth": 1},
            )

            assert response_no_depth.status_code == 200
            assert response_depth_1.status_code == 200

            data_no_depth = response_no_depth.json()
            data_depth_1 = response_depth_1.json()

            # Results should be identical
            callers_no_depth = {c["symbol"] for c in data_no_depth["callers"]}
            callers_depth_1 = {c["symbol"] for c in data_depth_1["callers"]}
            assert callers_no_depth == callers_depth_1, "Default depth should be 1"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_0_returns_empty(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth=0 returns empty results."""
        with httpx.Client() as client:
            # Test who_calls with depth=0
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level3_func", "depth": 0},
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
    async def test_nonexistent_symbol_depth_irrelevant(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth doesn't matter for non-existent symbols."""
        with httpx.Client() as client:
            for depth in [1, 3, 5]:
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/who_calls",
                    headers=auth_headers,
                    json={"symbol": "nonexistent_func", "depth": depth},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["callers"] == [], f"Should be empty for depth={depth}"


class TestPerformanceAtDepth:
    """Test performance implications of deep traversals."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_deep_traversal_within_timeout(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that deep traversals complete within timeout limits."""
        import time

        with httpx.Client() as client:
            # Test maximum depth traversal
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "level5_func", "depth": 5},
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 0.6, (
                f"Deep traversal took {elapsed:.3f}s, should be < 600ms"
            )

            # Test with list_dependencies as well
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "level0_func", "depth": 5},
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 0.6, (
                f"Deep traversal took {elapsed:.3f}s, should be < 600ms"
            )
