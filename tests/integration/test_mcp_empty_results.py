"""
Integration tests for MCP empty results handling.

These tests verify that all MCP endpoints handle empty results gracefully,
returning empty lists/objects rather than errors when data is not found.
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
async def empty_db(postgres_container: PostgresContainer) -> Database:
    """Create test database with no data for empty results testing."""
    db = Database(postgres_container.get_connection_url())
    await db.initialize()

    # Create schema only, no data
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

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entrypoint (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                symbol_id INTEGER REFERENCES symbol(id),
                file_id INTEGER REFERENCES file(id),
                line_number INTEGER,
                syscall_number INTEGER,
                config_bitmap INTEGER DEFAULT 1
            )
        """)

    yield db
    await db.close()


@pytest_asyncio.fixture
async def sparse_db(postgres_container: PostgresContainer) -> Database:
    """Create test database with minimal data for edge case testing."""
    db = Database(postgres_container.get_connection_url())
    await db.initialize()

    # Create schema with minimal data
    async with db.acquire() as conn:
        # Create tables (same as empty_db)
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

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entrypoint (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                symbol_id INTEGER REFERENCES symbol(id),
                file_id INTEGER REFERENCES file(id),
                line_number INTEGER,
                syscall_number INTEGER,
                config_bitmap INTEGER DEFAULT 1
            )
        """)

        # Add minimal data - just a few symbols with no connections
        file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('kernel/test_sparse.c', 'sha_sparse_test')
            RETURNING id
        """)

        # Add isolated functions (no call edges)
        isolated_funcs = ["orphan_func1", "orphan_func2", "orphan_func3"]
        for i, func_name in enumerate(isolated_funcs):
            await conn.fetchval(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
                VALUES ($1, 'function', $2, $3, $4, $5)
                RETURNING id
                """,
                func_name,
                file_id,
                100 + i * 100,
                100 + i * 100 + 50,
                f"void {func_name}(void)",
            )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


class TestEmptyDatabaseHandling:
    """Test API behavior with completely empty database."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_who_calls_empty_db(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test who_calls returns empty list with empty database."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "any_function", "depth": 3},
            )

            assert response.status_code == 200
            data = response.json()
            assert data == {"callers": []}, "Should return empty callers list"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_list_dependencies_empty_db(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test list_dependencies returns empty list with empty database."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "any_function", "depth": 3},
            )

            assert response.status_code == 200
            data = response.json()
            assert data == {"callees": []}, "Should return empty callees list"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_entrypoint_flow_empty_db(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test entrypoint_flow returns empty or error with empty database."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "sys_read"},
            )

            # Should either return empty steps or 404
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert "steps" in data
                assert len(data["steps"]) == 0 or data["steps"] is None

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_impact_of_empty_db(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test impact_of returns minimal impact with empty database."""
        with httpx.Client(timeout=10.0) as client:
            # Test with symbol input
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbol": "any_function"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should have required fields but minimal impact
            assert "affected_symbols" in data
            assert "subsystems" in data
            assert "risk_level" in data

            # Empty DB should show minimal or no impact
            assert len(data["affected_symbols"]) == 0
            assert data["risk_level"] in ["low", "minimal", "none"]


class TestNonExistentSymbols:
    """Test API behavior when querying for non-existent symbols."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_who_calls_nonexistent(
        self, auth_headers: dict[str, str], sparse_db: Database
    ) -> None:
        """Test who_calls with non-existent symbol returns empty list."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "definitely_not_exists_xyz123", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert data == {"callers": []}, (
                "Non-existent symbol should return empty list"
            )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_list_dependencies_nonexistent(
        self, auth_headers: dict[str, str], sparse_db: Database
    ) -> None:
        """Test list_dependencies with non-existent symbol returns empty list."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "definitely_not_exists_xyz123", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert data == {"callees": []}, (
                "Non-existent symbol should return empty list"
            )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_entrypoint_flow_nonexistent(
        self, auth_headers: dict[str, str], sparse_db: Database
    ) -> None:
        """Test entrypoint_flow with non-existent entry point."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "nonexistent_syscall_xyz"},
            )

            # Should gracefully handle non-existent entry
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert "steps" in data
                # Steps should be empty or minimal
                assert len(data.get("steps", [])) <= 1

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_impact_of_nonexistent(
        self, auth_headers: dict[str, str], sparse_db: Database
    ) -> None:
        """Test impact_of with non-existent symbol shows no impact."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbol": "definitely_not_exists_xyz123"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return valid structure but no impact
            assert "affected_symbols" in data
            assert len(data["affected_symbols"]) == 0
            assert data["risk_level"] in ["low", "minimal", "none"]


class TestIsolatedSymbols:
    """Test API behavior with symbols that have no connections."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_who_calls_orphan_function(
        self, auth_headers: dict[str, str], sparse_db: Database
    ) -> None:
        """Test who_calls for function with no callers."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "orphan_func1", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert data == {"callers": []}, "Orphan function should have no callers"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_list_dependencies_orphan_function(
        self, auth_headers: dict[str, str], sparse_db: Database
    ) -> None:
        """Test list_dependencies for function with no callees."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "orphan_func2", "depth": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert data == {"callees": []}, (
                "Orphan function should have no dependencies"
            )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_impact_of_orphan_function(
        self, auth_headers: dict[str, str], sparse_db: Database
    ) -> None:
        """Test impact_of for isolated function shows minimal impact."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbol": "orphan_func3"},
            )

            assert response.status_code == 200
            data = response.json()

            # Isolated function should have minimal impact
            assert "affected_symbols" in data
            # Might include the function itself or be empty
            assert len(data["affected_symbols"]) <= 1
            assert data["risk_level"] in ["low", "minimal", "none"]


class TestEdgeCaseInputs:
    """Test API behavior with edge case inputs."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_empty_symbol_name(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test endpoints with empty symbol name."""
        with httpx.Client(timeout=10.0) as client:
            # Test who_calls with empty symbol
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "", "depth": 3},
            )

            # Should either reject (422) or return empty
            assert response.status_code in [200, 422]
            if response.status_code == 200:
                data = response.json()
                assert data == {"callers": []}

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_special_characters_in_symbol(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test endpoints with special characters in symbol names."""
        with httpx.Client(timeout=10.0) as client:
            # Test with symbol containing special characters
            special_symbols = [
                "func::name",
                "func<template>",
                "func[array]",
                "func.member",
                "func->pointer",
                "func$dollar",
                "func@at",
                "func#hash",
            ]

            for symbol in special_symbols:
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/who_calls",
                    headers=auth_headers,
                    json={"symbol": symbol, "depth": 3},
                )

                # Should handle gracefully
                assert response.status_code in [200, 422]
                if response.status_code == 200:
                    data = response.json()
                    assert "callers" in data

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_very_long_symbol_name(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test endpoints with extremely long symbol names."""
        with httpx.Client(timeout=10.0) as client:
            # Create a very long symbol name
            long_symbol = "very_long_function_name_" * 100  # ~2500 characters

            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": long_symbol, "depth": 3},
            )

            # Should handle gracefully
            assert response.status_code in [200, 422]
            if response.status_code == 200:
                data = response.json()
                assert data == {"callees": []}

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_null_depth(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test endpoints when depth is null or missing."""
        with httpx.Client(timeout=10.0) as client:
            # Test without depth parameter
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "some_func"},
            )

            # Should use default depth
            assert response.status_code == 200
            data = response.json()
            assert "callers" in data

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_negative_depth(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test endpoints with negative depth values."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "some_func", "depth": -1},
            )

            # Should reject negative depth
            assert response.status_code in [200, 422]
            if response.status_code == 200:
                # If accepted, should treat as 0 or default
                data = response.json()
                assert "callees" in data


class TestMultipleEmptyQueries:
    """Test behavior with multiple empty result queries."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_batch_empty_queries(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test multiple empty queries in succession."""
        with httpx.Client(timeout=10.0) as client:
            # Make multiple queries that should all return empty
            queries = [
                ("who_calls", {"symbol": "func1", "depth": 3}),
                ("list_dependencies", {"symbol": "func2", "depth": 3}),
                ("who_calls", {"symbol": "func3", "depth": 5}),
                ("list_dependencies", {"symbol": "func4", "depth": 1}),
            ]

            for endpoint, params in queries:
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/{endpoint}",
                    headers=auth_headers,
                    json=params,
                )

                assert response.status_code == 200
                data = response.json()

                if endpoint == "who_calls":
                    assert data == {"callers": []}
                else:
                    assert data == {"callees": []}

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_all_endpoints_empty_results(
        self, auth_headers: dict[str, str], empty_db: Database
    ) -> None:
        """Test all four endpoints handle empty results correctly."""
        with httpx.Client(timeout=15.0) as client:
            # Test who_calls
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "test_func", "depth": 3},
            )
            assert response.status_code == 200
            assert response.json() == {"callers": []}

            # Test list_dependencies
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "test_func", "depth": 3},
            )
            assert response.status_code == 200
            assert response.json() == {"callees": []}

            # Test entrypoint_flow
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "sys_test"},
            )
            assert response.status_code in [200, 404]

            # Test impact_of
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbol": "test_func"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "affected_symbols" in data
            assert len(data["affected_symbols"]) == 0
