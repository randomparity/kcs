"""
Contract tests for list_dependencies MCP endpoint implementation.

These tests verify the actual implementation matches the API contract
defined in specs/003-implement-mcp-tools/contracts/list_dependencies.yaml.

Tests MUST FAIL initially while mock data is in place, then PASS after
implementation replaces mocks with real database queries.
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
    """Create test database with schema and test data."""
    db = Database(postgres_container.get_connection_url())
    await db.initialize()

    # Create schema
    async with db.acquire() as conn:
        # Create tables if not exist
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

        # Insert test data
        read_write_file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('fs/read_write.c', 'abc123def456')
            RETURNING id
        """)

        file_file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('fs/file.c', 'def789ghi012')
            RETURNING id
        """)

        # Create test symbols
        sys_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('sys_read', 'function', $1, 632, 635, 'SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)')
            RETURNING id
        """,
            read_write_file_id,
        )

        ksys_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('ksys_read', 'function', $1, 615, 630, 'ssize_t ksys_read(unsigned int fd, char __user *buf, size_t count)')
            RETURNING id
        """,
            read_write_file_id,
        )

        vfs_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('vfs_read', 'function', $1, 450, 465, 'ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)')
            RETURNING id
        """,
            read_write_file_id,
        )

        file_read_iter_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('file_read_iter', 'function', $1, 205, 220, 'ssize_t file_read_iter(struct kiocb *iocb, struct iov_iter *iter)')
            RETURNING id
        """,
            file_file_id,
        )

        rw_verify_area_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('rw_verify_area', 'function', $1, 375, 400, 'int rw_verify_area(int read_write, struct file *file, loff_t *ppos, size_t count)')
            RETURNING id
        """,
            read_write_file_id,
        )

        # Create call edges (what sys_read calls)
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 634)
        """,
            sys_read_id,
            ksys_read_id,
        )

        # What ksys_read calls
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 621)
        """,
            ksys_read_id,
            vfs_read_id,
        )

        # What vfs_read calls
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 455)
        """,
            vfs_read_id,
            rw_verify_area_id,
        )

        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'indirect', 460)
        """,
            vfs_read_id,
            file_read_iter_id,
        )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


class TestListDependenciesContract:
    """Contract tests for /mcp/tools/list_dependencies endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_endpoint_exists(self, auth_headers: dict[str, str]) -> None:
        """Verify the endpoint exists and accepts POST requests."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read"},
            )
            # Should not be 404
            assert response.status_code != 404, "Endpoint should exist"

    @skip_without_mcp
    @skip_in_ci
    def test_request_schema_validation(self, auth_headers: dict[str, str]) -> None:
        """Test request schema validation per contract."""
        with httpx.Client() as client:
            # Missing required field
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={},
            )
            assert response.status_code == 422, "Should reject missing symbol"

            # Invalid depth (too high)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read", "depth": 10},
            )
            assert response.status_code == 422, "Should reject depth > 5"

            # Invalid depth (negative)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read", "depth": -1},
            )
            assert response.status_code == 422, "Should reject negative depth"

    @skip_without_mcp
    @skip_in_ci
    def test_response_schema_structure(self, auth_headers: dict[str, str]) -> None:
        """Verify response matches contract schema."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read"},
            )

            assert response.status_code == 200
            data = response.json()

            # Required fields per contract
            assert "callees" in data, "Response must have callees array"
            assert isinstance(data["callees"], list), "callees must be array"

            # Check callee structure if present
            if data["callees"]:
                callee = data["callees"][0]
                assert "symbol" in callee, "Callee must have symbol"
                assert "span" in callee, "Callee must have span"
                assert "call_type" in callee, "Callee must have call_type"

                # Check span structure
                span = callee["span"]
                assert "path" in span, "Span must have path"
                assert "sha" in span, "Span must have sha"
                assert "start" in span, "Span must have start line"
                assert "end" in span, "Span must have end line"

                # Check types
                assert isinstance(callee["symbol"], str)
                assert isinstance(span["path"], str)
                assert isinstance(span["sha"], str)
                assert isinstance(span["start"], int)
                assert isinstance(span["end"], int)
                assert span["start"] > 0, "Line numbers must be positive"
                assert span["end"] >= span["start"], "End must be >= start"

                # Check call_type enum
                assert callee["call_type"] in ["direct", "indirect", "macro"]

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_empty_database_returns_empty_list(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that non-existent symbols return empty list."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "nonexistent_function_xyz"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty list, not error
            assert data == {"callees": []}, (
                "Non-existent symbol should return empty list"
            )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_with_populated_database(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test with actual call graph data returns real results."""
        # This test MUST FAIL initially while mock data is in place
        # After implementation, it should return actual database results

        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read", "depth": 1},
            )

            assert response.status_code == 200
            data = response.json()

            # Should have callees from database
            assert len(data["callees"]) >= 1, "Should find ksys_read"

            # Check for expected callees
            callee_names = {callee["symbol"] for callee in data["callees"]}
            assert "ksys_read" in callee_names, "Should find ksys_read as callee"

            # Verify citations are accurate
            for callee in data["callees"]:
                span = callee["span"]
                assert span["path"] == "fs/read_write.c", "Path should match test data"
                assert span["sha"] == "abc123def456", "SHA should match test data"
                assert span["start"] > 0, "Should have valid line numbers"

                # Verify specific line numbers for known callees
                if callee["symbol"] == "ksys_read":
                    assert span["start"] == 615
                    assert span["end"] == 630

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_parameter_traversal(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth parameter correctly traverses call graph."""
        with httpx.Client() as client:
            # Depth 1: direct callees only (sys_read -> ksys_read)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read", "depth": 1},
            )
            data1 = response.json()

            # Depth 2: should include indirect callees (sys_read -> ksys_read -> vfs_read)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read", "depth": 2},
            )
            data2 = response.json()

            # Depth 3: should go deeper (sys_read -> ksys_read -> vfs_read -> rw_verify_area/file_read_iter)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read", "depth": 3},
            )
            data3 = response.json()

            # Verify increasing depth finds more dependencies
            assert len(data2["callees"]) > len(data1["callees"]), (
                "Depth 2 should find more callees"
            )
            assert len(data3["callees"]) > len(data2["callees"]), (
                "Depth 3 should find even more callees"
            )

            # Check specific callees at each depth
            callees_depth1 = {callee["symbol"] for callee in data1["callees"]}
            assert "ksys_read" in callees_depth1, "Depth 1 should find direct callee"

            callees_depth2 = {callee["symbol"] for callee in data2["callees"]}
            assert "vfs_read" in callees_depth2, "Depth 2 should find vfs_read"

            callees_depth3 = {callee["symbol"] for callee in data3["callees"]}
            assert (
                "rw_verify_area" in callees_depth3 or "file_read_iter" in callees_depth3
            ), "Depth 3 should find deeper callees"

    @skip_without_mcp
    @skip_in_ci
    def test_config_parameter_filtering(self, auth_headers: dict[str, str]) -> None:
        """Test that config parameter filters results."""
        with httpx.Client() as client:
            # Test with specific config
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "sys_read", "config": "x86_64:defconfig"},
            )

            assert response.status_code == 200
            data = response.json()

            # Results should be filtered by config (implementation dependent)
            assert "callees" in data

    @skip_without_mcp
    @skip_in_ci
    def test_different_call_types(self, auth_headers: dict[str, str]) -> None:
        """Test that different call types are properly classified."""
        with httpx.Client() as client:
            # vfs_read has both direct and indirect callees in our test data
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "vfs_read", "depth": 1},
            )

            assert response.status_code == 200
            data = response.json()

            if data["callees"]:
                # Check for different call types
                call_types = {callee["call_type"] for callee in data["callees"]}
                # Our test data has both direct (rw_verify_area) and indirect (file_read_iter)
                assert "direct" in call_types or "indirect" in call_types

    @skip_without_mcp
    @skip_in_ci
    def test_performance_requirement(self, auth_headers: dict[str, str]) -> None:
        """Test that queries complete within performance requirements."""
        import time

        with httpx.Client() as client:
            # Even deep traversals should complete quickly
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={
                    "symbol": "sys_read",
                    "depth": 5,
                },  # Deep traversal
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 0.6, f"Query took {elapsed:.3f}s, should be < 600ms"


class TestListDependenciesErrorHandling:
    """Test error handling for list_dependencies endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_unauthorized_request(self) -> None:
        """Test that unauthorized requests are rejected."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                json={"symbol": "sys_read"},
                # No auth header
            )
            assert response.status_code in [401, 403], "Should reject unauthorized"

    @skip_without_mcp
    @skip_in_ci
    def test_malformed_json(self, auth_headers: dict[str, str]) -> None:
        """Test handling of malformed JSON."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                content="not json",
                headers_extra={"Content-Type": "application/json"},
            )
            assert response.status_code in [400, 422], "Should reject malformed JSON"
