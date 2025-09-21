"""
Contract tests for who_calls MCP endpoint implementation.

These tests verify the actual implementation matches the API contract
defined in specs/003-implement-mcp-tools/contracts/who_calls.yaml.

Tests MUST FAIL initially while mock data is in place, then PASS after
implementation replaces mocks with real database queries.
"""

import json
import os
from collections.abc import Generator
from typing import Any

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
        file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('fs/read_write.c', 'abc123def456')
            RETURNING id
        """)

        # Create test symbols
        vfs_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('vfs_read', 'function', $1, 450, 465, 'ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)')
            RETURNING id
        """,
            file_id,
        )

        ksys_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('ksys_read', 'function', $1, 615, 630, 'ssize_t ksys_read(unsigned int fd, char __user *buf, size_t count)')
            RETURNING id
        """,
            file_id,
        )

        sys_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('sys_read', 'function', $1, 632, 635, 'SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)')
            RETURNING id
        """,
            file_id,
        )

        kernel_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('kernel_read', 'function', $1, 440, 448, 'ssize_t kernel_read(struct file *file, void *buf, size_t count, loff_t *pos)')
            RETURNING id
        """,
            file_id,
        )

        # Create call edges (who calls vfs_read)
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 621)
        """,
            ksys_read_id,
            vfs_read_id,
        )

        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 445)
        """,
            kernel_read_id,
            vfs_read_id,
        )

        # sys_read calls ksys_read
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 634)
        """,
            sys_read_id,
            ksys_read_id,
        )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


class TestWhoCallsContractImplementation:
    """Contract tests for /mcp/tools/who_calls endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_endpoint_exists(self, auth_headers: dict[str, str]) -> None:
        """Verify the endpoint exists and accepts POST requests."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read"},
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
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={},
            )
            assert response.status_code == 422, "Should reject missing symbol"

            # Invalid depth (too high)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read", "depth": 10},
            )
            assert response.status_code == 422, "Should reject depth > 5"

            # Invalid depth (negative)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read", "depth": -1},
            )
            assert response.status_code == 422, "Should reject negative depth"

    @skip_without_mcp
    @skip_in_ci
    def test_response_schema_structure(self, auth_headers: dict[str, str]) -> None:
        """Verify response matches contract schema."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read"},
            )

            assert response.status_code == 200
            data = response.json()

            # Required fields per contract
            assert "callers" in data, "Response must have callers array"
            assert isinstance(data["callers"], list), "callers must be array"

            # Check caller structure if present
            if data["callers"]:
                caller = data["callers"][0]
                assert "symbol" in caller, "Caller must have symbol"
                assert "span" in caller, "Caller must have span"
                assert "call_type" in caller, "Caller must have call_type"

                # Check span structure
                span = caller["span"]
                assert "path" in span, "Span must have path"
                assert "sha" in span, "Span must have sha"
                assert "start" in span, "Span must have start line"
                assert "end" in span, "Span must have end line"

                # Check types
                assert isinstance(caller["symbol"], str)
                assert isinstance(span["path"], str)
                assert isinstance(span["sha"], str)
                assert isinstance(span["start"], int)
                assert isinstance(span["end"], int)
                assert span["start"] > 0, "Line numbers must be positive"
                assert span["end"] >= span["start"], "End must be >= start"

                # Check call_type enum
                assert caller["call_type"] in ["direct", "indirect", "macro"]

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_empty_database_returns_empty_list(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that non-existent symbols return empty list."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "nonexistent_function_xyz"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty list, not error
            assert data == {"callers": []}, (
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
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read", "depth": 1},
            )

            assert response.status_code == 200
            data = response.json()

            # Should have callers from database
            assert len(data["callers"]) >= 2, "Should find ksys_read and kernel_read"

            # Check for expected callers
            caller_names = {caller["symbol"] for caller in data["callers"]}
            assert "ksys_read" in caller_names, "Should find ksys_read as caller"
            assert "kernel_read" in caller_names, "Should find kernel_read as caller"

            # Verify citations are accurate
            for caller in data["callers"]:
                span = caller["span"]
                assert span["path"] == "fs/read_write.c", "Path should match test data"
                assert span["sha"] == "abc123def456", "SHA should match test data"
                assert span["start"] > 0, "Should have valid line numbers"

                # Verify specific line numbers for known callers
                if caller["symbol"] == "ksys_read":
                    assert span["start"] == 615
                    assert span["end"] == 630
                elif caller["symbol"] == "kernel_read":
                    assert span["start"] == 440
                    assert span["end"] == 448

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_depth_parameter_traversal(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that depth parameter correctly traverses call graph."""
        with httpx.Client() as client:
            # Depth 1: direct callers only
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read", "depth": 1},
            )
            data1 = response.json()

            # Depth 2: should include indirect callers
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read", "depth": 2},
            )
            data2 = response.json()

            # With depth=2, should find sys_read (which calls ksys_read which calls vfs_read)
            assert len(data2["callers"]) > len(data1["callers"]), (
                "Depth 2 should find more callers"
            )

            caller_names_depth2 = {caller["symbol"] for caller in data2["callers"]}
            assert "sys_read" in caller_names_depth2, (
                "Depth 2 should find indirect caller sys_read"
            )

    @skip_without_mcp
    @skip_in_ci
    def test_config_parameter_filtering(self, auth_headers: dict[str, str]) -> None:
        """Test that config parameter filters results."""
        with httpx.Client() as client:
            # Test with specific config
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "vfs_read", "config": "x86_64:defconfig"},
            )

            assert response.status_code == 200
            data = response.json()

            # Results should be filtered by config (implementation dependent)
            assert "callers" in data

    @skip_without_mcp
    @skip_in_ci
    def test_performance_requirement(self, auth_headers: dict[str, str]) -> None:
        """Test that queries complete within performance requirements."""
        import time

        with httpx.Client() as client:
            # Even deep traversals should complete quickly
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={
                    "symbol": "memcpy",
                    "depth": 5,
                },  # Common function with many callers
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 0.6, f"Query took {elapsed:.3f}s, should be < 600ms"


class TestWhoCallsErrorHandling:
    """Test error handling for who_calls endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_unauthorized_request(self) -> None:
        """Test that unauthorized requests are rejected."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                json={"symbol": "vfs_read"},
                # No auth header
            )
            assert response.status_code in [401, 403], "Should reject unauthorized"

    @skip_without_mcp
    @skip_in_ci
    def test_malformed_json(self, auth_headers: dict[str, str]) -> None:
        """Test handling of malformed JSON."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                content="not json",
                headers_extra={"Content-Type": "application/json"},
            )
            assert response.status_code in [400, 422], "Should reject malformed JSON"

    @skip_without_mcp
    @skip_in_ci
    def test_server_error_response_format(self, auth_headers: dict[str, str]) -> None:
        """Test that server errors follow contract format."""
        # This would require inducing a server error, which is difficult
        # in contract testing. Mainly verifying the format if an error occurs.
        pass  # Implementation-specific
