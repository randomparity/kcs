"""
Contract tests for entrypoint_flow MCP endpoint implementation.

These tests verify the actual implementation matches the API contract
defined in specs/003-implement-mcp-tools/contracts/entrypoint_flow.yaml.

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

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entrypoint (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                symbol_id INTEGER REFERENCES symbol(id),
                syscall_number INTEGER,
                config_bitmap INTEGER DEFAULT 1
            )
        """)

        # Insert test data for syscall flow
        await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('arch/x86/entry/syscalls/syscall_64.tbl', 'def456abc789')
            RETURNING id
        """)

        fs_file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('fs/read_write.c', 'abc123def456')
            RETURNING id
        """)

        # Create syscall-related symbols
        sys_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('sys_read', 'function', $1, 632, 635, 'SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)')
            RETURNING id
        """,
            fs_file_id,
        )

        ksys_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('ksys_read', 'function', $1, 615, 630, 'ssize_t ksys_read(unsigned int fd, char __user *buf, size_t count)')
            RETURNING id
        """,
            fs_file_id,
        )

        vfs_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('vfs_read', 'function', $1, 450, 465, 'ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)')
            RETURNING id
        """,
            fs_file_id,
        )

        # Create entry point for sys_read
        await conn.execute(
            """
            INSERT INTO entrypoint (name, type, symbol_id, syscall_number)
            VALUES ('read', 'syscall', $1, 0)
        """,
            sys_read_id,
        )

        # Create call edges for the flow
        # sys_read -> ksys_read
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 634)
        """,
            sys_read_id,
            ksys_read_id,
        )

        # ksys_read -> vfs_read
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 621)
        """,
            ksys_read_id,
            vfs_read_id,
        )

        # Add ioctl entry point for testing
        ioctl_symbol_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('device_ioctl', 'function', $1, 100, 150, 'long device_ioctl(struct file *file, unsigned int cmd, unsigned long arg)')
            RETURNING id
        """,
            fs_file_id,
        )

        await conn.execute(
            """
            INSERT INTO entrypoint (name, type, symbol_id)
            VALUES ('device_ioctl', 'ioctl', $1)
        """,
            ioctl_symbol_id,
        )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


class TestEntrypointFlowContractImplementation:
    """Contract tests for /mcp/tools/entrypoint_flow endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_endpoint_exists(self, auth_headers: dict[str, str]) -> None:
        """Verify the endpoint exists and accepts POST requests."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_read"},
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
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={},
            )
            assert response.status_code == 422, "Should reject missing entry field"

            # Valid with optional config
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_read", "config": "x86_64:defconfig"},
            )
            assert response.status_code in [200, 500], (
                "Should accept valid request with config"
            )

    @skip_without_mcp
    @skip_in_ci
    def test_response_schema_structure(self, auth_headers: dict[str, str]) -> None:
        """Verify response matches contract schema."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_read"},
            )

            assert response.status_code == 200
            data = response.json()

            # Required fields per contract
            assert "steps" in data, "Response must have steps array"
            assert isinstance(data["steps"], list), "steps must be array"

            # Check step structure if present
            if data["steps"]:
                step = data["steps"][0]
                assert "edge" in step, "Step must have edge type"
                assert "from" in step, "Step must have from field"
                assert "to" in step, "Step must have to field"
                assert "span" in step, "Step must have span"

                # Check edge enum values
                assert step["edge"] in [
                    "syscall",
                    "function_call",
                    "indirect_call",
                    "macro_expansion",
                ], f"Invalid edge type: {step['edge']}"

                # Check span structure
                span = step["span"]
                assert "path" in span, "Span must have path"
                assert "sha" in span, "Span must have sha"
                assert "start" in span, "Span must have start line"
                assert "end" in span, "Span must have end line"

                # Check types
                assert isinstance(step["from"], str)
                assert isinstance(step["to"], str)
                assert isinstance(span["path"], str)
                assert isinstance(span["sha"], str)
                assert isinstance(span["start"], int)
                assert isinstance(span["end"], int)
                assert span["start"] > 0, "Line numbers must be positive"
                assert span["end"] >= span["start"], "End must be >= start"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_syscall_mapping(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that syscall numbers are mapped to implementation functions."""
        # This test MUST FAIL initially while mock data is in place

        with httpx.Client() as client:
            # Test with syscall number format
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_read"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should have steps showing the flow
            assert len(data["steps"]) > 0, "Should have flow steps"

            # First step should be syscall mapping
            first_step = data["steps"][0]
            assert first_step["edge"] == "syscall", "First step should be syscall edge"
            assert "sys_read" in first_step["to"], (
                "Should map to sys_read implementation"
            )

            # Test with syscall name
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "read"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should also work with syscall name
            assert len(data["steps"]) > 0, "Should have flow steps for name"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_flow_steps_generation(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that flow steps correctly trace through call graph."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_read"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should have multiple steps (syscall -> sys_read -> ksys_read -> vfs_read)
            assert len(data["steps"]) >= 3, "Should trace through multiple functions"

            # Verify flow order
            step_targets = [step["to"] for step in data["steps"]]

            # Should include these functions in the flow
            expected_functions = ["sys_read", "ksys_read", "vfs_read"]
            for func in expected_functions:
                assert any(func in target for target in step_targets), (
                    f"Flow should include {func}"
                )

            # Verify citations are accurate
            for step in data["steps"]:
                span = step["span"]
                assert span["path"] in [
                    "arch/x86/entry/syscalls/syscall_64.tbl",
                    "fs/read_write.c",
                ], "Path should match test data"
                assert span["sha"] in ["def456abc789", "abc123def456"], (
                    "SHA should match test data"
                )
                assert span["start"] > 0, "Should have valid line numbers"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_empty_flow_for_unknown_entry(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that unknown entry points return empty flow."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_nonexistent"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty steps list, not error
            assert data == {"steps": []}, "Unknown entry should return empty steps"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_ioctl_entrypoint(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that ioctl entry points are handled correctly."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "device_ioctl"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should have at least the entry point itself
            if data["steps"]:
                # Verify it's recognized as an ioctl
                assert any(
                    step["edge"] in ["function_call", "indirect_call"]
                    for step in data["steps"]
                ), "Ioctl should be traced"

    @skip_without_mcp
    @skip_in_ci
    def test_config_parameter_filtering(self, auth_headers: dict[str, str]) -> None:
        """Test that config parameter filters results."""
        with httpx.Client() as client:
            # Test with specific config
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_read", "config": "x86_64:defconfig"},
            )

            assert response.status_code == 200
            data = response.json()

            # Results should be filtered by config (implementation dependent)
            assert "steps" in data

    @skip_without_mcp
    @skip_in_ci
    def test_performance_requirement(self, auth_headers: dict[str, str]) -> None:
        """Test that queries complete within performance requirements."""
        import time

        with httpx.Client() as client:
            # Entry point tracing should be fast
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "__NR_read"},
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 0.6, f"Query took {elapsed:.3f}s, should be < 600ms"


class TestEntrypointFlowErrorHandling:
    """Test error handling for entrypoint_flow endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_unauthorized_request(self) -> None:
        """Test that unauthorized requests are rejected."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                json={"entry": "__NR_read"},
                # No auth header
            )
            assert response.status_code in [401, 403], "Should reject unauthorized"

    @skip_without_mcp
    @skip_in_ci
    def test_malformed_json(self, auth_headers: dict[str, str]) -> None:
        """Test handling of malformed JSON."""
        with httpx.Client() as client:
            headers = auth_headers.copy()
            headers["Content-Type"] = "application/json"
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=headers,
                content="not json",
            )
            assert response.status_code in [400, 422], "Should reject malformed JSON"

    @skip_without_mcp
    @skip_in_ci
    def test_server_error_response_format(self, auth_headers: dict[str, str]) -> None:
        """Test that server errors follow contract format."""
        # This would require inducing a server error, which is difficult
        # in contract testing. Mainly verifying the format if an error occurs.
        pass  # Implementation-specific
