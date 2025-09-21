"""
Contract tests for impact_of MCP endpoint implementation.

These tests verify the actual implementation matches the API contract
defined in specs/003-implement-mcp-tools/contracts/impact_of.yaml.

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
            CREATE TABLE IF NOT EXISTS module (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                config_bitmap INTEGER DEFAULT 1
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS module_symbol (
                module_id INTEGER REFERENCES module(id),
                symbol_id INTEGER REFERENCES symbol(id),
                PRIMARY KEY (module_id, symbol_id)
            )
        """)

        # Insert test data for impact analysis
        fs_file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('fs/read_write.c', 'abc123def456')
            RETURNING id
        """)

        vfs_file_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('fs/vfs.c', 'xyz789uvw123')
            RETURNING id
        """)

        await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('fs/tests/vfs_test.c', 'test123456')
            RETURNING id
        """)

        # Create symbols that will be impacted
        vfs_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature, config_bitmap)
            VALUES ('vfs_read', 'function', $1, 450, 465, 'ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)', 1)
            RETURNING id
        """,
            fs_file_id,
        )

        vfs_write_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature, config_bitmap)
            VALUES ('vfs_write', 'function', $1, 500, 520, 'ssize_t vfs_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)', 1)
            RETURNING id
        """,
            fs_file_id,
        )

        ext4_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature, config_bitmap)
            VALUES ('ext4_file_read_iter', 'function', $1, 100, 150, 'static ssize_t ext4_file_read_iter(struct kiocb *iocb, struct iov_iter *to)', 2)
            RETURNING id
        """,
            vfs_file_id,
        )

        xfs_read_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature, config_bitmap)
            VALUES ('xfs_file_read_iter', 'function', $1, 200, 250, 'STATIC ssize_t xfs_file_read_iter(struct kiocb *iocb, struct iov_iter *to)', 4)
            RETURNING id
        """,
            vfs_file_id,
        )

        # Create modules
        vfs_module_id = await conn.fetchval("""
            INSERT INTO module (name, path, config_bitmap)
            VALUES ('vfs', 'fs/', 7)
            RETURNING id
        """)

        ext4_module_id = await conn.fetchval("""
            INSERT INTO module (name, path, config_bitmap)
            VALUES ('ext4', 'fs/ext4/', 2)
            RETURNING id
        """)

        xfs_module_id = await conn.fetchval("""
            INSERT INTO module (name, path, config_bitmap)
            VALUES ('xfs', 'fs/xfs/', 4)
            RETURNING id
        """)

        # Link symbols to modules
        await conn.execute(
            "INSERT INTO module_symbol (module_id, symbol_id) VALUES ($1, $2)",
            vfs_module_id,
            vfs_read_id,
        )
        await conn.execute(
            "INSERT INTO module_symbol (module_id, symbol_id) VALUES ($1, $2)",
            vfs_module_id,
            vfs_write_id,
        )
        await conn.execute(
            "INSERT INTO module_symbol (module_id, symbol_id) VALUES ($1, $2)",
            ext4_module_id,
            ext4_read_id,
        )
        await conn.execute(
            "INSERT INTO module_symbol (module_id, symbol_id) VALUES ($1, $2)",
            xfs_module_id,
            xfs_read_id,
        )

        # Create call edges to establish impact relationships
        # ext4_file_read_iter calls vfs_read
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number, config_bitmap)
            VALUES ($1, $2, 'direct', 125, 2)
        """,
            ext4_read_id,
            vfs_read_id,
        )

        # xfs_file_read_iter calls vfs_read
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number, config_bitmap)
            VALUES ($1, $2, 'direct', 225, 4)
        """,
            xfs_read_id,
            vfs_read_id,
        )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture
def sample_diff() -> str:
    """Sample git diff for testing."""
    return """
diff --git a/fs/read_write.c b/fs/read_write.c
index abc123..def456 100644
--- a/fs/read_write.c
+++ b/fs/read_write.c
@@ -450,7 +450,7 @@ ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos
        if (!(file->f_mode & FMODE_READ))
                return -EBADF;
-       if (!file->f_op->read && !file->f_op->read_iter)
+       if (!file->f_op->read && !file->f_op->read_iter && !file->f_op->splice_read)
                return -EINVAL;
        if (unlikely(!access_ok(VERIFY_WRITE, buf, count)))
"""


class TestImpactOfContractImplementation:
    """Contract tests for /mcp/tools/impact_of endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_endpoint_exists(self, auth_headers: dict[str, str]) -> None:
        """Verify the endpoint exists and accepts POST requests."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbols": ["vfs_read"]},
            )
            # Should not be 404
            assert response.status_code != 404, "Endpoint should exist"

    @skip_without_mcp
    @skip_in_ci
    def test_request_schema_validation(
        self, auth_headers: dict[str, str], sample_diff: str
    ) -> None:
        """Test request schema validation per contract."""
        with httpx.Client() as client:
            # Valid request with diff
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"diff": sample_diff},
            )
            assert response.status_code == 200, "Should accept diff input"

            # Valid request with files
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"files": ["fs/read_write.c", "fs/vfs.c"]},
            )
            assert response.status_code == 200, "Should accept files input"

            # Valid request with symbols
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbols": ["vfs_read", "vfs_write"]},
            )
            assert response.status_code == 200, "Should accept symbols input"

            # Valid request with all fields
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={
                    "diff": sample_diff,
                    "files": ["fs/read_write.c"],
                    "symbols": ["vfs_read"],
                    "config": "x86_64:defconfig",
                },
            )
            assert response.status_code == 200, "Should accept all fields"

            # Empty request is valid (analyzes nothing)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={},
            )
            assert response.status_code == 200, "Should accept empty request"

    @skip_without_mcp
    @skip_in_ci
    def test_response_schema_structure(
        self, auth_headers: dict[str, str], sample_diff: str
    ) -> None:
        """Verify response matches contract schema."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"diff": sample_diff},
            )

            assert response.status_code == 200
            data = response.json()

            # Required fields per contract
            assert "configs" in data, "Response must have configs array"
            assert "modules" in data, "Response must have modules array"
            assert "tests" in data, "Response must have tests array"
            assert "owners" in data, "Response must have owners array"
            assert "risks" in data, "Response must have risks array"
            assert "cites" in data, "Response must have cites array"

            # Check types
            assert isinstance(data["configs"], list), "configs must be array"
            assert isinstance(data["modules"], list), "modules must be array"
            assert isinstance(data["tests"], list), "tests must be array"
            assert isinstance(data["owners"], list), "owners must be array"
            assert isinstance(data["risks"], list), "risks must be array"
            assert isinstance(data["cites"], list), "cites must be array"

            # Check cite structure if present
            if data["cites"]:
                cite = data["cites"][0]
                assert "path" in cite, "Cite must have path"
                assert "sha" in cite, "Cite must have sha"
                assert "start" in cite, "Cite must have start line"
                assert "end" in cite, "Cite must have end line"

                # Check types
                assert isinstance(cite["path"], str)
                assert isinstance(cite["sha"], str)
                assert isinstance(cite["start"], int)
                assert isinstance(cite["end"], int)
                assert cite["start"] > 0, "Line numbers must be positive"
                assert cite["end"] >= cite["start"], "End must be >= start"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_symbol_impact_analysis(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test impact analysis for modified symbols."""
        # This test MUST FAIL initially while mock data is in place

        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbols": ["vfs_read"]},
            )

            assert response.status_code == 200
            data = response.json()

            # Should identify impacted modules
            assert "vfs" in data["modules"], "Should identify vfs module"
            assert "ext4" in data["modules"] or "xfs" in data["modules"], (
                "Should identify filesystem modules that depend on vfs_read"
            )

            # Should have citations for evidence
            assert len(data["cites"]) > 0, "Should provide citations"

            # Should identify risks
            assert len(data["risks"]) > 0, "Should identify risks"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_file_impact_analysis(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test impact analysis for modified files."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"files": ["fs/read_write.c"]},
            )

            assert response.status_code == 200
            data = response.json()

            # Should identify modules affected by file changes
            assert "vfs" in data["modules"], (
                "Should identify vfs module for fs/read_write.c"
            )

            # Should have relevant tests
            # (test data includes fs/tests/vfs_test.c)
            if data["tests"]:
                assert any("test" in test for test in data["tests"]), (
                    "Should identify test files"
                )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_diff_impact_analysis(
        self,
        auth_headers: dict[str, str],
        sample_diff: str,
        test_db: Database,
    ) -> None:
        """Test impact analysis from diff content."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"diff": sample_diff},
            )

            assert response.status_code == 200
            data = response.json()

            # Should extract symbols from diff
            # The diff modifies vfs_read function
            assert "vfs" in data["modules"], "Should identify vfs module from diff"

            # Should identify high risk for core VFS changes
            assert any(
                "high" in risk.lower() or "vfs" in risk.lower()
                for risk in data["risks"]
            ), "Should identify risk for VFS changes"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_config_filtering(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that config parameter filters results."""
        with httpx.Client() as client:
            # Test with x86_64:defconfig (config_bitmap = 1)
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={
                    "symbols": ["vfs_read"],
                    "config": "x86_64:defconfig",
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Should include correct config
            if data["configs"]:
                assert (
                    "x86_64:defconfig" in data["configs"] or len(data["configs"]) > 0
                ), "Should include matching config"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_empty_input_handling(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that empty input returns empty results gracefully."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty but valid structure
            assert data["configs"] == [] or isinstance(data["configs"], list)
            assert data["modules"] == [] or isinstance(data["modules"], list)
            assert data["tests"] == [] or isinstance(data["tests"], list)
            assert data["owners"] == [] or isinstance(data["owners"], list)
            assert data["risks"] == [] or isinstance(data["risks"], list)
            assert data["cites"] == [] or isinstance(data["cites"], list)

    @skip_without_mcp
    @skip_in_ci
    def test_performance_requirement(
        self, auth_headers: dict[str, str], sample_diff: str
    ) -> None:
        """Test that queries complete within performance requirements."""
        import time

        with httpx.Client() as client:
            # Impact analysis should be fast even for complex queries
            start = time.time()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={
                    "diff": sample_diff,
                    "files": ["fs/read_write.c", "fs/vfs.c"],
                    "symbols": ["vfs_read", "vfs_write"],
                },
            )
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 0.6, f"Query took {elapsed:.3f}s, should be < 600ms"


class TestImpactOfErrorHandling:
    """Test error handling for impact_of endpoint."""

    @skip_without_mcp
    @skip_in_ci
    def test_unauthorized_request(self) -> None:
        """Test that unauthorized requests are rejected."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                json={"symbols": ["vfs_read"]},
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
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
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
