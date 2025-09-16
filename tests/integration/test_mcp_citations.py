"""
Integration tests for MCP endpoint citation accuracy.

These tests verify that all MCP endpoints return valid Span objects
with accurate file paths, line numbers, and SHA values.
"""

import os
from collections.abc import Generator
from pathlib import Path

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

        # Insert test data with known citations
        file1_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('drivers/gpu/drm/drm_atomic.c', 'sha1_abc123')
            RETURNING id
        """)

        file2_id = await conn.fetchval("""
            INSERT INTO file (path, sha)
            VALUES ('kernel/sched/core.c', 'sha2_def456')
            RETURNING id
        """)

        # Create test symbols with specific line numbers
        await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('drm_atomic_state_init', 'function', $1, 100, 150, 'int drm_atomic_state_init(struct drm_device *dev, struct drm_atomic_state *state)')
            RETURNING id
            """,
            file1_id,
        )

        schedule_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('schedule', 'function', $1, 5000, 5050, 'void schedule(void)')
            RETURNING id
            """,
            file2_id,
        )

        __schedule_id = await conn.fetchval(
            """
            INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
            VALUES ('__schedule', 'function', $1, 4800, 4900, 'static void __schedule(bool preempt)')
            RETURNING id
            """,
            file2_id,
        )

        # Create call edges with line numbers
        await conn.execute(
            """
            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
            VALUES ($1, $2, 'direct', 5025)
            """,
            schedule_id,
            __schedule_id,
        )

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


def validate_span(span: dict, test_db_files: set[str] | None = None) -> None:
    """Validate a Span object has all required fields with valid values.

    Args:
        span: The span dictionary to validate
        test_db_files: Optional set of valid file paths from test database
    """
    # Required fields
    assert "path" in span, "Span must have path"
    assert "sha" in span, "Span must have sha"
    assert "start" in span, "Span must have start line"
    assert "end" in span, "Span must have end line"

    # Type checks
    assert isinstance(span["path"], str), "Path must be string"
    assert isinstance(span["sha"], str), "SHA must be string"
    assert isinstance(span["start"], int), "Start line must be integer"
    assert isinstance(span["end"], int), "End line must be integer"

    # Value checks
    assert len(span["path"]) > 0, "Path cannot be empty"
    assert len(span["sha"]) > 0, "SHA cannot be empty"
    assert span["start"] > 0, "Start line must be positive"
    assert span["end"] > 0, "End line must be positive"
    assert span["end"] >= span["start"], "End line must be >= start line"

    # Path validation if test data provided
    if test_db_files:
        assert span["path"] in test_db_files, f"Path {span['path']} not in test data"


class TestCitationAccuracy:
    """Test citation accuracy across all MCP endpoints."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_who_calls_citations(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that who_calls endpoint returns valid citations."""
        test_files = {"drivers/gpu/drm/drm_atomic.c", "kernel/sched/core.c"}

        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "__schedule"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check all returned callers have valid citations
            assert "callers" in data
            for caller in data["callers"]:
                assert "span" in caller, "Each caller must have a span"
                validate_span(caller["span"], test_files)

                # Verify SHA matches test data
                if caller["span"]["path"] == "kernel/sched/core.c":
                    assert caller["span"]["sha"] == "sha2_def456"
                elif caller["span"]["path"] == "drivers/gpu/drm/drm_atomic.c":
                    assert caller["span"]["sha"] == "sha1_abc123"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_list_dependencies_citations(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that list_dependencies endpoint returns valid citations."""
        test_files = {"drivers/gpu/drm/drm_atomic.c", "kernel/sched/core.c"}

        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "schedule"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check all returned callees have valid citations
            assert "callees" in data
            for callee in data["callees"]:
                assert "span" in callee, "Each callee must have a span"
                validate_span(callee["span"], test_files)

                # Verify line numbers are reasonable
                if callee["symbol"] == "__schedule":
                    assert callee["span"]["start"] == 4800
                    assert callee["span"]["end"] == 4900

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_entrypoint_flow_citations(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that entrypoint_flow endpoint returns valid citations."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "schedule"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check all flow steps have valid citations
            assert "steps" in data
            for step in data["steps"]:
                assert "span" in step, "Each step must have a span"
                validate_span(step["span"])

                # Additional checks for step structure
                assert "from" in step
                assert "to" in step
                assert "edge" in step

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_impact_of_citations(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that impact_of endpoint returns valid citations."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={
                    "symbols": ["schedule", "__schedule"],
                    "files": ["kernel/sched/core.c"],
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Check all citations are valid
            assert "cites" in data
            assert isinstance(data["cites"], list)
            for cite in data["cites"]:
                validate_span(cite)

    @skip_without_mcp
    @skip_in_ci
    def test_search_code_citations(self, auth_headers: dict[str, str]) -> None:
        """Test that search_code endpoint returns valid citations."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/search_code",
                headers=auth_headers,
                json={"query": "schedule", "limit": 5},
            )

            assert response.status_code == 200
            data = response.json()

            # Check all search results have valid citations
            assert "results" in data
            for result in data["results"]:
                assert "span" in result, "Each result must have a span"
                validate_span(result["span"])

                # Search results should also have relevance info
                assert "symbol" in result or "content" in result

    @skip_without_mcp
    @skip_in_ci
    def test_get_symbol_citations(self, auth_headers: dict[str, str]) -> None:
        """Test that get_symbol endpoint returns valid citations."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/get_symbol",
                headers=auth_headers,
                json={"symbol": "schedule"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check symbol has valid citation
            assert "span" in data, "Symbol must have a span"
            validate_span(data["span"])

            # Symbol should have additional metadata
            assert "name" in data
            assert "kind" in data
            assert "signature" in data or "summary" in data


class TestCitationConsistency:
    """Test that citations are consistent across related queries."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_same_symbol_same_citation(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that same symbol returns same citation across endpoints."""
        symbol = "schedule"
        citations = {}

        with httpx.Client() as client:
            # Get citation from get_symbol
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/get_symbol",
                headers=auth_headers,
                json={"symbol": symbol},
            )
            if response.status_code == 200:
                data = response.json()
                if "span" in data:
                    citations["get_symbol"] = data["span"]

            # Get citation from search_code
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/search_code",
                headers=auth_headers,
                json={"query": symbol, "limit": 1},
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    if "span" in data["results"][0]:
                        citations["search_code"] = data["results"][0]["span"]

            # Compare citations if we got multiple
            if len(citations) > 1:
                first_citation = None
                for _endpoint, citation in citations.items():
                    if first_citation is None:
                        first_citation = citation
                    else:
                        # Same symbol should have same file and line range
                        assert citation["path"] == first_citation["path"], (
                            f"Path mismatch for {symbol}"
                        )
                        assert citation["start"] == first_citation["start"], (
                            f"Start line mismatch for {symbol}"
                        )
                        assert citation["end"] == first_citation["end"], (
                            f"End line mismatch for {symbol}"
                        )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_caller_callee_relationship_citations(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that caller/callee relationships have consistent citations."""
        with httpx.Client() as client:
            # Get callers of __schedule
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "__schedule"},
            )
            assert response.status_code == 200
            callers_data = response.json()

            # Get callees of schedule
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                headers=auth_headers,
                json={"symbol": "schedule"},
            )
            assert response.status_code == 200
            callees_data = response.json()

            # If schedule calls __schedule, verify citations match
            schedule_caller = None
            for caller in callers_data.get("callers", []):
                if caller.get("symbol") == "schedule":
                    schedule_caller = caller
                    break

            __schedule_callee = None
            for callee in callees_data.get("callees", []):
                if callee.get("symbol") == "__schedule":
                    __schedule_callee = callee
                    break

            if schedule_caller and __schedule_callee:
                # The caller's span should be for schedule function
                assert schedule_caller["span"]["start"] == 5000
                assert schedule_caller["span"]["end"] == 5050

                # The callee's span should be for __schedule function
                assert __schedule_callee["span"]["start"] == 4800
                assert __schedule_callee["span"]["end"] == 4900


class TestCitationEdgeCases:
    """Test citation handling for edge cases."""

    @skip_without_mcp
    @skip_in_ci
    def test_nonexistent_symbol_no_citation(self, auth_headers: dict[str, str]) -> None:
        """Test that non-existent symbols don't return invalid citations."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/who_calls",
                headers=auth_headers,
                json={"symbol": "nonexistent_function_xyz123"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty list, not invalid citations
            assert data == {"callers": []}

    @skip_without_mcp
    @skip_in_ci
    def test_empty_database_no_citations(self, auth_headers: dict[str, str]) -> None:
        """Test that empty queries return no citations, not invalid ones."""
        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                headers=auth_headers,
                json={"entry": "nonexistent_entry"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty steps, not invalid citations
            assert data == {"steps": []}

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_large_line_numbers(
        self, auth_headers: dict[str, str], test_db: Database
    ) -> None:
        """Test that large line numbers are handled correctly."""
        # Add symbol with large line numbers
        async with test_db.acquire() as conn:
            file_id = await conn.fetchval("""
                INSERT INTO file (path, sha)
                VALUES ('mm/huge_file.c', 'sha_huge')
                RETURNING id
            """)

            await conn.execute(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
                VALUES ('huge_function', 'function', $1, 99999, 100001, 'void huge_function(void)')
                """,
                file_id,
            )

        with httpx.Client() as client:
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/get_symbol",
                headers=auth_headers,
                json={"symbol": "huge_function"},
            )

            if response.status_code == 200:
                data = response.json()
                if "span" in data:
                    # Large line numbers should be preserved
                    assert data["span"]["start"] == 99999
                    assert data["span"]["end"] == 100001
                    assert data["span"]["path"] == "mm/huge_file.c"
                    assert data["span"]["sha"] == "sha_huge"
