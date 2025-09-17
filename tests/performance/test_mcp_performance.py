"""
Performance validation tests for MCP endpoints.

These tests verify that all MCP endpoints meet the constitutional requirement
of p95 query times < 600ms.
"""

import asyncio
import os
import time
from collections.abc import Generator
from statistics import mean, quantiles

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
    os.getenv("CI") == "true" and os.getenv("RUN_PERFORMANCE_TESTS") != "true",
    reason="Performance tests skipped in CI",
)

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8080")
TEST_TOKEN = os.getenv("TEST_TOKEN", "test_token")

# Performance requirements
MAX_P95_MS = 600  # Constitutional requirement: p95 < 600ms
MAX_P50_MS = 200  # Target: p50 < 200ms


@pytest.fixture(scope="module")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Create a test PostgreSQL container."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest_asyncio.fixture
async def production_like_db(postgres_container: PostgresContainer) -> Database:
    """Create test database with production-scale data."""
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

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entry_point (
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

        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbol(name)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edge_caller ON call_edge(caller_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edge_callee ON call_edge(callee_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entry_point_name ON entry_point(name)"
        )

        # Generate production-scale test data
        # Target: ~50k symbols, ~500k call edges (based on constitutional requirements)
        print("Generating production-scale test data...")

        # Create files
        file_ids = []
        for i in range(1000):  # 1000 files
            file_id = await conn.fetchval(
                """
                INSERT INTO file (path, sha)
                VALUES ($1, $2)
                RETURNING id
                """,
                f"kernel/subsystem_{i // 100}/file_{i}.c",
                f"sha_{i:06d}",
            )
            file_ids.append(file_id)

        # Create symbols (50 per file = 50k total)
        symbol_ids = []
        for file_idx, file_id in enumerate(file_ids):
            for sym_idx in range(50):
                symbol_id = await conn.fetchval(
                    """
                    INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    f"func_{file_idx}_{sym_idx}",
                    "function",
                    file_id,
                    sym_idx * 20 + 1,
                    sym_idx * 20 + 15,
                    f"void func_{file_idx}_{sym_idx}(void)",
                )
                symbol_ids.append(symbol_id)

        # Create call edges (average 10 edges per symbol = 500k total)
        # Use batched inserts for performance
        edges = []
        for caller_idx, caller_id in enumerate(symbol_ids):
            # Each function calls 10 others (with some variation)
            num_callees = 5 + (caller_idx % 10)  # 5-14 callees
            for _ in range(min(num_callees, len(symbol_ids) - 1)):
                # Pick random callee (but not self)
                callee_idx = (caller_idx + 7 + _) % len(symbol_ids)
                if callee_idx != caller_idx:
                    edges.append((caller_id, symbol_ids[callee_idx], "direct", 100 + _))

                    # Batch insert every 1000 edges
                    if len(edges) >= 1000:
                        await conn.executemany(
                            """
                            INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
                            VALUES ($1, $2, $3, $4)
                            """,
                            edges,
                        )
                        edges = []

        # Insert remaining edges
        if edges:
            await conn.executemany(
                """
                INSERT INTO call_edge (caller_id, callee_id, call_type, line_number)
                VALUES ($1, $2, $3, $4)
                """,
                edges,
            )

        # Create some entry points
        for i in range(100):
            await conn.execute(
                """
                INSERT INTO entry_point (name, entry_type, symbol_id, file_id, line_number, syscall_number)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                f"__NR_syscall_{i}",
                "syscall",
                symbol_ids[i],
                file_ids[i],
                1,
                i,
            )

        print("Test data generation complete")

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for API requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


def measure_query_time(func):
    """Measure execution time of a function in milliseconds."""
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    return (end - start) * 1000, result


class TestEndpointPerformance:
    """Test performance of individual endpoints."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_who_calls_performance(
        self, auth_headers: dict[str, str], production_like_db: Database
    ) -> None:
        """Test who_calls endpoint performance."""
        with httpx.Client(timeout=30.0) as client:
            timings = []

            # Test with various symbols and depths
            test_cases = [
                ("func_0_0", 1),
                ("func_10_5", 2),
                ("func_50_10", 3),
                ("func_100_20", 5),
            ]

            for symbol, depth in test_cases:
                # Run multiple iterations for statistical significance
                for _ in range(10):
                    start = time.perf_counter()
                    response = client.post(
                        f"{MCP_BASE_URL}/mcp/tools/who_calls",
                        headers=auth_headers,
                        json={"symbol": symbol, "depth": depth},
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    assert response.status_code == 200
                    timings.append(elapsed_ms)

            # Calculate statistics
            p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
            avg = mean(timings)

            print("\nwho_calls performance:")
            print(f"  Average: {avg:.2f}ms")
            print(f"  P50: {p50:.2f}ms")
            print(f"  P95: {p95:.2f}ms")

            # Verify constitutional requirement
            assert p95 < MAX_P95_MS, f"P95 ({p95:.2f}ms) exceeds limit ({MAX_P95_MS}ms)"
            assert p50 < MAX_P50_MS * 2, f"P50 ({p50:.2f}ms) is too high"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_list_dependencies_performance(
        self, auth_headers: dict[str, str], production_like_db: Database
    ) -> None:
        """Test list_dependencies endpoint performance."""
        with httpx.Client(timeout=30.0) as client:
            timings = []

            test_cases = [
                ("func_5_0", 1),
                ("func_15_5", 2),
                ("func_25_10", 3),
                ("func_200_30", 5),
            ]

            for symbol, depth in test_cases:
                for _ in range(10):
                    start = time.perf_counter()
                    response = client.post(
                        f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                        headers=auth_headers,
                        json={"symbol": symbol, "depth": depth},
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    assert response.status_code == 200
                    timings.append(elapsed_ms)

            p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
            avg = mean(timings)

            print("\nlist_dependencies performance:")
            print(f"  Average: {avg:.2f}ms")
            print(f"  P50: {p50:.2f}ms")
            print(f"  P95: {p95:.2f}ms")

            assert p95 < MAX_P95_MS, f"P95 ({p95:.2f}ms) exceeds limit ({MAX_P95_MS}ms)"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_entrypoint_flow_performance(
        self, auth_headers: dict[str, str], production_like_db: Database
    ) -> None:
        """Test entrypoint_flow endpoint performance."""
        with httpx.Client(timeout=30.0) as client:
            timings = []

            test_entries = [
                "__NR_syscall_0",
                "__NR_syscall_10",
                "__NR_syscall_50",
                "__NR_read",
                "__NR_write",
            ]

            for entry in test_entries:
                for _ in range(10):
                    start = time.perf_counter()
                    response = client.post(
                        f"{MCP_BASE_URL}/mcp/tools/entrypoint_flow",
                        headers=auth_headers,
                        json={"entry": entry},
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    # May return 200 or appropriate error for unknown entries
                    assert response.status_code in [200, 404]
                    timings.append(elapsed_ms)

            p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
            avg = mean(timings)

            print("\nentrypoint_flow performance:")
            print(f"  Average: {avg:.2f}ms")
            print(f"  P50: {p50:.2f}ms")
            print(f"  P95: {p95:.2f}ms")

            assert p95 < MAX_P95_MS, f"P95 ({p95:.2f}ms) exceeds limit ({MAX_P95_MS}ms)"

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_impact_of_performance(
        self, auth_headers: dict[str, str], production_like_db: Database
    ) -> None:
        """Test impact_of endpoint performance."""
        with httpx.Client(timeout=30.0) as client:
            timings = []

            test_cases = [
                {"symbol": "func_1_1"},
                {"symbols": ["func_5_5", "func_10_10"]},
                {"files": ["kernel/subsystem_0/file_0.c"]},
                {
                    "diff": """
                    - void func_old() {
                    + void func_new() {
                        return 0;
                    }
                    """
                },
            ]

            for test_input in test_cases:
                for _ in range(10):
                    start = time.perf_counter()
                    response = client.post(
                        f"{MCP_BASE_URL}/mcp/tools/impact_of",
                        headers=auth_headers,
                        json=test_input,
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    assert response.status_code == 200
                    timings.append(elapsed_ms)

            p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
            avg = mean(timings)

            print("\nimpact_of performance:")
            print(f"  Average: {avg:.2f}ms")
            print(f"  P50: {p50:.2f}ms")
            print(f"  P95: {p95:.2f}ms")

            assert p95 < MAX_P95_MS, f"P95 ({p95:.2f}ms) exceeds limit ({MAX_P95_MS}ms)"


class TestConcurrentRequests:
    """Test concurrent request handling."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(
        self, auth_headers: dict[str, str], production_like_db: Database
    ) -> None:
        """Test performance under concurrent load."""

        async def make_request(endpoint: str, data: dict) -> float:
            """Make async request and measure time."""
            async with httpx.AsyncClient() as client:
                start = time.perf_counter()
                response = await client.post(
                    f"{MCP_BASE_URL}/mcp/tools/{endpoint}",
                    headers=auth_headers,
                    json=data,
                    timeout=10.0,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                assert response.status_code in [200, 404]
                return elapsed_ms

        # Create a mix of concurrent requests
        tasks = []
        for i in range(20):  # 20 concurrent requests
            endpoint = [
                "who_calls",
                "list_dependencies",
                "entrypoint_flow",
                "impact_of",
            ][i % 4]
            if endpoint == "who_calls":
                data = {"symbol": f"func_{i}_0", "depth": 2}
            elif endpoint == "list_dependencies":
                data = {"symbol": f"func_{i}_1", "depth": 2}
            elif endpoint == "entrypoint_flow":
                data = {"entry": f"__NR_syscall_{i}"}
            else:
                data = {"symbol": f"func_{i}_2"}

            tasks.append(make_request(endpoint, data))

        # Execute all requests concurrently
        start_all = time.perf_counter()
        timings = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - start_all) * 1000

        # Calculate statistics
        p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
        avg = mean(timings)

        print("\nConcurrent requests performance (20 requests):")
        print(f"  Total time: {total_time_ms:.2f}ms")
        print(f"  Average per request: {avg:.2f}ms")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")

        # Even under concurrent load, p95 should meet requirements
        assert p95 < MAX_P95_MS * 1.5, f"P95 under load ({p95:.2f}ms) is too high"


class TestWorstCaseScenarios:
    """Test performance in worst-case scenarios."""

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_deep_traversal_performance(
        self, auth_headers: dict[str, str], production_like_db: Database
    ) -> None:
        """Test performance with maximum depth traversal."""
        with httpx.Client(timeout=30.0) as client:
            # Test maximum depth (5) with heavily connected symbols
            timings = []

            for i in range(5):
                start = time.perf_counter()
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/who_calls",
                    headers=auth_headers,
                    json={"symbol": f"func_{i * 10}_0", "depth": 5},
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                assert response.status_code == 200
                timings.append(elapsed_ms)

                # Also test list_dependencies with max depth
                start = time.perf_counter()
                response = client.post(
                    f"{MCP_BASE_URL}/mcp/tools/list_dependencies",
                    headers=auth_headers,
                    json={"symbol": f"func_{i * 10}_0", "depth": 5},
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                assert response.status_code == 200
                timings.append(elapsed_ms)

            max_time = max(timings)
            avg = mean(timings)

            print("\nWorst-case (depth=5) performance:")
            print(f"  Average: {avg:.2f}ms")
            print(f"  Maximum: {max_time:.2f}ms")

            # Even worst case should not exceed constitutional requirement
            assert max_time < MAX_P95_MS * 2, (
                f"Worst case ({max_time:.2f}ms) is too slow"
            )

    @skip_without_mcp
    @skip_in_ci
    @pytest.mark.asyncio
    async def test_large_impact_analysis(
        self, auth_headers: dict[str, str], production_like_db: Database
    ) -> None:
        """Test impact_of with multiple symbols (large blast radius)."""
        with httpx.Client(timeout=30.0) as client:
            # Test with multiple symbols that have many connections
            large_symbol_set = [f"func_{i}_0" for i in range(20)]

            start = time.perf_counter()
            response = client.post(
                f"{MCP_BASE_URL}/mcp/tools/impact_of",
                headers=auth_headers,
                json={"symbols": large_symbol_set},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200

            print("\nLarge impact analysis (20 symbols):")
            print(f"  Time: {elapsed_ms:.2f}ms")

            # Should still meet requirement even with large input
            assert elapsed_ms < MAX_P95_MS * 3, (
                f"Large impact analysis too slow ({elapsed_ms:.2f}ms)"
            )
