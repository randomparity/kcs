"""
Performance tests for call graph traversal operations.

These tests verify that graph traversal operations meet constitutional
performance requirements under various loads and complexity scenarios.

Key test scenarios:
- Forward/backward/bidirectional traversal performance
- Deep traversal (max depth) performance
- Cycle detection performance with complex graphs
- Path finding between distant symbols
- Concurrent traversal requests
- Large graph traversal memory efficiency
- Incremental expansion performance
"""

import asyncio
import os
import time
from collections.abc import AsyncGenerator, Generator
from statistics import mean, quantiles
from typing import Any

import httpx
import pytest
import pytest_asyncio
import requests
from testcontainers.postgres import PostgresContainer


# Test infrastructure
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible for performance testing."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server required for performance tests"
)

# Skip performance tests in CI environments unless explicitly enabled
skip_performance_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_PERFORMANCE_TESTS") != "true",
    reason="Performance tests skipped in CI (set RUN_PERFORMANCE_TESTS=true to enable)",
)

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "performance_test_token"

# Constitutional performance requirements
MAX_P95_MS = 600  # Constitutional requirement: p95 < 600ms
MAX_P50_MS = 200  # Target: p50 < 200ms
MAX_DEEP_TRAVERSAL_MS = 1500  # Maximum time for depth=5 traversal
MAX_CONCURRENT_P95_MS = 1000  # P95 under concurrent load
MIN_THROUGHPUT_RPS = 10  # Minimum requests per second

# Graph complexity thresholds
MAX_NODES_DEPTH_3 = 1000  # Max nodes expected at depth 3
MAX_NODES_DEPTH_5 = 10000  # Max nodes expected at depth 5
MAX_EDGES_RATIO = 20  # Max edges per node ratio


# Test fixtures
@pytest.fixture(scope="module")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Create a test PostgreSQL container with graph data."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest_asyncio.fixture
async def graph_db(postgres_container: PostgresContainer) -> Any:
    """Create test database with complex call graph data."""
    from kcs_mcp.database import Database

    db = Database(postgres_container.get_connection_url())
    await db.initialize()

    # Create schema for graph traversal
    async with db.acquire() as conn:
        # Create core tables
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
                config_bitmap INTEGER DEFAULT 1,
                complexity_score INTEGER DEFAULT 1,
                is_exported BOOLEAN DEFAULT false,
                subsystem TEXT
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS call_edge (
                id SERIAL PRIMARY KEY,
                caller_id INTEGER REFERENCES symbol(id),
                callee_id INTEGER REFERENCES symbol(id),
                call_type TEXT DEFAULT 'direct',
                line_number INTEGER,
                config_bitmap INTEGER DEFAULT 1,
                weight FLOAT DEFAULT 1.0
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

        # Create performance-optimized indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_name_hash ON symbol USING hash(name)"
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_kind ON symbol(kind)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_subsystem ON symbol(subsystem)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edge_caller ON call_edge(caller_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edge_callee ON call_edge(callee_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edge_type ON call_edge(call_type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entrypoint_name ON entrypoint(name)"
        )

        print("Generating complex call graph data for performance testing...")

        # Create files representing different subsystems
        subsystems = ["fs", "mm", "net", "kernel", "arch", "drivers", "sound", "crypto"]
        file_ids = []

        for subsystem in subsystems:
            for i in range(200):  # 200 files per subsystem = 1600 files total
                file_id = await conn.fetchval(
                    """
                    INSERT INTO file (path, sha)
                    VALUES ($1, $2)
                    RETURNING id
                    """,
                    f"{subsystem}/module_{i // 20}/file_{i}.c",
                    f"sha_{subsystem}_{i:06d}",
                )
                file_ids.append((file_id, subsystem))

        # Create symbols with varying complexity (100k total symbols)
        symbol_ids = []
        batch_size = 1000

        symbol_batch = []
        for file_idx, (file_id, subsystem) in enumerate(file_ids):
            for sym_idx in range(62):  # ~62 symbols per file = ~100k total
                symbol_name = f"{subsystem}_func_{file_idx}_{sym_idx}"
                symbol_kind = "function" if sym_idx % 10 != 0 else "macro"
                complexity = 1 + (sym_idx % 15)  # Complexity 1-15
                is_exported = sym_idx % 7 == 0  # ~15% exported

                symbol_batch.append(
                    (
                        symbol_name,
                        symbol_kind,
                        file_id,
                        sym_idx * 20 + 1,
                        sym_idx * 20 + 15,
                        f"void {symbol_name}(void)",
                        complexity,
                        is_exported,
                        subsystem,
                    )
                )

                if len(symbol_batch) >= batch_size:
                    # Batch insert symbols
                    symbol_ids_batch = await conn.fetch(
                        """
                        INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature, complexity_score, is_exported, subsystem)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                        """,
                        *symbol_batch,
                    )
                    symbol_ids.extend([row["id"] for row in symbol_ids_batch])
                    symbol_batch = []

        # Insert remaining symbols
        if symbol_batch:
            symbol_ids_batch = await conn.fetch(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature, complexity_score, is_exported, subsystem)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                *symbol_batch,
            )
            symbol_ids.extend([row["id"] for row in symbol_ids_batch])

        print(f"Created {len(symbol_ids)} symbols")

        # Create complex call graph with cycles and multiple paths
        # Target: ~500k edges with varying patterns
        edge_batch = []
        edge_count = 0

        for caller_idx, caller_id in enumerate(symbol_ids):
            # Create different connection patterns:
            # 1. Linear chains (depth)
            # 2. Fan-out patterns (breadth)
            # 3. Cycles (complexity)
            # 4. Cross-subsystem calls

            if caller_idx % 1000 == 0:
                print(f"Creating edges for symbol {caller_idx}/{len(symbol_ids)}")

            # Regular forward calls (5-15 per symbol)
            num_callees = 5 + (caller_idx % 11)
            for call_idx in range(num_callees):
                callee_idx = (caller_idx + call_idx + 1) % len(symbol_ids)
                if callee_idx != caller_idx:
                    edge_batch.append(
                        (
                            caller_id,
                            symbol_ids[callee_idx],
                            "direct",
                            100 + call_idx,
                            1.0 + (call_idx * 0.1),
                        )
                    )

            # Create some backward edges for cycles (every 100th symbol)
            if caller_idx % 100 == 0 and caller_idx > 0:
                # Create cycle back to earlier symbol
                cycle_target_idx = max(0, caller_idx - 50)
                edge_batch.append(
                    (
                        caller_id,
                        symbol_ids[cycle_target_idx],
                        "direct",
                        150,
                        2.0,
                    )
                )

            # Cross-subsystem calls (every 50th symbol)
            if caller_idx % 50 == 0:
                # Find symbol from different subsystem
                for cross_idx in range(
                    caller_idx + 1000, min(caller_idx + 1100, len(symbol_ids))
                ):
                    if cross_idx < len(symbol_ids):
                        edge_batch.append(
                            (
                                caller_id,
                                symbol_ids[cross_idx],
                                "indirect",
                                200,
                                1.5,
                            )
                        )
                        break

            # Batch insert edges
            if len(edge_batch) >= batch_size:
                await conn.executemany(
                    """
                    INSERT INTO call_edge (caller_id, callee_id, call_type, line_number, weight)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    edge_batch,
                )
                edge_count += len(edge_batch)
                edge_batch = []

        # Insert remaining edges
        if edge_batch:
            await conn.executemany(
                """
                INSERT INTO call_edge (caller_id, callee_id, call_type, line_number, weight)
                VALUES ($1, $2, $3, $4, $5)
                """,
                edge_batch,
            )
            edge_count += len(edge_batch)

        print(f"Created {edge_count} call edges")

        # Create entry points for testing entry point traversal
        for i in range(50):
            symbol_idx = i * 100  # Every 100th symbol
            if symbol_idx < len(symbol_ids):
                await conn.execute(
                    """
                    INSERT INTO entrypoint (name, entry_type, symbol_id, file_id, line_number, syscall_number)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    f"__NR_perf_syscall_{i}",
                    "syscall",
                    symbol_ids[symbol_idx],
                    file_ids[symbol_idx % len(file_ids)][0],
                    1,
                    i + 400,  # Start from syscall 400
                )

        print("Complex graph data generation complete")

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for async requests."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


# Performance test classes
@skip_without_mcp_server
@skip_performance_in_ci
class TestBasicTraversalPerformance:
    """Test basic traversal performance across different directions and depths."""

    async def test_forward_traversal_performance(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test forward traversal performance across different depths."""
        test_cases = [
            ("fs_func_0_0", 1),
            ("fs_func_0_0", 2),
            ("fs_func_0_0", 3),
            ("fs_func_0_0", 5),
        ]

        depth_timings = {}

        for symbol, depth in test_cases:
            timings = []

            # Run multiple iterations for statistical significance
            for _ in range(10):
                request_data = {
                    "start_symbol": symbol,
                    "direction": "forward",
                    "max_depth": depth,
                    "detect_cycles": True,
                    "include_metrics": False,
                }

                start = time.perf_counter()
                response = await http_client.post(
                    f"{MCP_BASE_URL}/traverse_call_graph",
                    json=request_data,
                    headers=auth_headers,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                # Skip if endpoint not implemented
                if response.status_code == 404:
                    pytest.skip("traverse_call_graph endpoint not implemented")

                assert response.status_code == 200, f"Request failed: {response.text}"
                timings.append(elapsed_ms)

                # Verify response structure and content
                data = response.json()
                assert "nodes" in data
                assert "edges" in data
                assert "statistics" in data
                assert data["statistics"]["max_depth_reached"] <= depth

            # Calculate statistics for this depth
            p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
            avg = mean(timings)
            depth_timings[depth] = {"avg": avg, "p50": p50, "p95": p95}

            print(f"\nForward traversal depth={depth} performance:")
            print(f"  Average: {avg:.2f}ms")
            print(f"  P50: {p50:.2f}ms")
            print(f"  P95: {p95:.2f}ms")

            # Verify constitutional requirements
            assert p95 < MAX_P95_MS, f"Depth {depth} P95 ({p95:.2f}ms) exceeds limit"

        # Verify performance scales reasonably with depth
        assert depth_timings[1]["p95"] < depth_timings[5]["p95"], (
            "Performance should degrade with increased depth"
        )

    async def test_backward_traversal_performance(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test backward traversal performance (who calls this)."""
        # Use symbols that are likely to be called by many others
        test_symbols = ["mm_func_50_10", "net_func_100_5", "kernel_func_200_15"]

        timings = []

        for symbol in test_symbols:
            for depth in [1, 2, 3]:
                request_data = {
                    "start_symbol": symbol,
                    "direction": "backward",
                    "max_depth": depth,
                    "detect_cycles": True,
                }

                start = time.perf_counter()
                response = await http_client.post(
                    f"{MCP_BASE_URL}/traverse_call_graph",
                    json=request_data,
                    headers=auth_headers,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                if response.status_code == 404:
                    pytest.skip("traverse_call_graph endpoint not implemented")

                assert response.status_code == 200
                timings.append(elapsed_ms)

                # Verify backward traversal finds callers
                data = response.json()
                if len(data["nodes"]) > 1:  # Should find at least some callers
                    assert data["statistics"]["total_edges"] > 0

        p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
        avg = mean(timings)

        print("\nBackward traversal performance:")
        print(f"  Average: {avg:.2f}ms")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")

        assert p95 < MAX_P95_MS, f"Backward traversal P95 ({p95:.2f}ms) exceeds limit"

    async def test_bidirectional_traversal_performance(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test bidirectional traversal performance."""
        timings = []

        test_cases = [
            ("fs_func_10_5", 2),
            ("mm_func_20_10", 2),
            ("net_func_30_8", 3),
        ]

        for symbol, depth in test_cases:
            for _ in range(5):  # Fewer iterations due to higher complexity
                request_data = {
                    "start_symbol": symbol,
                    "direction": "bidirectional",
                    "max_depth": depth,
                    "detect_cycles": True,
                }

                start = time.perf_counter()
                response = await http_client.post(
                    f"{MCP_BASE_URL}/traverse_call_graph",
                    json=request_data,
                    headers=auth_headers,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                if response.status_code == 404:
                    pytest.skip("traverse_call_graph endpoint not implemented")

                assert response.status_code == 200
                timings.append(elapsed_ms)

                # Bidirectional should find more nodes than unidirectional
                data = response.json()
                assert data["statistics"]["total_nodes"] >= 1

        p50, p95 = quantiles(timings, n=20)[9], quantiles(timings, n=20)[18]
        avg = mean(timings)

        print("\nBidirectional traversal performance:")
        print(f"  Average: {avg:.2f}ms")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")

        # Bidirectional allowed higher latency due to complexity
        assert p95 < MAX_P95_MS * 1.5, f"Bidirectional P95 ({p95:.2f}ms) is too high"


@skip_without_mcp_server
@skip_performance_in_ci
class TestCycleDetectionPerformance:
    """Test cycle detection performance with complex graphs."""

    async def test_cycle_detection_performance(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test performance of cycle detection in complex graphs."""
        # Test symbols that are likely part of cycles
        test_symbols = ["fs_func_0_0", "mm_func_100_0", "net_func_200_0"]

        timings = []
        cycles_found = 0

        for symbol in test_symbols:
            request_data = {
                "start_symbol": symbol,
                "direction": "forward",
                "max_depth": 4,
                "detect_cycles": True,
                "include_metrics": True,
            }

            start = time.perf_counter()
            response = await http_client.post(
                f"{MCP_BASE_URL}/traverse_call_graph",
                json=request_data,
                headers=auth_headers,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 404:
                pytest.skip("traverse_call_graph endpoint not implemented")

            assert response.status_code == 200
            timings.append(elapsed_ms)

            data = response.json()
            if data.get("cycles"):
                cycles_found += len(data["cycles"])

        avg = mean(timings)
        max_time = max(timings)

        print("\nCycle detection performance:")
        print(f"  Average: {avg:.2f}ms")
        print(f"  Maximum: {max_time:.2f}ms")
        print(f"  Cycles found: {cycles_found}")

        # Cycle detection should not significantly impact performance
        assert max_time < MAX_P95_MS * 1.2, (
            f"Cycle detection too slow ({max_time:.2f}ms)"
        )

    async def test_large_graph_cycle_detection(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test cycle detection performance on large subgraphs."""
        request_data = {
            "start_symbol": "fs_func_0_0",
            "direction": "bidirectional",
            "max_depth": 5,  # Large traversal
            "detect_cycles": True,
            "include_metrics": True,
        }

        start = time.perf_counter()
        response = await http_client.post(
            f"{MCP_BASE_URL}/traverse_call_graph",
            json=request_data,
            headers=auth_headers,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 404:
            pytest.skip("traverse_call_graph endpoint not implemented")

        assert response.status_code == 200

        data = response.json()
        stats = data["statistics"]

        print("\nLarge graph cycle detection:")
        print(f"  Time: {elapsed_ms:.2f}ms")
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Edges: {stats['total_edges']}")
        print(f"  Cycles: {stats.get('cycles_detected', 0)}")

        # Should handle large graphs efficiently
        assert elapsed_ms < MAX_DEEP_TRAVERSAL_MS, (
            f"Large graph traversal too slow ({elapsed_ms:.2f}ms)"
        )

        # Verify reasonable graph size limits
        assert stats["total_nodes"] <= MAX_NODES_DEPTH_5, (
            f"Too many nodes returned: {stats['total_nodes']}"
        )


@skip_without_mcp_server
@skip_performance_in_ci
class TestPathFindingPerformance:
    """Test path finding performance between symbols."""

    async def test_path_finding_performance(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test performance of finding paths between symbols."""
        # Test path finding between symbols that are likely connected
        test_cases = [
            ("fs_func_0_0", "fs_func_10_5"),
            ("mm_func_0_0", "mm_func_50_10"),
            ("net_func_0_0", "kernel_func_20_5"),  # Cross-subsystem
        ]

        timings = []
        paths_found = 0

        for start_symbol, target_symbol in test_cases:
            request_data = {
                "start_symbol": start_symbol,
                "direction": "forward",
                "max_depth": 5,
                "find_all_paths": True,
                "target_symbol": target_symbol,
                "detect_cycles": False,  # Disable for faster path finding
            }

            start = time.perf_counter()
            response = await http_client.post(
                f"{MCP_BASE_URL}/traverse_call_graph",
                json=request_data,
                headers=auth_headers,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 404:
                pytest.skip("traverse_call_graph endpoint not implemented")

            assert response.status_code == 200
            timings.append(elapsed_ms)

            data = response.json()
            if data.get("paths"):
                paths_found += len(data["paths"])

        avg = mean(timings)
        max_time = max(timings)

        print("\nPath finding performance:")
        print(f"  Average: {avg:.2f}ms")
        print(f"  Maximum: {max_time:.2f}ms")
        print(f"  Paths found: {paths_found}")

        assert max_time < MAX_P95_MS * 1.3, f"Path finding too slow ({max_time:.2f}ms)"


@skip_without_mcp_server
@skip_performance_in_ci
class TestConcurrentTraversalPerformance:
    """Test performance under concurrent traversal requests."""

    async def test_concurrent_traversal_load(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test performance under concurrent traversal requests."""

        async def make_traversal_request(symbol: str, depth: int) -> float:
            """Make a single traversal request and measure time."""
            request_data = {
                "start_symbol": symbol,
                "direction": "forward",
                "max_depth": depth,
                "detect_cycles": True,
            }

            start = time.perf_counter()
            response = await http_client.post(
                f"{MCP_BASE_URL}/traverse_call_graph",
                json=request_data,
                headers=auth_headers,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code in [200, 404]
            return elapsed_ms

        # Create 20 concurrent requests with varying complexity
        tasks = []
        for i in range(20):
            symbol = f"fs_func_{i}_0"
            depth = 2 + (i % 3)  # Depth 2-4
            tasks.append(make_traversal_request(symbol, depth))

        # Execute all requests concurrently
        start_all = time.perf_counter()
        timings = await asyncio.gather(*tasks, return_exceptions=True)
        total_time_ms = (time.perf_counter() - start_all) * 1000

        # Filter out any exceptions (e.g., 404s)
        valid_timings = [t for t in timings if isinstance(t, (int, float))]

        if not valid_timings:
            pytest.skip("No successful traversal requests")

        p50, p95 = quantiles(valid_timings, n=20)[9], quantiles(valid_timings, n=20)[18]
        avg = mean(valid_timings)
        throughput = len(valid_timings) / (total_time_ms / 1000)

        print("\nConcurrent traversal performance (20 requests):")
        print(f"  Total time: {total_time_ms:.2f}ms")
        print(f"  Average per request: {avg:.2f}ms")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  Throughput: {throughput:.1f} req/s")

        # Verify performance under load
        assert p95 < MAX_CONCURRENT_P95_MS, f"Concurrent P95 ({p95:.2f}ms) too high"
        assert throughput >= MIN_THROUGHPUT_RPS, (
            f"Throughput ({throughput:.1f}) too low"
        )


@skip_without_mcp_server
@skip_performance_in_ci
class TestTraversalScalability:
    """Test traversal scalability with graph complexity."""

    async def test_complexity_scaling(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test how performance scales with graph complexity."""
        # Test with filters that create different complexity levels
        complexity_tests = [
            {
                "name": "simple",
                "filters": {
                    "exclude_patterns": [".*_test", ".*_debug"],
                    "include_subsystems": ["fs"],
                },
                "expected_max_nodes": 1000,
            },
            {
                "name": "medium",
                "filters": {
                    "include_subsystems": ["fs", "mm"],
                },
                "expected_max_nodes": 3000,
            },
            {
                "name": "complex",
                "filters": None,  # No filters = full complexity
                "expected_max_nodes": 10000,
            },
        ]

        results = {}

        for test_case in complexity_tests:
            request_data = {
                "start_symbol": "fs_func_0_0",
                "direction": "forward",
                "max_depth": 3,
                "detect_cycles": True,
                "filters": test_case["filters"],
            }

            start = time.perf_counter()
            response = await http_client.post(
                f"{MCP_BASE_URL}/traverse_call_graph",
                json=request_data,
                headers=auth_headers,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 404:
                pytest.skip("traverse_call_graph endpoint not implemented")

            assert response.status_code == 200

            data = response.json()
            stats = data["statistics"]

            results[test_case["name"]] = {
                "time_ms": elapsed_ms,
                "nodes": stats["total_nodes"],
                "edges": stats["total_edges"],
            }

            print(f"\n{test_case['name'].title()} complexity:")
            print(f"  Time: {elapsed_ms:.2f}ms")
            print(f"  Nodes: {stats['total_nodes']}")
            print(f"  Edges: {stats['total_edges']}")

            # Verify reasonable limits
            assert stats["total_nodes"] <= test_case["expected_max_nodes"], (
                f"Too many nodes in {test_case['name']} test: {stats['total_nodes']}"
            )

        # Verify performance scales reasonably
        assert results["simple"]["time_ms"] <= results["complex"]["time_ms"], (
            "Simple graphs should be faster than complex graphs"
        )

    async def test_memory_efficiency(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        graph_db: Any,
    ) -> None:
        """Test memory efficiency of large traversals."""
        # Large traversal to test memory usage
        request_data = {
            "start_symbol": "fs_func_0_0",
            "direction": "bidirectional",
            "max_depth": 4,
            "detect_cycles": True,
            "include_metrics": True,
            "include_visualization": False,  # Disable to save memory
        }

        start = time.perf_counter()
        response = await http_client.post(
            f"{MCP_BASE_URL}/traverse_call_graph",
            json=request_data,
            headers=auth_headers,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 404:
            pytest.skip("traverse_call_graph endpoint not implemented")

        assert response.status_code == 200

        data = response.json()
        stats = data["statistics"]

        # Estimate memory usage based on response size
        import sys

        response_size = sys.getsizeof(response.content)

        print("\nMemory efficiency test:")
        print(f"  Time: {elapsed_ms:.2f}ms")
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Edges: {stats['total_edges']}")
        print(f"  Response size: {response_size / 1024:.1f} KB")

        # Verify efficient memory usage
        bytes_per_node = (
            response_size / stats["total_nodes"] if stats["total_nodes"] > 0 else 0
        )
        assert bytes_per_node < 1000, (
            f"Memory usage per node too high: {bytes_per_node:.1f} bytes"
        )

        # Should complete within memory-conscious time limit
        assert elapsed_ms < MAX_DEEP_TRAVERSAL_MS, (
            f"Large traversal too slow ({elapsed_ms:.2f}ms)"
        )
