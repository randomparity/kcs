"""
Memory efficiency tests for large graph export operations.

These tests verify that graph export operations handle large datasets
efficiently without excessive memory usage or causing system instability.

Key test scenarios:
- Large graph export (100k+ nodes) memory efficiency
- Chunked export memory usage validation
- Compression effectiveness for different graph types
- Streaming export memory characteristics
- Concurrent export memory stability
- Memory leak detection during repeated exports
- Different format memory profiles (JSON vs GraphML vs CSV)
"""

import gc
import os
import sys
import time
from collections.abc import AsyncGenerator, Generator
from statistics import mean
from typing import Any

import httpx
import psutil
import pytest
import pytest_asyncio
import requests
from testcontainers.postgres import PostgresContainer


# Test infrastructure
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible for memory testing."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server required for memory tests"
)

# Skip performance tests in CI environments unless explicitly enabled
skip_memory_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_MEMORY_TESTS") != "true",
    reason="Memory tests skipped in CI (set RUN_MEMORY_TESTS=true to enable)",
)

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "memory_test_token"

# Memory thresholds
MAX_MEMORY_MB_SMALL = 100  # Max memory for small exports (MB)
MAX_MEMORY_MB_LARGE = 500  # Max memory for large exports (MB)
MAX_MEMORY_GROWTH_MB = 50  # Max memory growth during repeated exports (MB)
MIN_COMPRESSION_RATIO = 0.3  # Minimum compression effectiveness
MAX_EXPORT_TIME_SECONDS = 60  # Maximum export time for large graphs

# Graph size thresholds
SMALL_GRAPH_NODES = 1000
MEDIUM_GRAPH_NODES = 10000
LARGE_GRAPH_NODES = 50000
XLARGE_GRAPH_NODES = 100000


class MemoryMonitor:
    """Utility class for monitoring memory usage during tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.peak_memory = 0
        self.measurements = []

    def start_monitoring(self):
        """Start memory monitoring with baseline measurement."""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.process.memory_info().rss
        self.peak_memory = self.baseline_memory
        self.measurements = [self.baseline_memory]

    def measure(self):
        """Take a memory measurement."""
        current_memory = self.process.memory_info().rss
        self.measurements.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory

    def get_stats(self):
        """Get memory usage statistics."""
        if not self.baseline_memory:
            return {}

        current = self.process.memory_info().rss
        growth = current - self.baseline_memory
        peak_growth = self.peak_memory - self.baseline_memory

        return {
            "baseline_mb": self.baseline_memory / 1024 / 1024,
            "current_mb": current / 1024 / 1024,
            "peak_mb": self.peak_memory / 1024 / 1024,
            "growth_mb": growth / 1024 / 1024,
            "peak_growth_mb": peak_growth / 1024 / 1024,
            "measurements_count": len(self.measurements),
        }


# Test fixtures
@pytest.fixture(scope="module")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Create a test PostgreSQL container with large graph data."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest_asyncio.fixture
async def large_graph_db(postgres_container: PostgresContainer) -> Any:
    """Create test database with large graph for memory testing."""
    from kcs_mcp.database import Database

    db = Database(postgres_container.get_connection_url())
    await db.initialize()

    # Create schema optimized for large data
    async with db.acquire() as conn:
        # Create tables with optimizations for large datasets
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS file (
                id SERIAL PRIMARY KEY,
                path TEXT NOT NULL,
                sha TEXT NOT NULL,
                subsystem TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                complexity_score INTEGER DEFAULT 1,
                is_exported BOOLEAN DEFAULT false,
                subsystem TEXT,
                size_bytes INTEGER DEFAULT 0,
                memory_footprint INTEGER DEFAULT 0
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS call_edge (
                id SERIAL PRIMARY KEY,
                caller_id INTEGER REFERENCES symbol(id),
                callee_id INTEGER REFERENCES symbol(id),
                call_type TEXT DEFAULT 'direct',
                line_number INTEGER,
                weight FLOAT DEFAULT 1.0,
                frequency INTEGER DEFAULT 1,
                cost_estimate FLOAT DEFAULT 1.0
            )
        """)

        # Create indexes for large dataset performance
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_name_btree ON symbol(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_subsystem ON symbol(subsystem)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edge_caller_btree ON call_edge(caller_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_edge_callee_btree ON call_edge(callee_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_subsystem ON file(subsystem)"
        )

        print("Generating large graph data for memory testing...")

        # Create files for different subsystems with varied sizes
        subsystems = [
            "fs",
            "mm",
            "net",
            "kernel",
            "arch",
            "drivers",
            "sound",
            "crypto",
            "block",
            "security",
        ]
        file_batch = []
        batch_size = 500

        for subsystem in subsystems:
            for i in range(500):  # 500 files per subsystem = 5000 files total
                file_batch.append(
                    (
                        f"{subsystem}/level{i // 100}/module_{i // 20}/file_{i}.c",
                        f"sha_{subsystem}_{i:06d}",
                        subsystem,
                    )
                )

                if len(file_batch) >= batch_size:
                    await conn.executemany(
                        """
                        INSERT INTO file (path, sha, subsystem)
                        VALUES ($1, $2, $3)
                        """,
                        file_batch,
                    )
                    file_batch = []

        if file_batch:
            await conn.executemany(
                """
                INSERT INTO file (path, sha, subsystem)
                VALUES ($1, $2, $3)
                """,
                file_batch,
            )

        # Get file IDs for symbol creation
        file_rows = await conn.fetch("SELECT id, subsystem FROM file ORDER BY id")
        file_data = [(row["id"], row["subsystem"]) for row in file_rows]

        print(f"Created {len(file_data)} files")

        # Create large number of symbols (target: 100k symbols)
        symbol_batch = []
        symbols_per_file = XLARGE_GRAPH_NODES // len(file_data)

        for file_idx, (file_id, subsystem) in enumerate(file_data):
            if file_idx % 500 == 0:
                print(f"Creating symbols for file {file_idx}/{len(file_data)}")

            for sym_idx in range(symbols_per_file):
                symbol_name = f"{subsystem}_func_{file_idx}_{sym_idx}"
                symbol_kind = "function" if sym_idx % 8 != 0 else "macro"
                complexity = 1 + (sym_idx % 20)  # Complexity 1-20
                is_exported = sym_idx % 12 == 0  # ~8% exported
                size_bytes = 50 + (sym_idx % 500)  # 50-550 bytes
                memory_footprint = size_bytes * (1 + complexity * 0.1)

                symbol_batch.append(
                    (
                        symbol_name,
                        symbol_kind,
                        file_id,
                        sym_idx * 15 + 1,
                        sym_idx * 15 + 10,
                        f"void {symbol_name}(void)",
                        complexity,
                        is_exported,
                        subsystem,
                        size_bytes,
                        int(memory_footprint),
                    )
                )

                if len(symbol_batch) >= batch_size:
                    await conn.executemany(
                        """
                        INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature,
                                          complexity_score, is_exported, subsystem, size_bytes, memory_footprint)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """,
                        symbol_batch,
                    )
                    symbol_batch = []

            # Break early if we've created enough symbols
            total_created = (file_idx + 1) * symbols_per_file
            if total_created >= XLARGE_GRAPH_NODES:
                break

        if symbol_batch:
            await conn.executemany(
                """
                INSERT INTO symbol (name, kind, file_id, start_line, end_line, signature,
                                  complexity_score, is_exported, subsystem, size_bytes, memory_footprint)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                symbol_batch,
            )

        # Get symbol IDs for edge creation
        symbol_rows = await conn.fetch("SELECT id FROM symbol ORDER BY id")
        symbol_ids = [row["id"] for row in symbol_rows]

        print(f"Created {len(symbol_ids)} symbols")

        # Create call edges with different patterns for realistic graph structure
        edge_batch = []
        edge_count = 0
        target_edges = len(symbol_ids) * 8  # Average 8 edges per symbol

        for caller_idx, caller_id in enumerate(symbol_ids):
            if caller_idx % 5000 == 0:
                print(f"Creating edges for symbol {caller_idx}/{len(symbol_ids)}")

            # Create different edge patterns:
            # 1. Local calls (within same module) - high frequency
            num_local_calls = 3 + (caller_idx % 5)
            for call_idx in range(num_local_calls):
                callee_idx = (caller_idx + call_idx + 1) % len(symbol_ids)
                if callee_idx != caller_idx:
                    edge_batch.append(
                        (
                            caller_id,
                            symbol_ids[callee_idx],
                            "direct",
                            100 + call_idx,
                            1.0,
                            10 + (call_idx % 20),  # frequency
                            1.0 + (call_idx * 0.1),  # cost
                        )
                    )

            # 2. Cross-module calls - medium frequency
            if caller_idx % 20 == 0:
                for cross_idx in range(2):
                    target_idx = (caller_idx + 1000 + cross_idx * 500) % len(symbol_ids)
                    edge_batch.append(
                        (
                            caller_id,
                            symbol_ids[target_idx],
                            "indirect",
                            200,
                            2.0,
                            5,  # lower frequency
                            2.5,  # higher cost
                        )
                    )

            # 3. System calls - low frequency, high cost
            if caller_idx % 100 == 0:
                target_idx = (caller_idx + 10000) % len(symbol_ids)
                edge_batch.append(
                    (
                        caller_id,
                        symbol_ids[target_idx],
                        "syscall",
                        300,
                        5.0,
                        2,  # very low frequency
                        10.0,  # very high cost
                    )
                )

            # Batch insert edges
            if len(edge_batch) >= batch_size:
                await conn.executemany(
                    """
                    INSERT INTO call_edge (caller_id, callee_id, call_type, line_number,
                                         weight, frequency, cost_estimate)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    edge_batch,
                )
                edge_count += len(edge_batch)
                edge_batch = []

            # Stop when we reach target edge count
            if edge_count >= target_edges:
                break

        # Insert remaining edges
        if edge_batch:
            await conn.executemany(
                """
                INSERT INTO call_edge (caller_id, callee_id, call_type, line_number,
                                     weight, frequency, cost_estimate)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                edge_batch,
            )
            edge_count += len(edge_batch)

        print(f"Created {edge_count} call edges")
        print("Large graph data generation complete")

    yield db
    await db.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for async requests with extended timeout."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        yield client


# Memory test classes
@skip_without_mcp_server
@skip_memory_in_ci
class TestBasicExportMemory:
    """Test basic export memory usage patterns."""

    async def test_json_export_memory_efficiency(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        large_graph_db: Any,
    ) -> None:
        """Test JSON export memory usage with different graph sizes."""
        monitor = MemoryMonitor()
        monitor.start_monitoring()

        sizes_and_symbols = [
            (SMALL_GRAPH_NODES, "fs_func_0_0"),
            (MEDIUM_GRAPH_NODES, "fs_func_0_0"),
            (LARGE_GRAPH_NODES, "fs_func_0_0"),
        ]

        memory_results = {}

        for expected_size, root_symbol in sizes_and_symbols:
            gc.collect()  # Clean up before each test
            memory_before = monitor.measure()

            request_data = {
                "root_symbol": root_symbol,
                "format": "json",
                "depth": 3,  # Moderate depth to control size
                "include_metadata": True,
                "pretty": False,  # Compact format for memory efficiency
                "compress": False,  # Test uncompressed first
            }

            start_time = time.time()
            response = await http_client.post(
                f"{MCP_BASE_URL}/export_graph",
                json=request_data,
                headers=auth_headers,
            )
            export_time = time.time() - start_time

            if response.status_code == 404:
                pytest.skip("export_graph endpoint not implemented")

            assert response.status_code == 200, f"Export failed: {response.text}"

            memory_after = monitor.measure()
            memory_used = (memory_after - memory_before) / 1024 / 1024  # MB

            data = response.json()
            response_size = len(response.content)

            memory_results[expected_size] = {
                "memory_used_mb": memory_used,
                "response_size_mb": response_size / 1024 / 1024,
                "export_time": export_time,
                "nodes": len(data.get("graph", {}).get("nodes", []))
                if data.get("graph")
                else 0,
                "edges": len(data.get("graph", {}).get("edges", []))
                if data.get("graph")
                else 0,
            }

            print(f"\nJSON export (size ~{expected_size}):")
            print(f"  Memory used: {memory_used:.1f} MB")
            print(f"  Response size: {response_size / 1024 / 1024:.1f} MB")
            print(f"  Export time: {export_time:.2f}s")
            print(f"  Actual nodes: {memory_results[expected_size]['nodes']}")
            print(f"  Actual edges: {memory_results[expected_size]['edges']}")

            # Verify memory usage is reasonable for graph size
            if memory_results[expected_size]["nodes"] < SMALL_GRAPH_NODES:
                assert memory_used < MAX_MEMORY_MB_SMALL, (
                    f"Small export used too much memory: {memory_used:.1f}MB"
                )
            elif memory_results[expected_size]["nodes"] < LARGE_GRAPH_NODES:
                assert memory_used < MAX_MEMORY_MB_LARGE, (
                    f"Large export used too much memory: {memory_used:.1f}MB"
                )

            # Verify reasonable export time
            assert export_time < MAX_EXPORT_TIME_SECONDS, (
                f"Export took too long: {export_time:.2f}s"
            )

        # Verify memory usage scales reasonably
        stats = monitor.get_stats()
        print("\nOverall memory statistics:")
        print(f"  Peak memory growth: {stats['peak_growth_mb']:.1f} MB")

    async def test_compression_memory_efficiency(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        large_graph_db: Any,
    ) -> None:
        """Test that compression reduces memory usage and response size."""
        monitor = MemoryMonitor()

        formats_to_test = ["json", "graphml", "csv"]
        compression_results = {}

        for format_type in formats_to_test:
            # Test without compression
            monitor.start_monitoring()
            request_data = {
                "root_symbol": "fs_func_0_0",
                "format": format_type,
                "depth": 2,
                "compress": False,
                "include_metadata": True,
            }

            response_uncompressed = await http_client.post(
                f"{MCP_BASE_URL}/export_graph",
                json=request_data,
                headers=auth_headers,
            )

            if response_uncompressed.status_code == 404:
                pytest.skip("export_graph endpoint not implemented")

            assert response_uncompressed.status_code == 200
            uncompressed_size = len(response_uncompressed.content)
            uncompressed_memory = monitor.measure()

            # Test with compression
            monitor.start_monitoring()
            request_data["compress"] = True
            request_data["compression_format"] = "gzip"

            response_compressed = await http_client.post(
                f"{MCP_BASE_URL}/export_graph",
                json=request_data,
                headers=auth_headers,
            )

            assert response_compressed.status_code == 200
            compressed_data = response_compressed.json()
            compressed_memory = monitor.measure()

            # Calculate compression effectiveness
            compression_ratio = 1.0
            if compressed_data.get("size_info"):
                size_info = compressed_data["size_info"]
                if size_info["original_size"] > 0:
                    compression_ratio = (
                        size_info["compressed_size"] / size_info["original_size"]
                    )

            compression_results[format_type] = {
                "uncompressed_size_mb": uncompressed_size / 1024 / 1024,
                "compression_ratio": compression_ratio,
                "memory_difference": (uncompressed_memory - compressed_memory)
                / 1024
                / 1024,
            }

            print(f"\n{format_type.upper()} compression:")
            print(f"  Uncompressed size: {uncompressed_size / 1024 / 1024:.1f} MB")
            print(f"  Compression ratio: {compression_ratio:.3f}")
            print(
                f"  Memory difference: {compression_results[format_type]['memory_difference']:.1f} MB"
            )

            # Verify compression is effective
            assert compression_ratio < 1.0, (
                f"Compression should reduce size for {format_type}"
            )
            assert compression_ratio > MIN_COMPRESSION_RATIO, (
                f"Compression ratio too low for {format_type}: {compression_ratio:.3f}"
            )


@skip_without_mcp_server
@skip_memory_in_ci
class TestChunkedExportMemory:
    """Test memory usage with chunked exports."""

    async def test_chunked_vs_full_export_memory(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        large_graph_db: Any,
    ) -> None:
        """Test that chunked exports use less memory than full exports."""
        monitor = MemoryMonitor()

        # Test full export memory usage
        monitor.start_monitoring()
        full_request = {
            "root_symbol": "fs_func_0_0",
            "format": "json",
            "depth": 4,  # Larger depth for significant data
            "include_metadata": True,
            "compress": False,
        }

        full_response = await http_client.post(
            f"{MCP_BASE_URL}/export_graph",
            json=full_request,
            headers=auth_headers,
        )

        if full_response.status_code == 404:
            pytest.skip("export_graph endpoint not implemented")

        assert full_response.status_code == 200
        full_memory_peak = monitor.measure()
        full_size = len(full_response.content)

        # Test chunked export memory usage
        monitor.start_monitoring()
        chunk_size = 500  # Small chunks to test memory efficiency

        chunked_request = {
            "root_symbol": "fs_func_0_0",
            "format": "json",
            "depth": 4,
            "include_metadata": True,
            "compress": False,
            "chunk_size": chunk_size,
            "chunk_index": 0,
        }

        chunk_response = await http_client.post(
            f"{MCP_BASE_URL}/export_graph",
            json=chunked_request,
            headers=auth_headers,
        )

        assert chunk_response.status_code == 200
        chunk_memory_peak = monitor.measure()
        chunk_size_bytes = len(chunk_response.content)

        full_memory_mb = (full_memory_peak - monitor.baseline_memory) / 1024 / 1024
        chunk_memory_mb = (chunk_memory_peak - monitor.baseline_memory) / 1024 / 1024

        print("\nChunked vs Full Export Memory:")
        print(f"  Full export memory: {full_memory_mb:.1f} MB")
        print(f"  Chunked export memory: {chunk_memory_mb:.1f} MB")
        print(f"  Full response size: {full_size / 1024:.1f} KB")
        print(f"  Chunk response size: {chunk_size_bytes / 1024:.1f} KB")

        # Verify chunked exports use less memory
        assert chunk_memory_mb < full_memory_mb, (
            "Chunked export should use less memory than full export"
        )

        # Verify chunk is actually smaller
        assert chunk_size_bytes < full_size, "Chunk should be smaller than full export"

    async def test_multiple_chunk_memory_stability(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        large_graph_db: Any,
    ) -> None:
        """Test memory stability when fetching multiple chunks."""
        monitor = MemoryMonitor()
        monitor.start_monitoring()

        chunk_memories = []
        max_chunks_to_test = 5

        for chunk_index in range(max_chunks_to_test):
            request_data = {
                "root_symbol": "fs_func_0_0",
                "format": "json",
                "depth": 3,
                "chunk_size": 300,
                "chunk_index": chunk_index,
                "compress": True,  # Use compression for efficiency
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/export_graph",
                json=request_data,
                headers=auth_headers,
            )

            if response.status_code == 404:
                pytest.skip("export_graph endpoint not implemented")

            # Handle case where chunk_index is beyond available data
            if response.status_code != 200:
                break

            memory_after_chunk = monitor.measure()
            chunk_memories.append(memory_after_chunk)

            data = response.json()
            chunk_info = data.get("chunk_info")

            print(f"\nChunk {chunk_index}:")
            print(
                f"  Memory: {(memory_after_chunk - monitor.baseline_memory) / 1024 / 1024:.1f} MB"
            )
            print(f"  Response size: {len(response.content) / 1024:.1f} KB")

            if chunk_info and not chunk_info.get("has_more", False):
                print("  Last chunk reached")
                break

        if len(chunk_memories) > 1:
            # Calculate memory growth across chunks
            memory_growth = (chunk_memories[-1] - chunk_memories[0]) / 1024 / 1024

            print(f"\nMemory stability across {len(chunk_memories)} chunks:")
            print(f"  Memory growth: {memory_growth:.1f} MB")

            # Verify memory doesn't grow excessively
            assert memory_growth < MAX_MEMORY_GROWTH_MB, (
                f"Memory grew too much across chunks: {memory_growth:.1f}MB"
            )


@skip_without_mcp_server
@skip_memory_in_ci
class TestConcurrentExportMemory:
    """Test memory behavior under concurrent export requests."""

    async def test_concurrent_export_memory_usage(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        large_graph_db: Any,
    ) -> None:
        """Test memory usage with concurrent export requests."""
        import asyncio

        monitor = MemoryMonitor()
        monitor.start_monitoring()

        async def single_export(symbol_suffix: int) -> dict:
            """Perform a single export and return memory stats."""
            request_data = {
                "root_symbol": f"fs_func_{symbol_suffix}_0",
                "format": "json",
                "depth": 2,
                "compress": True,
                "include_metadata": False,  # Reduce memory usage
            }

            start_time = time.time()
            response = await http_client.post(
                f"{MCP_BASE_URL}/export_graph",
                json=request_data,
                headers=auth_headers,
            )
            export_time = time.time() - start_time

            assert response.status_code in [200, 404]
            if response.status_code == 404:
                return {"success": False, "export_time": export_time}

            return {
                "success": True,
                "export_time": export_time,
                "response_size": len(response.content),
            }

        # Launch concurrent exports
        num_concurrent = 5
        tasks = [single_export(i) for i in range(num_concurrent)]

        start_memory = monitor.measure()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_memory = monitor.measure()

        successful_exports = [
            r for r in results if isinstance(r, dict) and r.get("success")
        ]

        if not successful_exports:
            pytest.skip("No successful concurrent exports")

        memory_used = (end_memory - start_memory) / 1024 / 1024
        avg_export_time = mean([r["export_time"] for r in successful_exports])
        total_response_size = (
            sum([r["response_size"] for r in successful_exports]) / 1024 / 1024
        )

        print(f"\nConcurrent exports ({num_concurrent} requests):")
        print(f"  Successful exports: {len(successful_exports)}")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Average export time: {avg_export_time:.2f}s")
        print(f"  Total response size: {total_response_size:.1f} MB")

        # Verify memory usage is reasonable for concurrent requests
        expected_max_memory = (
            MAX_MEMORY_MB_SMALL * num_concurrent * 0.5
        )  # Allow some sharing
        assert memory_used < expected_max_memory, (
            f"Concurrent exports used too much memory: {memory_used:.1f}MB"
        )

        # Verify exports completed in reasonable time
        assert avg_export_time < 30.0, (
            f"Concurrent exports too slow: {avg_export_time:.2f}s"
        )


@skip_without_mcp_server
@skip_memory_in_ci
class TestMemoryLeakDetection:
    """Test for memory leaks during repeated export operations."""

    async def test_repeated_export_memory_stability(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        large_graph_db: Any,
    ) -> None:
        """Test for memory leaks during repeated export operations."""
        monitor = MemoryMonitor()
        monitor.start_monitoring()

        num_iterations = 10
        memory_measurements = []

        request_data = {
            "root_symbol": "fs_func_0_0",
            "format": "json",
            "depth": 2,
            "compress": True,
            "include_metadata": False,
        }

        for iteration in range(num_iterations):
            # Force garbage collection before each iteration
            gc.collect()

            response = await http_client.post(
                f"{MCP_BASE_URL}/export_graph",
                json=request_data,
                headers=auth_headers,
            )

            if response.status_code == 404:
                pytest.skip("export_graph endpoint not implemented")

            assert response.status_code == 200

            # Force garbage collection after response
            del response
            gc.collect()

            post_iteration_memory = monitor.measure()
            memory_measurements.append(post_iteration_memory)

            print(
                f"Iteration {iteration + 1}: {post_iteration_memory / 1024 / 1024:.1f} MB"
            )

        # Analyze memory trend
        first_half_avg = mean(memory_measurements[: num_iterations // 2])
        second_half_avg = mean(memory_measurements[num_iterations // 2 :])
        memory_growth = (second_half_avg - first_half_avg) / 1024 / 1024

        final_stats = monitor.get_stats()

        print(f"\nMemory leak analysis ({num_iterations} iterations):")
        print(f"  First half average: {first_half_avg / 1024 / 1024:.1f} MB")
        print(f"  Second half average: {second_half_avg / 1024 / 1024:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(f"  Peak growth: {final_stats['peak_growth_mb']:.1f} MB")

        # Verify no significant memory growth (indicating leaks)
        assert memory_growth < MAX_MEMORY_GROWTH_MB, (
            f"Possible memory leak detected: {memory_growth:.1f}MB growth"
        )

        # Verify peak memory growth is reasonable
        assert final_stats["peak_growth_mb"] < MAX_MEMORY_MB_LARGE, (
            f"Peak memory usage too high: {final_stats['peak_growth_mb']:.1f}MB"
        )

    async def test_format_specific_memory_profiles(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        large_graph_db: Any,
    ) -> None:
        """Test memory profiles for different export formats."""
        monitor = MemoryMonitor()
        formats = ["json", "graphml", "csv"]
        format_memory_profiles = {}

        for format_type in formats:
            monitor.start_monitoring()

            request_data = {
                "root_symbol": "fs_func_0_0",
                "format": format_type,
                "depth": 3,
                "include_metadata": True,
                "compress": False,
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/export_graph",
                json=request_data,
                headers=auth_headers,
            )

            if response.status_code == 404:
                pytest.skip("export_graph endpoint not implemented")

            assert response.status_code == 200

            memory_used = monitor.measure()
            response_size = len(response.content)

            format_memory_profiles[format_type] = {
                "memory_mb": (memory_used - monitor.baseline_memory) / 1024 / 1024,
                "response_size_mb": response_size / 1024 / 1024,
                "memory_per_mb": ((memory_used - monitor.baseline_memory) / 1024 / 1024)
                / (response_size / 1024 / 1024)
                if response_size > 0
                else 0,
            }

            print(f"\n{format_type.upper()} format memory profile:")
            print(
                f"  Memory used: {format_memory_profiles[format_type]['memory_mb']:.1f} MB"
            )
            print(
                f"  Response size: {format_memory_profiles[format_type]['response_size_mb']:.1f} MB"
            )
            print(
                f"  Memory efficiency: {format_memory_profiles[format_type]['memory_per_mb']:.2f} MB mem/MB response"
            )

        # Verify all formats use reasonable memory
        for format_type, profile in format_memory_profiles.items():
            assert profile["memory_mb"] < MAX_MEMORY_MB_SMALL, (
                f"{format_type} format used too much memory: {profile['memory_mb']:.1f}MB"
            )

            # Verify memory efficiency (memory usage shouldn't be much larger than response size)
            assert profile["memory_per_mb"] < 5.0, (
                f"{format_type} format memory efficiency too low: {profile['memory_per_mb']:.2f}"
            )
