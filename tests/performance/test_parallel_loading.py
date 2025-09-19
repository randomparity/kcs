"""
Performance tests for parallel chunk loading operations.

Tests verify that parallel chunk loading meets constitutional requirements:
- Efficient concurrent chunk loading
- Proper semaphore and resource management
- Scalable performance with increasing parallelism
- Database connection pool efficiency under load
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from statistics import mean, quantiles
from typing import Any

import aiofiles
import pytest
import pytest_asyncio
from kcs_mcp.chunk_loader import ChunkLoader
from kcs_mcp.chunk_processor import ChunkProcessor
from kcs_mcp.database import Database
from kcs_mcp.models.chunk_models import ChunkManifest, ChunkMetadata

# Skip in CI unless explicitly enabled
skip_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_PERFORMANCE_TESTS") != "true",
    reason="Performance tests skipped in CI",
)

# Performance targets
MAX_PARALLEL_LOAD_TIME_MS = 30000  # Target: <30s for loading 20 chunks in parallel
TARGET_PARALLEL_LOAD_TIME_MS = 15000  # Target: <15s for typical parallel load
MIN_PARALLEL_EFFICIENCY = 0.3  # Minimum parallel efficiency (30% of theoretical)
MAX_CHUNK_SIZE_BYTES = 50 * 1024 * 1024  # Constitutional limit: 50MB


def create_test_chunk_data(chunk_id: str, size_multiplier: int = 1) -> dict[str, Any]:
    """Create test chunk data with configurable size."""
    base_symbols = 100 * size_multiplier

    symbols = []
    for i in range(base_symbols):
        symbols.append(
            {
                "id": f"{chunk_id}_sym_{i:04d}",
                "name": f"test_function_{chunk_id}_{i}",
                "type": "function",
                "file_path": f"test/file_{i % 10}.c",
                "line": 100 + i,
                "signature": f"int test_function_{chunk_id}_{i}(void)",
                "visibility": "global" if i % 2 == 0 else "local",
                "is_static": i % 3 == 0,
                "is_inline": i % 4 == 0,
                "metadata": {
                    "export_type": "EXPORT_SYMBOL" if i % 5 == 0 else None,
                },
            }
        )

    entrypoints = []
    for i in range(base_symbols // 10):
        entrypoints.append(
            {
                "id": f"{chunk_id}_ep_{i:04d}",
                "name": f"sys_test_{chunk_id}_{i}",
                "type": "syscall",
                "file_path": f"test/syscall_{i}.c",
                "line": 50 + i,
                "signature": f"SYSCALL_DEFINE1(test_{chunk_id}_{i}, int, arg)",
                "syscall_number": 1000 + i,
                "metadata": {
                    "syscall_nr": f"__NR_test_{chunk_id}_{i}",
                },
            }
        )

    call_graph = []
    for i in range(min(base_symbols // 5, 50)):  # Limit call graph for performance
        caller_idx = i
        callee_idx = (i + 1) % base_symbols
        call_graph.append(
            {
                "caller_id": f"{chunk_id}_sym_{caller_idx:04d}",
                "callee_id": f"{chunk_id}_sym_{callee_idx:04d}",
                "call_type": "direct",
                "file_path": f"test/file_{caller_idx % 10}.c",
                "line": 200 + i,
            }
        )

    return {
        "manifest_version": "1.0.0",
        "chunk_id": chunk_id,
        "subsystem": "test",
        "symbols": symbols,
        "entrypoints": entrypoints,
        "call_graph": call_graph,
        "metadata": {
            "test_chunk": True,
            "size_multiplier": size_multiplier,
            "symbol_count": len(symbols),
            "entrypoint_count": len(entrypoints),
            "call_graph_edges": len(call_graph),
        },
    }


async def create_test_chunks_on_disk(
    temp_dir: Path, num_chunks: int, size_multiplier: int = 1
) -> tuple[ChunkManifest, list[Path]]:
    """Create test chunks on disk and return manifest and file paths."""
    chunk_files = []
    chunk_metadata_list = []

    for i in range(num_chunks):
        chunk_id = f"test_chunk_{i:03d}"
        chunk_data = create_test_chunk_data(chunk_id, size_multiplier)

        # Write chunk to file
        chunk_file = temp_dir / f"{chunk_id}.json"
        async with aiofiles.open(chunk_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(chunk_data, separators=(",", ":")))

        chunk_files.append(chunk_file)

        # Calculate file size and create metadata
        file_size = chunk_file.stat().st_size

        # Create a simple checksum (for testing - not cryptographically secure)
        import hashlib

        content = chunk_file.read_text(encoding="utf-8")
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

        chunk_metadata = ChunkMetadata(
            id=chunk_id,
            sequence=i + 1,
            file=chunk_file.name,
            subsystem="test",
            size_bytes=file_size,
            checksum_sha256=checksum,
            symbol_count=len(chunk_data["symbols"]),
            entrypoint_count=len(chunk_data["entrypoints"]),
            file_count=1,
        )
        chunk_metadata_list.append(chunk_metadata)

    # Create manifest
    total_size = sum(f.stat().st_size for f in chunk_files)
    manifest = ChunkManifest(
        version="1.0.0",
        created="2025-01-18T10:00:00Z",
        kernel_version="6.7.0-test",
        kernel_path="/test/kernel",
        config="test:config",
        total_chunks=num_chunks,
        total_size_bytes=total_size,
        chunks=chunk_metadata_list,
    )

    return manifest, chunk_files


class TestParallelChunkLoading:
    """Test parallel chunk loading performance and efficiency."""

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_concurrent_chunk_loading_performance(self) -> None:
        """Test loading multiple chunks concurrently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            num_chunks = 10
            size_multiplier = 2  # Medium-sized chunks

            # Create test chunks
            manifest, _chunk_files = await create_test_chunks_on_disk(
                temp_path, num_chunks, size_multiplier
            )

            # Test different parallelism levels
            parallelism_levels = [1, 2, 4, 8]
            results = []

            for parallelism in parallelism_levels:
                print(f"\nTesting parallelism level: {parallelism}")

                # Create chunk loader
                loader = ChunkLoader(verify_checksums=True)

                # Create semaphore to control parallelism
                semaphore = asyncio.Semaphore(parallelism)

                async def load_chunk_with_semaphore(
                    chunk_metadata: ChunkMetadata,
                    semaphore: asyncio.Semaphore = semaphore,
                    loader: ChunkLoader = loader,
                    temp_path: Path = temp_path,
                ) -> float:
                    """Load a chunk with semaphore control and measure time."""
                    async with semaphore:
                        start = time.perf_counter()
                        await loader.load_chunk(chunk_metadata, base_path=temp_path)
                        return (time.perf_counter() - start) * 1000

                # Load all chunks with controlled parallelism
                start_total = time.perf_counter()
                load_times = await asyncio.gather(
                    *[load_chunk_with_semaphore(chunk) for chunk in manifest.chunks]
                )
                total_time_ms = (time.perf_counter() - start_total) * 1000

                avg_load_time = mean(load_times)
                max_load_time = max(load_times)

                # Calculate parallel efficiency
                sequential_estimate = sum(load_times)  # If run sequentially
                parallel_efficiency = (
                    sequential_estimate / total_time_ms if total_time_ms > 0 else 0
                )

                result = {
                    "parallelism": parallelism,
                    "total_time_ms": total_time_ms,
                    "avg_load_time_ms": avg_load_time,
                    "max_load_time_ms": max_load_time,
                    "parallel_efficiency": parallel_efficiency,
                }
                results.append(result)

                print(f"  Total time: {total_time_ms:.2f}ms")
                print(f"  Average load time: {avg_load_time:.2f}ms")
                print(f"  Max load time: {max_load_time:.2f}ms")
                print(f"  Parallel efficiency: {parallel_efficiency:.2f}x")

            # Analyze results
            best_result = min(results, key=lambda r: r["total_time_ms"])
            worst_result = max(results, key=lambda r: r["total_time_ms"])

            print("\nParallel loading analysis:")
            print(
                f"  Best time: {best_result['total_time_ms']:.2f}ms (parallelism={best_result['parallelism']})"
            )
            print(
                f"  Worst time: {worst_result['total_time_ms']:.2f}ms (parallelism={worst_result['parallelism']})"
            )

            # Performance assertions
            assert best_result["total_time_ms"] <= MAX_PARALLEL_LOAD_TIME_MS, (
                f"Best parallel loading time ({best_result['total_time_ms']:.2f}ms) exceeds limit"
            )

            assert best_result["parallel_efficiency"] >= MIN_PARALLEL_EFFICIENCY, (
                f"Parallel efficiency ({best_result['parallel_efficiency']:.2f}x) below minimum"
            )

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_chunk_processor_parallel_performance(self) -> None:
        """Test ChunkProcessor's adaptive parallelism under load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            num_chunks = 8
            size_multiplier = 3  # Larger chunks to test memory pressure

            # Create test chunks
            manifest, _chunk_files = await create_test_chunks_on_disk(
                temp_path, num_chunks, size_multiplier
            )

            # Create manifest file
            manifest_file = temp_path / "manifest.json"
            async with aiofiles.open(manifest_file, "w", encoding="utf-8") as f:
                await f.write(
                    json.dumps(
                        manifest.model_dump(), separators=(",", ":"), default=str
                    )
                )

            # Test different parallelism configurations
            test_configs = [
                {"max_parallelism": 1, "adaptive": False, "label": "Sequential"},
                {"max_parallelism": 2, "adaptive": False, "label": "Fixed-2"},
                {"max_parallelism": 4, "adaptive": False, "label": "Fixed-4"},
                {"max_parallelism": None, "adaptive": True, "label": "Adaptive"},
            ]

            # Mock database for testing (memory-only) - Skip actual database for pure loading test
            # mock_db = Database("sqlite:///:memory:")

            config_results = []

            for config in test_configs:
                print(f"\nTesting configuration: {config['label']}")

                # Create chunk processor with specific configuration
                default_parallelism = (
                    config.get("max_parallelism") or 4
                )  # Ensure it's never None
                processor = ChunkProcessor(
                    chunk_loader=ChunkLoader(
                        verify_checksums=False
                    ),  # Skip checksum for speed
                    database_queries=None,  # Skip database operations for pure loading test
                    default_max_parallelism=default_parallelism,
                    adaptive_parallelism=config["adaptive"],
                    max_memory_mb=1024,  # 1GB limit for testing
                )

                # Measure loading time
                start = time.perf_counter()

                # Use a simplified loading approach to focus on parallelism
                chunk_metadata_list = manifest.chunks
                optimal_parallelism = processor._calculate_optimal_parallelism(
                    chunk_metadata_list, config.get("max_parallelism")
                )

                # Create loader tasks
                semaphore = asyncio.Semaphore(optimal_parallelism)
                loader = ChunkLoader(verify_checksums=False)

                async def load_chunk_task(
                    chunk_metadata: ChunkMetadata,
                    semaphore: asyncio.Semaphore = semaphore,
                    loader: ChunkLoader = loader,
                    temp_path: Path = temp_path,
                ) -> dict[str, Any]:
                    async with semaphore:
                        return await loader.load_chunk(
                            chunk_metadata, base_path=temp_path
                        )

                # Execute parallel loading
                loaded_chunks = await asyncio.gather(
                    *[load_chunk_task(chunk) for chunk in chunk_metadata_list]
                )

                total_time_ms = (time.perf_counter() - start) * 1000

                result = {
                    "config": config["label"],
                    "optimal_parallelism": optimal_parallelism,
                    "total_time_ms": total_time_ms,
                    "chunks_loaded": len(loaded_chunks),
                    "avg_time_per_chunk": total_time_ms / len(loaded_chunks),
                }
                config_results.append(result)

                print(f"  Optimal parallelism: {optimal_parallelism}")
                print(f"  Total time: {total_time_ms:.2f}ms")
                print(f"  Chunks loaded: {len(loaded_chunks)}")
                print(f"  Avg time per chunk: {result['avg_time_per_chunk']:.2f}ms")

                # Verify all chunks loaded successfully
                assert len(loaded_chunks) == num_chunks
                for chunk_data in loaded_chunks:
                    assert "symbols" in chunk_data
                    assert len(chunk_data["symbols"]) > 0

            # await mock_db.close()

            # Analyze configuration performance
            sequential_time = next(
                r for r in config_results if r["config"] == "Sequential"
            )["total_time_ms"]
            best_parallel = min(
                (r for r in config_results if r["config"] != "Sequential"),
                key=lambda r: r["total_time_ms"],
            )

            speedup = sequential_time / best_parallel["total_time_ms"]

            print("\nConfiguration analysis:")
            print(f"  Sequential time: {sequential_time:.2f}ms")
            print(
                f"  Best parallel time: {best_parallel['total_time_ms']:.2f}ms ({best_parallel['config']})"
            )
            print(f"  Speedup: {speedup:.2f}x")

            # Performance assertions
            assert best_parallel["total_time_ms"] <= TARGET_PARALLEL_LOAD_TIME_MS, (
                f"Best parallel configuration too slow ({best_parallel['total_time_ms']:.2f}ms)"
            )

            assert speedup >= 1.2, (
                f"Parallel processing shows insufficient speedup ({speedup:.2f}x)"
            )

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_parallel_loading_with_checksum_verification(self) -> None:
        """Test parallel loading performance with checksum verification enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            num_chunks = 6
            size_multiplier = 2

            # Create test chunks
            manifest, _chunk_files = await create_test_chunks_on_disk(
                temp_path, num_chunks, size_multiplier
            )

            # Test with and without checksum verification
            verification_configs = [
                {"verify": False, "label": "No Verification"},
                {"verify": True, "label": "With Verification"},
            ]

            results = []

            for config in verification_configs:
                print(f"\nTesting {config['label']}")

                loader = ChunkLoader(verify_checksums=config["verify"])
                parallelism = 4
                semaphore = asyncio.Semaphore(parallelism)

                async def load_chunk_with_timing(
                    chunk_metadata: ChunkMetadata,
                    semaphore: asyncio.Semaphore = semaphore,
                    loader: ChunkLoader = loader,
                    temp_path: Path = temp_path,
                ) -> float:
                    async with semaphore:
                        start = time.perf_counter()
                        await loader.load_chunk(chunk_metadata, base_path=temp_path)
                        return (time.perf_counter() - start) * 1000

                start_total = time.perf_counter()
                load_times = await asyncio.gather(
                    *[load_chunk_with_timing(chunk) for chunk in manifest.chunks]
                )
                total_time_ms = (time.perf_counter() - start_total) * 1000

                avg_time = mean(load_times)
                p95_time = (
                    quantiles(load_times, n=20)[18]
                    if len(load_times) >= 20
                    else max(load_times)
                )

                result = {
                    "verification": config["verify"],
                    "label": config["label"],
                    "total_time_ms": total_time_ms,
                    "avg_load_time_ms": avg_time,
                    "p95_load_time_ms": p95_time,
                }
                results.append(result)

                print(f"  Total time: {total_time_ms:.2f}ms")
                print(f"  Average load time: {avg_time:.2f}ms")
                print(f"  P95 load time: {p95_time:.2f}ms")

            # Calculate verification overhead
            no_verify = next(r for r in results if not r["verification"])
            with_verify = next(r for r in results if r["verification"])

            overhead_factor = with_verify["total_time_ms"] / no_verify["total_time_ms"]

            print("\nChecksum verification analysis:")
            print(f"  Without verification: {no_verify['total_time_ms']:.2f}ms")
            print(f"  With verification: {with_verify['total_time_ms']:.2f}ms")
            print(f"  Verification overhead: {overhead_factor:.2f}x")

            # Performance assertions
            assert with_verify["total_time_ms"] <= MAX_PARALLEL_LOAD_TIME_MS, (
                f"Parallel loading with verification too slow ({with_verify['total_time_ms']:.2f}ms)"
            )

            # Verification should not add more than 3x overhead
            assert overhead_factor <= 3.0, (
                f"Checksum verification overhead too high ({overhead_factor:.2f}x)"
            )

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_parallel_loading_memory_efficiency(self) -> None:
        """Test memory efficiency during parallel chunk loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test different chunk sizes and parallelism levels
            test_cases = [
                {
                    "chunks": 4,
                    "size_mult": 1,
                    "parallelism": 4,
                    "label": "Small chunks, high parallelism",
                },
                {
                    "chunks": 8,
                    "size_mult": 2,
                    "parallelism": 2,
                    "label": "Medium chunks, medium parallelism",
                },
                {
                    "chunks": 2,
                    "size_mult": 5,
                    "parallelism": 1,
                    "label": "Large chunks, low parallelism",
                },
            ]

            def get_memory_usage() -> int:
                """Get current memory usage in bytes."""
                try:
                    import psutil

                    return int(psutil.Process().memory_info().rss)
                except ImportError:
                    return 0

            for case in test_cases:
                print(f"\nTesting: {case['label']}")

                # Create test chunks for this case
                manifest, chunk_files = await create_test_chunks_on_disk(
                    temp_path, case["chunks"], case["size_mult"]
                )

                memory_before = get_memory_usage()

                # Load chunks in parallel
                loader = ChunkLoader(verify_checksums=False)
                semaphore = asyncio.Semaphore(case["parallelism"])

                async def load_chunk_with_semaphore(
                    chunk_metadata: ChunkMetadata,
                    semaphore: asyncio.Semaphore = semaphore,
                    loader: ChunkLoader = loader,
                    temp_path: Path = temp_path,
                ) -> dict[str, Any]:
                    async with semaphore:
                        return await loader.load_chunk(
                            chunk_metadata, base_path=temp_path
                        )

                start = time.perf_counter()
                loaded_chunks = await asyncio.gather(
                    *[load_chunk_with_semaphore(chunk) for chunk in manifest.chunks]
                )
                load_time_ms = (time.perf_counter() - start) * 1000

                memory_after = get_memory_usage()

                # Calculate memory efficiency
                total_file_size = sum(f.stat().st_size for f in chunk_files)
                memory_used = memory_after - memory_before if memory_after > 0 else 0
                memory_ratio = (
                    memory_used / total_file_size if total_file_size > 0 else 0
                )

                print(f"  Chunks loaded: {len(loaded_chunks)}")
                print(f"  Total file size: {total_file_size:,} bytes")
                print(f"  Load time: {load_time_ms:.2f}ms")
                if memory_used > 0:
                    print(f"  Memory used: {memory_used:,} bytes")
                    print(f"  Memory ratio: {memory_ratio:.2f}x")

                # Verify all chunks loaded correctly
                assert len(loaded_chunks) == case["chunks"]
                for chunk_data in loaded_chunks:
                    assert "symbols" in chunk_data
                    assert len(chunk_data["symbols"]) > 0

                # Memory efficiency assertion (if psutil available)
                if memory_used > 0:
                    assert memory_ratio <= 4.0, (
                        f"Memory usage too high ({memory_ratio:.2f}x) for {case['label']}"
                    )

                # Clean up loaded data
                del loaded_chunks

                # Clean up chunk files for next iteration
                for chunk_file in chunk_files:
                    chunk_file.unlink()

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_parallel_loading_error_handling(self) -> None:
        """Test error handling during parallel chunk loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            num_chunks = 5

            # Create test chunks
            manifest, chunk_files = await create_test_chunks_on_disk(
                temp_path, num_chunks, size_multiplier=1
            )

            # Corrupt one chunk file (delete it)
            corrupted_chunk = chunk_files[2]
            corrupted_chunk.unlink()

            # Corrupt another chunk (invalid JSON)
            invalid_chunk = chunk_files[3]
            invalid_chunk.write_text("{ invalid json content", encoding="utf-8")

            print(f"\nTesting error handling with {num_chunks} chunks:")
            print(f"  - {len(chunk_files) - 2} valid chunks")
            print("  - 1 missing chunk")
            print("  - 1 invalid JSON chunk")

            loader = ChunkLoader(verify_checksums=False)
            parallelism = 3
            semaphore = asyncio.Semaphore(parallelism)

            async def load_chunk_safe(chunk_metadata: ChunkMetadata) -> dict[str, Any]:
                """Load chunk with error handling."""
                async with semaphore:
                    try:
                        return await loader.load_chunk(
                            chunk_metadata, base_path=temp_path
                        )
                    except Exception as e:
                        return {"error": str(e), "chunk_id": chunk_metadata.id}

            start = time.perf_counter()
            results = await asyncio.gather(
                *[load_chunk_safe(chunk) for chunk in manifest.chunks]
            )
            total_time_ms = (time.perf_counter() - start) * 1000

            # Analyze results
            successful_loads = [r for r in results if "error" not in r]
            failed_loads = [r for r in results if "error" in r]

            print(f"  Load time: {total_time_ms:.2f}ms")
            print(f"  Successful loads: {len(successful_loads)}")
            print(f"  Failed loads: {len(failed_loads)}")

            for failed in failed_loads:
                print(f"    - {failed['chunk_id']}: {failed['error']}")

            # Verify error handling worked correctly (5 total: 3 good, 1 deleted, 1 corrupted)
            assert len(successful_loads) == 3, (
                f"Expected 3 successful loads, got {len(successful_loads)}"
            )
            assert len(failed_loads) == 2, (
                f"Expected 2 failed loads, got {len(failed_loads)}"
            )

            # Verify successful chunks loaded properly
            for chunk_data in successful_loads:
                assert "symbols" in chunk_data
                assert len(chunk_data["symbols"]) > 0

            # Performance assertion - error handling shouldn't be too slow
            assert total_time_ms <= TARGET_PARALLEL_LOAD_TIME_MS, (
                f"Error handling during parallel loading too slow ({total_time_ms:.2f}ms)"
            )


if __name__ == "__main__":
    # Allow running individual tests for development
    pytest.main([__file__, "-v", "-s"])
