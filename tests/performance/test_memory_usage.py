"""
Memory usage performance tests for chunk processing operations.

Tests verify memory efficiency during chunk operations:
- Memory usage per chunk during processing
- Memory leak detection across multiple chunks
- Memory pressure behavior under concurrent load
- Memory efficiency ratios within constitutional limits
"""

import asyncio
import gc
import json
import os
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

import pytest
import pytest_asyncio
from kcs_mcp.chunk_loader import ChunkLoader
from kcs_mcp.models.chunk_models import ChunkManifest, ChunkMetadata

# Skip in CI unless explicitly enabled
skip_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_PERFORMANCE_TESTS") != "true",
    reason="Performance tests skipped in CI",
)

# Memory efficiency targets
MAX_MEMORY_PER_CHUNK_MB = 150  # Constitutional limit: <150MB per 50MB chunk
TARGET_MEMORY_PER_CHUNK_MB = 100  # Target: <100MB per 50MB chunk
MEMORY_LEAK_THRESHOLD_MB = 50  # Max memory growth across chunks
MEMORY_EFFICIENCY_RATIO = 3.0  # Max memory usage should be <3x chunk size


def measure_memory_usage() -> int:
    """Get current memory usage in bytes (RSS)."""
    try:
        import psutil

        process = psutil.Process()
        return int(process.memory_info().rss)
    except ImportError:
        # Fallback: return 0 if psutil not available
        return 0


def generate_test_chunk_data(size_mb: int) -> dict[str, Any]:
    """Generate synthetic chunk data of approximately target size."""
    target_bytes = size_mb * 1024 * 1024
    base_symbol_size = 450  # Conservative estimate including JSON overhead
    symbol_count = target_bytes // base_symbol_size

    symbols = []
    for i in range(symbol_count):
        symbol = {
            "id": f"mem_test_sym_{i:06d}",
            "name": f"memory_test_function_{i}",
            "type": "function",
            "file_path": f"kernel/memory_test_{i % 100}.c",
            "line": 100 + (i % 1000),
            "signature": f"int memory_test_function_{i}(struct device *dev, unsigned long flags)",
            "visibility": "global" if i % 3 == 0 else "local",
            "is_static": i % 4 != 0,
            "metadata": {
                "complexity_score": i % 100,
                "dependencies": [f"dep_{j}" for j in range(i % 3)],
            },
        }
        symbols.append(symbol)

    return {
        "manifest_version": "1.0.0",
        "chunk_id": f"memory_test_chunk_{size_mb}mb",
        "subsystem": "kernel",
        "symbols": symbols,
        "entrypoints": [],
        "call_graph": [],
        "metadata": {
            "generation_timestamp": "2025-01-18T10:00:00Z",
            "target_size_mb": size_mb,
            "actual_symbol_count": len(symbols),
        },
    }


async def create_memory_test_chunks(
    temp_dir: Path, chunk_sizes: list[int]
) -> tuple[ChunkManifest, list[Path]]:
    """Create test chunks for memory usage testing."""
    chunk_files = []
    chunk_metadata_list = []

    for i, size_mb in enumerate(chunk_sizes):
        chunk_data = generate_test_chunk_data(size_mb)
        chunk_file = temp_dir / f"memory_test_chunk_{i:03d}.json"

        # Write chunk to file
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, separators=(",", ":"))

        file_size = chunk_file.stat().st_size
        chunk_files.append(chunk_file)

        # Create metadata
        import hashlib

        chunk_content = chunk_file.read_bytes()
        checksum = hashlib.sha256(chunk_content).hexdigest()

        metadata = ChunkMetadata(
            id=chunk_data["chunk_id"],
            sequence=i + 1,
            file=str(chunk_file.relative_to(temp_dir)),
            subsystem="kernel",
            size_bytes=file_size,
            checksum_sha256=checksum,
            symbol_count=len(chunk_data["symbols"]),
            entrypoint_count=len(chunk_data["entrypoints"]),
            file_count=1,
        )
        chunk_metadata_list.append(metadata)

    # Create manifest
    from datetime import datetime

    manifest = ChunkManifest(
        version="1.0.0",
        created=datetime.now(),
        total_chunks=len(chunk_sizes),
        chunks=chunk_metadata_list,
    )

    return manifest, chunk_files


class TestMemoryUsage:
    """Test memory usage during chunk processing operations."""

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_memory_usage_per_chunk(self) -> None:
        """Test memory usage for individual chunk processing."""
        chunk_sizes = [10, 25, 45]  # MB sizes
        memory_measurements = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            _manifest, chunk_files = await create_memory_test_chunks(
                temp_path, chunk_sizes
            )

            loader = ChunkLoader()

            for _i, (size_mb, chunk_file) in enumerate(
                zip(chunk_sizes, chunk_files, strict=True)
            ):
                print(f"\nTesting memory usage for {size_mb}MB chunk...")

                # Force garbage collection before measurement
                gc.collect()
                memory_before = measure_memory_usage()

                # Load chunk
                start_time = time.perf_counter()
                chunk_data = await loader.load_chunk(chunk_file)
                load_time_ms = (time.perf_counter() - start_time) * 1000

                # Measure memory after loading
                memory_after_load = measure_memory_usage()

                # Simulate processing (JSON parsing overhead)
                process_start = time.perf_counter()
                parsed_data = json.loads(json.dumps(chunk_data))
                process_time_ms = (time.perf_counter() - process_start) * 1000

                # Measure memory after processing
                gc.collect()
                memory_after_process = measure_memory_usage()

                if memory_before > 0:
                    memory_used_load = memory_after_load - memory_before
                    memory_used_process = memory_after_process - memory_before
                    memory_load_mb = memory_used_load / (1024 * 1024)
                    memory_process_mb = memory_used_process / (1024 * 1024)

                    print(f"  Chunk size: {size_mb}MB")
                    print(f"  Load time: {load_time_ms:.2f}ms")
                    print(f"  Process time: {process_time_ms:.2f}ms")
                    print(f"  Memory for loading: {memory_load_mb:.2f}MB")
                    print(f"  Memory for processing: {memory_process_mb:.2f}MB")
                    print(
                        f"  Memory efficiency (load): {memory_load_mb / size_mb:.2f}x"
                    )
                    print(
                        f"  Memory efficiency (process): {memory_process_mb / size_mb:.2f}x"
                    )

                    memory_measurements.append(
                        {
                            "chunk_size_mb": size_mb,
                            "memory_load_mb": memory_load_mb,
                            "memory_process_mb": memory_process_mb,
                            "load_time_ms": load_time_ms,
                            "process_time_ms": process_time_ms,
                        }
                    )

                    # Memory usage assertions
                    assert memory_load_mb <= MAX_MEMORY_PER_CHUNK_MB, (
                        f"Loading memory ({memory_load_mb:.2f}MB) exceeds limit ({MAX_MEMORY_PER_CHUNK_MB}MB)"
                    )

                    assert memory_process_mb <= MAX_MEMORY_PER_CHUNK_MB, (
                        f"Processing memory ({memory_process_mb:.2f}MB) exceeds limit ({MAX_MEMORY_PER_CHUNK_MB}MB)"
                    )

                    # Efficiency assertions
                    load_efficiency = memory_load_mb / size_mb
                    process_efficiency = memory_process_mb / size_mb

                    assert load_efficiency <= MEMORY_EFFICIENCY_RATIO, (
                        f"Load memory efficiency ({load_efficiency:.2f}x) exceeds target ({MEMORY_EFFICIENCY_RATIO}x)"
                    )

                    assert process_efficiency <= MEMORY_EFFICIENCY_RATIO, (
                        f"Process memory efficiency ({process_efficiency:.2f}x) exceeds target ({MEMORY_EFFICIENCY_RATIO}x)"
                    )

                # Clean up chunk data to prevent accumulation
                del chunk_data, parsed_data
                gc.collect()

        # Overall efficiency analysis
        if memory_measurements:
            avg_load_efficiency = mean(
                m["memory_load_mb"] / m["chunk_size_mb"] for m in memory_measurements
            )
            avg_process_efficiency = mean(
                m["memory_process_mb"] / m["chunk_size_mb"] for m in memory_measurements
            )

            print("\nOverall memory efficiency:")
            print(f"  Average load efficiency: {avg_load_efficiency:.2f}x")
            print(f"  Average process efficiency: {avg_process_efficiency:.2f}x")

            assert avg_load_efficiency <= TARGET_MEMORY_PER_CHUNK_MB / 50, (
                f"Average load efficiency too high: {avg_load_efficiency:.2f}x"
            )

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self) -> None:
        """Test for memory leaks across multiple chunk operations."""
        chunk_size = 20  # MB per chunk
        num_iterations = 5
        memory_snapshots = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple identical chunks
            chunk_sizes = [chunk_size] * num_iterations
            _manifest, chunk_files = await create_memory_test_chunks(
                temp_path, chunk_sizes
            )

            loader = ChunkLoader()

            for iteration in range(num_iterations):
                print(f"\nIteration {iteration + 1}/{num_iterations}")

                # Force garbage collection before measurement
                gc.collect()
                memory_before = measure_memory_usage()

                # Load and process chunk
                chunk_file = chunk_files[iteration]
                chunk_data = await loader.load_chunk(chunk_file)

                # Simulate some processing work
                processed_symbols = [
                    {"id": s["id"], "name": s["name"]}
                    for s in chunk_data.get("symbols", [])[:100]
                ]

                # Measure memory after processing
                gc.collect()
                memory_after = measure_memory_usage()

                if memory_before > 0:
                    memory_used_mb = (memory_after - memory_before) / (1024 * 1024)
                    memory_snapshots.append(memory_used_mb)

                    print(f"  Memory before: {memory_before / (1024 * 1024):.2f}MB")
                    print(f"  Memory after: {memory_after / (1024 * 1024):.2f}MB")
                    print(f"  Memory used: {memory_used_mb:.2f}MB")
                    print(f"  Symbols processed: {len(processed_symbols)}")

                # Clean up explicitly
                del chunk_data, processed_symbols
                gc.collect()

        # Analyze memory leak patterns
        if len(memory_snapshots) >= 3:
            # Check if memory usage is growing consistently
            memory_growth = memory_snapshots[-1] - memory_snapshots[0]
            max_memory = max(memory_snapshots)
            min_memory = min(memory_snapshots)
            memory_variance = max_memory - min_memory

            print("\nMemory leak analysis:")
            print(f"  Initial memory usage: {memory_snapshots[0]:.2f}MB")
            print(f"  Final memory usage: {memory_snapshots[-1]:.2f}MB")
            print(f"  Total memory growth: {memory_growth:.2f}MB")
            print(f"  Memory variance: {memory_variance:.2f}MB")
            print(f"  Memory snapshots: {[f'{m:.2f}' for m in memory_snapshots]}")

            # Assert no significant memory leak
            assert abs(memory_growth) <= MEMORY_LEAK_THRESHOLD_MB, (
                f"Potential memory leak detected: {memory_growth:.2f}MB growth over {num_iterations} iterations"
            )

            assert memory_variance <= MEMORY_LEAK_THRESHOLD_MB, (
                f"High memory variance detected: {memory_variance:.2f}MB"
            )

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_concurrent_memory_pressure(self) -> None:
        """Test memory usage under concurrent chunk processing."""
        chunk_size = 15  # MB per chunk
        num_concurrent = 4

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple chunks for concurrent processing
            chunk_sizes = [chunk_size] * num_concurrent
            _manifest, chunk_files = await create_memory_test_chunks(
                temp_path, chunk_sizes
            )

            # Measure baseline memory
            gc.collect()
            baseline_memory = measure_memory_usage()

            async def process_chunk_with_memory_tracking(
                chunk_file: Path, chunk_id: int
            ) -> dict[str, Any]:
                """Process a chunk and track memory usage."""
                loader = ChunkLoader()

                # Load chunk
                chunk_data = await loader.load_chunk(chunk_file)

                # Simulate processing work
                start_time = time.perf_counter()
                symbols = chunk_data.get("symbols", [])
                processed_count = 0

                # Simulate symbol processing
                for symbol in symbols[:1000]:  # Limit processing to avoid timeout
                    _ = {
                        "id": symbol["id"],
                        "name": symbol["name"],
                        "processed": True,
                    }
                    processed_count += 1

                processing_time_ms = (time.perf_counter() - start_time) * 1000

                return {
                    "chunk_id": chunk_id,
                    "symbols_processed": processed_count,
                    "processing_time_ms": processing_time_ms,
                    "chunk_size_mb": chunk_size,
                }

            # Process chunks concurrently
            print(f"\nProcessing {num_concurrent} chunks concurrently...")
            start_total = time.perf_counter()

            # Use semaphore to limit memory pressure
            semaphore = asyncio.Semaphore(num_concurrent)

            async def process_with_semaphore(chunk_file: Path, chunk_id: int) -> dict:
                async with semaphore:
                    return await process_chunk_with_memory_tracking(
                        chunk_file, chunk_id
                    )

            tasks = [
                process_with_semaphore(chunk_file, i)
                for i, chunk_file in enumerate(chunk_files)
            ]

            # Monitor memory during concurrent processing
            peak_memory = baseline_memory
            memory_samples = []

            async def memory_monitor():
                """Monitor memory usage during concurrent processing."""
                nonlocal peak_memory
                for _ in range(20):  # Sample for ~2 seconds
                    await asyncio.sleep(0.1)
                    current_memory = measure_memory_usage()
                    if current_memory > peak_memory:
                        peak_memory = current_memory
                    memory_samples.append(current_memory)

            # Run processing and monitoring concurrently
            monitor_task = asyncio.create_task(memory_monitor())
            results = await asyncio.gather(*tasks)
            monitor_task.cancel()

            total_time_ms = (time.perf_counter() - start_total) * 1000

            # Analyze results
            if baseline_memory > 0 and peak_memory > baseline_memory:
                memory_used_mb = (peak_memory - baseline_memory) / (1024 * 1024)
                memory_per_chunk_mb = memory_used_mb / num_concurrent
                total_data_size_mb = num_concurrent * chunk_size

                print(f"  Baseline memory: {baseline_memory / (1024 * 1024):.2f}MB")
                print(f"  Peak memory: {peak_memory / (1024 * 1024):.2f}MB")
                print(f"  Memory used: {memory_used_mb:.2f}MB")
                print(f"  Memory per chunk: {memory_per_chunk_mb:.2f}MB")
                print(f"  Total data processed: {total_data_size_mb}MB")
                print(
                    f"  Memory efficiency: {memory_used_mb / total_data_size_mb:.2f}x"
                )
                print(f"  Total processing time: {total_time_ms:.2f}ms")

                # Memory efficiency assertions
                assert memory_per_chunk_mb <= MAX_MEMORY_PER_CHUNK_MB, (
                    f"Memory per chunk ({memory_per_chunk_mb:.2f}MB) exceeds limit ({MAX_MEMORY_PER_CHUNK_MB}MB)"
                )

                memory_efficiency = memory_used_mb / total_data_size_mb
                assert memory_efficiency <= MEMORY_EFFICIENCY_RATIO, (
                    f"Memory efficiency ({memory_efficiency:.2f}x) exceeds target ({MEMORY_EFFICIENCY_RATIO}x)"
                )

            # Verify all chunks processed successfully
            assert len(results) == num_concurrent
            for result in results:
                assert result["symbols_processed"] > 0
                assert result["processing_time_ms"] > 0

    @skip_in_ci
    @pytest.mark.asyncio
    async def test_memory_efficiency_scaling(self) -> None:
        """Test memory efficiency as chunk size increases."""
        chunk_sizes = [5, 15, 30, 45]  # MB
        efficiency_measurements = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for chunk_size in chunk_sizes:
                print(f"\nTesting memory efficiency for {chunk_size}MB chunk...")

                # Create single chunk
                _manifest, chunk_files = await create_memory_test_chunks(
                    temp_path, [chunk_size]
                )
                chunk_file = chunk_files[0]

                # Measure memory usage
                gc.collect()
                memory_before = measure_memory_usage()

                loader = ChunkLoader()
                chunk_data = await loader.load_chunk(chunk_file)

                # Simulate realistic processing
                symbols = chunk_data.get("symbols", [])
                processed_data = []

                for i, symbol in enumerate(symbols):
                    if i % 100 == 0:  # Process every 100th symbol for efficiency
                        processed_symbol = {
                            "id": symbol["id"],
                            "name": symbol["name"],
                            "file": symbol["file_path"],
                            "line": symbol["line"],
                        }
                        processed_data.append(processed_symbol)

                gc.collect()
                memory_after = measure_memory_usage()

                if memory_before > 0:
                    memory_used_mb = (memory_after - memory_before) / (1024 * 1024)
                    efficiency_ratio = memory_used_mb / chunk_size

                    efficiency_measurements.append(
                        {
                            "chunk_size_mb": chunk_size,
                            "memory_used_mb": memory_used_mb,
                            "efficiency_ratio": efficiency_ratio,
                            "symbols_processed": len(processed_data),
                        }
                    )

                    print(f"  Chunk size: {chunk_size}MB")
                    print(f"  Memory used: {memory_used_mb:.2f}MB")
                    print(f"  Efficiency ratio: {efficiency_ratio:.2f}x")
                    print(f"  Symbols processed: {len(processed_data)}")

                    # Per-size efficiency check
                    assert efficiency_ratio <= MEMORY_EFFICIENCY_RATIO, (
                        f"Efficiency ratio ({efficiency_ratio:.2f}x) exceeds target for {chunk_size}MB chunk"
                    )

                # Clean up
                del chunk_data, processed_data
                gc.collect()

        # Analyze scaling behavior
        if len(efficiency_measurements) >= 2:
            efficiency_ratios = [m["efficiency_ratio"] for m in efficiency_measurements]
            min_efficiency = min(efficiency_ratios)
            max_efficiency = max(efficiency_ratios)
            efficiency_variance = max_efficiency - min_efficiency

            print("\nMemory efficiency scaling analysis:")
            print(f"  Min efficiency: {min_efficiency:.2f}x")
            print(f"  Max efficiency: {max_efficiency:.2f}x")
            print(f"  Efficiency variance: {efficiency_variance:.2f}x")

            # Efficiency should not degrade significantly with size
            assert efficiency_variance <= 1.0, (
                f"Memory efficiency varies too much across chunk sizes: {efficiency_variance:.2f}x"
            )

            # All measurements should be within targets
            assert max_efficiency <= MEMORY_EFFICIENCY_RATIO, (
                f"Worst efficiency ({max_efficiency:.2f}x) exceeds target ({MEMORY_EFFICIENCY_RATIO}x)"
            )


if __name__ == "__main__":
    # Allow running individual tests for development
    pytest.main([__file__, "-v", "-s"])
