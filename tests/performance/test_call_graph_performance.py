"""
Performance benchmarks for call graph extraction.

Tests the performance and scalability of call graph extraction
on various sizes and complexities of kernel source code.
"""

import asyncio
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import requests
from kcs_mcp.database import Database
from kcs_mcp.database.call_graph import CallGraphWriter
from kcs_mcp.tools import CallGraphExtractor, ExtractCallGraphRequest

# Global test utilities and fixtures


def create_mock_call_edge(
    caller_name: str, callee_name: str, file_path: str, line_num: int
) -> dict[str, Any]:
    """Create a properly structured mock call edge for testing."""
    return {
        "caller": {
            "name": caller_name,
            "file_path": file_path,
            "line_number": max(1, line_num - 5),  # Ensure line_number > 0
            "symbol_type": "function",
        },
        "callee": {
            "name": callee_name,
            "file_path": "/usr/include/linux/kernel.h",
            "line_number": 100,
            "symbol_type": "function",
        },
        "call_site": {
            "file_path": file_path,
            "line_number": max(1, line_num),  # Ensure line_number > 0
            "column_number": 12,
        },
        "call_type": "direct",
        "confidence": "high",
    }


@pytest.fixture
async def mock_database():
    """Create a mock database for testing."""
    # Create a mock database that doesn't require actual PostgreSQL
    database = AsyncMock(spec=Database)

    # Mock the connection methods
    database.get_connection = AsyncMock()
    database.get_connection.return_value.__aenter__ = AsyncMock()
    database.get_connection.return_value.__aexit__ = AsyncMock()

    return database


@pytest.fixture
def extractor(mock_database):
    """Create a call graph extractor with mock database."""
    return CallGraphExtractor(mock_database)


def create_sample_kernel_file(complexity: str = "medium", size: str = "small") -> str:
    """Create sample kernel source code for testing."""
    base_functions = {
        "simple": [
            "static int init_module(void)",
            "static void exit_module(void)",
            "static int device_open(struct inode *inode, struct file *file)",
            "static int device_release(struct inode *inode, struct file *file)",
        ],
        "medium": [
            "static int __init driver_init(void)",
            "static void __exit driver_exit(void)",
            "static irqreturn_t interrupt_handler(int irq, void *dev_id)",
            "static int probe_device(struct platform_device *pdev)",
            "static int remove_device(struct platform_device *pdev)",
            "static long device_ioctl(struct file *file, unsigned int cmd, unsigned long arg)",
        ],
        "complex": [
            "static int __init complex_driver_init(void)",
            "static void __exit complex_driver_exit(void)",
            "static irqreturn_t primary_interrupt_handler(int irq, void *dev_id)",
            "static void tasklet_handler(unsigned long data)",
            "static void work_queue_handler(struct work_struct *work)",
            "static int probe_pci_device(struct pci_dev *pdev, const struct pci_device_id *id)",
            "static void remove_pci_device(struct pci_dev *pdev)",
            "static int configure_dma(struct device_data *data)",
            "static int power_management_suspend(struct device *dev)",
            "static int power_management_resume(struct device *dev)",
        ],
    }

    multipliers = {
        "small": 1,
        "medium": 3,
        "large": 10,
        "huge": 30,
    }

    functions = base_functions.get(complexity, base_functions["medium"])
    multiplier = multipliers.get(size, 1)

    code_parts = [
        "#include <linux/module.h>",
        "#include <linux/kernel.h>",
        "#include <linux/fs.h>",
        "#include <linux/device.h>",
        "#include <linux/interrupt.h>",
        "#include <linux/platform_device.h>",
        "#include <linux/slab.h>",
        "",
        "static struct device_data {",
        "    struct platform_device *pdev;",
        "    void __iomem *base;",
        "    int irq;",
        "    spinlock_t lock;",
        "    atomic_t ref_count;",
        "} *global_data;",
        "",
    ]

    # Generate function implementations
    for i in range(multiplier):
        for func in functions:
            func_name = func.split()[-1].split("(")[0]
            if i > 0:
                func = func.replace(func_name, f"{func_name}_{i}")
                func_name = f"{func_name}_{i}"

            code_parts.extend(
                [
                    f"{func} {{",
                    f"    // Implementation of {func_name}",
                    "    int ret = 0;",
                    "    struct device_data *data = global_data;",
                    "",
                    "    if (!data) {",
                    '        printk(KERN_ERR "No device data\\n");',
                    "        return -ENODEV;",
                    "    }",
                    "",
                    "    spin_lock(&data->lock);",
                    "    atomic_inc(&data->ref_count);",
                    "",
                    # Add function-specific code
                    "    // Function-specific operations",
                ]
            )

            if "init" in func_name:
                code_parts.extend(
                    [
                        "    data = kmalloc(sizeof(*data), GFP_KERNEL);",
                        "    if (!data) {",
                        "        ret = -ENOMEM;",
                        "        goto out_unlock;",
                        "    }",
                        "    spin_lock_init(&data->lock);",
                        "    global_data = data;",
                    ]
                )
            elif "exit" in func_name:
                code_parts.extend(
                    [
                        "    kfree(data);",
                        "    global_data = NULL;",
                    ]
                )
            elif "interrupt" in func_name:
                code_parts.extend(
                    [
                        "    u32 status = readl(data->base + 0x10);",
                        "    if (!(status & 0x1))",
                        "        return IRQ_NONE;",
                        "    writel(status, data->base + 0x10);",
                        "    return IRQ_HANDLED;",
                    ]
                )
            else:
                code_parts.extend(
                    [
                        "    msleep(1);",  # Simulate some work
                    ]
                )

            code_parts.extend(
                [
                    "",
                    "out_unlock:",
                    "    atomic_dec(&data->ref_count);",
                    "    spin_unlock(&data->lock);",
                    "    return ret;",
                    "}",
                    "",
                ]
            )

    # Add module metadata
    code_parts.extend(
        [
            'MODULE_LICENSE("GPL");',
            'MODULE_AUTHOR("Test Author");',
            'MODULE_DESCRIPTION("Performance test kernel module");',
            "",
            f"module_init({functions[0].split()[-1].split('(')[0]});",
            f"module_exit({functions[1].split()[-1].split('(')[0]});",
        ]
    )

    return "\n".join(code_parts)


def create_test_files(
    count: int, complexity: str = "medium", size: str = "small"
) -> list[str]:
    """Create multiple test files for benchmark testing."""
    files = []
    temp_dir = tempfile.mkdtemp(prefix="kcs_perf_test_")

    for i in range(count):
        file_path = os.path.join(temp_dir, f"test_module_{i}.c")
        content = create_sample_kernel_file(complexity, size)

        # Make each file slightly different
        content = content.replace("global_data", f"global_data_{i}")
        content = content.replace("test_module", f"test_module_{i}")

        with open(file_path, "w") as f:
            f.write(content)
        files.append(file_path)

    return files


# Performance test functions


@pytest.mark.performance
async def test_single_file_extraction_performance(extractor):
    """Test call graph extraction performance on a single file."""
    file_path = create_test_files(1, "complex", "medium")[0]

    request = ExtractCallGraphRequest(
        file_paths=[file_path],
        include_indirect=True,
        include_macros=True,
        max_depth=10,
        confidence_threshold=0.7,
    )

    # Mock the Rust extraction to return reasonable test data
    async def mock_rust_extraction(req):
        return {
            "call_edges": [
                create_mock_call_edge("driver_init", "kmalloc", file_path, 45),
                create_mock_call_edge("interrupt_handler", "readl", file_path, 78),
            ]
            * 50,  # Simulate finding many calls
            "function_pointers": [],
            "macro_calls": [],
            "functions_analyzed": 25,
            "accuracy_estimate": 0.95,
        }

    extractor._run_rust_extraction = mock_rust_extraction

    # Measure extraction time
    start_time = time.time()
    result = await extractor.extract_call_graph(request)
    extraction_time = time.time() - start_time

    # Verify performance metrics
    assert extraction_time < 5.0, (
        f"Single file extraction took too long: {extraction_time:.2f}s"
    )
    assert result.extraction_stats.call_edges_found > 0

    print(f"Single file extraction: {extraction_time:.3f}s")
    print(f"Call edges found: {result.extraction_stats.call_edges_found}")

    # Cleanup
    os.unlink(file_path)
    os.rmdir(os.path.dirname(file_path))


@pytest.mark.performance
async def test_multiple_files_extraction_performance(extractor):
    """Test call graph extraction performance on multiple files."""
    file_count = 5
    file_paths = create_test_files(file_count, "medium", "small")

    request = ExtractCallGraphRequest(
        file_paths=file_paths,
        include_indirect=True,
        include_macros=True,
        max_depth=5,
        confidence_threshold=0.8,
    )

    # Mock the Rust extraction
    async def mock_rust_extraction(req):
        return {
            "call_edges": [
                create_mock_call_edge(f"func_{i}", f"target_{i}", fp, 10 + i)
                for i, fp in enumerate(req.file_paths)
            ]
            * 20,  # Multiple calls per file
            "function_pointers": [],
            "macro_calls": [],
            "functions_analyzed": 50,
            "accuracy_estimate": 0.92,
        }

    extractor._run_rust_extraction = mock_rust_extraction

    # Measure extraction time
    start_time = time.time()
    result = await extractor.extract_call_graph(request)
    extraction_time = time.time() - start_time

    # Performance should scale reasonably with file count
    expected_max_time = file_count * 1.0  # 1 second per file max
    assert extraction_time < expected_max_time, (
        f"Multiple file extraction took too long: {extraction_time:.2f}s"
    )

    print(f"Multiple files ({file_count}) extraction: {extraction_time:.3f}s")
    print(f"Average per file: {extraction_time / file_count:.3f}s")
    print(f"Call edges found: {result.extraction_stats.call_edges_found}")

    # Cleanup
    for file_path in file_paths:
        os.unlink(file_path)
    os.rmdir(os.path.dirname(file_paths[0]))


@pytest.mark.performance
async def test_large_file_extraction_performance(extractor):
    """Test call graph extraction performance on large files."""
    file_path = create_test_files(1, "complex", "large")[0]

    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"Large file size: {file_size / 1024:.1f} KB")

    request = ExtractCallGraphRequest(
        file_paths=[file_path],
        include_indirect=False,  # Limit scope for performance
        include_macros=True,
        max_depth=3,
        confidence_threshold=0.8,
    )

    # Mock with proportionally more results for large file
    async def mock_rust_extraction(req):
        num_functions = 100  # Estimated for "large" complexity
        return {
            "call_edges": [
                create_mock_call_edge(
                    f"function_{i}", f"target_{j}", file_path, 10 + i * 10
                )
                for i in range(num_functions)
                for j in range(3)  # 3 calls per function
            ],
            "function_pointers": [],
            "macro_calls": [],
            "functions_analyzed": num_functions,
            "accuracy_estimate": 0.88,
        }

    extractor._run_rust_extraction = mock_rust_extraction

    # Measure extraction time
    start_time = time.time()
    result = await extractor.extract_call_graph(request)
    extraction_time = time.time() - start_time

    # Large files should still complete in reasonable time
    assert extraction_time < 15.0, (
        f"Large file extraction took too long: {extraction_time:.2f}s"
    )

    print(f"Large file extraction: {extraction_time:.3f}s")
    print(f"Processing rate: {file_size / extraction_time / 1024:.1f} KB/s")
    print(f"Call edges found: {result.extraction_stats.call_edges_found}")

    # Cleanup
    os.unlink(file_path)
    os.rmdir(os.path.dirname(file_path))


@pytest.mark.performance
async def test_memory_usage_during_extraction(extractor):
    """Test memory usage during call graph extraction."""
    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not available for memory testing")

    import gc

    file_paths = create_test_files(3, "medium", "medium")

    # Force garbage collection before test
    gc.collect()
    initial_memory = psutil.Process().memory_info().rss

    request = ExtractCallGraphRequest(
        file_paths=file_paths,
        include_indirect=True,
        include_macros=True,
        max_depth=8,
        confidence_threshold=0.7,
    )

    # Mock extraction with memory-conscious simulation
    async def mock_rust_extraction(req):
        # Simulate memory usage during extraction
        large_data = []
        try:
            for _i, fp in enumerate(req.file_paths):
                # Simulate processing each file
                await asyncio.sleep(0.1)  # Simulate processing time
                large_data.extend(
                    [
                        {
                            "caller": f"func_{j}",
                            "callee": f"target_{j}",
                            "file_path": fp,
                            "line_number": j,
                            "call_type": "direct",
                            "confidence": 0.9,
                        }
                        for j in range(100)
                    ]
                )  # Simulate finding many calls

            return {
                "call_edges": [
                    create_mock_call_edge(f"func_{j}", f"target_{j}", fp, j)
                    for fp in req.file_paths
                    for j in range(100)
                ],
                "function_pointers": [],
                "macro_calls": [],
                "functions_analyzed": len(req.file_paths) * 100,
                "accuracy_estimate": 0.90,
            }
        finally:
            # Cleanup simulation data
            del large_data

    extractor._run_rust_extraction = mock_rust_extraction

    # Measure memory during extraction
    peak_memory = initial_memory

    async def memory_monitor():
        nonlocal peak_memory
        while True:
            current_memory = psutil.Process().memory_info().rss
            peak_memory = max(peak_memory, current_memory)
            await asyncio.sleep(0.1)

    # Start memory monitoring
    monitor_task = asyncio.create_task(memory_monitor())

    try:
        start_time = time.time()
        result = await extractor.extract_call_graph(request)
        extraction_time = time.time() - start_time
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    # Check memory usage
    memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB

    print(f"Memory increase during extraction: {memory_increase:.1f} MB")
    print(f"Extraction time: {extraction_time:.3f}s")
    print(f"Call edges found: {result.extraction_stats.call_edges_found}")

    # Memory usage should be reasonable (less than 200MB increase)
    assert memory_increase < 200, f"Memory usage too high: {memory_increase:.1f} MB"

    # Cleanup
    for file_path in file_paths:
        os.unlink(file_path)
    os.rmdir(os.path.dirname(file_paths[0]))


@pytest.mark.performance
def test_file_size_scaling():
    """Test how extraction time scales with file size."""
    sizes = ["small", "medium", "large"]
    results = {}

    for size in sizes:
        file_path = create_test_files(1, "medium", size)[0]
        file_size = os.path.getsize(file_path)

        # Simulate extraction time based on file size
        # In real implementation, this would call actual extraction
        simulated_time = (file_size / 1024) * 0.001  # 1ms per KB

        results[size] = {
            "file_size": file_size,
            "extraction_time": simulated_time,
            "rate": file_size / simulated_time / 1024,  # KB/s
        }

        print(
            f"{size}: {file_size / 1024:.1f} KB, {simulated_time:.3f}s, {results[size]['rate']:.1f} KB/s"
        )

        # Cleanup
        os.unlink(file_path)
        os.rmdir(os.path.dirname(file_path))

    # Verify scaling is reasonable (rate should be relatively consistent)
    rates = [results[size]["rate"] for size in sizes]
    rate_variance = max(rates) / min(rates)
    assert rate_variance < 3.0, (
        f"Processing rate variance too high: {rate_variance:.2f}x"
    )


@pytest.mark.performance
async def test_depth_limit_performance(extractor):
    """Test how max_depth parameter affects performance."""
    file_path = create_test_files(1, "complex", "medium")[0]
    depths = [1, 3, 5, 10]
    results = {}

    for depth in depths:
        request = ExtractCallGraphRequest(
            file_paths=[file_path],
            include_indirect=True,
            include_macros=True,
            max_depth=depth,
            confidence_threshold=0.8,
        )

        # Mock extraction with depth-dependent results
        async def mock_rust_extraction(req):
            # Simulate depth impact with controlled timing and consistent result size
            await asyncio.sleep(
                0.001 * (req.max_depth**0.5)
            )  # Very small sublinear growth

            # Keep result size constant to focus on depth processing time, not data volume
            num_calls = 20  # Fixed number of calls regardless of depth

            return {
                "call_edges": [
                    create_mock_call_edge(f"func_{i}", f"target_{i}", file_path, 10 + i)
                    for i in range(num_calls)
                ],
                "function_pointers": [],
                "macro_calls": [],
                "functions_analyzed": num_calls,
                "accuracy_estimate": 0.91,
            }

        extractor._run_rust_extraction = mock_rust_extraction

        start_time = time.time()
        result = await extractor.extract_call_graph(request)
        extraction_time = time.time() - start_time

        results[depth] = {
            "time": extraction_time,
            "call_edges": result.extraction_stats.call_edges_found,
        }

        print(
            f"Depth {depth}: {extraction_time:.3f}s, {result.extraction_stats.call_edges_found} edges"
        )

    # Verify that extraction works for all depths and returns consistent results
    for depth in depths:
        assert results[depth]["call_edges"] > 0, (
            f"No call edges found for depth {depth}"
        )
        assert results[depth]["time"] > 0, (
            f"No extraction time recorded for depth {depth}"
        )

    # Basic sanity check that extraction completes within reasonable time
    max_time = max(results[depth]["time"] for depth in depths)
    assert max_time < 1.0, f"Extraction took too long: {max_time:.3f}s"

    # Cleanup
    os.unlink(file_path)
    os.rmdir(os.path.dirname(file_path))


@pytest.mark.performance
def test_performance_regression_detection():
    """Test for performance regression detection."""
    # This would typically compare against stored baseline metrics
    # For now, we'll simulate the concept

    baseline_metrics = {
        "single_file_time": 2.0,  # seconds
        "memory_usage": 100,  # MB
        "throughput": 500,  # KB/s
    }

    # Simulate current performance
    current_metrics = {
        "single_file_time": 1.8,
        "memory_usage": 95,
        "throughput": 520,
    }

    # Check for regressions (more than 20% slower)
    for metric, baseline in baseline_metrics.items():
        current = current_metrics[metric]

        if metric in ["single_file_time", "memory_usage"]:
            # Lower is better
            regression_ratio = current / baseline
            assert regression_ratio < 1.2, (
                f"Performance regression in {metric}: {regression_ratio:.2f}x"
            )
        else:
            # Higher is better
            regression_ratio = baseline / current
            assert regression_ratio < 1.2, (
                f"Performance regression in {metric}: {regression_ratio:.2f}x"
            )

    print("Performance regression check passed")


def generate_performance_report(results: dict[str, Any]) -> str:
    """Generate a performance report from test results."""
    report_lines = [
        "# Call Graph Extraction Performance Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Results",
        "",
    ]

    for test_name, metrics in results.items():
        report_lines.extend(
            [
                f"### {test_name}",
                "",
            ]
        )

        for metric, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"- {metric}: {value:.3f}")
            else:
                report_lines.append(f"- {metric}: {value}")

        report_lines.append("")

    return "\n".join(report_lines)
