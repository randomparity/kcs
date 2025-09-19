"""
Performance tests for kernel pattern detection.

Tests the speed and scalability of pattern detection
and entry point extraction on real kernel code.
"""

import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pytest

# Import the Python bridge module if available
try:
    import kcs_python_bridge as kpb

    HAS_PYTHON_BRIDGE = True
except ImportError:
    HAS_PYTHON_BRIDGE = False


def create_sample_kernel_file(complexity: str = "medium") -> str:
    """Create a sample kernel source file for testing."""
    if complexity == "simple":
        return """
#include <linux/kernel.h>
#include <linux/module.h>

static int simple_param = 42;
module_param(simple_param, int, 0644);
MODULE_PARM_DESC(simple_param, "A simple parameter");

int simple_function(int x) {
    return x * 2;
}
EXPORT_SYMBOL(simple_function);

MODULE_LICENSE("GPL");
"""
    elif complexity == "medium":
        return """
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/debugfs.h>

static int debug_mode = 0;
module_param(debug_mode, int, 0644);
MODULE_PARM_DESC(debug_mode, "Enable debug mode");

static int buffer_size = 4096;
module_param(buffer_size, int, 0444);
MODULE_PARM_DESC(buffer_size, "Buffer size in bytes");

static long device_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    switch (cmd) {
        case 0x1234:
            return 0;
        default:
            return -EINVAL;
    }
}

static const struct file_operations device_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = device_ioctl,
};

int process_data(void *data, size_t len) {
    // Process the data
    return 0;
}
EXPORT_SYMBOL_GPL(process_data);

void cleanup_resources(void) {
    // Cleanup
}
EXPORT_SYMBOL(cleanup_resources);

static int __init test_init(void) {
    proc_create("test_entry", 0644, NULL, &device_fops);
    debugfs_create_file("test_debug", 0644, NULL, NULL, &device_fops);
    return 0;
}

static void __exit test_exit(void) {
    cleanup_resources();
}

module_init(test_init);
module_exit(test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Test Author");
MODULE_DESCRIPTION("Test module for pattern detection");
"""
    else:  # complex
        return """
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/debugfs.h>
#include <linux/interrupt.h>
#include <linux/netlink.h>

/* Module parameters */
static int param1 = 100;
static int param2 = 200;
static char *param_str = "default";
static int param_array[4] = {1, 2, 3, 4};
static int param_array_size = 4;

module_param(param1, int, 0644);
module_param(param2, int, 0444);
module_param(param_str, charp, 0644);
module_param_array(param_array, int, &param_array_size, 0644);

MODULE_PARM_DESC(param1, "First parameter");
MODULE_PARM_DESC(param2, "Second parameter");
MODULE_PARM_DESC(param_str, "String parameter");
MODULE_PARM_DESC(param_array, "Array parameter");

/* Exported symbols */
int exported_func1(int x) { return x; }
EXPORT_SYMBOL(exported_func1);

int exported_func2(int x) { return x * 2; }
EXPORT_SYMBOL_GPL(exported_func2);

int exported_func3(int x) { return x * 3; }
EXPORT_SYMBOL_NS(exported_func3, TEST_NS);

/* System calls */
SYSCALL_DEFINE2(test_syscall, int, arg1, int, arg2) {
    return arg1 + arg2;
}

/* IOCTL handlers */
static long test_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    return 0;
}

static long test_compat_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    return 0;
}

/* File operations */
static const struct file_operations test_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = test_ioctl,
    .compat_ioctl = test_compat_ioctl,
};

/* Procfs entries */
static int test_proc_show(struct seq_file *m, void *v) {
    seq_printf(m, "Test proc entry\\n");
    return 0;
}

static int test_proc_open(struct inode *inode, struct file *file) {
    return single_open(file, test_proc_show, NULL);
}

static const struct proc_ops test_proc_ops = {
    .proc_open = test_proc_open,
    .proc_read = seq_read,
};

/* Interrupt handler */
static irqreturn_t test_irq_handler(int irq, void *dev_id) {
    return IRQ_HANDLED;
}

/* Boot parameters */
static int __init test_setup(char *str) {
    return 1;
}
__setup("test_param=", test_setup);

static int __init test_early(char *str) {
    return 0;
}
early_param("test_early", test_early);

/* Module init/exit */
static int __init test_module_init(void) {
    struct proc_dir_entry *entry;

    /* Create proc entries */
    entry = proc_create("test_proc", 0644, NULL, &test_proc_ops);
    proc_create_data("test_proc_data", 0644, NULL, &test_proc_ops, NULL);

    /* Create debugfs entries */
    debugfs_create_file("test_debug", 0644, NULL, NULL, &test_fops);
    debugfs_create_u32("test_debug_u32", 0644, NULL, &param1);

    /* Request IRQ */
    request_irq(42, test_irq_handler, IRQF_SHARED, "test_irq", NULL);

    /* Netlink */
    netlink_kernel_create(&init_net, NETLINK_TEST, NULL);

    return 0;
}

static void __exit test_module_exit(void) {
    free_irq(42, NULL);
}

module_init(test_module_init);
module_exit(test_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Performance Test");
MODULE_DESCRIPTION("Complex module for performance testing");
"""


def measure_pattern_detection_performance(
    file_content: str, iterations: int = 100
) -> dict[str, float]:
    """Measure the performance of pattern detection."""
    if not HAS_PYTHON_BRIDGE:
        pytest.skip("Python bridge module not available")

    times = {"pattern_detection": [], "entrypoint_extraction": [], "combined": []}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
        f.write(file_content)
        temp_file = f.name

    try:
        for _ in range(iterations):
            # Measure pattern detection
            start = time.perf_counter()
            _ = kpb.detect_patterns(temp_file, file_content)
            pattern_time = time.perf_counter() - start
            times["pattern_detection"].append(pattern_time * 1000)  # Convert to ms

            # Measure entry point extraction (all types)
            start = time.perf_counter()
            _ = kpb.extract_entrypoints(
                os.path.dirname(temp_file),
                include_syscalls=True,
                include_ioctls=True,
                include_procfs=True,
                include_debugfs=True,
                include_netlink=True,
                include_interrupts=True,
            )
            entry_time = time.perf_counter() - start
            times["entrypoint_extraction"].append(entry_time * 1000)

            # Measure combined operation
            start = time.perf_counter()
            _ = kpb.detect_patterns(temp_file, file_content)
            _ = kpb.extract_entrypoints(
                os.path.dirname(temp_file), include_syscalls=True, include_ioctls=True
            )
            combined_time = time.perf_counter() - start
            times["combined"].append(combined_time * 1000)

    finally:
        os.unlink(temp_file)

    # Calculate statistics
    results = {}
    for operation, measurements in times.items():
        results[operation] = {
            "min": min(measurements),
            "max": max(measurements),
            "mean": sum(measurements) / len(measurements),
            "median": sorted(measurements)[len(measurements) // 2],
        }

    return results


def test_pattern_detection_speed_simple():
    """Test pattern detection speed on simple files."""
    content = create_sample_kernel_file("simple")
    results = measure_pattern_detection_performance(content, iterations=100)

    # Assert performance targets
    assert results["pattern_detection"]["mean"] < 10, (
        f"Simple pattern detection too slow: {results['pattern_detection']['mean']:.2f}ms"
    )
    assert results["pattern_detection"]["max"] < 20, (
        f"Simple pattern detection worst case too slow: {results['pattern_detection']['max']:.2f}ms"
    )


def test_pattern_detection_speed_medium():
    """Test pattern detection speed on medium complexity files."""
    content = create_sample_kernel_file("medium")
    results = measure_pattern_detection_performance(content, iterations=100)

    # Assert performance targets (100ms per file target)
    assert results["pattern_detection"]["mean"] < 50, (
        f"Medium pattern detection too slow: {results['pattern_detection']['mean']:.2f}ms"
    )
    assert results["pattern_detection"]["max"] < 100, (
        f"Medium pattern detection worst case exceeds target: {results['pattern_detection']['max']:.2f}ms"
    )


def test_pattern_detection_speed_complex():
    """Test pattern detection speed on complex files."""
    content = create_sample_kernel_file("complex")
    results = measure_pattern_detection_performance(content, iterations=50)

    # Assert performance targets (100ms per file target)
    assert results["pattern_detection"]["mean"] < 100, (
        f"Complex pattern detection exceeds target: {results['pattern_detection']['mean']:.2f}ms"
    )
    assert results["pattern_detection"]["max"] < 200, (
        f"Complex pattern detection worst case too slow: {results['pattern_detection']['max']:.2f}ms"
    )


def test_parallel_processing_scalability():
    """Test that parallel processing scales well."""
    if not HAS_PYTHON_BRIDGE:
        pytest.skip("Python bridge module not available")

    # Create multiple test files
    num_files = 20
    test_files = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_files):
            complexity = ["simple", "medium", "complex"][i % 3]
            content = create_sample_kernel_file(complexity)
            file_path = os.path.join(tmpdir, f"test_{i}.c")
            with open(file_path, "w") as f:
                f.write(content)
            test_files.append(file_path)

        # Measure sequential processing
        start = time.perf_counter()
        sequential_results = []
        for file_path in test_files:
            with open(file_path) as f:
                content = f.read()
            patterns = kpb.detect_patterns(file_path, content)
            sequential_results.append(len(patterns))
        sequential_time = time.perf_counter() - start

        # Measure parallel processing with different worker counts
        parallel_times = {}
        for num_workers in [2, 4, 8]:
            if num_workers > os.cpu_count():
                continue

            start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for file_path in test_files:
                    with open(file_path) as f:
                        content = f.read()
                    future = executor.submit(kpb.detect_patterns, file_path, content)
                    futures.append(future)

                parallel_results = []
                for future in as_completed(futures):
                    patterns = future.result()
                    parallel_results.append(len(patterns))

            parallel_times[num_workers] = time.perf_counter() - start

        # Assert that parallel processing is faster
        for num_workers, parallel_time in parallel_times.items():
            speedup = sequential_time / parallel_time
            print(f"Parallel speedup with {num_workers} workers: {speedup:.2f}x")

            # Expect at least 1.5x speedup with 2+ workers
            if num_workers >= 2:
                assert speedup > 1.3, (
                    f"Insufficient speedup with {num_workers} workers: {speedup:.2f}x"
                )


def test_memory_efficiency():
    """Test that pattern detection doesn't use excessive memory."""
    # This is a basic test - in production you'd use memory profiling tools
    import tracemalloc

    content = create_sample_kernel_file("complex")

    # Start tracing
    tracemalloc.start()
    initial = tracemalloc.get_traced_memory()[0]

    # Run pattern detection multiple times
    for _ in range(100):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=True) as f:
            f.write(content)
            if HAS_PYTHON_BRIDGE:
                _ = kpb.detect_patterns(f.name, content)

    # Check memory usage
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_used_mb = (peak - initial) / (1024 * 1024)
    print(f"Memory used: {memory_used_mb:.2f} MB")

    # Assert reasonable memory usage (less than 100MB for 100 iterations)
    assert memory_used_mb < 100, f"Excessive memory usage: {memory_used_mb:.2f} MB"


def test_performance_with_kernel_fixtures():
    """Test performance with real kernel code fixtures if available."""
    fixture_dir = Path("tests/fixtures/kernel")
    if not fixture_dir.exists():
        pytest.skip("Kernel fixtures not available")

    c_files = list(fixture_dir.glob("**/*.c"))[:10]  # Test first 10 files
    if not c_files:
        pytest.skip("No C files in kernel fixtures")

    times = []
    for c_file in c_files:
        with open(c_file) as f:
            content = f.read()

        start = time.perf_counter()
        if HAS_PYTHON_BRIDGE:
            _ = kpb.detect_patterns(str(c_file), content)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

        print(f"{c_file.name}: {elapsed:.2f}ms ({len(content)} bytes)")

    # Check average performance
    avg_time = sum(times) / len(times)
    assert avg_time < 100, f"Average processing time exceeds target: {avg_time:.2f}ms"

    # Check worst case
    max_time = max(times)
    assert max_time < 200, f"Worst case processing time too high: {max_time:.2f}ms"


if __name__ == "__main__":
    # Run basic performance test
    print("Running performance validation...")

    for complexity in ["simple", "medium", "complex"]:
        print(f"\n{complexity.upper()} complexity:")
        content = create_sample_kernel_file(complexity)
        results = measure_pattern_detection_performance(content, iterations=50)

        for operation, stats in results.items():
            print(f"  {operation}:")
            print(f"    Mean:   {stats['mean']:.2f}ms")
            print(f"    Median: {stats['median']:.2f}ms")
            print(f"    Min:    {stats['min']:.2f}ms")
            print(f"    Max:    {stats['max']:.2f}ms")

    print("\nPerformance validation complete!")
