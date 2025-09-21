#!/usr/bin/env python3
"""
Performance benchmarking script for semantic search engine.

Tests the semantic search system against constitutional requirements:
- p95 query response time ≤ 600ms
- System can handle concurrent users
- Memory usage remains stable

This script performs real end-to-end performance testing against a running
database with actual indexed content.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, quantiles
from typing import Any

import psutil

# Add semantic search module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from semantic_search.database.connection import DatabaseConfig, init_database_connection
from semantic_search.mcp.search_tool import SemanticSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Performance requirements (from constitutional requirements)
MAX_P95_MS = 600  # Constitutional requirement: p95 < 600ms
MAX_P50_MS = 200  # Target: p50 < 200ms
MIN_SAMPLE_SIZE = 100  # Minimum queries for statistical significance


class SemanticSearchBenchmark:
    """Performance benchmark suite for semantic search system."""

    def __init__(self, database_url: str | None = None):
        """Initialize benchmark with database connection."""
        self.database_url = database_url
        self.search_tool: SemanticSearchTool | None = None
        self.test_queries = self._get_test_queries()
        self.process = psutil.Process()

    def _get_test_queries(self) -> list[dict[str, Any]]:
        """Get realistic test queries for performance testing."""
        return [
            {
                "query": "memory allocation functions that can fail",
                "max_results": 10,
                "min_confidence": 0.5,
            },
            {
                "query": "buffer overflow vulnerability patterns",
                "max_results": 5,
                "min_confidence": 0.7,
            },
            {
                "query": "lock acquisition in scheduler",
                "max_results": 15,
                "min_confidence": 0.4,
                "content_types": ["SOURCE_CODE"],
            },
            {
                "query": "network packet processing",
                "max_results": 20,
                "min_confidence": 0.6,
                "file_patterns": ["drivers/net/*"],
            },
            {
                "query": "error handling patterns",
                "max_results": 8,
                "min_confidence": 0.5,
                "config_context": ["CONFIG_NET"],
            },
            {
                "query": "interrupt handler registration",
                "max_results": 12,
                "min_confidence": 0.6,
            },
            {
                "query": "file system operations",
                "max_results": 18,
                "min_confidence": 0.4,
            },
            {
                "query": "device driver initialization",
                "max_results": 25,
                "min_confidence": 0.5,
            },
            {
                "query": "kernel module loading",
                "max_results": 7,
                "min_confidence": 0.8,
            },
            {
                "query": "system call implementation",
                "max_results": 15,
                "min_confidence": 0.6,
            },
        ]

    async def initialize(self) -> None:
        """Initialize database connection and search tool."""
        try:
            # Initialize database connection
            if self.database_url:
                db_config = DatabaseConfig.from_url(self.database_url)
            else:
                db_config = DatabaseConfig.from_env()

            await init_database_connection(db_config)
            logger.info(f"Connected to database: {db_config.host}:{db_config.port}")

            # Initialize search tool
            self.search_tool = SemanticSearchTool()
            logger.info("Semantic search tool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize benchmark: {e}")
            raise

    async def single_query_benchmark(self, query_data: dict[str, Any]) -> tuple[float, bool, dict[str, Any]]:
        """Execute a single query and return response time, success status, and stats."""
        if not self.search_tool:
            raise RuntimeError("Search tool not initialized")

        start_time = time.perf_counter()
        try:
            result = await self.search_tool.execute(query_data)
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000  # Convert to ms

            # Extract search stats
            search_stats = result.get("search_stats", {})

            return response_time, True, search_stats

        except Exception as e:
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000
            logger.warning(f"Query failed: {e}")
            return response_time, False, {}

    async def basic_performance_test(self) -> dict[str, Any]:
        """Test basic performance requirements: p95 ≤ 600ms, p50 ≤ 200ms."""
        logger.info("Running basic performance test...")

        response_times = []
        search_stats_list = []
        success_count = 0

        # Run multiple iterations of each query type
        iterations = max(MIN_SAMPLE_SIZE // len(self.test_queries), 10)
        total_queries = iterations * len(self.test_queries)

        logger.info(f"Executing {total_queries} queries ({iterations} iterations x {len(self.test_queries)} query types)")

        for iteration in range(iterations):
            for i, query_data in enumerate(self.test_queries):
                response_time, success, search_stats = await self.single_query_benchmark(query_data)
                response_times.append(response_time)

                if success:
                    success_count += 1
                    search_stats_list.append(search_stats)

                # Progress indicator
                completed = iteration * len(self.test_queries) + i + 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total_queries} queries completed")

        # Calculate statistics
        if len(response_times) < MIN_SAMPLE_SIZE:
            raise RuntimeError(f"Insufficient samples: {len(response_times)} < {MIN_SAMPLE_SIZE}")

        mean_time = mean(response_times)

        # Calculate percentiles
        if len(response_times) >= 20:
            p50, p95 = quantiles(response_times, n=20)[9], quantiles(response_times, n=20)[18]
        else:
            sorted_times = sorted(response_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]

        success_rate = success_count / len(response_times)

        # Calculate average search stats
        avg_search_stats = {}
        if search_stats_list:
            avg_search_stats = {
                "avg_search_time_ms": mean([s.get("search_time_ms", 0) for s in search_stats_list]),
                "avg_embedding_time_ms": mean([s.get("embedding_time_ms", 0) for s in search_stats_list]),
                "avg_total_matches": mean([s.get("total_matches", 0) for s in search_stats_list]),
                "avg_filtered_matches": mean([s.get("filtered_matches", 0) for s in search_stats_list]),
            }

        results = {
            "total_queries": len(response_times),
            "success_rate": success_rate,
            "mean_response_time_ms": mean_time,
            "p50_response_time_ms": p50,
            "p95_response_time_ms": p95,
            "max_response_time_ms": max(response_times),
            "min_response_time_ms": min(response_times),
            "p95_requirement_met": p95 < MAX_P95_MS,
            "p50_target_met": p50 < MAX_P50_MS,
            "search_stats": avg_search_stats,
        }

        logger.info("Basic performance test completed")
        return results

    async def concurrent_users_test(self, num_users: int = 25, session_duration: int = 30) -> dict[str, Any]:
        """Test performance under concurrent user load."""
        logger.info(f"Running concurrent users test with {num_users} users for {session_duration}s...")

        async def simulate_user_session(user_id: int) -> list[tuple[float, bool]]:
            """Simulate a user session with multiple queries."""
            results = []
            end_time = time.time() + session_duration
            query_index = 0

            while time.time() < end_time:
                query = self.test_queries[query_index % len(self.test_queries)]
                response_time, success, _ = await self.single_query_benchmark(query)
                results.append((response_time, success))

                # Simulate user think time
                await asyncio.sleep(0.5 + (user_id % 3) * 0.2)  # 0.5-1.1s think time
                query_index += 1

            return results

        # Create concurrent user sessions
        tasks = []
        for user_id in range(num_users):
            task = asyncio.create_task(simulate_user_session(user_id))
            tasks.append(task)

        # Wait for all users to complete
        start_time = time.time()
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Collect all response times
        response_times = []
        success_count = 0
        total_queries = 0

        for user_result in user_results:
            if isinstance(user_result, Exception):
                logger.warning(f"User session failed: {user_result}")
                continue

            for response_time, success in user_result:
                response_times.append(response_time)
                total_queries += 1
                if success:
                    success_count += 1

        if not response_times:
            raise RuntimeError("No successful queries in concurrent test")

        # Calculate metrics
        mean_time = mean(response_times)
        p50 = quantiles(response_times, n=20)[9] if len(response_times) >= 20 else sorted(response_times)[len(response_times) // 2]
        p95 = quantiles(response_times, n=20)[18] if len(response_times) >= 20 else sorted(response_times)[int(len(response_times) * 0.95)]
        success_rate = success_count / total_queries if total_queries > 0 else 0

        results = {
            "concurrent_users": num_users,
            "session_duration": session_duration,
            "total_time": total_time,
            "total_queries": total_queries,
            "success_rate": success_rate,
            "mean_response_time_ms": mean_time,
            "p50_response_time_ms": p50,
            "p95_response_time_ms": p95,
            "p95_requirement_met": p95 < MAX_P95_MS,
            "throughput_qps": total_queries / total_time if total_time > 0 else 0,
        }

        logger.info("Concurrent users test completed")
        return results

    async def memory_usage_test(self) -> dict[str, Any]:
        """Test memory usage stability during sustained query load."""
        logger.info("Running memory usage test...")

        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]

        # Run sustained query load
        for i in range(50):
            for query_data in self.test_queries:
                await self.single_query_benchmark(query_data)

                # Sample memory every 10 queries
                query_count = i * len(self.test_queries) + self.test_queries.index(query_data) + 1
                if query_count % 10 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)

        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_samples)

        results = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "max_memory_mb": max_memory,
            "memory_growth_mb": memory_growth,
            "memory_stable": memory_growth < 50,  # < 50MB growth is acceptable
            "memory_samples": len(memory_samples),
        }

        logger.info("Memory usage test completed")
        return results

    async def run_full_benchmark(self) -> dict[str, Any]:
        """Run the complete performance benchmark suite."""
        logger.info("Starting full performance benchmark...")

        await self.initialize()

        benchmark_results = {
            "benchmark_timestamp": time.time(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": sys.version,
            },
            "performance_requirements": {
                "max_p95_ms": MAX_P95_MS,
                "max_p50_ms": MAX_P50_MS,
                "min_sample_size": MIN_SAMPLE_SIZE,
            }
        }

        try:
            # Basic performance test
            basic_results = await self.basic_performance_test()
            benchmark_results["basic_performance"] = basic_results

            # Concurrent users test
            concurrent_results = await self.concurrent_users_test()
            benchmark_results["concurrent_performance"] = concurrent_results

            # Memory usage test
            memory_results = await self.memory_usage_test()
            benchmark_results["memory_usage"] = memory_results

            # Overall assessment
            overall_pass = (
                basic_results["p95_requirement_met"] and
                concurrent_results["p95_requirement_met"] and
                memory_results["memory_stable"] and
                basic_results["success_rate"] > 0.95 and
                concurrent_results["success_rate"] > 0.90
            )

            benchmark_results["overall_assessment"] = {
                "constitutional_compliance": overall_pass,
                "p95_requirement_met": basic_results["p95_requirement_met"] and concurrent_results["p95_requirement_met"],
                "memory_stable": memory_results["memory_stable"],
                "high_success_rate": basic_results["success_rate"] > 0.95,
                "concurrent_capable": concurrent_results["success_rate"] > 0.90,
            }

            logger.info(f"Benchmark completed. Overall pass: {overall_pass}")
            return benchmark_results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            benchmark_results["error"] = str(e)
            return benchmark_results


def print_benchmark_results(results: dict[str, Any]) -> None:
    """Print benchmark results in a formatted way."""
    print("\n" + "="*80)
    print("SEMANTIC SEARCH PERFORMANCE BENCHMARK RESULTS")
    print("="*80)

    if "error" in results:
        print(f"❌ Benchmark failed: {results['error']}")
        return

    # Basic performance
    basic = results.get("basic_performance", {})
    print("\n📊 BASIC PERFORMANCE:")
    print(f"  Queries executed: {basic.get('total_queries', 0)}")
    print(f"  Success rate: {basic.get('success_rate', 0):.1%}")
    print(f"  Mean response time: {basic.get('mean_response_time_ms', 0):.1f}ms")
    print(f"  P50 response time: {basic.get('p50_response_time_ms', 0):.1f}ms (target: <{MAX_P50_MS}ms)")
    print(f"  P95 response time: {basic.get('p95_response_time_ms', 0):.1f}ms (requirement: <{MAX_P95_MS}ms)")

    p95_status = "✅" if basic.get("p95_requirement_met", False) else "❌"
    print(f"  P95 requirement met: {p95_status}")

    # Concurrent performance
    concurrent = results.get("concurrent_performance", {})
    print("\n👥 CONCURRENT PERFORMANCE:")
    print(f"  Concurrent users: {concurrent.get('concurrent_users', 0)}")
    print(f"  Total queries: {concurrent.get('total_queries', 0)}")
    print(f"  Success rate: {concurrent.get('success_rate', 0):.1%}")
    print(f"  P95 response time: {concurrent.get('p95_response_time_ms', 0):.1f}ms")
    print(f"  Throughput: {concurrent.get('throughput_qps', 0):.1f} queries/sec")

    concurrent_p95_status = "✅" if concurrent.get("p95_requirement_met", False) else "❌"
    print(f"  P95 requirement met: {concurrent_p95_status}")

    # Memory usage
    memory = results.get("memory_usage", {})
    print("\n💾 MEMORY USAGE:")
    print(f"  Initial memory: {memory.get('initial_memory_mb', 0):.1f}MB")
    print(f"  Final memory: {memory.get('final_memory_mb', 0):.1f}MB")
    print(f"  Memory growth: {memory.get('memory_growth_mb', 0):.1f}MB")

    memory_status = "✅" if memory.get("memory_stable", False) else "❌"
    print(f"  Memory stable: {memory_status}")

    # Overall assessment
    assessment = results.get("overall_assessment", {})
    print("\n🏆 OVERALL ASSESSMENT:")

    overall_status = "✅ PASS" if assessment.get("constitutional_compliance", False) else "❌ FAIL"
    print(f"  Constitutional compliance: {overall_status}")

    if assessment.get("constitutional_compliance", False):
        print("  🎉 Semantic search meets all performance requirements!")
    else:
        print("  ⚠️  Performance requirements not met. See details above.")

    print("\n" + "="*80)


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Semantic Search Performance Benchmark")
    parser.add_argument("--database-url", help="Database connection URL")
    parser.add_argument("--output", help="JSON output file for results")
    parser.add_argument("--basic-only", action="store_true", help="Run only basic performance test")
    parser.add_argument("--concurrent-only", action="store_true", help="Run only concurrent test")
    parser.add_argument("--memory-only", action="store_true", help="Run only memory test")
    parser.add_argument("--dry-run", action="store_true", help="Test script without database connection")

    args = parser.parse_args()

    async def run_benchmark():
        benchmark = SemanticSearchBenchmark(args.database_url)

        if args.basic_only:
            await benchmark.initialize()
            results = {"basic_performance": await benchmark.basic_performance_test()}
        elif args.concurrent_only:
            await benchmark.initialize()
            results = {"concurrent_performance": await benchmark.concurrent_users_test()}
        elif args.memory_only:
            await benchmark.initialize()
            results = {"memory_usage": await benchmark.memory_usage_test()}
        else:
            results = await benchmark.run_full_benchmark()

        return results

    try:
        results = asyncio.run(run_benchmark())

        # Print results to console
        print_benchmark_results(results)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        # Exit with appropriate code
        if results.get("overall_assessment", {}).get("constitutional_compliance", False):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
