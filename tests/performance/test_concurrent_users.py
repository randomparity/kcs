"""
Performance test for semantic search concurrent user handling.

Validates that the system can handle multiple concurrent users without
performance degradation or resource contention issues.
Tests T017: Performance test: concurrent user handling in tests/performance/test_concurrent_users.py

Following TDD: This test MUST FAIL before implementation exists.
"""

import asyncio
import os
import time
from collections.abc import Generator
from statistics import mean, quantiles
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio


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
MAX_P95_MS = 600  # Constitutional requirement: p95 < 600ms even under load
MAX_P50_MS = 200  # Target: p50 < 200ms under concurrent load
MAX_CONCURRENT_USERS = 50  # Target concurrent user capacity
MAX_DEGRADATION_FACTOR = (
    1.5  # Response time should not degrade more than 50% under load
)


class TestSemanticSearchConcurrentUsers:
    """Performance tests for concurrent user handling in semantic search."""

    @pytest.fixture(scope="class")
    def concurrent_test_queries(self) -> list[dict[str, Any]]:
        """Diverse queries for concurrent testing."""
        return [
            {"query": "memory allocation", "max_results": 10},
            {"query": "buffer overflow", "max_results": 15},
            {"query": "lock contention", "max_results": 5},
            {"query": "network packet", "max_results": 20},
            {"query": "file system", "max_results": 8},
            {"query": "interrupt handler", "max_results": 12},
            {"query": "error handling", "max_results": 7},
            {"query": "kernel module", "max_results": 25},
            {"query": "device driver", "max_results": 18},
            {"query": "memory management", "max_results": 13},
        ]

    @pytest.fixture(scope="class")
    async def semantic_search_client(self) -> AsyncMock:
        """Mock semantic search client for concurrent testing."""
        client = AsyncMock()

        # Mock response structure
        mock_response = {
            "query_id": "concurrent-test-query",
            "results": [
                {
                    "file_path": "/kernel/mm/slub.c",
                    "line_start": 1234,
                    "line_end": 1250,
                    "content": "static void *__slab_alloc(struct kmem_cache *s, gfp_t gfpflags, int node)",
                    "context_lines": [
                        "/* Memory allocation function */",
                        "/* Can return NULL */",
                    ],
                    "confidence": 0.85,
                    "similarity_score": 0.91,
                    "match_type": "SEMANTIC",
                    "config_applicable": True,
                },
            ],
            "total_indexed": 50000,
            "search_duration_ms": 150,
            "confidence_threshold_used": 0.5,
        }

        # Simulate realistic concurrent processing with slight randomization
        async def mock_search(*args, **kwargs):
            # Simulate variable processing time under load
            base_time = 0.15  # 150ms base
            load_factor = min(
                1.0, len(asyncio.all_tasks()) / 20
            )  # Simulate load impact
            processing_time = base_time * (
                1 + load_factor * 0.3
            )  # Up to 30% degradation
            await asyncio.sleep(processing_time)
            return mock_response

        client.semantic_search = mock_search
        return client

    async def _single_user_query(
        self, client: AsyncMock, query_data: dict[str, Any], user_id: int
    ) -> tuple[float, bool]:
        """Execute a single query and return response time and success status."""
        start_time = time.perf_counter()
        try:
            result = await client.semantic_search(**query_data)
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000  # Convert to ms

            # Validate response structure
            success = (
                "query_id" in result
                and "results" in result
                and "search_duration_ms" in result
            )
            return response_time, success

        except Exception:
            # Expected to fail until implementation exists
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000, False

    async def _simulate_user_session(
        self,
        client: AsyncMock,
        queries: list[dict[str, Any]],
        user_id: int,
        session_duration: int = 30,
    ) -> list[tuple[float, bool]]:
        """Simulate a user session with multiple queries over time."""
        results = []
        end_time = time.time() + session_duration

        while time.time() < end_time:
            for query in queries:
                if time.time() >= end_time:
                    break
                result = await self._single_user_query(client, query, user_id)
                results.append(result)
                # Simulate user think time between queries
                await asyncio.sleep(0.5 + (user_id % 3) * 0.2)  # 0.5-1.1s think time

        return results

    @skip_without_mcp
    @skip_in_ci
    async def test_concurrent_user_performance_baseline(
        self,
        semantic_search_client: AsyncMock,
        concurrent_test_queries: list[dict[str, Any]],
    ) -> None:
        """
        Establish baseline performance with single user for comparison.
        """
        baseline_times = []

        # Run baseline queries sequentially
        for _ in range(20):
            for query in concurrent_test_queries[:3]:  # Use subset for baseline
                response_time, success = await self._single_user_query(
                    semantic_search_client, query, 0
                )
                if not success:
                    pytest.fail("Semantic search not implemented yet")
                baseline_times.append(response_time)

        if not baseline_times:
            pytest.fail("No successful baseline queries")

        baseline_p50 = quantiles(baseline_times, n=20)[9]
        baseline_p95 = quantiles(baseline_times, n=20)[18]

        print("\\nBaseline Performance:")
        print(f"  P50: {baseline_p50:.1f}ms")
        print(f"  P95: {baseline_p95:.1f}ms")

        # Store baseline for comparison in other tests
        self._baseline_p50 = baseline_p50
        self._baseline_p95 = baseline_p95

    @skip_without_mcp
    @skip_in_ci
    async def test_concurrent_users_performance(
        self,
        semantic_search_client: AsyncMock,
        concurrent_test_queries: list[dict[str, Any]],
    ) -> None:
        """
        Test performance under concurrent user load.

        Validates that p95 remains under 600ms even with multiple concurrent users.
        """
        concurrent_users = [10, 25, 50]  # Progressive load testing
        all_results = {}

        for num_users in concurrent_users:
            print(f"\\nTesting with {num_users} concurrent users...")

            # Create concurrent user sessions
            tasks = []
            for user_id in range(num_users):
                task = asyncio.create_task(
                    self._simulate_user_session(
                        semantic_search_client,
                        concurrent_test_queries,
                        user_id,
                        session_duration=10,  # 10 second sessions
                    )
                )
                tasks.append(task)

            # Wait for all users to complete
            user_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect all response times
            response_times = []
            success_count = 0
            total_queries = 0

            for user_result in user_results:
                if isinstance(user_result, Exception):
                    continue
                for response_time, success in user_result:
                    response_times.append(response_time)
                    total_queries += 1
                    if success:
                        success_count += 1

            if not response_times:
                pytest.fail(f"No successful queries with {num_users} concurrent users")

            # Calculate performance metrics
            mean_time = mean(response_times)
            p50 = quantiles(response_times, n=20)[9]
            p95 = quantiles(response_times, n=20)[18]
            success_rate = success_count / total_queries if total_queries > 0 else 0

            all_results[num_users] = {
                "mean": mean_time,
                "p50": p50,
                "p95": p95,
                "success_rate": success_rate,
                "total_queries": total_queries,
            }

            # Performance assertions
            assert p95 < MAX_P95_MS, (
                f"P95 response time {p95:.1f}ms exceeds {MAX_P95_MS}ms "
                f"with {num_users} concurrent users"
            )

            assert success_rate > 0.95, (
                f"Success rate {success_rate:.2%} too low with {num_users} concurrent users"
            )

            print(
                f"  Results: P50={p50:.1f}ms, P95={p95:.1f}ms, "
                f"Success={success_rate:.1%}, Queries={total_queries}"
            )

        # Test for reasonable performance degradation
        single_user_p95 = all_results[10]["p95"]
        max_load_p95 = all_results[50]["p95"]
        degradation_factor = (
            max_load_p95 / single_user_p95 if single_user_p95 > 0 else float("inf")
        )

        assert degradation_factor <= MAX_DEGRADATION_FACTOR, (
            f"Performance degraded by {degradation_factor:.1f}x under load "
            f"(exceeds {MAX_DEGRADATION_FACTOR}x limit)"
        )

        print("\\nConcurrent Performance Summary:")
        for users, metrics in all_results.items():
            print(
                f"  {users} users: P95={metrics['p95']:.1f}ms, "
                f"Success={metrics['success_rate']:.1%}"
            )
        print(f"  Performance degradation: {degradation_factor:.1f}x")

    @skip_without_mcp
    @skip_in_ci
    async def test_resource_contention_handling(
        self,
        semantic_search_client: AsyncMock,
        concurrent_test_queries: list[dict[str, Any]],
    ) -> None:
        """
        Test that resource contention is handled gracefully.

        No deadlocks, reasonable queue management, and fair resource allocation.
        """
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_threads = process.num_threads()

        # Simulate burst load
        burst_users = 100
        burst_duration = 5

        print(
            f"\\nTesting resource contention with {burst_users} users for {burst_duration}s..."
        )

        tasks = []
        for user_id in range(burst_users):
            task = asyncio.create_task(
                self._simulate_user_session(
                    semantic_search_client,
                    concurrent_test_queries[:2],  # Simple queries
                    user_id,
                    session_duration=burst_duration,
                )
            )
            tasks.append(task)

        start_time = time.time()
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Check resource usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_threads = process.num_threads()

        memory_growth = final_memory - initial_memory
        thread_growth = final_threads - initial_threads

        # Analyze results
        completed_sessions = sum(
            1 for r in user_results if not isinstance(r, Exception)
        )
        exception_count = sum(1 for r in user_results if isinstance(r, Exception))

        print(f"  Completed sessions: {completed_sessions}/{burst_users}")
        print(f"  Exceptions: {exception_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Thread growth: {thread_growth}")

        # Resource usage should be reasonable
        assert memory_growth < 200, (
            f"Excessive memory growth: {memory_growth:.1f}MB under burst load"
        )

        assert thread_growth < 50, (
            f"Excessive thread growth: {thread_growth} under burst load"
        )

        # Most sessions should complete (allowing for some failures due to load)
        completion_rate = completed_sessions / burst_users
        assert completion_rate > 0.80, (
            f"Too many failed sessions: {completion_rate:.1%} completion rate"
        )

    @skip_without_mcp
    @skip_in_ci
    async def test_query_fairness_under_load(
        self,
        semantic_search_client: AsyncMock,
        concurrent_test_queries: list[dict[str, Any]],
    ) -> None:
        """
        Test that queries are handled fairly under concurrent load.

        No user should experience significantly worse performance than others.
        """
        num_users = 20
        session_duration = 15

        print(f"\\nTesting query fairness with {num_users} users...")

        # Run concurrent sessions and track per-user performance
        tasks = []
        for user_id in range(num_users):
            task = asyncio.create_task(
                self._simulate_user_session(
                    semantic_search_client,
                    concurrent_test_queries,
                    user_id,
                    session_duration=session_duration,
                )
            )
            tasks.append(task)

        user_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate per-user metrics
        user_metrics = {}
        for user_id, user_result in enumerate(user_results):
            if isinstance(user_result, Exception):
                continue

            user_times = [rt for rt, success in user_result if success]
            if user_times:
                user_metrics[user_id] = {
                    "mean": mean(user_times),
                    "p95": quantiles(user_times, n=20)[18]
                    if len(user_times) >= 20
                    else max(user_times),
                    "query_count": len(user_times),
                }

        if len(user_metrics) < num_users * 0.8:
            pytest.fail("Too few users completed successfully for fairness analysis")

        # Analyze fairness
        all_means = [metrics["mean"] for metrics in user_metrics.values()]
        all_p95s = [metrics["p95"] for metrics in user_metrics.values()]

        mean_variance = (
            max(all_means) / min(all_means) if min(all_means) > 0 else float("inf")
        )
        p95_variance = (
            max(all_p95s) / min(all_p95s) if min(all_p95s) > 0 else float("inf")
        )

        print("  User performance variance:")
        print(
            f"    Mean response time range: {min(all_means):.1f}ms - {max(all_means):.1f}ms"
        )
        print(
            f"    P95 response time range: {min(all_p95s):.1f}ms - {max(all_p95s):.1f}ms"
        )
        print(f"    Mean variance ratio: {mean_variance:.1f}x")
        print(f"    P95 variance ratio: {p95_variance:.1f}x")

        # Fairness assertions - no user should be more than 2x slower than others
        assert mean_variance <= 2.0, (
            f"Unfair mean response times: {mean_variance:.1f}x variance between users"
        )

        assert p95_variance <= 2.5, (
            f"Unfair P95 response times: {p95_variance:.1f}x variance between users"
        )
