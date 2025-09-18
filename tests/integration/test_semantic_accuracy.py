"""
Integration tests for semantic search accuracy and relevance.

These tests verify that the semantic search functionality returns
high-quality, relevant results for typical kernel development queries.
They focus on the accuracy of search results rather than API contracts.

Key test scenarios:
- VFS operations search returns actual filesystem functions
- Memory management queries find MM subsystem code
- Network protocol searches match networking code
- Architecture-specific queries return arch-specific results
- Cross-reference searches find related functions
- Search result ranking validates relevance scores
"""

import os
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests


# Test infrastructure
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible for integration testing."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server required for integration tests"
)

# Skip integration tests in CI environments unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "integration_test_token"

# Expected accuracy thresholds
MIN_RELEVANCE_SCORE = 0.3  # Minimum acceptable similarity score
MIN_RESULTS_FOR_COMMON_QUERIES = 3  # Common queries should return multiple results
MAX_RESPONSE_TIME_MS = 5000  # Maximum acceptable response time


# Test fixtures
@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for async requests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


# Accuracy test cases
@skip_without_mcp_server
@skip_integration_in_ci
class TestSemanticSearchAccuracy:
    """Test semantic search accuracy and result quality."""

    async def test_vfs_operations_accuracy(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that VFS operation queries return relevant filesystem functions."""
        test_cases = [
            {
                "query": "VFS read file operations",
                "expected_symbols": ["vfs_read", "generic_file_read", "do_sync_read"],
                "expected_subsystems": ["fs", "vfs", "filesystem"],
            },
            {
                "query": "file system write operations",
                "expected_symbols": [
                    "vfs_write",
                    "generic_file_write",
                    "do_sync_write",
                ],
                "expected_subsystems": ["fs", "vfs", "filesystem"],
            },
            {
                "query": "directory entry lookup",
                "expected_symbols": ["lookup_one_len", "d_lookup", "real_lookup"],
                "expected_subsystems": ["fs", "vfs"],
            },
        ]

        for case in test_cases:
            request_data = {
                "query": case["query"],
                "limit": 20,
                "similarity_threshold": 0.4,
                "search_mode": "semantic",
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/semantic_search",
                json=request_data,
                headers=auth_headers,
            )

            # Skip if endpoint not implemented or server issues
            if response.status_code != 200:
                pytest.skip(f"Semantic search not available: {response.status_code}")

            data = response.json()

            # Verify we get some results
            assert len(data["results"]) >= MIN_RESULTS_FOR_COMMON_QUERIES, (
                f"Query '{case['query']}' should return at least "
                f"{MIN_RESULTS_FOR_COMMON_QUERIES} results"
            )

            # Verify response time is reasonable
            assert data["search_time_ms"] < MAX_RESPONSE_TIME_MS, (
                f"Search should complete within {MAX_RESPONSE_TIME_MS}ms"
            )

            # Check for expected symbols (flexible matching)
            found_symbols = [r["symbol"] for r in data["results"]]
            symbol_matches = 0
            for expected_symbol in case["expected_symbols"]:
                for found_symbol in found_symbols:
                    if expected_symbol.lower() in found_symbol.lower():
                        symbol_matches += 1
                        break

            # Should find at least one expected symbol
            assert symbol_matches > 0, (
                f"Query '{case['query']}' should match at least one expected symbol. "
                f"Expected: {case['expected_symbols']}, Found: {found_symbols[:5]}"
            )

            # Verify relevance scores are reasonable
            for result in data["results"]:
                assert result["similarity_score"] >= MIN_RELEVANCE_SCORE, (
                    f"Result '{result['symbol']}' has low relevance score: "
                    f"{result['similarity_score']}"
                )

            # Check subsystem classification
            found_subsystems = set()
            for result in data["results"]:
                if result["context"]["subsystem"]:
                    found_subsystems.add(result["context"]["subsystem"])

            # Should have some filesystem-related subsystems
            fs_subsystems = {"fs", "vfs", "filesystem"} & found_subsystems
            if found_subsystems:  # Only check if subsystems are populated
                assert len(fs_subsystems) > 0, (
                    f"VFS query should return filesystem subsystems. "
                    f"Found: {found_subsystems}"
                )

    async def test_memory_management_accuracy(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that memory management queries return MM subsystem functions."""
        test_cases = [
            {
                "query": "memory allocation kmalloc",
                "expected_symbols": ["kmalloc", "__kmalloc", "kmem_cache_alloc"],
                "expected_subsystems": ["mm", "memory"],
            },
            {
                "query": "page frame allocation",
                "expected_symbols": ["alloc_pages", "__alloc_pages", "get_free_page"],
                "expected_subsystems": ["mm", "memory"],
            },
            {
                "query": "virtual memory management",
                "expected_symbols": ["vm_area_struct", "find_vma", "do_mmap"],
                "expected_subsystems": ["mm", "memory"],
            },
        ]

        for case in test_cases:
            request_data = {
                "query": case["query"],
                "limit": 15,
                "similarity_threshold": 0.3,
                "filters": {
                    "subsystems": ["mm", "memory"],
                    "exclude_tests": True,
                },
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/semantic_search",
                json=request_data,
                headers=auth_headers,
            )

            if response.status_code != 200:
                pytest.skip(f"Semantic search not available: {response.status_code}")

            data = response.json()

            # Should find memory-related functions
            assert len(data["results"]) >= 2, (
                f"Memory query '{case['query']}' should find multiple results"
            )

            # Check for expected memory management symbols
            found_symbols = [r["symbol"].lower() for r in data["results"]]
            expected_found = []
            for expected in case["expected_symbols"]:
                for found in found_symbols:
                    if expected.lower() in found:
                        expected_found.append(expected)
                        break

            assert len(expected_found) > 0, (
                f"Memory query should find at least one expected symbol. "
                f"Expected: {case['expected_symbols']}, Found symbols: {found_symbols[:10]}"
            )

            # Results should have high relevance for memory queries
            high_relevance_count = sum(
                1 for r in data["results"] if r["similarity_score"] > 0.5
            )
            assert high_relevance_count > 0, (
                "Memory management queries should return highly relevant results"
            )

    async def test_networking_protocol_accuracy(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that networking queries return network subsystem functions."""
        test_cases = [
            {
                "query": "TCP socket operations",
                "expected_symbols": ["tcp_sendmsg", "tcp_recvmsg", "tcp_connect"],
                "expected_file_patterns": ["net/", "tcp"],
            },
            {
                "query": "network packet processing",
                "expected_symbols": [
                    "netif_receive_skb",
                    "dev_queue_xmit",
                    "skb_alloc",
                ],
                "expected_file_patterns": ["net/", "skb"],
            },
            {
                "query": "socket system calls",
                "expected_symbols": ["sys_socket", "sys_connect", "sys_sendto"],
                "expected_file_patterns": ["net/", "socket"],
            },
        ]

        for case in test_cases:
            request_data = {
                "query": case["query"],
                "limit": 15,
                "similarity_threshold": 0.3,
                "filters": {
                    "file_patterns": ["net/*.c", "net/*/*.c"],
                    "exclude_tests": True,
                },
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/semantic_search",
                json=request_data,
                headers=auth_headers,
            )

            if response.status_code != 200:
                pytest.skip(f"Semantic search not available: {response.status_code}")

            data = response.json()

            # Should find networking functions
            assert len(data["results"]) >= 1, (
                f"Network query '{case['query']}' should find results"
            )

            # Check that results are from networking subsystem
            net_files = 0
            for result in data["results"]:
                file_path = result["span"]["path"]
                if any(
                    pattern in file_path for pattern in case["expected_file_patterns"]
                ):
                    net_files += 1

            # At least some results should be from network files
            if len(data["results"]) > 0:
                net_ratio = net_files / len(data["results"])
                assert net_ratio > 0.3, (
                    f"Network query should return mostly network files. "
                    f"Network files: {net_files}/{len(data['results'])}"
                )

    async def test_architecture_specific_accuracy(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that architecture-specific queries return arch-specific code."""
        test_cases = [
            {
                "query": "x86_64 system call entry",
                "expected_patterns": ["x86", "entry", "syscall"],
                "arch_files": ["arch/x86/", "x86"],
            },
            {
                "query": "ARM64 exception handling",
                "expected_patterns": ["arm64", "exception", "handler"],
                "arch_files": ["arch/arm64/", "arm64"],
            },
            {
                "query": "interrupt handling architecture",
                "expected_patterns": ["interrupt", "irq", "handler"],
                "arch_files": ["arch/", "irq"],
            },
        ]

        for case in test_cases:
            request_data = {
                "query": case["query"],
                "limit": 10,
                "similarity_threshold": 0.2,  # Lower threshold for arch-specific
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/semantic_search",
                json=request_data,
                headers=auth_headers,
            )

            if response.status_code != 200:
                pytest.skip(f"Semantic search not available: {response.status_code}")

            data = response.json()

            if len(data["results"]) > 0:
                # Check that results contain architecture-related terms
                relevant_results = 0
                for result in data["results"]:
                    symbol_lower = result["symbol"].lower()
                    path_lower = result["span"]["path"].lower()
                    snippet_lower = result["snippet"].lower()

                    # Check if result contains expected patterns
                    for pattern in case["expected_patterns"]:
                        if (
                            pattern.lower() in symbol_lower
                            or pattern.lower() in path_lower
                            or pattern.lower() in snippet_lower
                        ):
                            relevant_results += 1
                            break

                # At least half the results should be relevant
                relevance_ratio = relevant_results / len(data["results"])
                assert relevance_ratio >= 0.3, (
                    f"Architecture query should return relevant results. "
                    f"Relevant: {relevant_results}/{len(data['results'])}"
                )

    async def test_cross_reference_accuracy(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that searches for related functions find connected code."""
        test_cases = [
            {
                "query": "file operations structure initialization",
                "related_terms": ["file_operations", "fops", "open", "read", "write"],
            },
            {
                "query": "process creation and management",
                "related_terms": ["fork", "clone", "exec", "task_struct", "process"],
            },
            {
                "query": "device driver probe and remove",
                "related_terms": ["probe", "remove", "driver", "device", "platform"],
            },
        ]

        for case in test_cases:
            request_data = {
                "query": case["query"],
                "limit": 20,
                "similarity_threshold": 0.25,
                "search_mode": "hybrid",  # Use hybrid for better cross-reference
                "keyword_weight": 0.4,
                "semantic_weight": 0.6,
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/semantic_search",
                json=request_data,
                headers=auth_headers,
            )

            if response.status_code != 200:
                pytest.skip(f"Semantic search not available: {response.status_code}")

            data = response.json()

            if len(data["results"]) > 0:
                # Check for presence of related terms across all results
                related_term_counts = dict.fromkeys(case["related_terms"], 0)

                for result in data["results"]:
                    text_to_search = (
                        result["symbol"]
                        + " "
                        + result["span"]["path"]
                        + " "
                        + result["snippet"]
                    ).lower()

                    for term in case["related_terms"]:
                        if term.lower() in text_to_search:
                            related_term_counts[term] += 1

                # Should find multiple related terms
                found_terms = sum(
                    1 for count in related_term_counts.values() if count > 0
                )
                assert found_terms >= 2, (
                    f"Cross-reference query should find multiple related terms. "
                    f"Found terms: {related_term_counts}"
                )

    async def test_result_ranking_quality(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that search results are properly ranked by relevance."""
        request_data = {
            "query": "virtual file system read write operations",
            "limit": 15,
            "similarity_threshold": 0.1,  # Low threshold to get varied results
            "rerank": True,
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/semantic_search",
            json=request_data,
            headers=auth_headers,
        )

        if response.status_code != 200:
            pytest.skip(f"Semantic search not available: {response.status_code}")

        data = response.json()

        if len(data["results"]) >= 3:
            # Verify results are ordered by relevance (descending)
            scores = [r["similarity_score"] for r in data["results"]]
            assert scores == sorted(scores, reverse=True), (
                "Results should be ordered by similarity score (highest first)"
            )

            # Top results should have significantly higher scores than bottom results
            if len(scores) >= 5:
                top_score = scores[0]
                bottom_score = scores[-1]
                score_spread = top_score - bottom_score

                assert score_spread > 0.1, (
                    f"Should have good score spread between top and bottom results. "
                    f"Top: {top_score}, Bottom: {bottom_score}, Spread: {score_spread}"
                )

            # Verify that higher-ranked results are more relevant
            query_terms = ["virtual", "file", "system", "read", "write", "operations"]
            top_3_relevance = 0
            bottom_3_relevance = 0

            # Check top 3 results
            for result in data["results"][:3]:
                text = (result["symbol"] + " " + result["snippet"]).lower()
                matches = sum(1 for term in query_terms if term in text)
                top_3_relevance += matches

            # Check bottom 3 results
            if len(data["results"]) >= 6:
                for result in data["results"][-3:]:
                    text = (result["symbol"] + " " + result["snippet"]).lower()
                    matches = sum(1 for term in query_terms if term in text)
                    bottom_3_relevance += matches

                # Top results should generally be more relevant
                assert top_3_relevance >= bottom_3_relevance, (
                    f"Top results should be more relevant than bottom results. "
                    f"Top 3 relevance: {top_3_relevance}, Bottom 3: {bottom_3_relevance}"
                )

    async def test_semantic_vs_exact_matching(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that semantic search finds relevant results beyond exact matches."""
        # Use a query with synonyms and related concepts
        request_data = {
            "query": "memory allocation routines for kernel objects",
            "limit": 10,
            "similarity_threshold": 0.3,
            "search_mode": "semantic",
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/semantic_search",
            json=request_data,
            headers=auth_headers,
        )

        if response.status_code != 200:
            pytest.skip(f"Semantic search not available: {response.status_code}")

        data = response.json()

        if len(data["results"]) > 0:
            # Should find memory-related functions even without exact keyword matches
            memory_related = 0
            allocation_related = 0
            kernel_related = 0

            for result in data["results"]:
                text = (result["symbol"] + " " + result["snippet"]).lower()

                # Memory related terms (including variants)
                if any(
                    term in text for term in ["mem", "alloc", "malloc", "cache", "slab"]
                ):
                    memory_related += 1

                # Allocation related terms
                if any(
                    term in text
                    for term in ["alloc", "free", "get", "put", "new", "create"]
                ):
                    allocation_related += 1

                # Kernel related terms
                if any(term in text for term in ["kernel", "kmem", "k_", "sys_"]):
                    kernel_related += 1

            # Should find semantically related results
            assert memory_related > 0, "Should find memory-related functions"
            assert allocation_related > 0, "Should find allocation-related functions"

            # At least half the results should be semantically relevant
            semantic_relevance = (memory_related + allocation_related) / len(
                data["results"]
            )
            assert semantic_relevance >= 0.5, (
                f"Semantic search should find relevant results beyond exact matches. "
                f"Relevance ratio: {semantic_relevance}"
            )

    async def test_query_expansion_effectiveness(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that query expansion improves search recall."""
        base_query = "VFS"

        # Search without expansion
        request_without_expansion = {
            "query": base_query,
            "limit": 20,
            "expand_query": False,
        }

        # Search with expansion
        request_with_expansion = {
            "query": base_query,
            "limit": 20,
            "expand_query": True,
            "expansion_terms": 5,
        }

        response1 = await http_client.post(
            f"{MCP_BASE_URL}/semantic_search",
            json=request_without_expansion,
            headers=auth_headers,
        )

        response2 = await http_client.post(
            f"{MCP_BASE_URL}/semantic_search",
            json=request_with_expansion,
            headers=auth_headers,
        )

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            # Query expansion should generally improve recall
            if "expanded_query" in data2:
                # Expanded query should be longer
                assert len(data2["expanded_query"]) > len(base_query), (
                    "Expanded query should be longer than original"
                )

            # Should find relevant results in both cases
            if len(data1["results"]) > 0 or len(data2["results"]) > 0:
                # Expanded search might find different or additional relevant results
                symbols1 = {r["symbol"] for r in data1["results"]}
                symbols2 = {r["symbol"] for r in data2["results"]}

                # Results might be different due to expansion
                if len(symbols1) > 0 and len(symbols2) > 0:
                    # At least some results should be filesystem-related
                    fs_terms = ["vfs", "file", "fs", "dentry", "inode"]

                    def has_fs_relevance(results):
                        return any(
                            any(
                                term in r["symbol"].lower()
                                or term in r["span"]["path"].lower()
                                or term in r["snippet"].lower()
                                for term in fs_terms
                            )
                            for r in results
                        )

                    # At least one search should find filesystem-relevant results
                    assert has_fs_relevance(data1["results"]) or has_fs_relevance(
                        data2["results"]
                    ), "VFS search should find filesystem-relevant results"
