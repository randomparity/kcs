"""
Integration test for configuration filtering functionality.

This test validates the complete end-to-end functionality of semantic search
with kernel configuration filtering as specified in quickstart.md.

Scenario: User searches with configuration context like "CONFIG_NET" or "!CONFIG_EMBEDDED"
Expected: Results respect configuration filters and exclude/include appropriate code

Following TDD: This test MUST FAIL before implementation exists.
"""

import asyncio
import time
from typing import Any

import pytest


class TestConfigurationFilteringScenario:
    """Integration tests for configuration filtering scenarios."""

    @pytest.fixture
    async def search_engine(self):
        """Initialize semantic search engine for testing."""
        # This will fail until implementation exists
        from src.python.semantic_search.database.connection import (
            get_database_connection,
        )
        from src.python.semantic_search.services.embedding_service import (
            EmbeddingService,
        )
        from src.python.semantic_search.services.ranking_service import RankingService
        from src.python.semantic_search.services.vector_search_service import (
            VectorSearchService,
        )

        # Initialize services
        db_conn = await get_database_connection()
        embedding_service = EmbeddingService()
        vector_service = VectorSearchService(db_conn)
        ranking_service = RankingService()

        return {
            "embedding": embedding_service,
            "vector_search": vector_service,
            "ranking": ranking_service,
            "db": db_conn,
        }

    @pytest.fixture
    def sample_config_aware_content(self) -> list[dict[str, Any]]:
        """Sample kernel code with various configuration contexts."""
        return [
            {
                "file_path": "/net/socket.c",
                "line_start": 1234,
                "line_end": 1240,
                "content": """
#ifdef CONFIG_NET
static int __init sock_init(void)
{
    sk_init();
    protosw_init();
    return 0;
}
#endif /* CONFIG_NET */
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_NET"],
                "metadata": {
                    "function_name": "sock_init",
                    "symbols": ["sk_init", "protosw_init"],
                    "subsystem": "networking",
                },
            },
            {
                "file_path": "/kernel/power/suspend.c",
                "line_start": 567,
                "line_end": 574,
                "content": """
#ifdef CONFIG_EMBEDDED
static void suspend_prepare_embedded(void)
{
    /* Embedded specific power management */
    disable_non_essential_devices();
    reduce_cpu_frequency();
}
#endif
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_EMBEDDED", "CONFIG_PM"],
                "metadata": {
                    "function_name": "suspend_prepare_embedded",
                    "symbols": ["disable_non_essential_devices"],
                    "subsystem": "power_management",
                },
            },
            {
                "file_path": "/drivers/net/ethernet/intel/e1000/e1000_main.c",
                "line_start": 890,
                "line_end": 897,
                "content": """
#if defined(CONFIG_NET) && defined(CONFIG_PCI)
static int e1000_probe(struct pci_dev *pdev,
                      const struct pci_device_id *ent)
{
    struct net_device *netdev;
    return register_netdev(netdev);
}
#endif
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_NET", "CONFIG_PCI", "CONFIG_E1000"],
                "metadata": {
                    "function_name": "e1000_probe",
                    "symbols": ["pci_dev", "net_device"],
                    "subsystem": "networking",
                },
            },
            {
                "file_path": "/arch/arm/mach-omap2/pm.c",
                "line_start": 123,
                "line_end": 130,
                "content": """
#ifdef CONFIG_EMBEDDED
static void omap_pm_embedded_optimize(void)
{
    /* Embedded ARM power optimizations */
    omap_voltage_scale_down();
    disable_unused_clocks();
}
#endif
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_EMBEDDED", "CONFIG_ARM", "CONFIG_OMAP"],
                "metadata": {
                    "function_name": "omap_pm_embedded_optimize",
                    "symbols": ["omap_voltage_scale_down"],
                    "subsystem": "power_management",
                },
            },
            {
                "file_path": "/drivers/wireless/ath/ath9k/main.c",
                "line_start": 445,
                "line_end": 452,
                "content": """
#ifdef CONFIG_ATH9K_DEBUGFS
static void ath9k_debug_init(struct ath_softc *sc)
{
    debugfs_create_file("interrupt", S_IRUSR,
                       sc->debug.debugfs_phy, sc,
                       &fops_interrupt);
}
#endif
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": [
                    "CONFIG_ATH9K",
                    "CONFIG_ATH9K_DEBUGFS",
                    "CONFIG_WIRELESS",
                ],
                "metadata": {
                    "function_name": "ath9k_debug_init",
                    "symbols": ["ath_softc", "debugfs_create_file"],
                    "subsystem": "wireless",
                },
            },
            {
                "file_path": "/net/ipv4/tcp.c",
                "line_start": 2890,
                "line_end": 2897,
                "content": """
static int tcp_sendmsg(struct sock *sk, struct msghdr *msg, size_t size)
{
    struct tcp_sock *tp = tcp_sk(sk);
    int err, copied = 0;

    lock_sock(sk);
    err = tcp_sendmsg_locked(sk, msg, size);
    release_sock(sk);
    return err;
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_NET", "CONFIG_INET"],
                "metadata": {
                    "function_name": "tcp_sendmsg",
                    "symbols": ["tcp_sock", "msghdr"],
                    "subsystem": "networking",
                },
            },
        ]

    @pytest.fixture
    async def indexed_content(self, search_engine, sample_config_aware_content):
        """Index sample content for testing."""
        # This will fail until indexing implementation exists
        from src.python.semantic_search.services.indexing_service import IndexingService

        indexing_service = IndexingService(
            search_engine["embedding"], search_engine["vector_search"]
        )

        # Index the sample content
        for content in sample_config_aware_content:
            await indexing_service.index_content(
                file_path=content["file_path"],
                content=content["content"],
                line_start=content["line_start"],
                line_end=content["line_end"],
                content_type=content["content_type"],
                config_guards=content["config_guards"],
                metadata=content.get("metadata", {}),
            )

        return True

    async def test_config_net_positive_filtering(self, search_engine, indexed_content):
        """Test positive CONFIG_NET filtering from quickstart.md."""
        # This is the exact scenario from quickstart.md line 127
        query = "socket operations"

        # Search with CONFIG_NET filter
        results = await search_engine["vector_search"].search(
            query=query, config_context=["CONFIG_NET"], max_results=10
        )

        # All results should have CONFIG_NET in their configuration guards
        for result in results:
            if "metadata" in result and "config_guards" in result["metadata"]:
                config_guards = result["metadata"]["config_guards"]
                assert "CONFIG_NET" in config_guards, (
                    f"Result should have CONFIG_NET context: {config_guards}"
                )

        # Should find networking-related code
        if results:
            result_content = " ".join([r["content"] for r in results])
            networking_keywords = ["socket", "net", "tcp", "sock", "network"]
            found_keywords = [
                kw for kw in networking_keywords if kw.lower() in result_content.lower()
            ]
            assert len(found_keywords) > 0, "Should find networking-related keywords"

    async def test_config_embedded_negative_filtering(
        self, search_engine, indexed_content
    ):
        """Test negative CONFIG_EMBEDDED filtering from quickstart.md."""
        # This is the exact scenario from quickstart.md line 130
        query = "power management"

        # Search excluding embedded-specific code
        results = await search_engine["vector_search"].search(
            query=query, config_context=["!CONFIG_EMBEDDED"], max_results=10
        )

        # Results should not have CONFIG_EMBEDDED in their configuration guards
        for result in results:
            if "metadata" in result and "config_guards" in result["metadata"]:
                config_guards = result["metadata"]["config_guards"]
                assert "CONFIG_EMBEDDED" not in config_guards, (
                    f"Result should not have CONFIG_EMBEDDED context: {config_guards}"
                )

    async def test_multiple_config_filtering(self, search_engine, indexed_content):
        """Test filtering with multiple configuration contexts."""
        query = "network driver initialization"

        # Search with multiple positive config filters
        results = await search_engine["vector_search"].search(
            query=query, config_context=["CONFIG_NET", "CONFIG_PCI"], max_results=10
        )

        # Results should have both CONFIG_NET and CONFIG_PCI
        for result in results:
            if "metadata" in result and "config_guards" in result["metadata"]:
                config_guards = result["metadata"]["config_guards"]
                has_net = "CONFIG_NET" in config_guards
                has_pci = "CONFIG_PCI" in config_guards
                assert has_net and has_pci, (
                    f"Result should have both CONFIG_NET and CONFIG_PCI: {config_guards}"
                )

    async def test_mixed_positive_negative_config_filtering(
        self, search_engine, indexed_content
    ):
        """Test mixed positive and negative configuration filtering."""
        query = "system initialization"

        # Search with both positive and negative filters
        results = await search_engine["vector_search"].search(
            query=query,
            config_context=["CONFIG_NET", "!CONFIG_EMBEDDED"],
            max_results=10,
        )

        # Results should have CONFIG_NET but not CONFIG_EMBEDDED
        for result in results:
            if "metadata" in result and "config_guards" in result["metadata"]:
                config_guards = result["metadata"]["config_guards"]
                has_net = "CONFIG_NET" in config_guards
                has_embedded = "CONFIG_EMBEDDED" in config_guards

                # Should have CONFIG_NET (positive filter)
                assert has_net, f"Result should have CONFIG_NET: {config_guards}"
                # Should not have CONFIG_EMBEDDED (negative filter)
                assert not has_embedded, (
                    f"Result should not have CONFIG_EMBEDDED: {config_guards}"
                )

    async def test_config_filtering_performance(self, search_engine, indexed_content):
        """Test that configuration filtering doesn't impact performance significantly."""
        query = "network operations"

        # Test without config filtering
        start_time = time.time()
        await search_engine["vector_search"].search(query=query, max_results=5)
        no_filter_time = time.time() - start_time

        # Test with config filtering
        start_time = time.time()
        await search_engine["vector_search"].search(
            query=query, config_context=["CONFIG_NET"], max_results=5
        )
        filter_time = time.time() - start_time

        # Both should complete within performance limit
        assert no_filter_time * 1000 < 600, "Search without filter should be fast"
        assert filter_time * 1000 < 600, "Search with filter should be fast"

        # Filtering shouldn't add more than 100ms overhead
        overhead = filter_time - no_filter_time
        assert overhead < 0.1, f"Config filtering overhead {overhead * 1000}ms too high"

    async def test_config_filtering_metadata_inclusion(
        self, search_engine, indexed_content
    ):
        """Test that config filtering properly includes metadata in results."""
        query = "wireless debugging"

        results = await search_engine["vector_search"].search(
            query=query, config_context=["CONFIG_WIRELESS"], max_results=5
        )

        # Results should include metadata with config_guards information
        for result in results:
            assert "metadata" in result, "Results should include metadata"

            metadata = result["metadata"]
            assert "config_guards" in metadata, "Metadata should include config_guards"

            config_guards = metadata["config_guards"]
            assert isinstance(config_guards, list), "config_guards should be a list"
            assert len(config_guards) > 0, "config_guards should not be empty"

            # Should have wireless-related configuration
            wireless_configs = [
                g for g in config_guards if "WIRELESS" in g or "ATH" in g
            ]
            assert len(wireless_configs) > 0, (
                f"Should have wireless configs: {config_guards}"
            )

    async def test_config_filtering_exact_match_scenarios(
        self, search_engine, indexed_content
    ):
        """Test configuration filtering for exact scenarios from quickstart.md."""
        # Test scenario 1: Network-specific code with CONFIG_NET
        result1 = await search_engine["vector_search"].search(
            query="socket operations", config_context=["CONFIG_NET"], max_results=5
        )

        # Should find socket-related networking code
        socket_results = [r for r in result1 if "socket" in r["content"].lower()]
        assert len(socket_results) > 0, "Should find socket operations with CONFIG_NET"

        # Test scenario 2: Exclude embedded-specific power management
        result2 = await search_engine["vector_search"].search(
            query="power management", config_context=["!CONFIG_EMBEDDED"], max_results=5
        )

        # Should not return embedded-specific power management
        for result in result2:
            if "metadata" in result and "config_guards" in result["metadata"]:
                assert "CONFIG_EMBEDDED" not in result["metadata"]["config_guards"]

    async def test_config_filtering_error_handling(
        self, search_engine, indexed_content
    ):
        """Test error handling for invalid configuration contexts."""
        query = "test query"

        # Test invalid config format (missing CONFIG_ prefix)
        with pytest.raises(ValueError):
            await search_engine["vector_search"].search(
                query=query,
                config_context=["INVALID_CONFIG"],  # Should start with CONFIG_
            )

        # Test empty config context
        with pytest.raises(ValueError):
            await search_engine["vector_search"].search(
                query=query,
                config_context=[""],  # Empty config should fail
            )

        # Test malformed negative config
        with pytest.raises(ValueError):
            await search_engine["vector_search"].search(
                query=query,
                config_context=["!!CONFIG_NET"],  # Double negative invalid
            )

    async def test_config_filtering_subsystem_awareness(
        self, search_engine, indexed_content
    ):
        """Test that config filtering respects kernel subsystem boundaries."""
        # Test wireless subsystem filtering
        wireless_results = await search_engine["vector_search"].search(
            query="driver initialization",
            config_context=["CONFIG_WIRELESS"],
            max_results=10,
        )

        # All wireless results should be from wireless subsystem
        for result in wireless_results:
            if "metadata" in result and "subsystem" in result["metadata"]:
                subsystem = result["metadata"]["subsystem"]
                assert subsystem == "wireless", (
                    f"Expected wireless subsystem, got {subsystem}"
                )

        # Test networking subsystem filtering
        net_results = await search_engine["vector_search"].search(
            query="protocol initialization",
            config_context=["CONFIG_NET"],
            max_results=10,
        )

        # Networking results should be from networking subsystem
        for result in net_results:
            if "metadata" in result and "subsystem" in result["metadata"]:
                subsystem = result["metadata"]["subsystem"]
                assert subsystem == "networking", (
                    f"Expected networking subsystem, got {subsystem}"
                )

    async def test_config_filtering_mcp_integration(
        self, search_engine, indexed_content
    ):
        """Test configuration filtering through MCP interface."""
        # This will fail until MCP integration exists
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Test positive config filtering via MCP
        result = await semantic_search(
            query="socket operations", config_context=["CONFIG_NET"], max_results=5
        )

        # Validate MCP response structure
        assert "query_id" in result
        assert "results" in result
        assert "search_stats" in result

        # All results should respect CONFIG_NET filter
        for search_result in result["results"]:
            if (
                "metadata" in search_result
                and "config_guards" in search_result["metadata"]
            ):
                config_guards = search_result["metadata"]["config_guards"]
                assert "CONFIG_NET" in config_guards

        # Test negative config filtering via MCP
        result2 = await semantic_search(
            query="power management", config_context=["!CONFIG_EMBEDDED"], max_results=5
        )

        # Results should not have CONFIG_EMBEDDED
        for search_result in result2["results"]:
            if (
                "metadata" in search_result
                and "config_guards" in search_result["metadata"]
            ):
                config_guards = search_result["metadata"]["config_guards"]
                assert "CONFIG_EMBEDDED" not in config_guards

    async def test_config_filtering_concurrent_queries(
        self, search_engine, indexed_content
    ):
        """Test configuration filtering under concurrent load."""
        # Define multiple concurrent queries with different config contexts
        queries = [
            ("networking code", ["CONFIG_NET"]),
            ("wireless drivers", ["CONFIG_WIRELESS"]),
            ("power management", ["!CONFIG_EMBEDDED"]),
            ("debug features", ["CONFIG_DEBUG"]),
            ("PCI drivers", ["CONFIG_PCI"]),
        ]

        async def single_query(query, config_context):
            start_time = time.time()
            results = await search_engine["vector_search"].search(
                query=query, config_context=config_context, max_results=3
            )
            end_time = time.time()
            return (end_time - start_time) * 1000, len(results), config_context

        # Execute concurrent queries
        tasks = [single_query(q, c) for q, c in queries]
        results = await asyncio.gather(*tasks)

        # All queries should complete within performance limit
        for search_time, _result_count, config_context in results:
            assert search_time < 600, (
                f"Query with config {config_context} took {search_time}ms, exceeds 600ms limit"
            )

    async def test_config_filtering_complex_scenarios(
        self, search_engine, indexed_content
    ):
        """Test complex configuration filtering scenarios."""
        # Scenario 1: Multiple positive configs (AND logic)
        results1 = await search_engine["vector_search"].search(
            query="network device",
            config_context=["CONFIG_NET", "CONFIG_PCI"],
            max_results=5,
        )

        # Should find results that have BOTH configs
        for result in results1:
            if "metadata" in result and "config_guards" in result["metadata"]:
                guards = result["metadata"]["config_guards"]
                assert "CONFIG_NET" in guards and "CONFIG_PCI" in guards

        # Scenario 2: Mixed positive and multiple negative configs
        results2 = await search_engine["vector_search"].search(
            query="system code",
            config_context=["CONFIG_NET", "!CONFIG_EMBEDDED", "!CONFIG_DEBUG"],
            max_results=5,
        )

        # Should have CONFIG_NET but neither CONFIG_EMBEDDED nor CONFIG_DEBUG
        for result in results2:
            if "metadata" in result and "config_guards" in result["metadata"]:
                guards = result["metadata"]["config_guards"]
                assert "CONFIG_NET" in guards
                assert "CONFIG_EMBEDDED" not in guards
                assert "CONFIG_DEBUG" not in guards

    async def test_config_filtering_edge_cases(self, search_engine, indexed_content):
        """Test edge cases for configuration filtering."""
        # Test with non-existent config
        results1 = await search_engine["vector_search"].search(
            query="test code", config_context=["CONFIG_NONEXISTENT"], max_results=5
        )

        # Should return empty results for non-existent config
        assert len(results1) == 0, "Non-existent config should return no results"

        # Test with config that exists but no matching content
        results2 = await search_engine["vector_search"].search(
            query="completely unrelated query that matches nothing",
            config_context=["CONFIG_NET"],
            max_results=5,
        )

        # May return no results due to semantic mismatch, which is fine
        # Just ensure no errors are thrown
        assert isinstance(results2, list), "Should return list even if empty"
