"""
Integration test for buffer overflow search scenario.

This test validates the complete end-to-end functionality of semantic search
for buffer overflow queries as specified in quickstart.md.

Scenario: User searches for "buffer overflow in network drivers"
Expected: Returns network driver code with buffer bounds checking and validation

Following TDD: This test MUST FAIL before implementation exists.
"""

import asyncio
import time
from typing import Any

import pytest


class TestBufferOverflowSearchScenario:
    """Integration tests for buffer overflow search scenarios."""

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
    def sample_buffer_overflow_content(self) -> list[dict[str, Any]]:
        """Sample network driver code with buffer overflow vulnerabilities/fixes."""
        return [
            {
                "file_path": "/drivers/net/ethernet/realtek/r8169.c",
                "line_start": 1234,
                "line_end": 1240,
                "content": """
static void rtl8169_rx_interrupt(struct net_device *dev,
                                struct rtl8169_private *tp)
{
    if (pkt_size > rx_buf_sz) {
        netdev_err(dev, "Packet size exceeds buffer bounds\n");
        return;
    }
    memcpy(skb->data, data, pkt_size);
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_NET", "CONFIG_R8169"],
                "metadata": {
                    "function_name": "rtl8169_rx_interrupt",
                    "symbols": ["rtl8169_private", "net_device", "pkt_size"],
                    "vulnerability_pattern": "buffer_bounds_check",
                },
            },
            {
                "file_path": "/drivers/net/wireless/ath/ath9k/recv.c",
                "line_start": 567,
                "line_end": 573,
                "content": """
static bool ath_rx_prepare(struct ath_hw *ah, struct sk_buff *skb)
{
    if (skb->len < ATH_RX_MIN_SIZE || skb->len > ATH_RX_MAX_SIZE) {
        ath_err(ah, "Invalid SKB buffer length: %d\n", skb->len);
        return false;
    }
    return ath_validate_rx_buffer(skb);
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_ATH9K", "CONFIG_WIRELESS"],
                "metadata": {
                    "function_name": "ath_rx_prepare",
                    "symbols": ["ath_hw", "sk_buff", "ATH_RX_MIN_SIZE"],
                    "vulnerability_pattern": "skb_length_validation",
                },
            },
            {
                "file_path": "/drivers/net/ethernet/intel/e1000/e1000_main.c",
                "line_start": 2890,
                "line_end": 2897,
                "content": """
static void e1000_clean_rx_irq(struct e1000_adapter *adapter)
{
    /* Buffer overflow protection */
    if (length > adapter->rx_buffer_len) {
        e1000_err("Received packet length exceeds allocated buffer\n");
        adapter->stats.rx_length_errors++;
        goto next_desc;
    }
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_E1000"],
                "metadata": {
                    "function_name": "e1000_clean_rx_irq",
                    "symbols": ["e1000_adapter", "rx_buffer_len"],
                    "vulnerability_pattern": "buffer_overflow_protection",
                },
            },
            {
                "file_path": "/drivers/net/wireless/intel/iwlwifi/rx.c",
                "line_start": 445,
                "line_end": 452,
                "content": """
static int iwl_rx_handle_bad_length(struct iwl_priv *priv, u32 len)
{
    if (len > IWL_RX_BUF_SIZE_4K) {
        IWL_ERR(priv, "Packet length %u exceeds max buffer size\\n", len);
        priv->rx_stats.buffer_overflow++;
        return -EINVAL;
    }
    return 0;
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_IWLWIFI"],
                "metadata": {
                    "function_name": "iwl_rx_handle_bad_length",
                    "symbols": ["iwl_priv", "IWL_RX_BUF_SIZE_4K"],
                    "vulnerability_pattern": "rx_buffer_overflow_check",
                },
            },
            {
                "file_path": "/drivers/net/ethernet/broadcom/bnx2.c",
                "line_start": 1567,
                "line_end": 1575,
                "content": """
static int bnx2_rx_int(struct bnx2 *bp, int budget)
{
    /* Vulnerable code - no bounds checking */
    unsigned char *data = skb_put(skb, len);
    if (!data) {
        /* This should check len before skb_put */
        bnx2_reuse_rx_skb(bp, skb, sw_ring_cons);
        goto next_rx;
    }
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_BNX2"],
                "metadata": {
                    "function_name": "bnx2_rx_int",
                    "symbols": ["bnx2", "skb_put"],
                    "vulnerability_pattern": "missing_bounds_check",
                },
            },
        ]

    @pytest.fixture
    async def indexed_content(self, search_engine, sample_buffer_overflow_content):
        """Index sample content for testing."""
        # This will fail until indexing implementation exists
        from src.python.semantic_search.services.indexing_service import IndexingService

        indexing_service = IndexingService(
            search_engine["embedding"], search_engine["vector_search"]
        )

        # Index the sample content
        for content in sample_buffer_overflow_content:
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

    async def test_basic_buffer_overflow_search(self, search_engine, indexed_content):
        """Test basic buffer overflow search functionality."""
        # This is the core scenario from quickstart.md
        query = "buffer overflow in network drivers"

        # Perform the search
        start_time = time.time()
        results = await search_engine["vector_search"].search(
            query=query, max_results=5, min_confidence=0.5
        )
        end_time = time.time()

        # Performance requirement: Response time < 600ms
        search_time_ms = (end_time - start_time) * 1000
        assert search_time_ms < 600, (
            f"Search took {search_time_ms}ms, exceeds 600ms limit"
        )

        # Should return results
        assert len(results) >= 2, "Should return at least 2 results for buffer overflow"

        # All results should have confidence > 0.5
        for result in results:
            assert result["confidence"] > 0.5, (
                f"Result confidence {result['confidence']} below 0.5"
            )

        # Results should be from network drivers
        result_files = [r["file_path"] for r in results]
        network_driver_files = [f for f in result_files if "/drivers/net/" in f]
        assert len(network_driver_files) > 0, "Should find results in network drivers"

        # Results should contain buffer-related keywords
        result_content = " ".join([r["content"] for r in results])
        buffer_keywords = ["buffer", "length", "size", "bounds", "overflow"]
        found_keywords = [
            kw for kw in buffer_keywords if kw.lower() in result_content.lower()
        ]

        assert len(found_keywords) >= 2, (
            f"Expected buffer-related keywords, found: {found_keywords}"
        )

    async def test_buffer_overflow_search_with_confidence_threshold(
        self, search_engine, indexed_content
    ):
        """Test buffer overflow search with confidence threshold from quickstart.md."""
        # This exact scenario is mentioned in quickstart.md line 73
        query = "buffer overflow in network drivers"

        # Search with confidence threshold of 0.7
        results = await search_engine["vector_search"].search(
            query=query, max_results=10, min_confidence=0.7
        )

        # All results should meet the confidence threshold
        for result in results:
            assert result["confidence"] >= 0.7

        # Should still find relevant buffer overflow patterns
        if results:
            result_content = " ".join([r["content"] for r in results])
            security_patterns = [
                "bounds",
                "length",
                "validate",
                "check",
                "overflow",
                "exceed",
            ]
            found_patterns = [
                p for p in security_patterns if p in result_content.lower()
            ]
            assert len(found_patterns) > 0, "Should find security-related patterns"

    async def test_buffer_overflow_vulnerability_detection(
        self, search_engine, indexed_content
    ):
        """Test detection of specific buffer overflow vulnerability patterns."""
        query = "packet length exceeds buffer bounds"

        results = await search_engine["vector_search"].search(
            query=query, max_results=5, min_confidence=0.6
        )

        # Should find the specific vulnerability patterns from our sample data
        vulnerability_indicators = ["exceeds", "buffer", "length", "bounds", "validate"]

        for result in results:
            content_lower = result["content"].lower()
            found_indicators = [
                ind for ind in vulnerability_indicators if ind in content_lower
            ]

            # Each result should contain multiple vulnerability indicators
            assert len(found_indicators) >= 2, (
                f"Result should contain vulnerability indicators: {content_lower}"
            )

    async def test_buffer_overflow_search_network_driver_filtering(
        self, search_engine, indexed_content
    ):
        """Test buffer overflow search filtered to network drivers."""
        query = "buffer overflow protection"

        # Filter to only network driver files
        results = await search_engine["vector_search"].search(
            query=query, max_results=10, file_patterns=["/drivers/net/*"]
        )

        # All results should be from network drivers
        for result in results:
            assert "/drivers/net/" in result["file_path"], (
                f"Result {result['file_path']} not from network drivers"
            )

        # Should find protection mechanisms
        if results:
            result_content = " ".join([r["content"] for r in results])
            protection_keywords = ["check", "validate", "protect", "bounds", "length"]
            found_keywords = [
                kw for kw in protection_keywords if kw in result_content.lower()
            ]
            assert len(found_keywords) > 0, "Should find protection mechanisms"

    async def test_buffer_overflow_search_config_awareness(
        self, search_engine, indexed_content
    ):
        """Test buffer overflow search with network configuration context."""
        query = "wireless driver buffer validation"

        # Search with wireless configuration context
        results = await search_engine["vector_search"].search(
            query=query, config_context=["CONFIG_WIRELESS"]
        )

        # Results should respect wireless configuration context
        for result in results:
            if "metadata" in result and "config_guards" in result["metadata"]:
                config_guards = result["metadata"]["config_guards"]
                wireless_configs = [
                    g
                    for g in config_guards
                    if "WIRELESS" in g or "ATH" in g or "IWL" in g
                ]
                assert len(wireless_configs) > 0, (
                    f"Result should have wireless context: {config_guards}"
                )

    async def test_buffer_overflow_search_semantic_ranking(
        self, search_engine, indexed_content
    ):
        """Test semantic ranking for buffer overflow searches."""
        query = "SKB buffer length validation"

        results = await search_engine["vector_search"].search(
            query=query, max_results=5, min_confidence=0.4
        )

        # Results should be ranked by relevance
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True), (
            "Results should be ordered by confidence"
        )

        # Top results should be most relevant to SKB validation
        if results:
            top_result = results[0]
            content_lower = top_result["content"].lower()
            skb_related = ["skb", "socket", "buffer", "length", "validate"]
            found_skb_terms = [term for term in skb_related if term in content_lower]
            assert len(found_skb_terms) >= 2, (
                f"Top result should be SKB-related: {content_lower}"
            )

    async def test_buffer_overflow_search_vulnerability_patterns(
        self, search_engine, indexed_content
    ):
        """Test detection of various buffer overflow vulnerability patterns."""
        test_queries = [
            ("packet size exceeds buffer", "buffer_bounds_check"),
            ("invalid SKB buffer length", "skb_length_validation"),
            ("received packet length exceeds", "buffer_overflow_protection"),
            ("length exceeds max buffer size", "rx_buffer_overflow_check"),
            ("no bounds checking", "missing_bounds_check"),
        ]

        for query, expected_pattern in test_queries:
            results = await search_engine["vector_search"].search(
                query=query, max_results=3, min_confidence=0.5
            )

            # Should find results relevant to the pattern
            if results:
                # Check if any result contains the expected vulnerability pattern
                found_pattern = False
                for result in results:
                    if (
                        "metadata" in result
                        and "vulnerability_pattern" in result["metadata"]
                    ):
                        if (
                            result["metadata"]["vulnerability_pattern"]
                            == expected_pattern
                        ):
                            found_pattern = True
                            break

                # At least one result should match the expected pattern for high-confidence queries
                if any(r["confidence"] > 0.8 for r in results):
                    assert found_pattern, (
                        f"High-confidence results should include pattern {expected_pattern} for query '{query}'"
                    )

    async def test_buffer_overflow_search_error_handling(self, search_engine):
        """Test error handling for buffer overflow searches."""
        # Test empty query
        with pytest.raises((ValueError, TypeError)):
            await search_engine["vector_search"].search(query="")

        # Test query too long
        with pytest.raises(ValueError):
            await search_engine["vector_search"].search(query="x" * 1001)

        # Test invalid file patterns
        with pytest.raises(ValueError):
            await search_engine["vector_search"].search(
                query="buffer overflow",
                file_patterns=[""],  # Empty pattern should fail
            )

    async def test_buffer_overflow_search_performance_under_load(
        self, search_engine, indexed_content
    ):
        """Test buffer overflow search performance under concurrent load."""
        query = "buffer overflow in network drivers"

        # Run multiple concurrent searches
        async def single_search():
            start_time = time.time()
            results = await search_engine["vector_search"].search(
                query=query, max_results=5
            )
            end_time = time.time()
            return (end_time - start_time) * 1000, len(results)

        # Execute concurrent searches
        tasks = [single_search() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        search_times = [r[0] for r in results]
        result_counts = [r[1] for r in results]

        # All searches should complete within performance limit
        max_time = max(search_times)
        assert max_time < 600, f"Slowest search took {max_time}ms, exceeds 600ms limit"

        # All searches should return results
        min_results = min(result_counts)
        assert min_results > 0, "All searches should return at least one result"

    async def test_buffer_overflow_mcp_integration(
        self, search_engine, indexed_content
    ):
        """Test buffer overflow search through MCP interface."""
        # This will fail until MCP integration exists
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Test the exact scenario mentioned in quickstart.md
        result = await semantic_search(
            query="buffer overflow vulnerabilities in network drivers",
            max_results=5,
            min_confidence=0.7,
            content_types=["SOURCE_CODE"],
            file_patterns=["/drivers/net/*"],
        )

        # Validate MCP response structure
        assert "query_id" in result
        assert "results" in result
        assert "search_stats" in result

        # Validate results match expected format from quickstart.md
        assert len(result["results"]) >= 2
        assert all(r["confidence"] >= 0.7 for r in result["results"])
        assert result["search_stats"]["search_time_ms"] < 600

        # Results should match the expected pattern from quickstart.md
        expected_files = ["r8169.c", "recv.c"]  # From the example output
        result_files = [r["file_path"] for r in result["results"]]
        matching_files = [
            ef for ef in expected_files if any(ef in rf for rf in result_files)
        ]

        # Should find at least one of the expected file types
        assert len(matching_files) > 0 or len(result["results"]) > 0, (
            "Should find results matching quickstart.md examples or other relevant results"
        )

    async def test_buffer_overflow_search_metadata_extraction(
        self, search_engine, indexed_content
    ):
        """Test metadata extraction for buffer overflow search results."""
        query = "buffer bounds checking in packet processing"

        results = await search_engine["vector_search"].search(
            query=query, max_results=5
        )

        # Results should have proper metadata
        for result in results:
            # Basic required fields
            assert "file_path" in result
            assert "line_start" in result
            assert "line_end" in result
            assert "content" in result
            assert "confidence" in result
            assert "content_type" in result

            # Network driver files should have relevant metadata
            if "/drivers/net/" in result["file_path"] and "metadata" in result:
                metadata = result["metadata"]

                if "function_name" in metadata:
                    # Function names should be extracted
                    assert isinstance(metadata["function_name"], str)
                    assert len(metadata["function_name"]) > 0

                if "symbols" in metadata:
                    # Should extract network-related symbols
                    assert isinstance(metadata["symbols"], list)

                if "vulnerability_pattern" in metadata:
                    # Should identify vulnerability patterns
                    assert isinstance(metadata["vulnerability_pattern"], str)
                    pattern_keywords = [
                        "buffer",
                        "bounds",
                        "check",
                        "validation",
                        "overflow",
                    ]
                    pattern_lower = metadata["vulnerability_pattern"].lower()
                    found_pattern_keywords = [
                        kw for kw in pattern_keywords if kw in pattern_lower
                    ]
                    assert len(found_pattern_keywords) > 0, (
                        f"Vulnerability pattern should contain relevant keywords: {metadata['vulnerability_pattern']}"
                    )
