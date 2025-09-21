"""
Integration test for memory allocation search scenario.

This test validates the complete end-to-end functionality of semantic search
for memory allocation queries as specified in quickstart.md.

Scenario: User searches for "memory allocation functions that can fail"
Expected: Returns kmalloc, vmalloc, kzalloc functions with high confidence

Following TDD: This test MUST FAIL before implementation exists.
"""

import asyncio
import time
from typing import Any

import pytest


class TestMemoryAllocationSearchScenario:
    """Integration tests for memory allocation search scenarios."""

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
    def sample_memory_allocation_content(self) -> list[dict[str, Any]]:
        """Sample kernel memory allocation code for testing."""
        return [
            {
                "file_path": "/kernel/mm/slab.c",
                "line_start": 3421,
                "line_end": 3425,
                "content": """
static inline void *kmalloc(size_t size, gfp_t flags)
{
    if (__builtin_constant_p(size) && size > KMALLOC_MAX_CACHE_SIZE)
        return kmalloc_large(size, flags);
    return __kmalloc(size, flags);
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_SLAB"],
            },
            {
                "file_path": "/kernel/mm/vmalloc.c",
                "line_start": 2567,
                "line_end": 2572,
                "content": """
void *vmalloc(unsigned long size)
{
    return __vmalloc_node(size, 1, GFP_KERNEL, NUMA_NO_NODE,
                         __builtin_return_address(0));
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_MMU"],
            },
            {
                "file_path": "/kernel/mm/slab.c",
                "line_start": 3500,
                "line_end": 3504,
                "content": """
static inline void *kzalloc(size_t size, gfp_t flags)
{
    return kmalloc(size, flags | __GFP_ZERO);
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_SLAB"],
            },
            {
                "file_path": "/kernel/mm/page_alloc.c",
                "line_start": 5123,
                "line_end": 5128,
                "content": """
struct page *alloc_pages(gfp_t gfp, unsigned int order)
{
    struct page *page = alloc_pages_current(gfp, order);
    if (unlikely(page == NULL))
        return NULL;
    return page;
}
                """.strip(),
                "content_type": "SOURCE_CODE",
                "config_guards": ["CONFIG_NUMA"],
            },
        ]

    @pytest.fixture
    async def indexed_content(self, search_engine, sample_memory_allocation_content):
        """Index sample content for testing."""
        # This will fail until indexing implementation exists
        from src.python.semantic_search.services.indexing_service import IndexingService

        indexing_service = IndexingService(
            search_engine["embedding"], search_engine["vector_search"]
        )

        # Index the sample content
        for content in sample_memory_allocation_content:
            await indexing_service.index_content(
                file_path=content["file_path"],
                content=content["content"],
                line_start=content["line_start"],
                line_end=content["line_end"],
                content_type=content["content_type"],
                config_guards=content["config_guards"],
            )

        return True

    async def test_basic_memory_allocation_search(self, search_engine, indexed_content):
        """Test basic memory allocation search functionality."""
        # This is the core scenario from quickstart.md
        query = "memory allocation functions that can fail"

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

        # Success criteria from quickstart.md
        assert len(results) >= 3, "Should return at least 3 results"

        # All results should have confidence > 0.5
        for result in results:
            assert result["confidence"] > 0.5, (
                f"Result confidence {result['confidence']} below 0.5"
            )

        # Results should include memory allocation functions
        result_content = " ".join([r["content"] for r in results])
        expected_functions = ["kmalloc", "vmalloc", "kzalloc", "alloc_pages"]
        found_functions = [
            func for func in expected_functions if func in result_content
        ]

        assert len(found_functions) >= 2, (
            f"Expected memory allocation functions, found: {found_functions}"
        )

    async def test_memory_allocation_search_with_confidence_threshold(
        self, search_engine, indexed_content
    ):
        """Test memory allocation search with higher confidence threshold."""
        query = "allocate memory that might fail"

        # Search with higher confidence threshold
        results = await search_engine["vector_search"].search(
            query=query, max_results=10, min_confidence=0.7
        )

        # All results should meet the confidence threshold
        for result in results:
            assert result["confidence"] >= 0.7

        # Should still find relevant allocation functions
        result_files = [r["file_path"] for r in results]
        memory_management_files = [f for f in result_files if "/mm/" in f]
        assert len(memory_management_files) > 0, (
            "Should find results in memory management code"
        )

    async def test_memory_allocation_search_semantic_ranking(
        self, search_engine, indexed_content
    ):
        """Test that results are properly ranked by semantic relevance."""
        query = "kernel memory allocator with failure handling"

        results = await search_engine["vector_search"].search(
            query=query, max_results=5, min_confidence=0.4
        )

        # Results should be ordered by confidence/relevance
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True), (
            "Results should be ordered by confidence"
        )

        # Top result should be highly relevant
        if results:
            top_result = results[0]
            assert top_result["confidence"] > 0.6, (
                "Top result should have high confidence"
            )

            # Should contain allocation-related keywords
            content_lower = top_result["content"].lower()
            allocation_keywords = ["alloc", "malloc", "kmalloc", "vmalloc", "kzalloc"]
            found_keywords = [kw for kw in allocation_keywords if kw in content_lower]
            assert len(found_keywords) > 0, (
                f"Top result should contain allocation keywords, content: {content_lower}"
            )

    async def test_memory_allocation_search_file_filtering(
        self, search_engine, indexed_content
    ):
        """Test memory allocation search with file pattern filtering."""
        query = "memory allocation functions"

        # Filter to only memory management files
        results = await search_engine["vector_search"].search(
            query=query, max_results=10, file_patterns=["/kernel/mm/*"]
        )

        # All results should be from memory management subsystem
        for result in results:
            assert "/mm/" in result["file_path"], (
                f"Result {result['file_path']} not from mm subsystem"
            )

    async def test_memory_allocation_search_config_awareness(
        self, search_engine, indexed_content
    ):
        """Test memory allocation search with configuration context."""
        query = "slab memory allocation"

        # Search with SLAB configuration context
        results = await search_engine["vector_search"].search(
            query=query, config_context=["CONFIG_SLAB"]
        )

        # Results should respect configuration context
        for result in results:
            if "metadata" in result and "config_guards" in result["metadata"]:
                config_guards = result["metadata"]["config_guards"]
                assert "CONFIG_SLAB" in config_guards, (
                    f"Result should have CONFIG_SLAB context: {config_guards}"
                )

    async def test_memory_allocation_search_error_handling(self, search_engine):
        """Test error handling for memory allocation searches."""
        # Test empty query
        with pytest.raises((ValueError, TypeError)):
            await search_engine["vector_search"].search(query="")

        # Test invalid max_results
        with pytest.raises(ValueError):
            await search_engine["vector_search"].search(
                query="memory allocation", max_results=0
            )

        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            await search_engine["vector_search"].search(
                query="memory allocation", min_confidence=1.5
            )

    async def test_memory_allocation_search_performance_under_load(
        self, search_engine, indexed_content
    ):
        """Test memory allocation search performance under concurrent load."""
        query = "memory allocation functions that can fail"

        # Run multiple concurrent searches
        async def single_search():
            start_time = time.time()
            results = await search_engine["vector_search"].search(
                query=query, max_results=5
            )
            end_time = time.time()
            return (end_time - start_time) * 1000, len(results)

        # Execute 10 concurrent searches
        tasks = [single_search() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        search_times = [r[0] for r in results]
        result_counts = [r[1] for r in results]

        # All searches should complete within performance limit
        max_time = max(search_times)
        assert max_time < 600, f"Slowest search took {max_time}ms, exceeds 600ms limit"

        # All searches should return results
        min_results = min(result_counts)
        assert min_results > 0, "All searches should return at least one result"

        # Performance should be consistent (95th percentile test)
        p95_time = sorted(search_times)[int(0.95 * len(search_times))]
        assert p95_time <= 600, (
            f"95th percentile time {p95_time}ms exceeds 600ms requirement"
        )

    async def test_memory_allocation_mcp_integration(
        self, search_engine, indexed_content
    ):
        """Test memory allocation search through MCP interface."""
        # This will fail until MCP integration exists
        from src.python.semantic_search.mcp.search_tool import semantic_search

        # Test the exact scenario from quickstart.md
        result = await semantic_search(
            query="memory allocation functions that can fail",
            max_results=3,
            min_confidence=0.5,
        )

        # Validate MCP response structure
        assert "query_id" in result
        assert "results" in result
        assert "search_stats" in result

        # Validate success criteria
        assert len(result["results"]) >= 3
        assert all(r["confidence"] > 0.5 for r in result["results"])
        assert result["search_stats"]["search_time_ms"] < 600

        # Results should include memory allocation functions
        result_content = " ".join([r["content"] for r in result["results"]])
        memory_allocation_indicators = ["alloc", "malloc", "kmalloc", "vmalloc"]
        found_indicators = [
            ind for ind in memory_allocation_indicators if ind in result_content.lower()
        ]
        assert len(found_indicators) > 0, (
            f"Should find memory allocation indicators: {found_indicators}"
        )

    async def test_memory_allocation_search_metadata_extraction(
        self, search_engine, indexed_content
    ):
        """Test that memory allocation search properly extracts metadata."""
        query = "kmalloc memory allocation"

        results = await search_engine["vector_search"].search(
            query=query, max_results=5
        )

        # Results should have proper metadata
        for result in results:
            # Should have basic required fields
            assert "file_path" in result
            assert "line_start" in result
            assert "line_end" in result
            assert "content" in result
            assert "confidence" in result
            assert "content_type" in result

            # Should have metadata with function information
            if "metadata" in result:
                metadata = result["metadata"]
                if "function_name" in metadata:
                    # Function name should be extracted for code results
                    assert isinstance(metadata["function_name"], str)
                    assert len(metadata["function_name"]) > 0

                if "symbols" in metadata:
                    # Should extract relevant symbols
                    assert isinstance(metadata["symbols"], list)

                if "config_guards" in metadata:
                    # Configuration guards should be valid
                    assert isinstance(metadata["config_guards"], list)
                    for guard in metadata["config_guards"]:
                        assert guard.startswith("CONFIG_")
