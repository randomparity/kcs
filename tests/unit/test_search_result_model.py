"""
Unit tests for SearchResult model.

Tests the SearchResult data model as defined in
specs/008-semantic-search-engine/data-model.md

Following TDD: This test MUST FAIL before implementation exists.
"""

from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestSearchResultModel:
    """Unit tests for SearchResult model."""

    def test_search_result_model_exists(self):
        """Test that SearchResult model class exists and is importable."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Should be a class that can be instantiated
        assert isinstance(SearchResult, type)

    def test_search_result_basic_creation(self):
        """Test basic SearchResult creation with required fields."""
        from src.python.semantic_search.models.search_result import SearchResult

        result = SearchResult(
            query_id="query_123",
            content_id="content_456",
            similarity_score=0.85,
            bm25_score=0.72,
            combined_score=0.809,  # (0.7 * 0.85) + (0.3 * 0.72)
            confidence=0.91,
        )

        # Required fields should be set
        assert result.query_id == "query_123"
        assert result.content_id == "content_456"
        assert result.similarity_score == 0.85
        assert result.bm25_score == 0.72
        assert result.combined_score == 0.809
        assert result.confidence == 0.91

    def test_search_result_field_types(self):
        """Test that SearchResult fields have correct types."""
        from src.python.semantic_search.models.search_result import SearchResult

        result = SearchResult(
            query_id="test_query",
            content_id="test_content",
            similarity_score=0.8,
            bm25_score=0.6,
            combined_score=0.74,
            confidence=0.85,
            context_lines=["previous line", "matching line", "next line"],
            explanation="This matches because it contains memory allocation patterns",
        )

        # Type validations
        assert isinstance(result.query_id, str)
        assert isinstance(result.content_id, str)
        assert isinstance(result.similarity_score, (int, float))
        assert isinstance(result.bm25_score, (int, float))
        assert isinstance(result.combined_score, (int, float))
        assert isinstance(result.confidence, (int, float))
        assert isinstance(result.context_lines, list)
        assert isinstance(result.explanation, str)

    def test_search_result_score_validation(self):
        """Test that all scores are in valid range 0.0-1.0."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Valid scores should work
        valid_scores = [
            (0.0, 0.0, 0.0, 0.0),  # Minimum values
            (1.0, 1.0, 1.0, 1.0),  # Maximum values
            (0.5, 0.5, 0.5, 0.5),  # Middle values
            (0.85, 0.72, 0.809, 0.91),  # Realistic values
        ]

        for sim, bm25, combined, conf in valid_scores:
            result = SearchResult(
                query_id="test",
                content_id="test",
                similarity_score=sim,
                bm25_score=bm25,
                combined_score=combined,
                confidence=conf,
            )
            assert 0.0 <= result.similarity_score <= 1.0
            assert 0.0 <= result.bm25_score <= 1.0
            assert 0.0 <= result.combined_score <= 1.0
            assert 0.0 <= result.confidence <= 1.0

        # Invalid scores should fail
        invalid_score_sets = [
            (-0.1, 0.5, 0.5, 0.5),  # Negative similarity
            (0.5, -0.1, 0.5, 0.5),  # Negative bm25
            (0.5, 0.5, -0.1, 0.5),  # Negative combined
            (0.5, 0.5, 0.5, -0.1),  # Negative confidence
            (1.1, 0.5, 0.5, 0.5),  # similarity > 1.0
            (0.5, 1.1, 0.5, 0.5),  # bm25 > 1.0
            (0.5, 0.5, 1.1, 0.5),  # combined > 1.0
            (0.5, 0.5, 0.5, 1.1),  # confidence > 1.0
        ]

        for sim, bm25, combined, conf in invalid_score_sets:
            with pytest.raises(ValueError):
                SearchResult(
                    query_id="test",
                    content_id="test",
                    similarity_score=sim,
                    bm25_score=bm25,
                    combined_score=combined,
                    confidence=conf,
                )

    def test_search_result_combined_score_calculation(self):
        """Test combined_score calculation: (0.7 * similarity) + (0.3 * bm25)."""
        from src.python.semantic_search.models.search_result import SearchResult

        test_cases = [
            (0.8, 0.6, 0.74),  # (0.7 * 0.8) + (0.3 * 0.6) = 0.56 + 0.18 = 0.74
            (1.0, 0.0, 0.7),  # (0.7 * 1.0) + (0.3 * 0.0) = 0.7
            (0.0, 1.0, 0.3),  # (0.7 * 0.0) + (0.3 * 1.0) = 0.3
            (0.5, 0.5, 0.5),  # (0.7 * 0.5) + (0.3 * 0.5) = 0.35 + 0.15 = 0.5
        ]

        for similarity, bm25, expected_combined in test_cases:
            result = SearchResult(
                query_id="test",
                content_id="test",
                similarity_score=similarity,
                bm25_score=bm25,
                combined_score=expected_combined,
                confidence=0.8,
            )

            # If auto-calculation is implemented, test it
            if hasattr(result, "calculate_combined_score"):
                calculated = result.calculate_combined_score()
                assert abs(calculated - expected_combined) < 1e-6

            # Or if it's validated on construction
            assert abs(result.combined_score - expected_combined) < 1e-6

    def test_search_result_auto_combined_score(self):
        """Test automatic combined_score calculation if not provided."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Test if combined_score is auto-calculated when not provided
        try:
            result = SearchResult(
                query_id="test",
                content_id="test",
                similarity_score=0.8,
                bm25_score=0.6,
                confidence=0.75,
                # combined_score not provided
            )

            # Should auto-calculate: (0.7 * 0.8) + (0.3 * 0.6) = 0.74
            expected_combined = (0.7 * 0.8) + (0.3 * 0.6)
            assert abs(result.combined_score - expected_combined) < 1e-6

        except TypeError:
            # If combined_score is required, that's also valid
            pass

    def test_search_result_context_lines_validation(self):
        """Test context_lines validation (max 10 lines)."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Valid context lines
        valid_context_sets = [
            [],  # Empty
            ["single line"],  # Single line
            ["line 1", "line 2", "line 3"],  # Multiple lines
            ["line " + str(i) for i in range(10)],  # Maximum 10 lines
        ]

        for context_lines in valid_context_sets:
            result = SearchResult(
                query_id="test",
                content_id="test",
                similarity_score=0.8,
                bm25_score=0.6,
                combined_score=0.74,
                confidence=0.8,
                context_lines=context_lines,
            )
            assert result.context_lines == context_lines
            assert len(result.context_lines) <= 10

        # Too many context lines should fail
        too_many_lines = ["line " + str(i) for i in range(11)]  # 11 lines
        with pytest.raises(ValueError):
            SearchResult(
                query_id="test",
                content_id="test",
                similarity_score=0.8,
                bm25_score=0.6,
                combined_score=0.74,
                confidence=0.8,
                context_lines=too_many_lines,
            )

    def test_search_result_explanation_generation(self):
        """Test explanation field behavior."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Explicit explanation
        explicit_explanation = (
            "This result matches the query about memory allocation functions"
        )
        result = SearchResult(
            query_id="test",
            content_id="test",
            similarity_score=0.8,
            bm25_score=0.6,
            combined_score=0.74,
            confidence=0.8,
            explanation=explicit_explanation,
        )
        assert result.explanation == explicit_explanation

        # Auto-generated explanation (if supported)
        result_auto = SearchResult(
            query_id="test",
            content_id="test",
            similarity_score=0.8,
            bm25_score=0.6,
            combined_score=0.74,
            confidence=0.8,
            # explanation not provided
        )

        # Should either have auto-generated explanation or require explicit one
        if hasattr(result_auto, "explanation"):
            assert isinstance(result_auto.explanation, str)
            assert len(result_auto.explanation) > 0

    def test_search_result_confidence_derivation(self):
        """Test confidence derivation from distance metrics and context."""
        from src.python.semantic_search.models.search_result import SearchResult

        # High similarity and BM25 should yield high confidence
        high_quality = SearchResult(
            query_id="test",
            content_id="test",
            similarity_score=0.95,
            bm25_score=0.90,
            combined_score=0.935,
            confidence=0.92,
        )

        # Low scores should yield lower confidence
        low_quality = SearchResult(
            query_id="test",
            content_id="test",
            similarity_score=0.45,
            bm25_score=0.40,
            combined_score=0.435,
            confidence=0.48,
        )

        # Confidence should generally correlate with combined score
        assert high_quality.confidence > low_quality.confidence

        # Test auto-confidence calculation if implemented
        if hasattr(SearchResult, "calculate_confidence"):
            calculated_conf = SearchResult.calculate_confidence(
                similarity_score=0.8, bm25_score=0.7, context_quality=0.9
            )
            assert 0.0 <= calculated_conf <= 1.0

    def test_search_result_relationships(self):
        """Test relationships with SearchQuery and VectorEmbedding."""
        from src.python.semantic_search.models.search_result import SearchResult

        result = SearchResult(
            query_id="query_abc123",
            content_id="content_def456",
            similarity_score=0.8,
            bm25_score=0.7,
            combined_score=0.77,
            confidence=0.82,
        )

        # Should reference other entities
        assert result.query_id == "query_abc123"
        assert result.content_id == "content_def456"

        # Should support relationship methods
        if hasattr(result, "get_query"):
            query = result.get_query()
            assert query is not None

        if hasattr(result, "get_content"):
            content = result.get_content()
            assert content is not None

    def test_search_result_serialization(self):
        """Test serialization and deserialization of SearchResult."""
        from src.python.semantic_search.models.search_result import SearchResult

        original_result = SearchResult(
            query_id="serialization_test",
            content_id="content_serialize",
            similarity_score=0.87,
            bm25_score=0.64,
            combined_score=0.801,
            confidence=0.89,
            context_lines=[
                "int alloc_pages(gfp_t gfp, unsigned int order)",
                "{",
                "    // allocation logic",
            ],
            explanation="Memory allocation function matching the query pattern",
        )

        # Should be serializable to dict
        result_dict = original_result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["query_id"] == "serialization_test"
        assert result_dict["content_id"] == "content_serialize"
        assert result_dict["similarity_score"] == 0.87
        assert result_dict["context_lines"] == original_result.context_lines

        # Should be deserializable from dict
        restored_result = SearchResult.from_dict(result_dict)
        assert restored_result.query_id == original_result.query_id
        assert restored_result.content_id == original_result.content_id
        assert restored_result.similarity_score == original_result.similarity_score
        assert restored_result.context_lines == original_result.context_lines

    def test_search_result_comparison(self):
        """Test comparison and sorting of SearchResult instances."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Create results with different scores
        high_result = SearchResult(
            query_id="test",
            content_id="high",
            similarity_score=0.9,
            bm25_score=0.8,
            combined_score=0.86,
            confidence=0.92,
        )

        low_result = SearchResult(
            query_id="test",
            content_id="low",
            similarity_score=0.6,
            bm25_score=0.5,
            combined_score=0.57,
            confidence=0.62,
        )

        # Should support comparison by combined_score
        assert high_result > low_result
        assert low_result < high_result
        assert high_result != low_result

        # Should support sorting
        results = [low_result, high_result]
        sorted_results = sorted(results, reverse=True)  # Highest first
        assert sorted_results[0] == high_result
        assert sorted_results[1] == low_result

    def test_search_result_immutability(self):
        """Test that core fields are immutable after creation."""
        from src.python.semantic_search.models.search_result import SearchResult

        result = SearchResult(
            query_id="immutable_test",
            content_id="test_content",
            similarity_score=0.8,
            bm25_score=0.7,
            combined_score=0.77,
            confidence=0.8,
        )

        original_query_id = result.query_id
        original_similarity = result.similarity_score

        # Attempting to modify should fail or be ignored
        with pytest.raises((AttributeError, TypeError)):
            result.query_id = "modified_query"

        with pytest.raises((AttributeError, TypeError)):
            result.similarity_score = 0.9

        # Values should remain unchanged
        assert result.query_id == original_query_id
        assert result.similarity_score == original_similarity

    def test_search_result_validation_edge_cases(self):
        """Test edge cases in SearchResult validation."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Empty strings should be validated
        with pytest.raises(ValueError):
            SearchResult(
                query_id="",  # Empty query_id
                content_id="test",
                similarity_score=0.8,
                bm25_score=0.7,
                combined_score=0.77,
                confidence=0.8,
            )

        with pytest.raises(ValueError):
            SearchResult(
                query_id="test",
                content_id="",  # Empty content_id
                similarity_score=0.8,
                bm25_score=0.7,
                combined_score=0.77,
                confidence=0.8,
            )

        # None values should fail
        with pytest.raises((ValueError, TypeError)):
            SearchResult(
                query_id=None,
                content_id="test",
                similarity_score=0.8,
                bm25_score=0.7,
                combined_score=0.77,
                confidence=0.8,
            )

    def test_search_result_context_lines_types(self):
        """Test that context_lines contains only strings."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Valid string context lines
        valid_context = ["line 1", "line 2", "line 3"]
        result = SearchResult(
            query_id="test",
            content_id="test",
            similarity_score=0.8,
            bm25_score=0.7,
            combined_score=0.77,
            confidence=0.8,
            context_lines=valid_context,
        )
        assert result.context_lines == valid_context

        # Invalid non-string context lines should fail
        invalid_contexts = [
            [123, "line 2"],  # Number in list
            ["line 1", None],  # None in list
            [{"line": 1}],  # Dict in list
            "not a list",  # Not a list at all
        ]

        for invalid_context in invalid_contexts:
            with pytest.raises((ValueError, TypeError)):
                SearchResult(
                    query_id="test",
                    content_id="test",
                    similarity_score=0.8,
                    bm25_score=0.7,
                    combined_score=0.77,
                    confidence=0.8,
                    context_lines=invalid_context,
                )

    def test_search_result_performance_ranking(self):
        """Test that SearchResult supports performance-oriented ranking."""
        from src.python.semantic_search.models.search_result import SearchResult

        # Create multiple results
        results = []
        for i in range(10):
            similarity = 0.5 + (i * 0.05)  # 0.5 to 0.95
            bm25 = 0.4 + (i * 0.04)  # 0.4 to 0.76
            combined = (0.7 * similarity) + (0.3 * bm25)
            confidence = min(0.95, 0.5 + (i * 0.05))

            result = SearchResult(
                query_id="perf_test",
                content_id=f"content_{i}",
                similarity_score=similarity,
                bm25_score=bm25,
                combined_score=combined,
                confidence=confidence,
            )
            results.append(result)

        # Should be sortable by combined_score
        sorted_results = sorted(results, key=lambda r: r.combined_score, reverse=True)

        # Verify ranking order
        for i in range(len(sorted_results) - 1):
            assert (
                sorted_results[i].combined_score >= sorted_results[i + 1].combined_score
            )

    def test_search_result_memory_efficiency(self):
        """Test memory efficiency of SearchResult instances."""
        import sys

        from src.python.semantic_search.models.search_result import SearchResult

        result = SearchResult(
            query_id="memory_test",
            content_id="test_content",
            similarity_score=0.8,
            bm25_score=0.7,
            combined_score=0.77,
            confidence=0.8,
            context_lines=["line 1", "line 2", "line 3"],
            explanation="Test explanation for memory efficiency",
        )

        # Should be reasonably sized
        result_size = sys.getsizeof(result)

        # Should be less than 5KB
        assert result_size < 5000
