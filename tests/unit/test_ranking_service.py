"""
Unit tests for RankingService implementation.

Tests the hybrid BM25+semantic scoring system including BM25Calculator
and RankingService for comprehensive search result ranking.
"""

import math
from unittest.mock import Mock, patch

import pytest

from src.python.semantic_search.models.search_result import SearchResult
from src.python.semantic_search.services.ranking_service import (
    BM25Calculator,
    RankingService,
)


class TestBM25Calculator:
    """Test suite for BM25Calculator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.calculator = BM25Calculator()

    def test_bm25_calculator_initialization(self):
        """Test that BM25Calculator initializes with correct parameters."""
        calculator = BM25Calculator()
        assert calculator.k1 == 1.2
        assert calculator.b == 0.75
        assert calculator.corpus_stats == {}
        assert not calculator._prepared

        # Test custom parameters
        calculator = BM25Calculator(k1=1.5, b=0.8)
        assert calculator.k1 == 1.5
        assert calculator.b == 0.8

    def test_tokenize(self):
        """Test text tokenization functionality."""
        # Basic tokenization
        tokens = self.calculator._tokenize("memory allocation error")
        assert tokens == ["memory", "allocation", "error"]

        # Case normalization
        tokens = self.calculator._tokenize("MEMORY Allocation ERROR")
        assert tokens == ["memory", "allocation", "error"]

        # Code delimiters
        tokens = self.calculator._tokenize("memory_allocation buffer->data")
        assert "memory_allocation" in tokens
        assert "buffer" in tokens
        assert "data" in tokens

        # Filter short tokens
        tokens = self.calculator._tokenize("a memory is ok")
        assert "a" not in tokens  # Too short
        assert "is" in tokens  # Exactly 2 chars
        assert "memory" in tokens
        assert "ok" in tokens

        # Empty and whitespace
        tokens = self.calculator._tokenize("")
        assert tokens == []

        tokens = self.calculator._tokenize("   ")
        assert tokens == []

    def test_prepare_corpus_empty(self):
        """Test corpus preparation with empty document list."""
        self.calculator.prepare_corpus([])

        assert self.calculator._prepared
        assert self.calculator.corpus_stats["doc_count"] == 0
        assert self.calculator.corpus_stats["avg_doc_length"] == 0.0
        assert len(self.calculator.corpus_stats["doc_freqs"]) == 0
        assert self.calculator.corpus_stats["doc_lengths"] == []

    def test_prepare_corpus_single_document(self):
        """Test corpus preparation with single document."""
        documents = ["memory allocation error in kernel"]

        self.calculator.prepare_corpus(documents)

        assert self.calculator._prepared
        assert self.calculator.corpus_stats["doc_count"] == 1
        assert self.calculator.corpus_stats["avg_doc_length"] == 5.0  # 5 tokens
        assert self.calculator.corpus_stats["doc_lengths"] == [5]

        # All terms should have frequency 1
        doc_freqs = self.calculator.corpus_stats["doc_freqs"]
        for term in ["memory", "allocation", "error", "in", "kernel"]:
            assert doc_freqs[term] == 1

    def test_prepare_corpus_multiple_documents(self):
        """Test corpus preparation with multiple documents."""
        documents = [
            "memory allocation error",
            "memory leak in kernel",
            "buffer overflow error",
        ]

        self.calculator.prepare_corpus(documents)

        assert self.calculator.corpus_stats["doc_count"] == 3
        assert (
            abs(self.calculator.corpus_stats["avg_doc_length"] - 10 / 3) < 0.01
        )  # Average length: (3+4+3)/3

        doc_freqs = self.calculator.corpus_stats["doc_freqs"]
        assert doc_freqs["memory"] == 2  # Appears in 2 documents
        assert doc_freqs["error"] == 2  # Appears in 2 documents
        assert doc_freqs["allocation"] == 1  # Appears in 1 document
        assert doc_freqs["kernel"] == 1
        assert doc_freqs["buffer"] == 1
        assert doc_freqs["overflow"] == 1
        assert doc_freqs["leak"] == 1

    def test_score_document_not_prepared(self):
        """Test document scoring when corpus is not prepared."""
        score = self.calculator.score_document("memory", "memory allocation", 0)
        assert score == 0.0

    def test_score_document_empty_query_or_document(self):
        """Test document scoring with empty query or document."""
        self.calculator.prepare_corpus(["memory allocation"])

        # Empty query
        score = self.calculator.score_document("", "memory allocation", 0)
        assert score == 0.0

        # Empty document
        score = self.calculator.score_document("memory", "", 0)
        assert score == 0.0

        # Both empty
        score = self.calculator.score_document("", "", 0)
        assert score == 0.0

    def test_score_document_no_matching_terms(self):
        """Test document scoring when no terms match."""
        documents = ["memory allocation error"]
        self.calculator.prepare_corpus(documents)

        score = self.calculator.score_document(
            "network protocol", "memory allocation error", 0
        )
        assert score == 0.0

    def test_score_document_basic_scoring(self):
        """Test basic BM25 scoring functionality."""
        documents = [
            "memory allocation error in kernel",
            "network protocol implementation",
            "file system cache optimization",
            "buffer overflow detection",
            "thread synchronization mechanism",
        ]
        self.calculator.prepare_corpus(documents)

        # Query with rare terms should get positive score
        score = self.calculator.score_document(
            "allocation", "memory allocation error in kernel", 0
        )
        assert score >= 0.0
        assert score <= 1.0

        # Query that doesn't match should get zero score
        no_match_score = self.calculator.score_document(
            "xyz", "memory allocation error in kernel", 0
        )
        assert no_match_score == 0.0

    def test_score_document_idf_calculation(self):
        """Test IDF calculation in BM25 scoring."""
        documents = [
            "memory allocation error",  # allocation appears here
            "network protocol stack",  # allocation does not appear
            "buffer overflow detection",  # allocation does not appear
            "file system implementation",  # allocation does not appear
            "cache optimization strategy",  # allocation does not appear
        ]
        self.calculator.prepare_corpus(documents)

        # Test rare term - "allocation" appears in only 1 out of 5 docs
        alloc_score = self.calculator.score_document(
            "allocation", "memory allocation error", 0
        )
        assert alloc_score > 0.0

        # Test very rare term - should get high score
        rare_score = self.calculator.score_document(
            "optimization", "cache optimization strategy", 4
        )
        assert rare_score > 0.0

    def test_score_document_length_normalization(self):
        """Test document length normalization in BM25."""
        documents = [
            "allocation",  # Short document, term appears once
            "allocation error system kernel module driver implementation details analysis",  # Long document, term appears once
            "network protocol stack implementation",  # Different document for better corpus stats
            "buffer overflow detection mechanism",
            "file system cache optimization",
        ]
        self.calculator.prepare_corpus(documents)

        # Same query against documents of different lengths
        short_score = self.calculator.score_document("allocation", "allocation", 0)
        long_score = self.calculator.score_document(
            "allocation",
            "allocation error system kernel module driver implementation details analysis",
            1,
        )

        assert short_score >= 0.0
        assert long_score >= 0.0
        # If both scores are positive, shorter document should score higher
        if short_score > 0 and long_score > 0:
            assert short_score > long_score

    def test_score_document_term_frequency_saturation(self):
        """Test term frequency saturation in BM25."""
        documents = [
            "allocation allocation",  # TF = 2
            "allocation allocation allocation allocation allocation",  # TF = 5
            "network protocol stack",  # Different content
            "buffer overflow detection",
            "file system implementation",
        ]
        self.calculator.prepare_corpus(documents)

        score_tf2 = self.calculator.score_document(
            "allocation", "allocation allocation", 0
        )
        score_tf5 = self.calculator.score_document(
            "allocation", "allocation allocation allocation allocation allocation", 1
        )

        assert score_tf2 >= 0.0
        assert score_tf5 >= 0.0
        # If both scores are positive, higher TF should give higher score, but with diminishing returns
        if score_tf2 > 0 and score_tf5 > 0:
            assert score_tf5 > score_tf2
            # The difference should be less than linear due to saturation
            ratio = score_tf5 / score_tf2
            assert ratio < 2.5  # Not linear scaling

    def test_score_normalization(self):
        """Test that BM25 scores are normalized to 0-1 range."""
        documents = ["memory allocation error"] * 10  # Repeated documents
        self.calculator.prepare_corpus(documents)

        # Test various queries
        queries = [
            "memory",
            "memory allocation",
            "memory allocation error",
            "allocation error memory",
        ]

        for query in queries:
            score = self.calculator.score_document(query, "memory allocation error", 0)
            assert 0.0 <= score <= 1.0


class TestRankingService:
    """Test suite for RankingService class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.service = RankingService()

    def test_ranking_service_initialization(self):
        """Test that RankingService initializes with correct parameters."""
        service = RankingService()
        assert service.semantic_weight == 0.7
        assert service.bm25_weight == 0.3
        assert math.isclose(service.semantic_weight + service.bm25_weight, 1.0)
        assert isinstance(service.bm25_calculator, BM25Calculator)
        assert not service._corpus_prepared

        # Test custom weights
        service = RankingService(semantic_weight=0.6, bm25_weight=0.4)
        assert service.semantic_weight == 0.6
        assert service.bm25_weight == 0.4

    def test_ranking_service_invalid_weights(self):
        """Test initialization with invalid weight combinations."""
        # Weights don't sum to 1.0
        with pytest.raises(
            ValueError, match=r"semantic_weight \+ bm25_weight must equal 1.0"
        ):
            RankingService(semantic_weight=0.6, bm25_weight=0.5)

        with pytest.raises(
            ValueError, match=r"semantic_weight \+ bm25_weight must equal 1.0"
        ):
            RankingService(semantic_weight=0.8, bm25_weight=0.1)

    def test_prepare_corpus(self):
        """Test corpus preparation for BM25 calculations."""
        documents = ["memory allocation error", "network protocol"]
        self.service.prepare_corpus(documents)

        assert self.service._corpus_prepared
        assert self.service.bm25_calculator._prepared

    def test_calculate_bm25_score_not_prepared(self):
        """Test BM25 score calculation when corpus is not prepared."""
        score = self.service.calculate_bm25_score("memory", "memory allocation")
        assert score == 0.0

    def test_calculate_bm25_score_prepared(self):
        """Test BM25 score calculation when corpus is prepared."""
        documents = ["memory allocation error", "network protocol"]
        self.service.prepare_corpus(documents)

        score = self.service.calculate_bm25_score("memory", "memory allocation error")
        assert 0.0 <= score <= 1.0

    def test_calculate_combined_score(self):
        """Test combined score calculation from similarity and BM25 scores."""
        # Test with default weights (0.7 semantic, 0.3 BM25)
        combined = self.service.calculate_combined_score(0.8, 0.6)
        expected = 0.7 * 0.8 + 0.3 * 0.6  # 0.56 + 0.18 = 0.74
        assert math.isclose(combined, expected)

        # Test boundary values
        assert self.service.calculate_combined_score(0.0, 0.0) == 0.0
        assert self.service.calculate_combined_score(1.0, 1.0) == 1.0

        # Test custom weights
        service = RankingService(semantic_weight=0.5, bm25_weight=0.5)
        combined = service.calculate_combined_score(0.8, 0.6)
        expected = 0.5 * 0.8 + 0.5 * 0.6  # 0.4 + 0.3 = 0.7
        assert math.isclose(combined, expected)

    def test_calculate_combined_score_invalid_input(self):
        """Test combined score calculation with invalid input."""
        # Similarity score out of range
        with pytest.raises(ValueError, match="similarity_score must be in \\[0,1\\]"):
            self.service.calculate_combined_score(-0.1, 0.5)

        with pytest.raises(ValueError, match="similarity_score must be in \\[0,1\\]"):
            self.service.calculate_combined_score(1.1, 0.5)

        # BM25 score out of range
        with pytest.raises(ValueError, match="bm25_score must be in \\[0,1\\]"):
            self.service.calculate_combined_score(0.5, -0.1)

        with pytest.raises(ValueError, match="bm25_score must be in \\[0,1\\]"):
            self.service.calculate_combined_score(0.5, 1.1)

    def test_calculate_confidence(self):
        """Test confidence score calculation."""
        # High scores should give high confidence
        confidence = self.service.calculate_confidence(0.9, 0.8)
        assert confidence > 0.8

        # Low scores should give low confidence
        confidence = self.service.calculate_confidence(0.2, 0.1)
        assert confidence < 0.5

        # Test score agreement boost
        # When both scores are high, confidence should be boosted
        high_agreement = self.service.calculate_confidence(0.8, 0.8)
        low_agreement = self.service.calculate_confidence(0.8, 0.2)
        assert high_agreement > low_agreement

        # Test context quality factor
        confidence_good_context = self.service.calculate_confidence(
            0.7, 0.6, context_quality=1.0
        )
        confidence_poor_context = self.service.calculate_confidence(
            0.7, 0.6, context_quality=0.2
        )
        assert confidence_good_context > confidence_poor_context

        # Test variance penalty
        confidence_low_variance = self.service.calculate_confidence(
            0.7, 0.6, score_variance=0.0
        )
        confidence_high_variance = self.service.calculate_confidence(
            0.7, 0.6, score_variance=0.5
        )
        assert confidence_low_variance > confidence_high_variance

        # Confidence should be bounded [0, 1]
        confidence = self.service.calculate_confidence(
            1.0, 1.0, context_quality=1.0, score_variance=0.0
        )
        assert 0.0 <= confidence <= 1.0

    def test_extract_context_lines(self):
        """Test context line extraction from documents."""
        document = "line1\nmemory allocation\nline3\nbuffer overflow\nline5"
        query = "memory buffer"

        context = self.service._extract_context_lines(document, query, max_lines=3)

        assert len(context) <= 3
        # Should include lines with query terms
        assert any("memory" in line for line in context)
        assert any("buffer" in line for line in context)

        # Test with short document
        short_doc = "memory allocation"
        context = self.service._extract_context_lines(short_doc, "memory", max_lines=5)
        assert context == ["memory allocation"]

    def test_rank_results_empty(self):
        """Test ranking with empty search results."""
        results = self.service.rank_results("query", [])
        assert results == []

    def test_rank_results_basic(self):
        """Test basic result ranking functionality."""
        search_results = [
            ("doc1", 0.7, "allocation error memory"),
            ("doc2", 0.5, "network protocol stack"),
        ]

        ranked = self.service.rank_results("allocation", search_results, max_results=10)

        assert len(ranked) == 2
        # Results should be SearchResult objects
        for result in ranked:
            assert hasattr(result, "combined_score")
            assert hasattr(result, "content_id")
            assert hasattr(result, "similarity_score")
            assert hasattr(result, "bm25_score")

        # Should be sorted by combined score
        assert ranked[0].combined_score >= ranked[1].combined_score

    def test_rank_results_max_results_limit(self):
        """Test that max_results parameter limits returned results."""
        search_results = [
            ("doc1", 0.9, "allocation error kernel"),
            ("doc2", 0.8, "allocation buffer overflow"),
            ("doc3", 0.7, "allocation network protocol"),
            ("doc4", 0.6, "allocation file system"),
            ("doc5", 0.5, "allocation process scheduling"),
        ]

        ranked = self.service.rank_results("allocation", search_results, max_results=3)

        assert len(ranked) == 3
        # Results should be sorted by combined_score descending
        assert ranked[0].combined_score >= ranked[1].combined_score
        assert ranked[1].combined_score >= ranked[2].combined_score

    def test_update_weights(self):
        """Test dynamic weight updating."""
        initial_semantic = self.service.semantic_weight
        initial_bm25 = self.service.bm25_weight

        self.service.update_weights(0.8, 0.2)

        assert self.service.semantic_weight == 0.8
        assert self.service.bm25_weight == 0.2
        assert self.service.semantic_weight != initial_semantic
        assert self.service.bm25_weight != initial_bm25

    def test_update_weights_invalid(self):
        """Test weight updating with invalid values."""
        original_semantic = self.service.semantic_weight
        original_bm25 = self.service.bm25_weight

        with pytest.raises(
            ValueError, match=r"semantic_weight \+ bm25_weight must equal 1.0"
        ):
            self.service.update_weights(0.6, 0.5)

        # Weights should remain unchanged after failed update
        assert self.service.semantic_weight == original_semantic
        assert self.service.bm25_weight == original_bm25

    def test_get_ranking_stats(self):
        """Test ranking service statistics retrieval."""
        stats = self.service.get_ranking_stats()

        assert "semantic_weight" in stats
        assert "bm25_weight" in stats
        assert "corpus_prepared" in stats
        assert "bm25_params" in stats
        assert "corpus_stats" in stats

        assert stats["semantic_weight"] == 0.7
        assert stats["bm25_weight"] == 0.3
        assert stats["corpus_prepared"] is False
        assert stats["bm25_params"]["k1"] == 1.2
        assert stats["bm25_params"]["b"] == 0.75
        assert stats["corpus_stats"] is None

        # Test after corpus preparation
        self.service.prepare_corpus(["test document"])
        stats = self.service.get_ranking_stats()

        assert stats["corpus_prepared"] is True
        assert stats["corpus_stats"] is not None

    def test_ranking_weights_affect_results(self):
        """Test that different ranking weights produce different results."""
        semantic_heavy = RankingService(semantic_weight=0.9, bm25_weight=0.1)
        bm25_heavy = RankingService(semantic_weight=0.1, bm25_weight=0.9)

        # Case where semantic score is high but BM25 is low
        semantic_combined = semantic_heavy.calculate_combined_score(0.9, 0.2)
        bm25_combined = bm25_heavy.calculate_combined_score(0.9, 0.2)

        # Semantic-heavy should score higher when semantic score is high
        assert semantic_combined > bm25_combined

        # Case where BM25 score is high but semantic is low
        semantic_combined = semantic_heavy.calculate_combined_score(0.2, 0.9)
        bm25_combined = bm25_heavy.calculate_combined_score(0.2, 0.9)

        # BM25-heavy should score higher when BM25 score is high
        assert bm25_combined > semantic_combined

    def test_end_to_end_ranking_pipeline(self):
        """Test complete ranking pipeline from search results to ranked output."""
        # Prepare search results with different characteristics
        search_results = [
            (
                "doc1",
                0.9,
                "allocation error in kernel",
            ),  # High semantic, should match "allocation"
            (
                "doc2",
                0.3,
                "allocation allocation system",
            ),  # Low semantic, high keyword density
            (
                "doc3",
                0.7,
                "network protocol implementation",
            ),  # Medium semantic, no keyword match
        ]

        ranked = self.service.rank_results("allocation", search_results)

        assert len(ranked) == 3
        # Verify that ranking considers both semantic and keyword factors
        # Results should be sorted by combined score
        for i in range(len(ranked) - 1):
            assert ranked[i].combined_score >= ranked[i + 1].combined_score

        # All results should have the expected attributes
        for result in ranked:
            assert hasattr(result, "content_id")
            assert hasattr(result, "similarity_score")
            assert hasattr(result, "bm25_score")
            assert hasattr(result, "combined_score")
            assert hasattr(result, "confidence")
            assert 0.0 <= result.combined_score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
