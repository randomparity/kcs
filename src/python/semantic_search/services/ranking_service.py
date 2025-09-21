"""
RankingService implementation with hybrid BM25+semantic scoring.

Provides hybrid ranking combining BM25 keyword matching with semantic similarity
scores for comprehensive search result ranking.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..models.search_result import SearchResult

logger = logging.getLogger(__name__)


class BM25Calculator:
    """
    BM25 scoring calculator for keyword-based ranking.

    Implements Okapi BM25 ranking function for combining with semantic similarity.
    Optimized for code search with sensible parameter defaults.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        """
        Initialize BM25 calculator with standard parameters.

        Args:
            k1: Term frequency saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus_stats: dict[str, Any] = {}
        self._prepared = False

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 scoring.

        Optimized for code content with identifier handling.

        Args:
            text: Text to tokenize

        Returns:
            List of normalized tokens
        """
        # Convert to lowercase for case-insensitive matching
        text = text.lower()

        # Split on whitespace and common code delimiters
        tokens = re.findall(r"\b\w+\b", text)

        # Filter out very short tokens (less than 2 chars)
        tokens = [token for token in tokens if len(token) >= 2]

        return tokens

    def prepare_corpus(self, documents: list[str]) -> None:
        """
        Prepare corpus statistics for BM25 calculations.

        Args:
            documents: List of document texts to analyze
        """
        if not documents:
            self.corpus_stats = {
                "doc_count": 0,
                "avg_doc_length": 0.0,
                "doc_freqs": defaultdict(int),
                "doc_lengths": [],
            }
            self._prepared = True
            return

        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]

        # Calculate document frequencies
        doc_freqs: defaultdict[str, int] = defaultdict(int)
        doc_lengths = []

        for tokens in tokenized_docs:
            doc_length = len(tokens)
            doc_lengths.append(doc_length)

            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freqs[term] += 1

        # Calculate average document length
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

        self.corpus_stats = {
            "doc_count": len(documents),
            "avg_doc_length": avg_doc_length,
            "doc_freqs": doc_freqs,
            "doc_lengths": doc_lengths,
        }
        self._prepared = True

    def score_document(self, query: str, document: str, doc_index: int = 0) -> float:
        """
        Calculate BM25 score for a document against a query.

        Args:
            query: Search query text
            document: Document text to score
            doc_index: Document index in corpus (for length lookup)

        Returns:
            BM25 score (higher is better)
        """
        if not self._prepared:
            logger.warning("BM25Calculator not prepared with corpus statistics")
            return 0.0

        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        if not query_tokens or not doc_tokens:
            return 0.0

        # Get document length
        if doc_index < len(self.corpus_stats["doc_lengths"]):
            doc_length = self.corpus_stats["doc_lengths"][doc_index]
        else:
            doc_length = len(doc_tokens)

        # Calculate term frequencies in document
        doc_term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            # Term frequency in document
            tf = doc_term_freqs.get(term, 0)
            if tf == 0:
                continue

            # Document frequency for IDF calculation
            df = self.corpus_stats["doc_freqs"].get(term, 0)
            if df == 0:
                continue

            # IDF calculation
            idf = math.log((self.corpus_stats["doc_count"] - df + 0.5) / (df + 0.5))

            # BM25 component calculation
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.corpus_stats["avg_doc_length"])
            )

            score += idf * (numerator / denominator)

        # Normalize to 0-1 range (approximate)
        # Use tanh for smooth normalization
        normalized_score = math.tanh(score / 10.0)
        return max(0.0, min(1.0, normalized_score))


class RankingService:
    """
    Hybrid ranking service combining BM25 and semantic similarity.

    Provides comprehensive ranking that balances keyword matching (BM25) with
    semantic similarity for optimal search result quality.
    """

    def __init__(
        self,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """
        Initialize ranking service with configurable weights.

        Args:
            semantic_weight: Weight for semantic similarity (default: 0.7)
            bm25_weight: Weight for BM25 score (default: 0.3)
            k1: BM25 k1 parameter (default: 1.2)
            b: BM25 b parameter (default: 0.75)
        """
        if not math.isclose(semantic_weight + bm25_weight, 1.0, rel_tol=1e-9):
            raise ValueError("semantic_weight + bm25_weight must equal 1.0")

        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.bm25_calculator = BM25Calculator(k1=k1, b=b)
        self._corpus_prepared = False

    def prepare_corpus(self, documents: list[str]) -> None:
        """
        Prepare corpus for BM25 calculations.

        Args:
            documents: List of document texts to prepare
        """
        self.bm25_calculator.prepare_corpus(documents)
        self._corpus_prepared = True
        logger.info(f"Prepared corpus with {len(documents)} documents for BM25 scoring")

    def calculate_bm25_score(
        self, query: str, document: str, doc_index: int = 0
    ) -> float:
        """
        Calculate BM25 score for a query-document pair.

        Args:
            query: Search query text
            document: Document text to score
            doc_index: Document index in corpus

        Returns:
            BM25 score (0.0-1.0)
        """
        if not self._corpus_prepared:
            logger.warning("Corpus not prepared for BM25 scoring")
            return 0.0

        return self.bm25_calculator.score_document(query, document, doc_index)

    def calculate_combined_score(
        self, similarity_score: float, bm25_score: float
    ) -> float:
        """
        Calculate combined score from similarity and BM25 scores.

        Args:
            similarity_score: Semantic similarity score (0.0-1.0)
            bm25_score: BM25 keyword score (0.0-1.0)

        Returns:
            Combined ranking score (0.0-1.0)
        """
        if not (0.0 <= similarity_score <= 1.0):
            raise ValueError(
                f"similarity_score must be in [0,1], got: {similarity_score}"
            )
        if not (0.0 <= bm25_score <= 1.0):
            raise ValueError(f"bm25_score must be in [0,1], got: {bm25_score}")

        combined = (self.semantic_weight * similarity_score) + (
            self.bm25_weight * bm25_score
        )
        return max(0.0, min(1.0, combined))

    def calculate_confidence(
        self,
        similarity_score: float,
        bm25_score: float,
        context_quality: float = 1.0,
        score_variance: float = 0.0,
    ) -> float:
        """
        Calculate confidence score based on multiple factors.

        Args:
            similarity_score: Semantic similarity score
            bm25_score: BM25 keyword score
            context_quality: Quality of surrounding context (0.0-1.0)
            score_variance: Variance in similar results (higher = less confident)

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from combined score
        combined = self.calculate_combined_score(similarity_score, bm25_score)

        # Score agreement boost - when both scores are high, confidence increases
        score_agreement = min(similarity_score, bm25_score)
        agreement_boost = score_agreement * 0.15

        # Context quality factor
        context_factor = context_quality * 0.1

        # Score variance penalty - higher variance reduces confidence
        variance_penalty = score_variance * 0.2

        # Calculate final confidence
        confidence = combined + agreement_boost + context_factor - variance_penalty
        return max(0.0, min(1.0, confidence))

    def rank_results(
        self,
        query: str,
        search_results: list[
            tuple[str, float, str]
        ],  # (content_id, similarity, document)
        max_results: int = 20,
    ) -> list[SearchResult]:
        """
        Rank search results using hybrid BM25+semantic scoring.

        Args:
            query: Original search query
            search_results: List of (content_id, similarity_score, document_text) tuples
            max_results: Maximum number of results to return

        Returns:
            Ranked list of SearchResult objects
        """
        if not search_results:
            return []

        # Prepare corpus for BM25 scoring
        documents = [doc for _, _, doc in search_results]
        self.prepare_corpus(documents)

        # Calculate combined scores for all results
        ranked_results = []

        for i, (content_id, similarity_score, document) in enumerate(search_results):
            # Calculate BM25 score
            bm25_score = self.calculate_bm25_score(query, document, doc_index=i)

            # Calculate combined score
            combined_score = self.calculate_combined_score(similarity_score, bm25_score)

            # Calculate confidence
            confidence = self.calculate_confidence(similarity_score, bm25_score)

            # Extract context lines (simple implementation)
            context_lines = self._extract_context_lines(document, query)

            # Create SearchResult
            result = SearchResult(
                query_id=f"query_{hash(query)}",  # Simple query ID generation
                content_id=content_id,
                similarity_score=similarity_score,
                bm25_score=bm25_score,
                combined_score=combined_score,
                confidence=confidence,
                context_lines=context_lines,
                explanation="",  # Will be auto-generated by SearchResult
            )

            ranked_results.append(result)

        # Sort by combined score (descending) and limit results
        ranked_results.sort(key=lambda x: x.combined_score, reverse=True)
        return ranked_results[:max_results]

    def _extract_context_lines(
        self, document: str, query: str, max_lines: int = 5
    ) -> list[str]:
        """
        Extract relevant context lines from document.

        Args:
            document: Document text
            query: Search query
            max_lines: Maximum context lines to extract

        Returns:
            List of relevant context lines
        """
        lines = document.split("\n")
        if len(lines) <= max_lines:
            return lines

        # Simple relevance-based extraction
        query_tokens = set(self.bm25_calculator._tokenize(query))
        scored_lines = []

        for i, line in enumerate(lines):
            line_tokens = set(self.bm25_calculator._tokenize(line))
            overlap = len(query_tokens & line_tokens)
            scored_lines.append((overlap, i, line))

        # Sort by relevance and take top lines
        scored_lines.sort(key=lambda x: x[0], reverse=True)
        top_lines = scored_lines[:max_lines]

        # Sort back by original order for readability
        top_lines.sort(key=lambda x: x[1])

        return [line for _, _, line in top_lines]

    def update_weights(self, semantic_weight: float, bm25_weight: float) -> None:
        """
        Update ranking weights dynamically.

        Args:
            semantic_weight: New weight for semantic similarity
            bm25_weight: New weight for BM25 score

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if not math.isclose(semantic_weight + bm25_weight, 1.0, rel_tol=1e-9):
            raise ValueError("semantic_weight + bm25_weight must equal 1.0")

        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        logger.info(
            f"Updated ranking weights: semantic={semantic_weight}, bm25={bm25_weight}"
        )

    def get_ranking_stats(self) -> dict[str, Any]:
        """
        Get ranking service statistics.

        Returns:
            Dictionary with ranking statistics
        """
        return {
            "semantic_weight": self.semantic_weight,
            "bm25_weight": self.bm25_weight,
            "corpus_prepared": self._corpus_prepared,
            "bm25_params": {"k1": self.bm25_calculator.k1, "b": self.bm25_calculator.b},
            "corpus_stats": self.bm25_calculator.corpus_stats
            if self._corpus_prepared
            else None,
        }
