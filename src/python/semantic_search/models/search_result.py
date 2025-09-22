"""
SearchResult model implementation.

Individual result returned from semantic search with ranking logic
and confidence scoring.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator


@dataclass(frozen=True, order=True)
class SearchResult:
    """
    Immutable SearchResult model with ranking logic.

    Represents an individual result from semantic search with
    similarity scoring, BM25 ranking, and confidence assessment.

    Attributes:
        query_id: Reference to originating SearchQuery
        content_id: Reference to VectorEmbedding
        similarity_score: Semantic similarity (0.0-1.0)
        bm25_score: Keyword matching score (0.0-1.0)
        combined_score: Final ranking score (0.0-1.0)
        confidence: Result confidence (0.0-1.0)
        context_lines: Surrounding code lines (max 10)
        explanation: Why this result matches the query
    """

    # For ordering, we want highest combined_score first
    combined_score: float = field(compare=True)
    query_id: str = field(compare=False)
    content_id: str = field(compare=False)
    similarity_score: float = field(compare=False)
    bm25_score: float = field(compare=False)
    confidence: float = field(compare=False)
    context_lines: list[str] = field(default_factory=list, compare=False)
    explanation: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        """Validate all fields after object creation."""
        self._validate_ids()
        self._validate_scores()
        self._validate_context_lines()
        self._validate_explanation()

        # Auto-generate explanation if empty
        if not self.explanation:
            object.__setattr__(self, "explanation", self._generate_explanation())

    def _validate_ids(self) -> None:
        """Validate query_id and content_id fields."""
        if not isinstance(self.query_id, str):
            raise TypeError("query_id must be a string")
        if not self.query_id or len(self.query_id.strip()) == 0:
            raise ValueError("query_id cannot be empty")

        if not isinstance(self.content_id, str):
            raise TypeError("content_id must be a string")
        if not self.content_id or len(self.content_id.strip()) == 0:
            raise ValueError("content_id cannot be empty")

    def _validate_scores(self) -> None:
        """Validate all score fields are in range 0.0-1.0."""
        scores = {
            "similarity_score": self.similarity_score,
            "bm25_score": self.bm25_score,
            "combined_score": self.combined_score,
            "confidence": self.confidence,
        }

        for score_name, score_value in scores.items():
            if not isinstance(score_value, (int, float)):
                raise TypeError(f"{score_name} must be a number")

            if not (0.0 <= score_value <= 1.0):
                raise ValueError(
                    f"{score_name} must be in range 0.0-1.0, got: {score_value}"
                )

    def _validate_context_lines(self) -> None:
        """Validate context_lines field."""
        if not isinstance(self.context_lines, list):
            raise TypeError("context_lines must be a list")

        if len(self.context_lines) > 10:
            raise ValueError("context_lines cannot exceed 10 lines")

        for i, line in enumerate(self.context_lines):
            if not isinstance(line, str):
                raise ValueError(
                    f"context_lines[{i}] must be a string, got: {type(line)}"
                )

    def _validate_explanation(self) -> None:
        """Validate explanation field."""
        if not isinstance(self.explanation, str):
            raise TypeError("explanation must be a string")

    def _generate_explanation(self) -> str:
        """Generate automatic explanation based on scores."""
        if self.combined_score >= 0.9:
            quality = "excellent"
        elif self.combined_score >= 0.7:
            quality = "good"
        elif self.combined_score >= 0.5:
            quality = "moderate"
        else:
            quality = "weak"

        return (
            f"This result shows {quality} relevance "
            f"(similarity: {self.similarity_score:.2f}, "
            f"keyword: {self.bm25_score:.2f}, "
            f"confidence: {self.confidence:.2f})"
        )

    def calculate_combined_score(self) -> float:
        """
        Calculate combined score using standard formula.

        Formula: (0.7 * similarity_score) + (0.3 * bm25_score)

        Returns:
            Combined ranking score
        """
        return (0.7 * self.similarity_score) + (0.3 * self.bm25_score)

    @classmethod
    def calculate_combined_score_static(
        cls, similarity_score: float, bm25_score: float
    ) -> float:
        """
        Calculate combined score using standard formula (static version).

        Formula: (0.7 * similarity_score) + (0.3 * bm25_score)

        Args:
            similarity_score: Semantic similarity score
            bm25_score: Keyword matching score

        Returns:
            Combined ranking score
        """
        return (0.7 * similarity_score) + (0.3 * bm25_score)

    @classmethod
    def calculate_confidence(
        cls,
        similarity_score: float,
        bm25_score: float,
        context_quality: float = 1.0,
    ) -> float:
        """
        Calculate confidence based on score quality and context.

        Args:
            similarity_score: Semantic similarity score
            bm25_score: Keyword matching score
            context_quality: Quality of surrounding context (0.0-1.0)

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from combined score
        combined = cls.calculate_combined_score_static(similarity_score, bm25_score)

        # Boost confidence when both scores are high
        score_agreement = min(similarity_score, bm25_score)
        agreement_boost = score_agreement * 0.1

        # Factor in context quality
        context_factor = context_quality * 0.05

        # Calculate final confidence (clamped to [0, 1])
        confidence = combined + agreement_boost + context_factor
        return min(1.0, max(0.0, confidence))

    def to_dict(self) -> dict[str, Any]:
        """Serialize SearchResult to dictionary."""
        return {
            "query_id": self.query_id,
            "content_id": self.content_id,
            "similarity_score": self.similarity_score,
            "bm25_score": self.bm25_score,
            "combined_score": self.combined_score,
            "confidence": self.confidence,
            "context_lines": self.context_lines.copy(),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        """Deserialize SearchResult from dictionary."""
        return cls(
            query_id=data["query_id"],
            content_id=data["content_id"],
            similarity_score=data["similarity_score"],
            bm25_score=data["bm25_score"],
            combined_score=data["combined_score"],
            confidence=data["confidence"],
            context_lines=data.get("context_lines", []),
            explanation=data.get("explanation", ""),
        )

    def __str__(self) -> str:
        """String representation of SearchResult."""
        return (
            f"SearchResult(score={self.combined_score:.3f}, "
            f"query={self.query_id}, "
            f"content={self.content_id})"
        )

    def __repr__(self) -> str:
        """Detailed representation of SearchResult."""
        return (
            f"SearchResult("
            f"query_id={self.query_id!r}, "
            f"content_id={self.content_id!r}, "
            f"similarity_score={self.similarity_score}, "
            f"bm25_score={self.bm25_score}, "
            f"combined_score={self.combined_score}, "
            f"confidence={self.confidence})"
        )

    # Relationship methods (placeholders for database integration)
    def get_query(self) -> Any:
        """Get the associated SearchQuery object."""

        # Placeholder - will be implemented in database layer tasks
        # Return a mock object for testing
        class MockQuery:
            def __init__(self, query_id: str):
                self.query_id = query_id

        return MockQuery(self.query_id)

    def get_content(self) -> Any:
        """Get the associated VectorEmbedding object."""

        # Placeholder - will be implemented in database layer tasks
        # Return a mock object for testing
        class MockContent:
            def __init__(self, content_id: str):
                self.content_id = content_id

        return MockContent(self.content_id)


class SearchResultPydantic(BaseModel):
    """
    Pydantic model for SearchResult API validation.

    Used for MCP endpoint validation and serialization.
    Complements the immutable dataclass for different use cases.
    """

    query_id: str = Field(
        ..., min_length=1, description="Reference to originating SearchQuery"
    )
    content_id: str = Field(
        ..., min_length=1, description="Reference to VectorEmbedding"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic similarity (0.0-1.0)"
    )
    bm25_score: float = Field(
        ..., ge=0.0, le=1.0, description="Keyword matching score (0.0-1.0)"
    )
    combined_score: float = Field(
        ..., ge=0.0, le=1.0, description="Final ranking score (0.0-1.0)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Result confidence (0.0-1.0)"
    )
    context_lines: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Surrounding code lines (max 10)",
    )
    explanation: str = Field(
        default="", description="Why this result matches the query"
    )

    @field_validator("context_lines")
    @classmethod
    def validate_context_lines(cls, v: list[str]) -> list[str]:
        """Validate context lines are all strings."""
        for i, line in enumerate(v):
            if not isinstance(line, str):
                raise ValueError(f"context_lines[{i}] must be a string")
        return v

    def to_search_result(self) -> SearchResult:
        """Convert to immutable SearchResult dataclass."""
        return SearchResult(
            query_id=self.query_id,
            content_id=self.content_id,
            similarity_score=self.similarity_score,
            bm25_score=self.bm25_score,
            combined_score=self.combined_score,
            confidence=self.confidence,
            context_lines=self.context_lines,
            explanation=self.explanation,
        )
