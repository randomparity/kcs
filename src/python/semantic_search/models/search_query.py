"""
SearchQuery model implementation.

Represents a user's natural language search request with automatic processing
and embedding generation using BAAI/bge-small-en-v1.5 model.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..services.embedding_service import EmbeddingService
from ..services.query_preprocessor import QueryPreprocessor


@dataclass(frozen=True)
class SearchQuery:
    """
    Immutable SearchQuery model with automatic processing and embedding generation.

    Represents a user's natural language search request with validation,
    normalization, and vector embedding generation.

    Attributes:
        query_text: Original user query (1-1000 characters)
        processed_query: Normalized query text (auto-generated)
        embedding: 384-dimensional vector representation (auto-generated)
        timestamp: Query creation time (auto-generated)
        user_id: Optional authenticated user identifier
        config_context: Kernel configuration filters
    """

    query_text: str
    user_id: str | None = None
    config_context: list[str] = field(default_factory=list)

    # Auto-generated fields
    processed_query: str = field(init=False)
    embedding: list[float] = field(init=False)
    timestamp: datetime = field(init=False)

    def __post_init__(self) -> None:
        """Initialize auto-generated fields after object creation."""
        # Validate input fields
        self._validate_query_text()
        self._validate_config_context()

        # Generate auto fields using object.__setattr__ for frozen dataclass
        object.__setattr__(self, "timestamp", datetime.now())

        # Generate processed query
        preprocessor = QueryPreprocessor()
        processed = preprocessor.preprocess(self.query_text)
        object.__setattr__(self, "processed_query", processed)

        # Generate embedding
        embedding_service = EmbeddingService()
        embedding_vector = embedding_service.embed_query(self.processed_query)

        # Ensure embedding is normalized (unit vector for cosine similarity)
        embedding_array = np.array(embedding_vector)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_vector = (embedding_array / norm).tolist()

        object.__setattr__(self, "embedding", embedding_vector)

    def _validate_query_text(self) -> None:
        """Validate query_text field."""
        if not isinstance(self.query_text, str):
            raise TypeError("query_text must be a string")

        if not self.query_text or len(self.query_text.strip()) == 0:
            raise ValueError("query_text cannot be empty")

        if len(self.query_text) > 1000:
            raise ValueError("query_text cannot exceed 1000 characters")

    def _validate_config_context(self) -> None:
        """Validate config_context field."""
        if not isinstance(self.config_context, list):
            raise TypeError("config_context must be a list")

        for config in self.config_context:
            if not isinstance(config, str):
                raise ValueError("All config_context items must be strings")

            # Validate config format: CONFIG_* or !CONFIG_*
            if config.startswith("!"):
                config_name = config[1:]
            else:
                config_name = config

            if not config_name.startswith("CONFIG_"):
                raise ValueError(
                    f"Invalid config format: {config}. Must start with CONFIG_"
                )

            if config_name == "CONFIG_":
                raise ValueError(
                    f"Invalid config format: {config}. CONFIG_ prefix requires a name"
                )

            # Check case sensitivity (should be uppercase)
            if config_name != config_name.upper():
                raise ValueError(f"Invalid config format: {config}. Must be uppercase")

    def to_dict(self) -> dict[str, Any]:
        """Serialize SearchQuery to dictionary."""
        return {
            "query_text": self.query_text,
            "processed_query": self.processed_query,
            "embedding": self.embedding,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "config_context": self.config_context.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchQuery":
        """Deserialize SearchQuery from dictionary."""
        # Extract constructor args
        query_text = data["query_text"]
        user_id = data.get("user_id")
        config_context = data.get("config_context", [])

        # Create instance (will auto-generate processed_query, embedding, timestamp)
        return cls(
            query_text=query_text, user_id=user_id, config_context=config_context
        )

    def __str__(self) -> str:
        """String representation of SearchQuery."""
        user_part = f" (user: {self.user_id})" if self.user_id else ""
        config_part = (
            f" [configs: {', '.join(self.config_context)}]"
            if self.config_context
            else ""
        )
        return f"SearchQuery: '{self.query_text}'{user_part}{config_part}"

    def __repr__(self) -> str:
        """Detailed representation of SearchQuery."""
        return (
            f"SearchQuery("
            f"query_text={self.query_text!r}, "
            f"user_id={self.user_id!r}, "
            f"config_context={self.config_context!r}, "
            f"timestamp={self.timestamp!r})"
        )


class SearchQueryPydantic(BaseModel):
    """
    Pydantic model for SearchQuery API validation.

    Used for MCP endpoint validation and serialization.
    Complements the immutable dataclass for different use cases.
    """

    query_text: str = Field(
        ..., min_length=1, max_length=1000, description="User's search query"
    )
    user_id: str | None = Field(None, description="Authenticated user identifier")
    config_context: list[str] = Field(
        default_factory=list, description="Kernel configuration filters"
    )

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Validate query text is not just whitespace."""
        if not v.strip():
            raise ValueError("query_text cannot be empty or just whitespace")
        return v

    @field_validator("config_context")
    @classmethod
    def validate_config_context(cls, v: list[str]) -> list[str]:
        """Validate config context format."""
        for config in v:
            # Validate config format: CONFIG_* or !CONFIG_*
            if config.startswith("!"):
                config_name = config[1:]
            else:
                config_name = config

            if not config_name.startswith("CONFIG_"):
                raise ValueError(
                    f"Invalid config format: {config}. Must start with CONFIG_"
                )

            if config_name == "CONFIG_":
                raise ValueError(
                    f"Invalid config format: {config}. CONFIG_ prefix requires a name"
                )

            # Check case sensitivity (should be uppercase)
            if config_name != config_name.upper():
                raise ValueError(f"Invalid config format: {config}. Must be uppercase")

        return v

    def to_search_query(self) -> SearchQuery:
        """Convert to immutable SearchQuery dataclass."""
        return SearchQuery(
            query_text=self.query_text,
            user_id=self.user_id,
            config_context=self.config_context,
        )
