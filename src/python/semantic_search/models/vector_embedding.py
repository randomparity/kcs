"""
VectorEmbedding model implementation.

Stores vector representations of indexed content with pgvector integration
for semantic similarity search.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ContentType(str, Enum):
    """Valid content types for vector embeddings."""

    SOURCE_CODE = "SOURCE_CODE"
    DOCUMENTATION = "DOCUMENTATION"
    HEADER = "HEADER"
    COMMENT = "COMMENT"


@dataclass(frozen=True)
class VectorEmbedding:
    """
    Immutable VectorEmbedding model with pgvector integration.

    Stores vector representations of indexed content for semantic search
    with metadata and configuration awareness.

    Attributes:
        content_id: Unique identifier for source content
        file_path: Absolute path to source file
        content_type: Type of content (SOURCE_CODE, DOCUMENTATION, etc.)
        content_text: Original text content
        embedding: 384-dimensional vector representation
        line_start: Starting line number
        line_end: Ending line number
        metadata: Additional context (function names, symbols, etc.)
        created_at: When embedding was generated (auto-generated)
        config_guards: Configuration dependencies
    """

    content_id: str
    file_path: str
    content_type: str
    content_text: str
    embedding: list[float]
    line_start: int
    line_end: int
    metadata: dict[str, Any] = field(default_factory=dict)
    config_guards: list[str] = field(default_factory=list)

    # Auto-generated fields
    created_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        """Initialize auto-generated fields and validate inputs after object creation."""
        # Validate all fields
        self._validate_content_id()
        self._validate_file_path()
        self._validate_content_type()
        self._validate_content_text()
        self._validate_embedding()
        self._validate_line_numbers()
        self._validate_metadata()
        self._validate_config_guards()

        # Generate auto fields using object.__setattr__ for frozen dataclass
        object.__setattr__(self, "created_at", datetime.now())

    def _validate_content_id(self) -> None:
        """Validate content_id field."""
        if not isinstance(self.content_id, str):
            raise TypeError("content_id must be a string")

        if not self.content_id or len(self.content_id.strip()) == 0:
            raise ValueError("content_id cannot be empty")

    def _validate_file_path(self) -> None:
        """Validate file_path field."""
        if not isinstance(self.file_path, str):
            raise TypeError("file_path must be a string")

        if not self.file_path:
            raise ValueError("file_path cannot be empty")

        if not self.file_path.startswith("/"):
            raise ValueError("file_path must be an absolute path")

        if self.file_path == "/":
            raise ValueError("file_path cannot be root directory only")

    def _validate_content_type(self) -> None:
        """Validate content_type field."""
        if not isinstance(self.content_type, str):
            raise TypeError("content_type must be a string")

        valid_types = [ct.value for ct in ContentType]
        if self.content_type not in valid_types:
            raise ValueError(
                f"content_type must be one of {valid_types}, got: {self.content_type}"
            )

    def _validate_content_text(self) -> None:
        """Validate content_text field."""
        if not isinstance(self.content_text, str):
            raise TypeError("content_text must be a string")

        # Content text can be empty for some cases (e.g., binary files)
        # but the field must exist

    def _validate_embedding(self) -> None:
        """Validate embedding field."""
        if not isinstance(self.embedding, (list, np.ndarray)):
            raise TypeError("embedding must be a list or numpy array")

        if len(self.embedding) != 384:
            raise ValueError(
                f"embedding must be 384-dimensional, got: {len(self.embedding)}"
            )

        # Ensure all values are numeric
        for i, value in enumerate(self.embedding):
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"embedding[{i}] must be numeric, got: {type(value)}")

    def _validate_line_numbers(self) -> None:
        """Validate line_start and line_end fields."""
        if not isinstance(self.line_start, int):
            raise TypeError("line_start must be an integer")

        if not isinstance(self.line_end, int):
            raise TypeError("line_end must be an integer")

        if self.line_start < 1:
            raise ValueError("line_start must be >= 1")

        if self.line_end < 1:
            raise ValueError("line_end must be >= 1")

        if self.line_start > self.line_end:
            raise ValueError("line_start must be <= line_end")

    def _validate_metadata(self) -> None:
        """Validate metadata field."""
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

        # Metadata can be empty or contain any JSON-serializable data
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(f"metadata must be JSON-serializable: {e}") from e

    def _validate_config_guards(self) -> None:
        """Validate config_guards field."""
        if not isinstance(self.config_guards, list):
            raise TypeError("config_guards must be a list")

        for config in self.config_guards:
            if not isinstance(config, str):
                raise ValueError("All config_guards items must be strings")

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

    def calculate_similarity(self, query_embedding: list[float]) -> float:
        """
        Calculate cosine similarity with query embedding.

        Args:
            query_embedding: Query vector to compare against

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        if len(query_embedding) != 384:
            raise ValueError("Query embedding must be 384-dimensional")

        # Convert to numpy arrays
        vec1 = np.array(self.embedding)
        vec2 = np.array(query_embedding)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Clamp to [0, 1] range (cosine similarity can be [-1, 1])
        return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize VectorEmbedding to dictionary."""
        return {
            "content_id": self.content_id,
            "file_path": self.file_path,
            "content_type": self.content_type,
            "content_text": self.content_text,
            "embedding": self.embedding,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "metadata": self.metadata.copy(),
            "created_at": self.created_at.isoformat(),
            "config_guards": self.config_guards.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorEmbedding":
        """Deserialize VectorEmbedding from dictionary."""
        # Extract constructor args
        content_id = data["content_id"]
        file_path = data["file_path"]
        content_type = data["content_type"]
        content_text = data["content_text"]
        embedding = data["embedding"]
        line_start = data["line_start"]
        line_end = data["line_end"]
        metadata = data.get("metadata", {})
        config_guards = data.get("config_guards", [])

        # Create instance (will auto-generate created_at)
        return cls(
            content_id=content_id,
            file_path=file_path,
            content_type=content_type,
            content_text=content_text,
            embedding=embedding,
            line_start=line_start,
            line_end=line_end,
            metadata=metadata,
            config_guards=config_guards,
        )

    def __str__(self) -> str:
        """String representation of VectorEmbedding."""
        line_range = (
            f"{self.line_start}-{self.line_end}"
            if self.line_start != self.line_end
            else str(self.line_start)
        )
        return f"VectorEmbedding({self.content_id}: {self.file_path}:{line_range})"

    def __repr__(self) -> str:
        """Detailed representation of VectorEmbedding."""
        return (
            f"VectorEmbedding("
            f"content_id={self.content_id!r}, "
            f"file_path={self.file_path!r}, "
            f"content_type={self.content_type!r}, "
            f"line_start={self.line_start}, "
            f"line_end={self.line_end}, "
            f"created_at={self.created_at!r})"
        )

    # Database integration methods (placeholders for T026-T028)
    def save(self) -> None:
        """Save VectorEmbedding to database."""
        # Placeholder - will be implemented in database layer tasks
        raise NotImplementedError(
            "Database operations will be implemented in T026-T028"
        )

    def delete(self) -> None:
        """Delete VectorEmbedding from database."""
        # Placeholder - will be implemented in database layer tasks
        raise NotImplementedError(
            "Database operations will be implemented in T026-T028"
        )

    @classmethod
    def find_by_content_id(cls, content_id: str) -> "VectorEmbedding | None":
        """Find VectorEmbedding by content_id."""
        # Placeholder - will be implemented in database layer tasks
        raise NotImplementedError(
            "Database operations will be implemented in T026-T028"
        )

    @classmethod
    def find_by_file_path(cls, file_path: str) -> list["VectorEmbedding"]:
        """Find VectorEmbeddings by file_path."""
        # Placeholder - will be implemented in database layer tasks
        raise NotImplementedError(
            "Database operations will be implemented in T026-T028"
        )

    @classmethod
    def find_similar(
        cls, query_embedding: list[float], limit: int = 10
    ) -> list["VectorEmbedding"]:
        """Find similar VectorEmbeddings using vector search."""
        # Placeholder - will be implemented in database layer tasks
        raise NotImplementedError(
            "Database operations will be implemented in T026-T028"
        )


class VectorEmbeddingPydantic(BaseModel):
    """
    Pydantic model for VectorEmbedding API validation.

    Used for MCP endpoint validation and serialization.
    Complements the immutable dataclass for different use cases.
    """

    content_id: str = Field(
        ..., min_length=1, description="Unique identifier for source content"
    )
    file_path: str = Field(..., description="Absolute path to source file")
    content_type: ContentType = Field(..., description="Type of content")
    content_text: str = Field(..., description="Original text content")
    embedding: list[float] = Field(
        ..., description="384-dimensional vector representation"
    )
    line_start: int = Field(..., ge=1, description="Starting line number")
    line_end: int = Field(..., ge=1, description="Ending line number")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    config_guards: list[str] = Field(
        default_factory=list, description="Configuration dependencies"
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path is absolute."""
        if not v.startswith("/"):
            raise ValueError("file_path must be an absolute path")
        if v == "/":
            raise ValueError("file_path cannot be root directory only")
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(cls, v: list[float]) -> list[float]:
        """Validate embedding is 384-dimensional."""
        if len(v) != 384:
            raise ValueError(f"embedding must be 384-dimensional, got: {len(v)}")
        return v

    @field_validator("line_end")
    @classmethod
    def validate_line_range(cls, v: int, info: Any) -> int:
        """Validate line_end >= line_start."""
        if hasattr(info.data, "line_start") and v < info.data["line_start"]:
            raise ValueError("line_end must be >= line_start")
        return v

    @field_validator("config_guards")
    @classmethod
    def validate_config_guards(cls, v: list[str]) -> list[str]:
        """Validate config guards format."""
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

    def to_vector_embedding(self) -> VectorEmbedding:
        """Convert to immutable VectorEmbedding dataclass."""
        return VectorEmbedding(
            content_id=self.content_id,
            file_path=self.file_path,
            content_type=self.content_type.value,
            content_text=self.content_text,
            embedding=self.embedding,
            line_start=self.line_start,
            line_end=self.line_end,
            metadata=self.metadata,
            config_guards=self.config_guards,
        )
