"""
Placeholder EmbeddingService for SearchQuery model.

This is a temporary implementation to allow SearchQuery tests to pass.
Will be properly implemented in T022.
"""

from typing import cast

import numpy as np


class EmbeddingService:
    """Placeholder embedding service using BAAI/bge-small-en-v1.5 model."""

    def __init__(self) -> None:
        """Initialize embedding service."""
        # Placeholder initialization
        self._model_name = "BAAI/bge-small-en-v1.5"
        self._dimension = 384

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for query text.

        Args:
            text: Query text to embed

        Returns:
            384-dimensional embedding vector
        """
        # Placeholder implementation - generates deterministic random vector
        # based on text hash for consistent test results
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension).astype(np.float32)

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return cast(list[float], embedding.tolist())
