"""
EmbeddingService implementation using BAAI/bge-small-en-v1.5 model.

Provides semantic text embeddings for the kernel context search system
using CPU-optimized sentence transformers.
"""

import logging
import threading
from typing import ClassVar, cast

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service using BAAI/bge-small-en-v1.5 model for semantic search."""

    # Class-level model cache for thread safety and efficiency
    _model_cache: ClassVar[dict[str, SentenceTransformer]] = {}
    _model_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        """Initialize embedding service with BAAI/bge-small-en-v1.5 model."""
        self._model_name = "BAAI/bge-small-en-v1.5"
        self._dimension = 384

        # Initialize model lazily for performance
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Ensure the embedding model is loaded and ready for use."""
        # Check cache first without lock for performance
        if self._model_name in self._model_cache:
            return

        # Use lock for thread-safe model loading
        with self._model_lock:
            # Double-check pattern - another thread might have loaded it
            if self._model_name in self._model_cache:
                return

            try:
                logger.info(f"Loading embedding model: {self._model_name}")
                # Load model with CPU device for constitutional compliance
                model = SentenceTransformer(
                    self._model_name,
                    device="cpu",
                    trust_remote_code=False,  # Security best practice
                )
                # Cache the model for reuse across instances
                self._model_cache[self._model_name] = model
                logger.info(f"Successfully loaded {self._model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model {self._model_name}: {e}")
                raise RuntimeError(f"Could not initialize embedding model: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for query text using BAAI/bge-small-en-v1.5.

        Args:
            text: Query text to embed (preprocessed)

        Returns:
            384-dimensional embedding vector as list of floats

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If model fails to generate embedding
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text or len(text.strip()) == 0:
            raise ValueError("text cannot be empty")

        self._ensure_model_loaded()

        try:
            # Get model from cache
            model = self._model_cache[self._model_name]

            # Generate embedding using sentence-transformers
            # convert_to_numpy=True for efficiency, normalize_embeddings=True for cosine similarity
            embedding = model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]  # Get first (and only) embedding

            # Verify dimensions
            if len(embedding) != self._dimension:
                raise RuntimeError(
                    f"Model returned {len(embedding)}-dimensional vector, "
                    f"expected {self._dimension}"
                )

            # Convert to list of floats for consistency
            return cast(list[float], embedding.tolist())

        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embedding vectors

        Raises:
            ValueError: If texts list is empty or contains invalid text
            RuntimeError: If model fails to generate embeddings
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        if not texts:
            raise ValueError("texts list cannot be empty")

        # Validate all texts
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}] must be a string")
            if not text or len(text.strip()) == 0:
                raise ValueError(f"texts[{i}] cannot be empty")

        self._ensure_model_loaded()

        try:
            # Get model from cache
            model = self._model_cache[self._model_name]

            # Generate embeddings in batch for efficiency
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Verify dimensions for all embeddings
            for i, embedding in enumerate(embeddings):
                if len(embedding) != self._dimension:
                    raise RuntimeError(
                        f"Model returned {len(embedding)}-dimensional vector for texts[{i}], "
                        f"expected {self._dimension}"
                    )

            # Convert to list of lists of floats
            return [cast(list[float], embedding.tolist()) for embedding in embeddings]

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self._dimension

    def is_model_loaded(self) -> bool:
        """Check if the embedding model is currently loaded."""
        return self._model_name in self._model_cache
