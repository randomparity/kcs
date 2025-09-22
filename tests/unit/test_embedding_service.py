"""
Unit tests for EmbeddingService implementation.

Tests the BAAI/bge-small-en-v1.5 embedding service for proper functionality,
error handling, and performance characteristics.
"""

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.python.semantic_search.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService class."""

    def setup_method(self):
        """Reset class-level caches before each test."""
        # Clear the model cache to ensure clean test state
        EmbeddingService._model_cache.clear()

    def test_embedding_service_initialization(self):
        """Test that EmbeddingService initializes correctly."""
        service = EmbeddingService()

        assert service.model_name == "BAAI/bge-small-en-v1.5"
        assert service.dimension == 384
        assert isinstance(service._model_name, str)
        assert isinstance(service._dimension, int)

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_model_loading_success(self, mock_sentence_transformer):
        """Test successful model loading and caching."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        service = EmbeddingService()

        # Verify model was loaded
        mock_sentence_transformer.assert_called_once_with(
            "BAAI/bge-small-en-v1.5", device="cpu", trust_remote_code=False
        )

        # Verify model is cached
        assert service.is_model_loaded()
        assert "BAAI/bge-small-en-v1.5" in EmbeddingService._model_cache

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_model_loading_failure(self, mock_sentence_transformer):
        """Test proper error handling when model loading fails."""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")

        with pytest.raises(RuntimeError, match="Could not initialize embedding model"):
            EmbeddingService()

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_embed_query_success(self, mock_sentence_transformer):
        """Test successful query embedding generation."""
        # Mock model and its encode method
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3] + [0.0] * 381)  # 384 dimensions
        mock_model.encode.return_value = np.array([mock_embedding])
        mock_sentence_transformer.return_value = mock_model

        service = EmbeddingService()
        result = service.embed_query("test query")

        # Verify method calls
        mock_model.encode.assert_called_once_with(
            ["test query"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_embed_query_invalid_input(self):
        """Test embed_query with invalid input types and values."""
        service = EmbeddingService()

        # Test non-string input
        with pytest.raises(TypeError, match="text must be a string"):
            service.embed_query(123)

        # Test empty string
        with pytest.raises(ValueError, match="text cannot be empty"):
            service.embed_query("")

        # Test whitespace-only string
        with pytest.raises(ValueError, match="text cannot be empty"):
            service.embed_query("   ")

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_embed_query_wrong_dimensions(self, mock_sentence_transformer):
        """Test error handling when model returns wrong dimensions."""
        mock_model = Mock()
        # Return wrong number of dimensions
        mock_embedding = np.array([0.1, 0.2, 0.3])  # Only 3 dimensions instead of 384
        mock_model.encode.return_value = np.array([mock_embedding])
        mock_sentence_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(
            RuntimeError, match="Model returned 3-dimensional vector, expected 384"
        ):
            service.embed_query("test")

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_embed_batch_success(self, mock_sentence_transformer):
        """Test successful batch embedding generation."""
        mock_model = Mock()
        # Create two embeddings with correct dimensions
        mock_embedding1 = np.array([0.1, 0.2] + [0.0] * 382)
        mock_embedding2 = np.array([0.3, 0.4] + [0.0] * 382)
        mock_model.encode.return_value = np.array([mock_embedding1, mock_embedding2])
        mock_sentence_transformer.return_value = mock_model

        service = EmbeddingService()
        texts = ["first text", "second text"]
        result = service.embed_batch(texts)

        # Verify method calls
        mock_model.encode.assert_called_once_with(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(len(embedding) == 384 for embedding in result)
        assert all(isinstance(x, float) for embedding in result for x in embedding)

    def test_embed_batch_invalid_input(self):
        """Test embed_batch with invalid input types and values."""
        service = EmbeddingService()

        # Test non-list input
        with pytest.raises(TypeError, match="texts must be a list"):
            service.embed_batch("not a list")

        # Test empty list
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            service.embed_batch([])

        # Test list with non-string elements
        with pytest.raises(TypeError, match="texts\\[0\\] must be a string"):
            service.embed_batch([123, "valid text"])

        # Test list with empty string
        with pytest.raises(ValueError, match="texts\\[1\\] cannot be empty"):
            service.embed_batch(["valid text", ""])

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_embed_batch_wrong_dimensions(self, mock_sentence_transformer):
        """Test error handling when batch model returns wrong dimensions."""
        mock_model = Mock()
        # Return wrong dimensions for second embedding using a list to avoid numpy shape issues
        mock_embedding1 = [0.1] + [0.0] * 383  # Correct: 384 dimensions
        mock_embedding2 = [0.2, 0.3]  # Wrong: only 2 dimensions
        mock_model.encode.return_value = [mock_embedding1, mock_embedding2]
        mock_sentence_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(
            RuntimeError,
            match="Model returned 2-dimensional vector for texts\\[1\\], expected 384",
        ):
            service.embed_batch(["text1", "text2"])

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_model_caching_behavior(self, mock_sentence_transformer):
        """Test that model is properly cached and reused across instances."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Create first service instance
        service1 = EmbeddingService()
        assert mock_sentence_transformer.call_count == 1

        # Create second service instance - should use cached model
        service2 = EmbeddingService()
        assert mock_sentence_transformer.call_count == 1  # No additional calls

        # Both services should report model as loaded
        assert service1.is_model_loaded()
        assert service2.is_model_loaded()

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_thread_safety(self, mock_sentence_transformer):
        """Test that model loading is thread-safe."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Clear cache to ensure clean state
        EmbeddingService._model_cache.clear()

        services = []
        errors = []

        def create_service():
            try:
                service = EmbeddingService()
                services.append(service)
            except Exception as e:
                errors.append(e)

        # Create multiple threads that try to initialize service simultaneously
        threads = [threading.Thread(target=create_service) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify all services were created successfully
        assert len(services) == 10

        # Verify model was only loaded once despite multiple threads
        assert mock_sentence_transformer.call_count == 1

        # Verify all services report model as loaded
        assert all(service.is_model_loaded() for service in services)

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_model_properties(self, mock_sentence_transformer):
        """Test that model properties return correct values."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        service = EmbeddingService()

        assert service.model_name == "BAAI/bge-small-en-v1.5"
        assert service.dimension == 384
        assert isinstance(service.model_name, str)
        assert isinstance(service.dimension, int)

    @patch("src.python.semantic_search.services.embedding_service.SentenceTransformer")
    def test_encode_error_handling(self, mock_sentence_transformer):
        """Test proper error handling when encode method fails."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_sentence_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            service.embed_query("test")

        with pytest.raises(RuntimeError, match="Batch embedding generation failed"):
            service.embed_batch(["test"])

    def test_is_model_loaded_before_loading(self):
        """Test is_model_loaded returns False before model is loaded."""
        # Clear cache to ensure clean state
        EmbeddingService._model_cache.clear()

        service = EmbeddingService.__new__(
            EmbeddingService
        )  # Create without calling __init__
        service._model_name = "BAAI/bge-small-en-v1.5"

        assert not service.is_model_loaded()
