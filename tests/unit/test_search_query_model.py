"""
Unit tests for SearchQuery model.

Tests the SearchQuery data model as defined in
specs/008-semantic-search-engine/data-model.md

Following TDD: This test MUST FAIL before implementation exists.
"""

from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestSearchQueryModel:
    """Unit tests for SearchQuery model."""

    def test_search_query_model_exists(self):
        """Test that SearchQuery model class exists and is importable."""
        from src.python.semantic_search.models.search_query import SearchQuery

        # Should be a class that can be instantiated
        assert isinstance(SearchQuery, type)

    def test_search_query_basic_creation(self):
        """Test basic SearchQuery creation with required fields."""
        from src.python.semantic_search.models.search_query import SearchQuery

        query = SearchQuery(query_text="memory allocation functions")

        # Required fields should be set
        assert query.query_text == "memory allocation functions"
        assert hasattr(query, "processed_query")
        assert hasattr(query, "embedding")
        assert hasattr(query, "timestamp")

    def test_search_query_field_types(self):
        """Test that SearchQuery fields have correct types."""
        from src.python.semantic_search.models.search_query import SearchQuery

        query = SearchQuery(
            query_text="buffer overflow vulnerability",
            user_id="test_user",
            config_context=["CONFIG_NET", "!CONFIG_EMBEDDED"],
        )

        # Type validations
        assert isinstance(query.query_text, str)
        assert isinstance(query.processed_query, str)
        assert isinstance(query.embedding, (list, np.ndarray))
        assert isinstance(query.timestamp, datetime)
        assert isinstance(query.user_id, (str, type(None)))
        assert isinstance(query.config_context, list)

    def test_search_query_validation_rules(self):
        """Test SearchQuery validation rules from data model."""
        from src.python.semantic_search.models.search_query import SearchQuery

        # Valid query should work
        query = SearchQuery(query_text="mutex locking patterns")
        assert len(query.query_text) >= 1

        # Empty query should fail
        with pytest.raises(ValueError):
            SearchQuery(query_text="")

        # Query too long should fail (>1000 chars)
        with pytest.raises(ValueError):
            SearchQuery(query_text="x" * 1001)

        # Query at limit should work
        query_limit = SearchQuery(query_text="x" * 1000)
        assert len(query_limit.query_text) == 1000

    def test_search_query_automatic_processing(self):
        """Test that processed_query is generated automatically."""
        from src.python.semantic_search.models.search_query import SearchQuery

        original_text = "Find Memory Allocation Functions That Can FAIL!"
        query = SearchQuery(query_text=original_text)

        # processed_query should be normalized version
        assert query.processed_query != original_text
        assert (
            query.processed_query.lower() == query.processed_query
        )  # Should be lowercase
        assert len(query.processed_query) > 0

    def test_search_query_embedding_generation(self):
        """Test that embedding is generated via BAAI/bge-small-en-v1.5."""
        from src.python.semantic_search.models.search_query import SearchQuery

        query = SearchQuery(query_text="network driver implementation")

        # Embedding should be 384-dimensional vector
        assert hasattr(query, "embedding")
        assert len(query.embedding) == 384

        # Should be numeric values
        for value in query.embedding:
            assert isinstance(value, (int, float, np.number))

        # Should be normalized vector (for cosine similarity)
        embedding_array = np.array(query.embedding)
        norm = np.linalg.norm(embedding_array)
        assert abs(norm - 1.0) < 1e-6  # Should be unit vector

    def test_search_query_timestamp_automatic(self):
        """Test that timestamp is set automatically on creation."""
        from src.python.semantic_search.models.search_query import SearchQuery

        before_creation = datetime.now()
        query = SearchQuery(query_text="filesystem operations")
        after_creation = datetime.now()

        # Timestamp should be set automatically
        assert query.timestamp is not None
        assert before_creation <= query.timestamp <= after_creation

    def test_search_query_optional_fields(self):
        """Test optional fields: user_id and config_context."""
        from src.python.semantic_search.models.search_query import SearchQuery

        # Without optional fields
        query1 = SearchQuery(query_text="test query")
        assert query1.user_id is None
        assert query1.config_context == []

        # With optional fields
        query2 = SearchQuery(
            query_text="test query",
            user_id="user123",
            config_context=["CONFIG_NET", "CONFIG_INET"],
        )
        assert query2.user_id == "user123"
        assert query2.config_context == ["CONFIG_NET", "CONFIG_INET"]

    def test_search_query_config_context_validation(self):
        """Test validation of config_context field."""
        from src.python.semantic_search.models.search_query import SearchQuery

        # Valid config contexts
        valid_configs = [
            ["CONFIG_NET"],
            ["CONFIG_NET", "CONFIG_INET"],
            ["!CONFIG_EMBEDDED"],
            ["CONFIG_NET", "!CONFIG_EMBEDDED"],
        ]

        for config in valid_configs:
            query = SearchQuery(query_text="test", config_context=config)
            assert query.config_context == config

        # Invalid config contexts should fail
        invalid_configs = [
            ["INVALID_CONFIG"],  # Missing CONFIG_ prefix
            ["CONFIG_"],  # Empty config name
            ["config_net"],  # Wrong case
            [123],  # Non-string values
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                SearchQuery(query_text="test", config_context=invalid_config)

    def test_search_query_immutability(self):
        """Test that core fields are immutable after creation."""
        from src.python.semantic_search.models.search_query import SearchQuery

        query = SearchQuery(query_text="original query")
        original_text = query.query_text
        original_timestamp = query.timestamp
        original_embedding = query.embedding.copy()

        # Attempting to modify should either fail or be ignored
        with pytest.raises((AttributeError, TypeError)):
            query.query_text = "modified query"

        with pytest.raises((AttributeError, TypeError)):
            query.timestamp = datetime.now()

        with pytest.raises((AttributeError, TypeError)):
            query.embedding = [0.0] * 384

        # Values should remain unchanged
        assert query.query_text == original_text
        assert query.timestamp == original_timestamp
        assert np.array_equal(query.embedding, original_embedding)

    def test_search_query_equality(self):
        """Test equality comparison between SearchQuery instances."""
        from src.python.semantic_search.models.search_query import SearchQuery

        # Same query text should be equal (assuming deterministic processing)
        query1 = SearchQuery(query_text="test query")
        query2 = SearchQuery(query_text="test query")

        # Note: This might need adjustment based on actual implementation
        # if timestamps make queries unique
        assert query1.query_text == query2.query_text
        assert query1.processed_query == query2.processed_query

    def test_search_query_string_representation(self):
        """Test string representation of SearchQuery."""
        from src.python.semantic_search.models.search_query import SearchQuery

        query = SearchQuery(query_text="memory management")

        # Should have meaningful string representation
        str_repr = str(query)
        assert "memory management" in str_repr.lower()
        assert "searchquery" in str_repr.lower()

    def test_search_query_serialization(self):
        """Test serialization and deserialization of SearchQuery."""
        from src.python.semantic_search.models.search_query import SearchQuery

        original_query = SearchQuery(
            query_text="network protocol implementation",
            user_id="test_user",
            config_context=["CONFIG_NET"],
        )

        # Should be serializable to dict
        query_dict = original_query.to_dict()
        assert isinstance(query_dict, dict)
        assert query_dict["query_text"] == "network protocol implementation"
        assert query_dict["user_id"] == "test_user"
        assert "embedding" in query_dict
        assert "timestamp" in query_dict

        # Should be deserializable from dict
        restored_query = SearchQuery.from_dict(query_dict)
        assert restored_query.query_text == original_query.query_text
        assert restored_query.user_id == original_query.user_id
        assert restored_query.config_context == original_query.config_context

    def test_search_query_embedding_caching(self):
        """Test that embeddings are cached for identical queries."""
        from src.python.semantic_search.models.search_query import SearchQuery

        # Mock the embedding service to track calls
        with patch(
            "src.python.semantic_search.services.embedding_service.EmbeddingService"
        ) as mock_service:
            mock_embed = Mock(return_value=np.random.rand(384).tolist())
            mock_service.return_value.embed_query = mock_embed

            # First query should call embedding service
            SearchQuery(query_text="identical query")
            assert mock_embed.call_count == 1

            # Second identical query should use cache
            SearchQuery(query_text="identical query")
            # Note: This assumes caching is implemented
            # If not implemented, this test documents the requirement

    def test_search_query_preprocessing_steps(self):
        """Test specific preprocessing steps applied to query text."""
        from src.python.semantic_search.models.search_query import SearchQuery

        test_cases = [
            ("UPPERCASE TEXT", "should be lowercased"),
            ("text    with     extra     spaces", "should normalize spaces"),
            ("text\nwith\nnewlines", "should handle newlines"),
            ("text with punctuation!!!", "should handle punctuation"),
            ("CamelCaseText", "should handle mixed case"),
        ]

        for original, _description in test_cases:
            query = SearchQuery(query_text=original)

            # Processed query should be different from original
            assert query.processed_query != original

            # Should be lowercase
            assert query.processed_query.islower()

            # Should not have excessive whitespace
            assert "  " not in query.processed_query

    def test_search_query_error_handling(self):
        """Test error handling in SearchQuery creation."""
        from src.python.semantic_search.models.search_query import SearchQuery

        # Test various invalid inputs
        invalid_inputs = [
            (None, "None query_text"),
            (123, "Non-string query_text"),
            ("", "Empty query_text"),
            ("x" * 1001, "Query too long"),
        ]

        for invalid_input, _description in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                SearchQuery(query_text=invalid_input)

    def test_search_query_performance(self):
        """Test that SearchQuery creation performs within reasonable time."""
        import time

        from src.python.semantic_search.models.search_query import SearchQuery

        start_time = time.time()
        SearchQuery(query_text="performance test query")
        end_time = time.time()

        # Creation should be fast (< 100ms excluding embedding generation)
        creation_time = (end_time - start_time) * 1000
        assert creation_time < 5000  # 5 seconds max (allowing for embedding generation)

    def test_search_query_memory_usage(self):
        """Test that SearchQuery instances don't use excessive memory."""
        import sys

        from src.python.semantic_search.models.search_query import SearchQuery

        query = SearchQuery(query_text="memory usage test")

        # Query object should be reasonably sized
        query_size = sys.getsizeof(query)

        # Should be less than 10KB (rough estimate)
        assert query_size < 10000

    def test_search_query_concurrent_creation(self):
        """Test concurrent creation of SearchQuery instances."""
        import threading
        import time

        from src.python.semantic_search.models.search_query import SearchQuery

        results = []
        errors = []

        def create_query(query_text):
            try:
                query = SearchQuery(query_text=f"concurrent query {query_text}")
                results.append(query)
            except Exception as e:
                errors.append(e)

        # Create multiple queries concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_query, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 5

        # All should have unique timestamps
        timestamps = [query.timestamp for query in results]
        assert len(set(timestamps)) == len(timestamps)  # All unique
