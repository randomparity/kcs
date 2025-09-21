"""
Unit tests for VectorEmbedding model.

Tests the VectorEmbedding data model as defined in
specs/008-semantic-search-engine/data-model.md

Following TDD: This test MUST FAIL before implementation exists.
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestVectorEmbeddingModel:
    """Unit tests for VectorEmbedding model."""

    def test_vector_embedding_model_exists(self):
        """Test that VectorEmbedding model class exists and is importable."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Should be a class that can be instantiated
        assert isinstance(VectorEmbedding, type)

    def test_vector_embedding_basic_creation(self):
        """Test basic VectorEmbedding creation with required fields."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        embedding = VectorEmbedding(
            content_id="content_123",
            file_path="/kernel/mm/slub.c",
            content_type="SOURCE_CODE",
            content_text="static void *__slab_alloc(struct kmem_cache *s, gfp_t gfpflags, int node)",
            embedding=np.random.rand(384).tolist(),
            line_start=1234,
            line_end=1240,
        )

        # Required fields should be set
        assert embedding.content_id == "content_123"
        assert embedding.file_path == "/kernel/mm/slub.c"
        assert embedding.content_type == "SOURCE_CODE"
        assert embedding.content_text.startswith("static void *__slab_alloc")
        assert len(embedding.embedding) == 384
        assert embedding.line_start == 1234
        assert embedding.line_end == 1240

    def test_vector_embedding_field_types(self):
        """Test that VectorEmbedding fields have correct types."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        embedding = VectorEmbedding(
            content_id="test_content",
            file_path="/drivers/net/e1000/e1000_main.c",
            content_type="SOURCE_CODE",
            content_text="int e1000_setup_rx_resources(struct e1000_adapter *adapter)",
            embedding=np.random.rand(384).tolist(),
            line_start=500,
            line_end=520,
            metadata={
                "function_name": "e1000_setup_rx_resources",
                "symbols": ["e1000_adapter"],
            },
            config_guards=["CONFIG_NET", "CONFIG_E1000"],
        )

        # Type validations
        assert isinstance(embedding.content_id, str)
        assert isinstance(embedding.file_path, str)
        assert isinstance(embedding.content_type, str)
        assert isinstance(embedding.content_text, str)
        assert isinstance(embedding.embedding, (list, np.ndarray))
        assert isinstance(embedding.line_start, int)
        assert isinstance(embedding.line_end, int)
        assert isinstance(embedding.metadata, dict)
        assert isinstance(embedding.created_at, datetime)
        assert isinstance(embedding.config_guards, list)

    def test_vector_embedding_content_type_validation(self):
        """Test content_type enum validation."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        valid_content_types = ["SOURCE_CODE", "DOCUMENTATION", "HEADER", "COMMENT"]

        for content_type in valid_content_types:
            embedding = VectorEmbedding(
                content_id="test",
                file_path="/test.c",
                content_type=content_type,
                content_text="test content",
                embedding=np.random.rand(384).tolist(),
                line_start=1,
                line_end=1,
            )
            assert embedding.content_type == content_type

        # Invalid content type should fail
        with pytest.raises(ValueError):
            VectorEmbedding(
                content_id="test",
                file_path="/test.c",
                content_type="INVALID_TYPE",
                content_text="test content",
                embedding=np.random.rand(384).tolist(),
                line_start=1,
                line_end=1,
            )

    def test_vector_embedding_file_path_validation(self):
        """Test file_path validation rules."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Valid absolute paths should work
        valid_paths = [
            "/kernel/mm/slub.c",
            "/drivers/net/ethernet/intel/e1000/e1000_main.c",
            "/fs/ext4/inode.c",
            "/arch/x86/kernel/setup.c",
        ]

        for path in valid_paths:
            embedding = VectorEmbedding(
                content_id="test",
                file_path=path,
                content_type="SOURCE_CODE",
                content_text="test content",
                embedding=np.random.rand(384).tolist(),
                line_start=1,
                line_end=1,
            )
            assert embedding.file_path == path

        # Invalid paths should fail
        invalid_paths = [
            "relative/path.c",  # Not absolute
            "",  # Empty
            "/",  # Root only
            "invalid path with spaces",  # Relative with spaces
        ]

        for invalid_path in invalid_paths:
            with pytest.raises(ValueError):
                VectorEmbedding(
                    content_id="test",
                    file_path=invalid_path,
                    content_type="SOURCE_CODE",
                    content_text="test content",
                    embedding=np.random.rand(384).tolist(),
                    line_start=1,
                    line_end=1,
                )

    def test_vector_embedding_line_number_validation(self):
        """Test line number validation: line_start ≤ line_end."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Valid line ranges
        valid_ranges = [
            (1, 1),  # Single line
            (10, 20),  # Multi-line
            (100, 150),  # Larger range
        ]

        for start, end in valid_ranges:
            embedding = VectorEmbedding(
                content_id="test",
                file_path="/test.c",
                content_type="SOURCE_CODE",
                content_text="test content",
                embedding=np.random.rand(384).tolist(),
                line_start=start,
                line_end=end,
            )
            assert embedding.line_start == start
            assert embedding.line_end == end

        # Invalid ranges should fail
        invalid_ranges = [
            (20, 10),  # start > end
            (0, 5),  # start < 1
            (5, 0),  # end < 1
            (-1, 5),  # negative start
            (5, -1),  # negative end
        ]

        for start, end in invalid_ranges:
            with pytest.raises(ValueError):
                VectorEmbedding(
                    content_id="test",
                    file_path="/test.c",
                    content_type="SOURCE_CODE",
                    content_text="test content",
                    embedding=np.random.rand(384).tolist(),
                    line_start=start,
                    line_end=end,
                )

    def test_vector_embedding_dimension_validation(self):
        """Test that embedding must be 384-dimensional vector."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Valid 384-dimensional embedding
        valid_embedding = np.random.rand(384).tolist()
        embedding = VectorEmbedding(
            content_id="test",
            file_path="/test.c",
            content_type="SOURCE_CODE",
            content_text="test content",
            embedding=valid_embedding,
            line_start=1,
            line_end=1,
        )
        assert len(embedding.embedding) == 384

        # Invalid dimensions should fail
        invalid_embeddings = [
            np.random.rand(128).tolist(),  # Too small
            np.random.rand(512).tolist(),  # Too large
            [],  # Empty
            np.random.rand(383).tolist(),  # Off by one
            np.random.rand(385).tolist(),  # Off by one
        ]

        for invalid_emb in invalid_embeddings:
            with pytest.raises(ValueError):
                VectorEmbedding(
                    content_id="test",
                    file_path="/test.c",
                    content_type="SOURCE_CODE",
                    content_text="test content",
                    embedding=invalid_emb,
                    line_start=1,
                    line_end=1,
                )

    def test_vector_embedding_metadata_structure(self):
        """Test metadata field structure and validation."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Valid metadata structures
        valid_metadata = [
            {"function_name": "kmalloc"},
            {"function_name": "e1000_setup", "symbols": ["adapter", "ring"]},
            {"symbols": ["mutex", "lock"], "config_guards": ["CONFIG_PREEMPT"]},
            {},  # Empty metadata
        ]

        for metadata in valid_metadata:
            embedding = VectorEmbedding(
                content_id="test",
                file_path="/test.c",
                content_type="SOURCE_CODE",
                content_text="test content",
                embedding=np.random.rand(384).tolist(),
                line_start=1,
                line_end=1,
                metadata=metadata,
            )
            assert embedding.metadata == metadata

        # Invalid metadata should fail
        invalid_metadata = [
            "not a dict",  # Wrong type
            123,  # Wrong type
            None,  # None not allowed for metadata field
        ]

        for invalid_meta in invalid_metadata:
            with pytest.raises((ValueError, TypeError)):
                VectorEmbedding(
                    content_id="test",
                    file_path="/test.c",
                    content_type="SOURCE_CODE",
                    content_text="test content",
                    embedding=np.random.rand(384).tolist(),
                    line_start=1,
                    line_end=1,
                    metadata=invalid_meta,
                )

    def test_vector_embedding_config_guards_validation(self):
        """Test config_guards field validation."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Valid config guards
        valid_configs = [
            ["CONFIG_NET"],
            ["CONFIG_NET", "CONFIG_INET"],
            ["!CONFIG_EMBEDDED"],
            [],  # Empty list
        ]

        for config_guards in valid_configs:
            embedding = VectorEmbedding(
                content_id="test",
                file_path="/test.c",
                content_type="SOURCE_CODE",
                content_text="test content",
                embedding=np.random.rand(384).tolist(),
                line_start=1,
                line_end=1,
                config_guards=config_guards,
            )
            assert embedding.config_guards == config_guards

        # Invalid config guards should fail
        invalid_configs = [
            ["invalid_config"],  # Missing CONFIG_ prefix
            ["CONFIG_"],  # Empty config name
            [123],  # Non-string values
            "CONFIG_NET",  # Not a list
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                VectorEmbedding(
                    content_id="test",
                    file_path="/test.c",
                    content_type="SOURCE_CODE",
                    content_text="test content",
                    embedding=np.random.rand(384).tolist(),
                    line_start=1,
                    line_end=1,
                    config_guards=invalid_config,
                )

    def test_vector_embedding_created_at_automatic(self):
        """Test that created_at is set automatically."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        before_creation = datetime.now()
        embedding = VectorEmbedding(
            content_id="test",
            file_path="/test.c",
            content_type="SOURCE_CODE",
            content_text="test content",
            embedding=np.random.rand(384).tolist(),
            line_start=1,
            line_end=1,
        )
        after_creation = datetime.now()

        # created_at should be set automatically
        assert embedding.created_at is not None
        assert before_creation <= embedding.created_at <= after_creation

    def test_vector_embedding_unique_content_id(self):
        """Test that content_id generation ensures uniqueness."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # If content_id is auto-generated, test uniqueness
        embeddings = []
        for i in range(10):
            embedding = VectorEmbedding(
                content_id=f"auto_generated_{i}",  # Or auto-generate if supported
                file_path=f"/test_{i}.c",
                content_type="SOURCE_CODE",
                content_text=f"test content {i}",
                embedding=np.random.rand(384).tolist(),
                line_start=1,
                line_end=1,
            )
            embeddings.append(embedding)

        # All content_ids should be unique
        content_ids = [emb.content_id for emb in embeddings]
        assert len(set(content_ids)) == len(content_ids)

    def test_vector_embedding_serialization(self):
        """Test serialization and deserialization of VectorEmbedding."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        original_embedding = VectorEmbedding(
            content_id="serialization_test",
            file_path="/kernel/sched/core.c",
            content_type="SOURCE_CODE",
            content_text="void schedule(void)",
            embedding=np.random.rand(384).tolist(),
            line_start=100,
            line_end=120,
            metadata={"function_name": "schedule", "symbols": ["task_struct"]},
            config_guards=["CONFIG_PREEMPT"],
        )

        # Should be serializable to dict
        embedding_dict = original_embedding.to_dict()
        assert isinstance(embedding_dict, dict)
        assert embedding_dict["content_id"] == "serialization_test"
        assert embedding_dict["file_path"] == "/kernel/sched/core.c"
        assert embedding_dict["content_type"] == "SOURCE_CODE"
        assert len(embedding_dict["embedding"]) == 384

        # Should be deserializable from dict
        restored_embedding = VectorEmbedding.from_dict(embedding_dict)
        assert restored_embedding.content_id == original_embedding.content_id
        assert restored_embedding.file_path == original_embedding.file_path
        assert restored_embedding.content_type == original_embedding.content_type
        assert restored_embedding.content_text == original_embedding.content_text
        assert np.array_equal(
            restored_embedding.embedding, original_embedding.embedding
        )

    def test_vector_embedding_database_integration(self):
        """Test database integration aspects."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        embedding = VectorEmbedding(
            content_id="db_test",
            file_path="/fs/ext4/super.c",
            content_type="SOURCE_CODE",
            content_text="struct super_block *ext4_alloc_super(void)",
            embedding=np.random.rand(384).tolist(),
            line_start=200,
            line_end=210,
        )

        # Should support database operations
        assert hasattr(embedding, "save")  # Save to database
        assert hasattr(embedding, "delete")  # Delete from database

        # Should support querying
        assert hasattr(VectorEmbedding, "find_by_content_id")
        assert hasattr(VectorEmbedding, "find_by_file_path")
        assert hasattr(VectorEmbedding, "find_similar")

    def test_vector_embedding_similarity_search(self):
        """Test vector similarity search functionality."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Create test embeddings
        query_embedding = np.random.rand(384).tolist()

        embedding = VectorEmbedding(
            content_id="similarity_test",
            file_path="/test.c",
            content_type="SOURCE_CODE",
            content_text="test content",
            embedding=np.random.rand(384).tolist(),
            line_start=1,
            line_end=1,
        )

        # Should support similarity calculation
        similarity = embedding.calculate_similarity(query_embedding)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_vector_embedding_immutability(self):
        """Test that core fields are immutable after creation."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        embedding = VectorEmbedding(
            content_id="immutable_test",
            file_path="/test.c",
            content_type="SOURCE_CODE",
            content_text="original content",
            embedding=np.random.rand(384).tolist(),
            line_start=1,
            line_end=1,
        )

        original_id = embedding.content_id
        original_path = embedding.file_path
        original_embedding = embedding.embedding.copy()

        # Attempting to modify should fail or be ignored
        with pytest.raises((AttributeError, TypeError)):
            embedding.content_id = "modified_id"

        with pytest.raises((AttributeError, TypeError)):
            embedding.file_path = "/modified.c"

        with pytest.raises((AttributeError, TypeError)):
            embedding.embedding = [0.0] * 384

        # Values should remain unchanged
        assert embedding.content_id == original_id
        assert embedding.file_path == original_path
        assert np.array_equal(embedding.embedding, original_embedding)

    def test_vector_embedding_content_chunking(self):
        """Test handling of content chunking for large files."""
        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        # Large content that might be chunked
        large_content = """
        static int ext4_fill_super(struct super_block *sb, void *data, int silent)
        {
            struct buffer_head *bh;
            struct ext4_super_block *es = NULL;
            struct ext4_sb_info *sbi = kzalloc(sizeof(*sbi), GFP_KERNEL);
            // ... many more lines of code ...
        }
        """.strip()

        embedding = VectorEmbedding(
            content_id="chunking_test",
            file_path="/fs/ext4/super.c",
            content_type="SOURCE_CODE",
            content_text=large_content,
            embedding=np.random.rand(384).tolist(),
            line_start=1000,
            line_end=1050,
        )

        # Should handle large content appropriately
        assert len(embedding.content_text) > 100
        assert embedding.line_start < embedding.line_end

    def test_vector_embedding_performance(self):
        """Test VectorEmbedding creation and operation performance."""
        import time

        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        start_time = time.time()

        # Create multiple embeddings
        embeddings = []
        for i in range(100):
            embedding = VectorEmbedding(
                content_id=f"perf_test_{i}",
                file_path=f"/test_{i}.c",
                content_type="SOURCE_CODE",
                content_text=f"test content {i}",
                embedding=np.random.rand(384).tolist(),
                line_start=i + 1,  # Line numbers start at 1
                line_end=i + 11,  # Ensure line_end > line_start
            )
            embeddings.append(embedding)

        end_time = time.time()

        # Should create quickly
        creation_time = (end_time - start_time) * 1000
        assert creation_time < 1000  # Less than 1 second for 100 embeddings

    def test_vector_embedding_memory_efficiency(self):
        """Test memory efficiency of VectorEmbedding instances."""
        import sys

        from src.python.semantic_search.models.vector_embedding import VectorEmbedding

        embedding = VectorEmbedding(
            content_id="memory_test",
            file_path="/test.c",
            content_type="SOURCE_CODE",
            content_text="test content",
            embedding=np.random.rand(384).tolist(),
            line_start=1,
            line_end=1,
        )

        # Should be reasonably sized
        embedding_size = sys.getsizeof(embedding)

        # Should be less than 20KB (rough estimate including embedding vector)
        assert embedding_size < 20000
