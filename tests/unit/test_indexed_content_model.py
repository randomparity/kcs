"""
Unit tests for IndexedContent model.

Tests the IndexedContent data model as defined in
specs/008-semantic-search-engine/data-model.md

Following TDD: This test MUST FAIL before implementation exists.
"""

import os
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest


class TestIndexedContentModel:
    """Unit tests for IndexedContent model."""

    def test_indexed_content_model_exists(self):
        """Test that IndexedContent model class exists and is importable."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # Should be a class that can be instantiated
        assert isinstance(IndexedContent, type)

    def test_indexed_content_basic_creation(self):
        """Test basic IndexedContent creation with required fields."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        content = IndexedContent(
            file_path="/kernel/mm/slub.c",
            file_type="C_SOURCE",
            file_size=65536,
            last_modified=datetime(2024, 1, 15, 10, 30, 0),
            chunk_count=25,
            processing_status="COMPLETED",
        )

        # Required fields should be set
        assert content.file_path == "/kernel/mm/slub.c"
        assert content.file_type == "C_SOURCE"
        assert content.file_size == 65536
        assert content.last_modified == datetime(2024, 1, 15, 10, 30, 0)
        assert content.chunk_count == 25
        assert content.processing_status == "COMPLETED"

    def test_indexed_content_field_types(self):
        """Test that IndexedContent fields have correct types."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        content = IndexedContent(
            file_path="/drivers/net/e1000/e1000_main.c",
            file_type="C_SOURCE",
            file_size=120000,
            last_modified=datetime.now(),
            chunk_count=45,
            processing_status="PROCESSING",
            error_message=None,
        )

        # Type validations
        assert isinstance(content.file_path, str)
        assert isinstance(content.file_type, str)
        assert isinstance(content.file_size, int)
        assert isinstance(content.last_modified, datetime)
        assert isinstance(content.indexed_at, datetime)
        assert isinstance(content.chunk_count, int)
        assert isinstance(content.processing_status, str)
        assert content.error_message is None or isinstance(content.error_message, str)

    def test_indexed_content_file_type_validation(self):
        """Test file_type enum validation."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        valid_file_types = ["C_SOURCE", "C_HEADER", "DOCUMENTATION", "MAKEFILE"]

        for file_type in valid_file_types:
            content = IndexedContent(
                file_path=f"/test/file.{file_type.lower()}",
                file_type=file_type,
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status="COMPLETED",
            )
            assert content.file_type == file_type

        # Invalid file type should fail
        with pytest.raises(ValueError):
            IndexedContent(
                file_path="/test/file.txt",
                file_type="INVALID_TYPE",
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status="COMPLETED",
            )

    def test_indexed_content_processing_status_validation(self):
        """Test processing_status enum validation."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        valid_statuses = ["PENDING", "PROCESSING", "COMPLETED", "FAILED"]

        for status in valid_statuses:
            # FAILED status requires error_message
            error_message = "Test error" if status == "FAILED" else None

            content = IndexedContent(
                file_path="/test.c",
                file_type="C_SOURCE",
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status=status,
                error_message=error_message,
            )
            assert content.processing_status == status

        # Invalid status should fail
        with pytest.raises(ValueError):
            IndexedContent(
                file_path="/test.c",
                file_type="C_SOURCE",
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status="INVALID_STATUS",
            )

    def test_indexed_content_file_path_validation(self):
        """Test file_path validation rules."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # Valid absolute paths should work
        valid_paths = [
            "/kernel/mm/slub.c",
            "/drivers/net/ethernet/intel/e1000/e1000_main.c",
            "/fs/ext4/inode.c",
            "/Documentation/kernel-hacking/locking.rst",
        ]

        for path in valid_paths:
            content = IndexedContent(
                file_path=path,
                file_type="C_SOURCE",
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status="COMPLETED",
            )
            assert content.file_path == path

        # Invalid paths should fail
        invalid_paths = [
            "relative/path.c",  # Not absolute
            "",  # Empty
            "/",  # Root only
            None,  # None
        ]

        for invalid_path in invalid_paths:
            with pytest.raises((ValueError, TypeError)):
                IndexedContent(
                    file_path=invalid_path,
                    file_type="C_SOURCE",
                    file_size=1024,
                    last_modified=datetime.now(),
                    chunk_count=5,
                    processing_status="COMPLETED",
                )

    def test_indexed_content_file_size_validation(self):
        """Test file_size validation (must be positive)."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # Valid file sizes
        valid_sizes = [1, 1024, 65536, 1048576]  # 1B to 1MB

        for size in valid_sizes:
            content = IndexedContent(
                file_path="/test.c",
                file_type="C_SOURCE",
                file_size=size,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status="COMPLETED",
            )
            assert content.file_size == size
            assert content.file_size > 0

        # Invalid file sizes should fail
        invalid_sizes = [0, -1, -1024]

        for size in invalid_sizes:
            with pytest.raises(ValueError):
                IndexedContent(
                    file_path="/test.c",
                    file_type="C_SOURCE",
                    file_size=size,
                    last_modified=datetime.now(),
                    chunk_count=5,
                    processing_status="COMPLETED",
                )

    def test_indexed_content_chunk_count_validation(self):
        """Test chunk_count validation (must be non-negative)."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # Valid chunk counts
        valid_counts = [0, 1, 10, 100, 1000]

        for count in valid_counts:
            content = IndexedContent(
                file_path="/test.c",
                file_type="C_SOURCE",
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=count,
                processing_status="COMPLETED",
            )
            assert content.chunk_count == count
            assert content.chunk_count >= 0

        # Invalid chunk counts should fail
        invalid_counts = [-1, -10]

        for count in invalid_counts:
            with pytest.raises(ValueError):
                IndexedContent(
                    file_path="/test.c",
                    file_type="C_SOURCE",
                    file_size=1024,
                    last_modified=datetime.now(),
                    chunk_count=count,
                    processing_status="COMPLETED",
                )

    def test_indexed_content_error_message_requirement(self):
        """Test error_message requirement when status is FAILED."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # FAILED status should require error_message
        content_failed = IndexedContent(
            file_path="/test.c",
            file_type="C_SOURCE",
            file_size=1024,
            last_modified=datetime.now(),
            chunk_count=0,
            processing_status="FAILED",
            error_message="File encoding not supported",
        )
        assert content_failed.error_message == "File encoding not supported"

        # FAILED status without error_message should fail
        with pytest.raises(ValueError):
            IndexedContent(
                file_path="/test.c",
                file_type="C_SOURCE",
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=0,
                processing_status="FAILED",
                error_message=None,
            )

        # Other statuses should not require error_message
        statuses_without_error = ["PENDING", "PROCESSING", "COMPLETED"]
        for status in statuses_without_error:
            content = IndexedContent(
                file_path="/test.c",
                file_type="C_SOURCE",
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status=status,
                error_message=None,
            )
            assert content.error_message is None

    def test_indexed_content_indexed_at_automatic(self):
        """Test that indexed_at is set automatically."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        before_creation = datetime.now()
        content = IndexedContent(
            file_path="/test.c",
            file_type="C_SOURCE",
            file_size=1024,
            last_modified=datetime(2024, 1, 1, 12, 0, 0),
            chunk_count=5,
            processing_status="COMPLETED",
        )
        after_creation = datetime.now()

        # indexed_at should be set automatically
        assert content.indexed_at is not None
        assert before_creation <= content.indexed_at <= after_creation

    def test_indexed_content_state_transitions(self):
        """Test state transition validation."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # Test valid transitions
        content = IndexedContent(
            file_path="/test.c",
            file_type="C_SOURCE",
            file_size=1024,
            last_modified=datetime.now(),
            chunk_count=0,
            processing_status="PENDING",
        )

        # Should support status transitions
        if hasattr(content, "transition_to"):
            # PENDING → PROCESSING
            content.transition_to("PROCESSING")
            assert content.processing_status == "PROCESSING"

            # PROCESSING → COMPLETED
            content.transition_to("COMPLETED")
            assert content.processing_status == "COMPLETED"

        # Test PROCESSING → FAILED transition
        content_processing = IndexedContent(
            file_path="/test2.c",
            file_type="C_SOURCE",
            file_size=1024,
            last_modified=datetime.now(),
            chunk_count=0,
            processing_status="PROCESSING",
        )

        if hasattr(content_processing, "transition_to"):
            content_processing.transition_to("FAILED", error_message="Processing error")
            assert content_processing.processing_status == "FAILED"
            assert content_processing.error_message == "Processing error"

    def test_indexed_content_file_existence_check(self):
        """Test file existence validation."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # If file existence checking is implemented
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            # Existing file should work
            content = IndexedContent(
                file_path="/kernel/mm/slub.c",
                file_type="C_SOURCE",
                file_size=65536,
                last_modified=datetime.now(),
                chunk_count=25,
                processing_status="COMPLETED",
            )
            assert content.file_path == "/kernel/mm/slub.c"

        # Non-existent file handling
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            # Depending on implementation, this might fail or be allowed
            try:
                IndexedContent(
                    file_path="/non/existent/file.c",
                    file_type="C_SOURCE",
                    file_size=1024,
                    last_modified=datetime.now(),
                    chunk_count=5,
                    processing_status="COMPLETED",
                )
            except ValueError:
                # If file existence is validated, this is expected
                pass

    def test_indexed_content_file_type_inference(self):
        """Test file type inference from file extension."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # Test if file_type can be inferred from file_path
        inference_cases = [
            ("/kernel/mm/slub.c", "C_SOURCE"),
            ("/include/linux/mm.h", "C_HEADER"),
            ("/Documentation/mm/slub.rst", "DOCUMENTATION"),
            ("/Makefile", "MAKEFILE"),
            ("/kernel/Makefile", "MAKEFILE"),
        ]

        for file_path, expected_type in inference_cases:
            if hasattr(IndexedContent, "infer_file_type"):
                inferred_type = IndexedContent.infer_file_type(file_path)
                assert inferred_type == expected_type

            # Test creation with inferred type
            content = IndexedContent(
                file_path=file_path,
                file_type=expected_type,  # Explicitly provide expected type
                file_size=1024,
                last_modified=datetime.now(),
                chunk_count=5,
                processing_status="COMPLETED",
            )
            assert content.file_type == expected_type

    def test_indexed_content_serialization(self):
        """Test serialization and deserialization of IndexedContent."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        original_content = IndexedContent(
            file_path="/kernel/sched/core.c",
            file_type="C_SOURCE",
            file_size=98304,
            last_modified=datetime(2024, 1, 10, 14, 30, 0),
            chunk_count=42,
            processing_status="COMPLETED",
        )

        # Should be serializable to dict
        content_dict = original_content.to_dict()
        assert isinstance(content_dict, dict)
        assert content_dict["file_path"] == "/kernel/sched/core.c"
        assert content_dict["file_type"] == "C_SOURCE"
        assert content_dict["file_size"] == 98304
        assert content_dict["chunk_count"] == 42
        assert content_dict["processing_status"] == "COMPLETED"

        # Should be deserializable from dict
        restored_content = IndexedContent.from_dict(content_dict)
        assert restored_content.file_path == original_content.file_path
        assert restored_content.file_type == original_content.file_type
        assert restored_content.file_size == original_content.file_size
        assert restored_content.chunk_count == original_content.chunk_count
        assert restored_content.processing_status == original_content.processing_status

    def test_indexed_content_relationships(self):
        """Test relationship with VectorEmbedding (one-to-many)."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        content = IndexedContent(
            file_path="/fs/ext4/inode.c",
            file_type="C_SOURCE",
            file_size=76800,
            last_modified=datetime.now(),
            chunk_count=35,
            processing_status="COMPLETED",
        )

        # Should support relationship methods
        if hasattr(content, "get_embeddings"):
            embeddings = content.get_embeddings()
            assert isinstance(embeddings, list)

        if hasattr(content, "add_embedding"):
            # Should support adding embeddings
            mock_embedding = Mock()
            content.add_embedding(mock_embedding)

    def test_indexed_content_query_methods(self):
        """Test query and filtering methods."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        # Should support class-level query methods
        if hasattr(IndexedContent, "find_by_path"):
            result = IndexedContent.find_by_path("/kernel/mm/slub.c")
            assert result is None or isinstance(result, IndexedContent)

        if hasattr(IndexedContent, "find_by_status"):
            results = IndexedContent.find_by_status("COMPLETED")
            assert isinstance(results, list)

        if hasattr(IndexedContent, "find_pending"):
            pending = IndexedContent.find_pending()
            assert isinstance(pending, list)

        if hasattr(IndexedContent, "find_failed"):
            failed = IndexedContent.find_failed()
            assert isinstance(failed, list)

    def test_indexed_content_update_operations(self):
        """Test update operations for IndexedContent."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        content = IndexedContent(
            file_path="/test.c",
            file_type="C_SOURCE",
            file_size=1024,
            last_modified=datetime.now(),
            chunk_count=0,
            processing_status="PENDING",
        )

        # Should support updating chunk count
        if hasattr(content, "update_chunk_count"):
            content.update_chunk_count(15)
            assert content.chunk_count == 15

        # Should support updating processing status
        if hasattr(content, "update_status"):
            content.update_status("COMPLETED")
            assert content.processing_status == "COMPLETED"

        # Should support updating last_modified
        if hasattr(content, "update_last_modified"):
            new_time = datetime.now()
            content.update_last_modified(new_time)
            assert content.last_modified == new_time

    def test_indexed_content_comparison(self):
        """Test comparison of IndexedContent instances."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        content1 = IndexedContent(
            file_path="/kernel/mm/slub.c",
            file_type="C_SOURCE",
            file_size=65536,
            last_modified=datetime(2024, 1, 1, 12, 0, 0),
            chunk_count=25,
            processing_status="COMPLETED",
        )

        content2 = IndexedContent(
            file_path="/kernel/mm/slub.c",
            file_type="C_SOURCE",
            file_size=65536,
            last_modified=datetime(2024, 1, 1, 12, 0, 0),
            chunk_count=25,
            processing_status="COMPLETED",
        )

        content3 = IndexedContent(
            file_path="/drivers/net/e1000.c",
            file_type="C_SOURCE",
            file_size=32768,
            last_modified=datetime(2024, 1, 2, 12, 0, 0),
            chunk_count=15,
            processing_status="COMPLETED",
        )

        # Same file path should be considered equal
        assert content1 == content2
        assert content1 != content3

    def test_indexed_content_immutability(self):
        """Test that core fields are immutable after creation."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        content = IndexedContent(
            file_path="/test.c",
            file_type="C_SOURCE",
            file_size=1024,
            last_modified=datetime.now(),
            chunk_count=5,
            processing_status="COMPLETED",
        )

        original_path = content.file_path
        original_size = content.file_size

        # Attempting to modify core fields should fail
        with pytest.raises((AttributeError, TypeError)):
            content.file_path = "/modified.c"

        with pytest.raises((AttributeError, TypeError)):
            content.file_size = 2048

        # Values should remain unchanged
        assert content.file_path == original_path
        assert content.file_size == original_size

    def test_indexed_content_string_representation(self):
        """Test string representation of IndexedContent."""
        from src.python.semantic_search.models.indexed_content import IndexedContent

        content = IndexedContent(
            file_path="/kernel/mm/slub.c",
            file_type="C_SOURCE",
            file_size=65536,
            last_modified=datetime.now(),
            chunk_count=25,
            processing_status="COMPLETED",
        )

        # Should have meaningful string representation
        str_repr = str(content)
        assert "/kernel/mm/slub.c" in str_repr
        assert "COMPLETED" in str_repr
        assert "indexedcontent" in str_repr.lower()

    def test_indexed_content_performance(self):
        """Test performance of IndexedContent operations."""
        import time

        from src.python.semantic_search.models.indexed_content import IndexedContent

        start_time = time.time()

        # Create multiple IndexedContent instances
        contents = []
        for i in range(100):
            content = IndexedContent(
                file_path=f"/test_{i}.c",
                file_type="C_SOURCE",
                file_size=1024 * (i + 1),
                last_modified=datetime.now(),
                chunk_count=i + 1,
                processing_status="COMPLETED",
            )
            contents.append(content)

        end_time = time.time()

        # Should create quickly
        creation_time = (end_time - start_time) * 1000
        assert creation_time < 1000  # Less than 1 second for 100 instances

    def test_indexed_content_memory_efficiency(self):
        """Test memory efficiency of IndexedContent instances."""
        import sys

        from src.python.semantic_search.models.indexed_content import IndexedContent

        content = IndexedContent(
            file_path="/kernel/mm/slub.c",
            file_type="C_SOURCE",
            file_size=65536,
            last_modified=datetime.now(),
            chunk_count=25,
            processing_status="COMPLETED",
        )

        # Should be reasonably sized
        content_size = sys.getsizeof(content)

        # Should be less than 2KB
        assert content_size < 2000
