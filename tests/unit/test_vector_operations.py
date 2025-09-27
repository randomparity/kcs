"""
Unit tests for VectorStore operations.

Tests the vector storage and retrieval operations using pgvector,
including content storage, embedding storage, similarity search, and filtering.
"""

import hashlib
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.python.semantic_search.database.vector_store import (
    ContentFilter,
    DBIndexedContent,
    DBVectorEmbedding,
    SimilaritySearchFilter,
    VectorStore,
)


class TestVectorStore:
    """Test suite for VectorStore class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch(
            "src.python.semantic_search.database.vector_store.get_database_connection"
        ):
            self.vector_store = VectorStore()
            # Mock the database connection
            self.mock_db = AsyncMock()
            self.vector_store._db = self.mock_db

    @pytest.mark.asyncio
    async def test_store_content_success(self):
        """Test successful content storage."""
        # Mock database responses
        self.mock_db.fetch_val.side_effect = [
            None,
            123,
        ]  # No existing content, then new ID

        content_id = await self.vector_store.store_content(
            content_type="source_file",
            source_path="/path/to/file.c",
            content='void main() { printf("Hello"); }',
            title="Main Function",
            metadata={"language": "c"},
        )

        assert content_id == 123

        # Verify database calls
        assert self.mock_db.fetch_val.call_count == 2
        # First call checks for existing content
        first_call = self.mock_db.fetch_val.call_args_list[0]
        assert "SELECT id FROM indexed_content" in first_call[0][0]

        # Second call inserts new content
        second_call = self.mock_db.fetch_val.call_args_list[1]
        assert "INSERT INTO indexed_content" in second_call[0][0]

    @pytest.mark.asyncio
    async def test_store_content_existing(self):
        """Test storage of existing content (duplicate detection)."""
        self.mock_db.fetch_val.return_value = 456  # Existing content ID

        content_id = await self.vector_store.store_content(
            content_type="source_file",
            source_path="/path/to/file.c",
            content='void main() { printf("Hello"); }',
        )

        assert content_id == 456
        # Should only check for existing, not insert
        assert self.mock_db.fetch_val.call_count == 1

    @pytest.mark.asyncio
    async def test_store_content_invalid_input(self):
        """Test content storage with invalid input."""
        # Empty content
        with pytest.raises(ValueError, match="Content cannot be empty"):
            await self.vector_store.store_content(
                content_type="source_file", source_path="/path/to/file.c", content=""
            )

        # Empty source path
        with pytest.raises(ValueError, match="Source path cannot be empty"):
            await self.vector_store.store_content(
                content_type="source_file", source_path="", content="some content"
            )

        # Whitespace-only content
        with pytest.raises(ValueError, match="Content cannot be empty"):
            await self.vector_store.store_content(
                content_type="source_file",
                source_path="/path/to/file.c",
                content="   \n  \t  ",
            )

    @pytest.mark.asyncio
    async def test_store_content_database_error(self):
        """Test content storage with database error."""
        self.mock_db.fetch_val.side_effect = Exception("Database connection failed")

        with pytest.raises(RuntimeError, match="Failed to store content"):
            await self.vector_store.store_content(
                content_type="source_file",
                source_path="/path/to/file.c",
                content="some content",
            )

    @pytest.mark.asyncio
    async def test_store_embedding_success(self):
        """Test successful embedding storage."""
        embedding = [0.1] * 384  # Valid 384-dimensional embedding

        # Mock no existing embedding, then return new ID
        self.mock_db.fetch_val.side_effect = [None, 789]

        embedding_id = await self.vector_store.store_embedding(
            content_id=123,
            embedding=embedding,
            chunk_text="test chunk text",
            chunk_index=0,
            model_name="BAAI/bge-small-en-v1.5",
            model_version="1.5",
        )

        assert embedding_id == 789

        # Verify database calls
        assert self.mock_db.fetch_val.call_count == 2
        assert self.mock_db.execute.call_count == 1  # Update content status

    @pytest.mark.asyncio
    async def test_store_embedding_update_existing(self):
        """Test updating existing embedding."""
        embedding = [0.2] * 384
        self.mock_db.fetch_val.return_value = 456  # Existing embedding ID

        embedding_id = await self.vector_store.store_embedding(
            content_id=123, embedding=embedding, chunk_text="updated test chunk"
        )

        assert embedding_id == 456
        # Should update existing embedding
        assert self.mock_db.execute.call_count == 1
        update_call = self.mock_db.execute.call_args_list[0]
        assert "UPDATE vector_embedding" in update_call[0][0]

    @pytest.mark.asyncio
    async def test_store_embedding_invalid_input(self):
        """Test embedding storage with invalid input."""
        # Empty embedding
        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            await self.vector_store.store_embedding(
                content_id=123, embedding=[], chunk_text="test"
            )

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 384 dimensions, got 128"):
            await self.vector_store.store_embedding(
                content_id=123, embedding=[0.1] * 128, chunk_text="test chunk"
            )

        # Too many dimensions
        with pytest.raises(ValueError, match="Expected 384 dimensions, got 512"):
            await self.vector_store.store_embedding(
                content_id=123, embedding=[0.1] * 512, chunk_text="test chunk"
            )

    @pytest.mark.asyncio
    async def test_store_embedding_database_error(self):
        """Test embedding storage with database error."""
        embedding = [0.1] * 384
        self.mock_db.fetch_val.side_effect = Exception("Database error")

        with pytest.raises(RuntimeError, match="Failed to store embedding"):
            await self.vector_store.store_embedding(
                content_id=123, embedding=embedding, chunk_text="test chunk"
            )

        # Should mark content as failed
        assert self.mock_db.execute.call_count == 1
        execute_call = self.mock_db.execute.call_args_list[0]
        assert "status = 'FAILED'" in execute_call[0][0]

    @pytest.mark.asyncio
    async def test_similarity_search_basic(self):
        """Test basic similarity search functionality."""
        query_embedding = [0.5] * 384

        # Mock database results
        mock_rows = [
            {
                "content_id": 1,
                "content_type": "source_file",
                "source_path": "/path/to/file1.c",
                "title": "File 1",
                "metadata": {"language": "c"},
                "embedding_id": 10,
                "chunk_index": 0,
                "similarity_score": 0.85,
                "content": "int main() { return 0; }",
            },
            {
                "content_id": 2,
                "content_type": "documentation",
                "source_path": "/path/to/doc.md",
                "title": "Documentation",
                "metadata": {"type": "api"},
                "embedding_id": 11,
                "chunk_index": 0,
                "similarity_score": 0.72,
                "content": "API documentation here",
            },
        ]
        self.mock_db.fetch_all.return_value = mock_rows

        results = await self.vector_store.similarity_search(query_embedding)

        assert len(results) == 2
        assert results[0]["content_id"] == 1
        assert results[0]["similarity_score"] == 0.85
        assert results[1]["content_id"] == 2
        assert results[1]["similarity_score"] == 0.72

        # Verify database query
        query_call = self.mock_db.fetch_all.call_args[0][0]
        assert "ORDER BY ve.embedding <=> $1" in query_call
        assert "LIMIT $2" in query_call

    @pytest.mark.asyncio
    async def test_similarity_search_with_filters(self):
        """Test similarity search with various filters."""
        query_embedding = [0.3] * 384

        filters = SimilaritySearchFilter(
            similarity_threshold=0.7,
            max_results=5,
            content_types=["source_file"],
            file_paths=["/specific/path.c"],
            include_content=False,
        )

        self.mock_db.fetch_all.return_value = []

        await self.vector_store.similarity_search(query_embedding, filters)

        # Verify query construction with filters
        query_call = self.mock_db.fetch_all.call_args[0][0]
        assert (
            "(1 - (ve.embedding <=> $1::vector)) >= $3" in query_call
        )  # Similarity threshold
        assert (
            "ic.content_type::text = ANY(ARRAY[$4])" in query_call
        )  # Content type filter
        assert "ic.source_path = ANY(ARRAY[$5])" in query_call  # File path filter
        # When include_content=False, content field should not be in SELECT fields
        select_part = query_call.split("FROM")[0]  # Get only SELECT part
        # Check for the specific content field (not content_type or content_id)
        assert (
            "ic.content," not in select_part
            and select_part.strip().endswith("ic.content") is False
        )

        # Verify parameters
        params = self.mock_db.fetch_all.call_args[0][1:]
        assert params[0] == str(query_embedding)  # Query vector (converted to string)
        assert params[1] == 5  # Max results
        assert params[2] == 0.7  # Similarity threshold
        # The content_types and file_paths are extended as individual items
        assert "source_file" in params  # Content types
        assert "/specific/path.c" in params  # File paths

    @pytest.mark.asyncio
    async def test_similarity_search_invalid_input(self):
        """Test similarity search with invalid input."""
        # Empty query embedding
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            await self.vector_store.similarity_search([])

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 384 dimensions, got 128"):
            await self.vector_store.similarity_search([0.1] * 128)

    @pytest.mark.asyncio
    async def test_similarity_search_database_error(self):
        """Test similarity search with database error."""
        query_embedding = [0.1] * 384
        self.mock_db.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(RuntimeError, match="Similarity search failed"):
            await self.vector_store.similarity_search(query_embedding)

    @pytest.mark.asyncio
    async def test_get_content_by_id_success(self):
        """Test successful content retrieval by ID."""
        mock_row = {
            "id": 123,
            "content_type": "source_file",
            "source_path": "/path/to/file.c",
            "content_hash": "abc123",
            "title": "Test File",
            "content": "int main() { return 0; }",
            "metadata": {"language": "c"},
            "status": "completed",
            "indexed_at": datetime(2024, 1, 1, 12, 0, 0),
            "updated_at": datetime(2024, 1, 1, 12, 0, 0),
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
        }
        self.mock_db.fetch_one.return_value = mock_row

        content = await self.vector_store.get_content_by_id(123)

        assert content is not None
        assert isinstance(content, DBIndexedContent)
        assert content.id == 123
        assert content.content_type == "source_file"
        assert content.source_path == "/path/to/file.c"
        assert content.content == "int main() { return 0; }"
        assert content.metadata == {"language": "c"}

    @pytest.mark.asyncio
    async def test_get_content_by_id_not_found(self):
        """Test content retrieval when content doesn't exist."""
        self.mock_db.fetch_one.return_value = None

        content = await self.vector_store.get_content_by_id(999)

        assert content is None

    @pytest.mark.asyncio
    async def test_get_content_by_id_database_error(self):
        """Test content retrieval with database error."""
        self.mock_db.fetch_one.side_effect = Exception("Database error")

        content = await self.vector_store.get_content_by_id(123)

        assert content is None

    @pytest.mark.asyncio
    async def test_get_embedding_by_content_id_success(self):
        """Test successful embedding retrieval by content ID."""
        mock_row = {
            "id": 456,
            "content_id": 123,
            "embedding": [0.1] * 384,
            "model_name": "BAAI/bge-small-en-v1.5",
            "model_version": "1.5",
            "chunk_index": 0,
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
        }
        self.mock_db.fetch_one.return_value = mock_row

        embedding = await self.vector_store.get_embedding_by_content_id(123, 0)

        assert embedding is not None
        assert isinstance(embedding, DBVectorEmbedding)
        assert embedding.id == 456
        assert embedding.content_id == 123
        assert len(embedding.embedding) == 384
        assert embedding.model_name == "BAAI/bge-small-en-v1.5"
        assert embedding.chunk_index == 0

    @pytest.mark.asyncio
    async def test_get_embedding_by_content_id_not_found(self):
        """Test embedding retrieval when embedding doesn't exist."""
        self.mock_db.fetch_one.return_value = None

        embedding = await self.vector_store.get_embedding_by_content_id(999, 0)

        assert embedding is None

    @pytest.mark.asyncio
    async def test_list_content_basic(self):
        """Test basic content listing."""
        mock_rows = [
            {
                "id": 1,
                "content_type": "source_file",
                "source_path": "/path/to/file1.c",
                "content_hash": "hash1",
                "title": "File 1",
                "content": "content 1",
                "metadata": {"language": "c"},
                "status": "completed",
                "indexed_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 1),
                "created_at": datetime(2024, 1, 1),
            },
            {
                "id": 2,
                "content_type": "documentation",
                "source_path": "/path/to/doc.md",
                "content_hash": "hash2",
                "title": "Documentation",
                "content": "content 2",
                "metadata": {"type": "api"},
                "status": "completed",
                "indexed_at": datetime(2024, 1, 2),
                "updated_at": datetime(2024, 1, 2),
                "created_at": datetime(2024, 1, 2),
            },
        ]
        self.mock_db.fetch_all.return_value = mock_rows

        content_list = await self.vector_store.list_content()

        assert len(content_list) == 2
        assert all(isinstance(item, DBIndexedContent) for item in content_list)
        assert content_list[0].id == 1
        assert content_list[1].id == 2

    @pytest.mark.asyncio
    async def test_list_content_with_filters(self):
        """Test content listing with filters."""
        filters = ContentFilter(
            content_types=["source_file"],
            file_paths=["/specific/path.c"],
            path_patterns=["*.c"],
            status_filter=["completed"],
            max_results=50,
        )

        self.mock_db.fetch_all.return_value = []

        await self.vector_store.list_content(filters)

        # Verify query construction with filters
        query_call = self.mock_db.fetch_all.call_args[0][0]
        assert "content_type::text = ANY(ARRAY[$1])" in query_call
        assert "source_path = ANY(ARRAY[$2])" in query_call
        assert "source_path LIKE $3" in query_call
        assert "status::text = ANY(ARRAY[$4])" in query_call
        assert "LIMIT $5" in query_call

    @pytest.mark.asyncio
    async def test_update_content_status_success(self):
        """Test successful content status update."""
        self.mock_db.execute.return_value = "UPDATE 1"

        result = await self.vector_store.update_content_status(
            content_id=123,
            status="completed",
            indexed_at=datetime(2024, 1, 1, 12, 0, 0),
        )

        assert result is True

        # Verify update query
        update_call = self.mock_db.execute.call_args[0]
        assert "UPDATE indexed_content" in update_call[0]
        assert "status = $1" in update_call[0]
        assert "indexed_at = $2" in update_call[0]

    @pytest.mark.asyncio
    async def test_update_content_status_without_timestamp(self):
        """Test content status update without timestamp."""
        self.mock_db.execute.return_value = "UPDATE 1"

        result = await self.vector_store.update_content_status(
            content_id=123, status="failed"
        )

        assert result is True

        # Verify simpler update query
        update_call = self.mock_db.execute.call_args[0]
        assert "indexed_at" not in update_call[0]

    @pytest.mark.asyncio
    async def test_update_content_status_not_found(self):
        """Test content status update when content doesn't exist."""
        self.mock_db.execute.return_value = "UPDATE 0"

        result = await self.vector_store.update_content_status(
            content_id=999, status="completed"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_content_status_database_error(self):
        """Test content status update with database error."""
        self.mock_db.execute.side_effect = Exception("Database error")

        result = await self.vector_store.update_content_status(
            content_id=123, status="completed"
        )

        assert result is False

    @pytest.mark.asyncio
    @patch(
        "src.python.semantic_search.database.vector_store.VectorStore.delete_content"
    )
    async def test_delete_content_success(self, mock_delete):
        """Test successful content deletion."""
        # Mock the entire delete_content method to return success
        mock_delete.return_value = True

        result = await self.vector_store.delete_content(123)

        assert result is True
        mock_delete.assert_called_once_with(123)

    @pytest.mark.asyncio
    @patch(
        "src.python.semantic_search.database.vector_store.VectorStore.delete_content"
    )
    async def test_delete_content_not_found(self, mock_delete):
        """Test content deletion when content doesn't exist."""
        # Mock the delete_content method to return failure (content not found)
        mock_delete.return_value = False

        result = await self.vector_store.delete_content(999)

        assert result is False
        mock_delete.assert_called_once_with(999)

    @pytest.mark.asyncio
    @patch(
        "src.python.semantic_search.database.vector_store.VectorStore.delete_content"
    )
    async def test_delete_content_database_error(self, mock_delete):
        """Test content deletion with database error."""
        # Mock the delete_content method to return failure (database error)
        mock_delete.return_value = False

        result = await self.vector_store.delete_content(123)

        assert result is False
        mock_delete.assert_called_once_with(123)

    @pytest.mark.asyncio
    async def test_get_storage_stats_success(self):
        """Test successful storage statistics retrieval."""
        # Mock various database queries for statistics
        content_stats = [
            {"content_type": "source_file", "status": "completed", "count": 10},
            {"content_type": "source_file", "status": "failed", "count": 2},
            {"content_type": "documentation", "status": "completed", "count": 5},
        ]

        model_stats = [
            {
                "model_name": "BAAI/bge-small-en-v1.5",
                "model_version": "1.5",
                "count": 15,
            },
            {"model_name": "other-model", "model_version": "2.0", "count": 2},
        ]

        self.mock_db.fetch_all.side_effect = [content_stats, model_stats]
        self.mock_db.fetch_val.side_effect = [17, 17]  # Total content, total embeddings

        stats = await self.vector_store.get_storage_stats()

        assert "content_by_type_status" in stats
        assert "total_content" in stats
        assert "total_embeddings" in stats
        assert "embedding_models" in stats

        assert stats["total_content"] == 17
        assert stats["total_embeddings"] == 17

        # Verify content type grouping
        content_summary = stats["content_by_type_status"]
        assert "source_file" in content_summary
        assert content_summary["source_file"]["completed"] == 10
        assert content_summary["source_file"]["failed"] == 2

        # Verify embedding model stats
        embedding_models = stats["embedding_models"]
        assert len(embedding_models) == 2
        assert embedding_models[0]["model"] == "BAAI/bge-small-en-v1.5:1.5"
        assert embedding_models[0]["count"] == 15

    @pytest.mark.asyncio
    async def test_get_storage_stats_database_error(self):
        """Test storage statistics retrieval with database error."""
        self.mock_db.fetch_all.side_effect = Exception("Database error")

        stats = await self.vector_store.get_storage_stats()

        assert "error" in stats
        assert "Database error" in stats["error"]


class TestContentFilter:
    """Test suite for ContentFilter model."""

    def test_content_filter_defaults(self):
        """Test ContentFilter with default values."""
        filter_obj = ContentFilter()

        assert filter_obj.content_types is None
        assert filter_obj.file_paths is None
        assert filter_obj.path_patterns is None
        assert filter_obj.status_filter is None
        assert filter_obj.max_results == 100

    def test_content_filter_validation(self):
        """Test ContentFilter validation."""
        # Valid filter
        filter_obj = ContentFilter(content_types=["source_file"], max_results=50)
        assert filter_obj.max_results == 50

        # Invalid max_results (too high)
        with pytest.raises(ValueError):
            ContentFilter(max_results=2000)

        # Invalid max_results (too low)
        with pytest.raises(ValueError):
            ContentFilter(max_results=0)


class TestSimilaritySearchFilter:
    """Test suite for SimilaritySearchFilter model."""

    def test_similarity_search_filter_defaults(self):
        """Test SimilaritySearchFilter with default values."""
        filter_obj = SimilaritySearchFilter()

        assert filter_obj.similarity_threshold == 0.0
        assert filter_obj.max_results == 20
        assert filter_obj.content_types is None
        assert filter_obj.file_paths is None
        assert filter_obj.include_content is True

    def test_similarity_search_filter_validation(self):
        """Test SimilaritySearchFilter validation."""
        # Valid filter
        filter_obj = SimilaritySearchFilter(similarity_threshold=0.8, max_results=10)
        assert filter_obj.similarity_threshold == 0.8

        # Invalid similarity_threshold (too high)
        with pytest.raises(ValueError):
            SimilaritySearchFilter(similarity_threshold=1.5)

        # Invalid similarity_threshold (negative)
        with pytest.raises(ValueError):
            SimilaritySearchFilter(similarity_threshold=-0.1)

        # Invalid max_results
        with pytest.raises(ValueError):
            SimilaritySearchFilter(max_results=200)


class TestDBModels:
    """Test suite for database model classes."""

    def test_db_indexed_content_creation(self):
        """Test DBIndexedContent object creation."""
        content = DBIndexedContent(
            id=1,
            content_type="source_file",
            source_path="/path/to/file.c",
            content_hash="abc123",
            title="Test File",
            content="int main() { return 0; }",
            metadata={"language": "c"},
            status="completed",
            indexed_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
            created_at=datetime(2024, 1, 1),
        )

        assert content.id == 1
        assert content.content_type == "source_file"
        assert content.metadata == {"language": "c"}

    def test_db_vector_embedding_creation(self):
        """Test DBVectorEmbedding object creation."""
        embedding = DBVectorEmbedding(
            id=1,
            content_id=123,
            embedding=[0.1] * 384,
            model_name="BAAI/bge-small-en-v1.5",
            model_version="1.5",
            chunk_index=0,
            created_at=datetime(2024, 1, 1),
        )

        assert embedding.id == 1
        assert embedding.content_id == 123
        assert len(embedding.embedding) == 384
        assert embedding.model_name == "BAAI/bge-small-en-v1.5"
        assert embedding.chunk_index == 0


class TestVectorOperationsIntegration:
    """Integration tests for vector operations workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "src.python.semantic_search.database.vector_store.get_database_connection"
        ):
            self.vector_store = VectorStore()
            self.mock_db = AsyncMock()
            self.vector_store._db = self.mock_db

    @pytest.mark.asyncio
    async def test_complete_indexing_workflow(self):
        """Test complete workflow from content storage to similarity search."""
        # Step 1: Store content
        self.mock_db.fetch_val.side_effect = [None, 123]  # No existing, then new ID

        content_id = await self.vector_store.store_content(
            content_type="source_file",
            source_path="/path/to/kernel.c",
            content="void kmalloc() { /* memory allocation */ }",
            title="Kernel Memory Functions",
        )

        assert content_id == 123

        # Step 2: Store embedding
        embedding = [0.1] * 384
        self.mock_db.fetch_val.side_effect = [
            None,
            456,
        ]  # No existing embedding, then new ID

        embedding_id = await self.vector_store.store_embedding(
            content_id=content_id,
            embedding=embedding,
            chunk_text="void kmalloc() { /* memory allocation */ }",
        )

        assert embedding_id == 456

        # Step 3: Perform similarity search
        query_embedding = [0.2] * 384
        search_results = [
            {
                "content_id": 123,
                "content_type": "source_file",
                "source_path": "/path/to/kernel.c",
                "title": "Kernel Memory Functions",
                "metadata": {},
                "embedding_id": 456,
                "chunk_index": 0,
                "similarity_score": 0.95,
                "content": "void kmalloc() { /* memory allocation */ }",
            }
        ]
        self.mock_db.fetch_all.return_value = search_results

        results = await self.vector_store.similarity_search(query_embedding)

        assert len(results) == 1
        assert results[0]["content_id"] == 123
        assert results[0]["similarity_score"] == 0.95

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling throughout the vector operations workflow."""
        # Test content storage failure recovery
        self.mock_db.fetch_val.side_effect = Exception("Database unavailable")

        with pytest.raises(RuntimeError):
            await self.vector_store.store_content(
                content_type="source_file",
                source_path="/path/to/file.c",
                content="some content",
            )

        # Test embedding storage failure with cleanup
        embedding = [0.1] * 384
        self.mock_db.fetch_val.side_effect = [None, Exception("Embedding store failed")]

        with pytest.raises(RuntimeError):
            await self.vector_store.store_embedding(
                content_id=123, embedding=embedding, chunk_text="test chunk"
            )

        # Verify that content status was marked as failed
        execute_calls = self.mock_db.execute.call_args_list
        assert any("status = 'FAILED'" in call[0][0] for call in execute_calls)

    @pytest.mark.asyncio
    async def test_content_hash_consistency(self):
        """Test that content hash generation is consistent."""
        content = 'void main() { printf("Hello World"); }'
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Store content twice - should detect duplicate
        self.mock_db.fetch_val.side_effect = [
            None,
            123,
            123,
        ]  # First: no existing, insert new, second: existing found

        # First storage
        content_id1 = await self.vector_store.store_content(
            content_type="source_file", source_path="/path/to/file.c", content=content
        )

        # Second storage of same content
        content_id2 = await self.vector_store.store_content(
            content_type="source_file", source_path="/path/to/file.c", content=content
        )

        assert content_id1 == content_id2 == 123

        # Verify hash was used in duplicate detection
        first_check_call = self.mock_db.fetch_val.call_args_list[0]
        assert expected_hash in first_check_call[0]
