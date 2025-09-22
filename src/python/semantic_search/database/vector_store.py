"""
Vector storage operations using pgvector.

Provides CRUD operations for vector embeddings and content storage
with efficient similarity search using PostgreSQL and pgvector.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# We'll define database-compatible models here rather than importing
# the existing application models that have different field structures
from .connection import get_database_connection

logger = logging.getLogger(__name__)


# Database-compatible models
class DBIndexedContent:
    """Database representation of indexed content."""

    def __init__(
        self,
        id: int,
        content_type: str,
        source_path: str,
        content_hash: str,
        title: str | None,
        content: str,
        metadata: dict[str, Any],
        status: str,
        indexed_at: datetime | None,
        updated_at: datetime,
        created_at: datetime,
    ):
        self.id = id
        self.content_type = content_type
        self.source_path = source_path
        self.content_hash = content_hash
        self.title = title
        self.content = content
        self.metadata = metadata
        self.status = status
        self.indexed_at = indexed_at
        self.updated_at = updated_at
        self.created_at = created_at


class DBVectorEmbedding:
    """Database representation of vector embedding."""

    def __init__(
        self,
        id: int,
        content_id: int,
        embedding: list[float],
        chunk_index: int,
        created_at: datetime,
        model_name: str = "BAAI/bge-small-en-v1.5",
        model_version: str = "1.0",
    ):
        self.id = id
        self.content_id = content_id
        self.embedding = embedding
        self.model_name = model_name
        self.model_version = model_version
        self.chunk_index = chunk_index
        self.created_at = created_at


class ContentFilter(BaseModel):
    """Filters for content queries."""

    content_types: list[str] | None = Field(None, description="Filter by content types")
    file_paths: list[str] | None = Field(
        None, description="Filter by specific file paths"
    )
    path_patterns: list[str] | None = Field(
        None, description="Path patterns for LIKE matching"
    )
    status_filter: list[str] | None = Field(
        None, description="Filter by indexing status"
    )
    max_results: int = Field(100, ge=1, le=1000, description="Maximum results")


class SimilaritySearchFilter(BaseModel):
    """Filters for similarity search operations."""

    similarity_threshold: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    max_results: int = Field(20, ge=1, le=100, description="Maximum results")
    content_types: list[str] | None = Field(None, description="Filter by content types")
    file_paths: list[str] | None = Field(None, description="Filter by file paths")
    include_content: bool = Field(True, description="Include full content in results")


class VectorStore:
    """
    Vector storage operations using pgvector.

    Provides high-performance vector storage and similarity search operations
    for semantic search functionality using PostgreSQL with pgvector extension.
    """

    def __init__(self) -> None:
        """Initialize vector store."""
        self._db = get_database_connection()

    async def store_content(
        self,
        content_type: str,
        source_path: str,
        content: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Store content for indexing.

        Args:
            content_type: Type of content (source_file, documentation, etc.)
            source_path: Path to source file
            content: Content text
            title: Optional title
            metadata: Additional metadata

        Returns:
            Content ID

        Raises:
            ValueError: If content is invalid
            RuntimeError: If storage fails
        """
        if not content.strip():
            raise ValueError("Content cannot be empty")

        if not source_path.strip():
            raise ValueError("Source path cannot be empty")

        # Generate content hash for change detection
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        try:
            # Check if content already exists
            existing_id = await self._db.fetch_val(
                """
                SELECT id FROM indexed_content
                WHERE source_path = $1 AND content_hash = $2
                """,
                source_path,
                content_hash,
            )

            if existing_id:
                logger.info(f"Content already exists with ID {existing_id}")
                return int(existing_id)

            # Insert new content
            import json

            metadata_json = json.dumps(metadata or {})
            content_id = await self._db.fetch_val(
                """
                INSERT INTO indexed_content (
                    content_type, source_path, content_hash, title, content, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                content_type,
                source_path,
                content_hash,
                title,
                content,
                metadata_json,
            )

            logger.info(f"Stored content with ID {content_id}")
            return int(content_id)

        except Exception as e:
            logger.error(f"Failed to store content: {e}")
            raise RuntimeError(f"Failed to store content: {e}") from e

    async def store_embedding(
        self,
        content_id: int,
        embedding: list[float],
        chunk_text: str,
        chunk_index: int = 0,
        model_name: str = "BAAI/bge-small-en-v1.5",
        model_version: str = "1.5",
    ) -> int:
        """
        Store vector embedding for content.

        Args:
            content_id: Referenced content ID
            embedding: Vector embedding
            chunk_text: Text content of the chunk
            chunk_index: Index of the chunk within content
            model_name: Model used for embedding
            model_version: Model version

        Returns:
            Embedding ID

        Raises:
            ValueError: If embedding is invalid
            RuntimeError: If storage fails
        """
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        if len(embedding) != 384:  # BAAI/bge-small-en-v1.5 dimension
            raise ValueError(f"Expected 384 dimensions, got {len(embedding)}")

        try:
            # Check if embedding already exists
            existing_id = await self._db.fetch_val(
                """
                SELECT id FROM vector_embedding
                WHERE content_id = $1 AND chunk_index = $2
                """,
                content_id,
                chunk_index,
            )

            if existing_id:
                # Update existing embedding
                await self._db.execute(
                    """
                    UPDATE vector_embedding
                    SET embedding = $1::vector, chunk_text = $2
                    WHERE id = $3
                    """,
                    str(embedding),
                    chunk_text,
                    existing_id,
                )
                logger.info(f"Updated embedding with ID {existing_id}")
                return int(existing_id)

            # Insert new embedding
            embedding_id = await self._db.fetch_val(
                """
                INSERT INTO vector_embedding (
                    content_id, embedding, chunk_index, chunk_text
                ) VALUES ($1, $2::vector, $3, $4)
                RETURNING id
                """,
                content_id,
                str(embedding),
                chunk_index,
                chunk_text,
            )

            # Update content status to completed
            await self._db.execute(
                """
                UPDATE indexed_content
                SET status = 'COMPLETED', indexed_at = NOW(), updated_at = NOW()
                WHERE id = $1
                """,
                content_id,
            )

            logger.info(f"Stored embedding with ID {embedding_id}")
            return int(embedding_id)

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            # Mark content as failed
            await self._db.execute(
                """
                UPDATE indexed_content
                SET status = 'FAILED', updated_at = NOW()
                WHERE id = $1
                """,
                content_id,
            )
            raise RuntimeError(f"Failed to store embedding: {e}") from e

    async def similarity_search(
        self,
        query_embedding: list[float],
        filters: SimilaritySearchFilter | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search using vector embeddings.

        Args:
            query_embedding: Query vector embedding
            filters: Optional search filters

        Returns:
            List of search results with similarity scores

        Raises:
            ValueError: If query embedding is invalid
            RuntimeError: If search fails
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if len(query_embedding) != 384:
            raise ValueError(f"Expected 384 dimensions, got {len(query_embedding)}")

        filters = filters or SimilaritySearchFilter(
            similarity_threshold=0.0,
            max_results=20,
            content_types=None,
            file_paths=None,
            include_content=True,
        )

        try:
            # Build dynamic query with filters
            where_conditions = ["ve.embedding IS NOT NULL"]
            params = [query_embedding, filters.max_results]

            if filters.similarity_threshold > 0:
                where_conditions.append("(1 - (ve.embedding <=> $1)) >= $3")
                params.append(filters.similarity_threshold)

            if filters.content_types:
                placeholder = ",".join(
                    f"${i}"
                    for i in range(
                        len(params) + 1, len(params) + 1 + len(filters.content_types)
                    )
                )
                where_conditions.append(
                    f"ic.content_type::text = ANY(ARRAY[{placeholder}])"
                )
                params.extend(filters.content_types)

            if filters.file_paths:
                placeholder = ",".join(
                    f"${i}"
                    for i in range(
                        len(params) + 1, len(params) + 1 + len(filters.file_paths)
                    )
                )
                where_conditions.append(f"ic.source_path = ANY(ARRAY[{placeholder}])")
                params.extend(filters.file_paths)

            # Select fields based on include_content
            select_fields = [
                "ic.id as content_id",
                "ic.content_type",
                "ic.source_path",
                "ic.title",
                "ic.metadata",
                "ve.id as embedding_id",
                "ve.chunk_index",
                "(1 - (ve.embedding <=> $1)) as similarity_score",
            ]

            if filters.include_content:
                select_fields.append("ic.content")

            query = f"""
            SELECT {", ".join(select_fields)}
            FROM vector_embedding ve
            JOIN indexed_content ic ON ve.content_id = ic.id
            WHERE {" AND ".join(where_conditions)}
            ORDER BY ve.embedding <=> $1
            LIMIT $2
            """

            results = await self._db.fetch_all(query, *params)

            # Convert to list of dictionaries
            search_results = []
            for row in results:
                result = {
                    "content_id": row["content_id"],
                    "content_type": row["content_type"],
                    "source_path": row["source_path"],
                    "title": row["title"],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    "embedding_id": row["embedding_id"],
                    "chunk_index": row["chunk_index"],
                    "similarity_score": float(row["similarity_score"]),
                }

                if filters.include_content:
                    result["content"] = row["content"]

                search_results.append(result)

            logger.info(f"Found {len(search_results)} results for similarity search")
            return search_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Similarity search failed: {e}") from e

    async def get_content_by_id(self, content_id: int) -> DBIndexedContent | None:
        """
        Get content by ID.

        Args:
            content_id: Content ID

        Returns:
            DBIndexedContent object or None if not found
        """
        try:
            row = await self._db.fetch_one(
                """
                SELECT id, content_type, source_path, content_hash, title, content,
                       metadata, status, indexed_at, updated_at, created_at
                FROM indexed_content
                WHERE id = $1
                """,
                content_id,
            )

            if not row:
                return None

            return DBIndexedContent(
                id=row["id"],
                content_type=row["content_type"],
                source_path=row["source_path"],
                content_hash=row["content_hash"],
                title=row["title"],
                content=row["content"],
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                status=row["status"],
                indexed_at=row["indexed_at"],
                updated_at=row["updated_at"],
                created_at=row["created_at"],
            )

        except Exception as e:
            logger.error(f"Failed to get content by ID {content_id}: {e}")
            return None

    async def get_embedding_by_content_id(
        self, content_id: int, chunk_index: int = 0
    ) -> DBVectorEmbedding | None:
        """
        Get embedding by content ID and chunk index.

        Args:
            content_id: Content ID
            chunk_index: Chunk index

        Returns:
            DBVectorEmbedding object or None if not found
        """
        try:
            row = await self._db.fetch_one(
                """
                SELECT id, content_id, embedding, chunk_index, chunk_text,
                       line_start, line_end, metadata, created_at
                FROM vector_embedding
                WHERE content_id = $1 AND chunk_index = $2
                """,
                content_id,
                chunk_index,
            )

            if not row:
                return None

            return DBVectorEmbedding(
                id=row["id"],
                content_id=row["content_id"],
                embedding=list(row["embedding"]) if row["embedding"] else [],
                chunk_index=row["chunk_index"],
                created_at=row["created_at"],
            )

        except Exception as e:
            logger.error(f"Failed to get embedding for content {content_id}: {e}")
            return None

    async def list_content(
        self, filters: ContentFilter | None = None
    ) -> list[DBIndexedContent]:
        """
        List content with optional filters.

        Args:
            filters: Optional content filters

        Returns:
            List of DBIndexedContent objects
        """
        filters = filters or ContentFilter(
            content_types=None,
            file_paths=None,
            path_patterns=None,
            status_filter=None,
            max_results=100,
        )

        try:
            where_conditions = ["1=1"]
            params: list[Any] = []

            if filters.content_types:
                placeholder = ",".join(
                    f"${i}"
                    for i in range(
                        len(params) + 1, len(params) + 1 + len(filters.content_types)
                    )
                )
                where_conditions.append(
                    f"content_type::text = ANY(ARRAY[{placeholder}])"
                )
                params.extend(filters.content_types)

            if filters.file_paths:
                placeholder = ",".join(
                    f"${i}"
                    for i in range(
                        len(params) + 1, len(params) + 1 + len(filters.file_paths)
                    )
                )
                where_conditions.append(f"source_path = ANY(ARRAY[{placeholder}])")
                params.extend(filters.file_paths)

            if filters.path_patterns:
                pattern_conditions = []
                for pattern in filters.path_patterns:
                    params.append(f"%{pattern}%")
                    pattern_conditions.append(f"source_path LIKE ${len(params)}")
                where_conditions.append(f"({' OR '.join(pattern_conditions)})")

            if filters.status_filter:
                placeholder = ",".join(
                    f"${i}"
                    for i in range(
                        len(params) + 1, len(params) + 1 + len(filters.status_filter)
                    )
                )
                where_conditions.append(f"status::text = ANY(ARRAY[{placeholder}])")
                params.extend(filters.status_filter)

            params.append(filters.max_results)

            query = f"""
            SELECT id, content_type, source_path, content_hash, title, content,
                   metadata, status, indexed_at, updated_at, created_at
            FROM indexed_content
            WHERE {" AND ".join(where_conditions)}
            ORDER BY created_at DESC
            LIMIT ${len(params)}
            """

            rows = await self._db.fetch_all(query, *params)

            results = []
            for row in rows:
                content = DBIndexedContent(
                    id=row["id"],
                    content_type=row["content_type"],
                    source_path=row["source_path"],
                    content_hash=row["content_hash"],
                    title=row["title"],
                    content=row["content"],
                    metadata=dict(row["metadata"]) if row["metadata"] else {},
                    status=row["status"],
                    indexed_at=row["indexed_at"],
                    updated_at=row["updated_at"],
                    created_at=row["created_at"],
                )
                results.append(content)

            return results

        except Exception as e:
            logger.error(f"Failed to list content: {e}")
            raise RuntimeError(f"Failed to list content: {e}") from e

    async def update_content_status(
        self, content_id: int, status: str, indexed_at: datetime | None = None
    ) -> bool:
        """
        Update content indexing status.

        Args:
            content_id: Content ID
            status: New status (pending, indexing, completed, failed, stale)
            indexed_at: Optional indexing timestamp

        Returns:
            True if update was successful
        """
        try:
            if indexed_at:
                result = await self._db.execute(
                    """
                    UPDATE indexed_content
                    SET status = $1, indexed_at = $2, updated_at = NOW()
                    WHERE id = $3
                    """,
                    status,
                    indexed_at,
                    content_id,
                )
            else:
                result = await self._db.execute(
                    """
                    UPDATE indexed_content
                    SET status = $1, updated_at = NOW()
                    WHERE id = $2
                    """,
                    status,
                    content_id,
                )

            # Check if any rows were affected
            return "UPDATE 1" in result

        except Exception as e:
            logger.error(f"Failed to update content status: {e}")
            return False

    async def delete_content(self, content_id: int) -> bool:
        """
        Delete content and associated embeddings.

        Args:
            content_id: Content ID to delete

        Returns:
            True if deletion was successful
        """
        try:
            async with self._db.transaction() as conn:
                # Delete embeddings first (foreign key constraint)
                await conn.execute(
                    "DELETE FROM vector_embedding WHERE content_id = $1", content_id
                )

                # Delete content
                result = await conn.execute(
                    "DELETE FROM indexed_content WHERE id = $1", content_id
                )

                return "DELETE 1" in result

        except Exception as e:
            logger.error(f"Failed to delete content {content_id}: {e}")
            return False

    async def get_storage_stats(self) -> dict[str, Any]:
        """
        Get vector storage statistics.

        Returns:
            Storage statistics dictionary
        """
        try:
            stats = {}

            # Content counts by type and status
            content_stats = await self._db.fetch_all(
                """
                SELECT content_type, status, COUNT(*) as count
                FROM indexed_content
                GROUP BY content_type, status
                ORDER BY content_type, status
                """
            )

            content_summary: dict[str, dict[str, int]] = {}
            for row in content_stats:
                content_type = row["content_type"]
                if content_type not in content_summary:
                    content_summary[content_type] = {}
                content_summary[content_type][row["status"]] = row["count"]

            stats["content_by_type_status"] = content_summary

            # Total counts
            total_content = await self._db.fetch_val(
                "SELECT COUNT(*) FROM indexed_content"
            )
            total_embeddings = await self._db.fetch_val(
                "SELECT COUNT(*) FROM vector_embedding"
            )

            stats["total_content"] = total_content
            stats["total_embeddings"] = total_embeddings

            # Embedding model distribution (using default model since columns don't exist)
            total_embeddings = await self._db.fetch_val(
                "SELECT COUNT(*) FROM vector_embedding"
            )

            embedding_models = [
                {
                    "model": "BAAI/bge-small-en-v1.5:1.0",
                    "count": total_embeddings,
                }
            ]
            stats["embedding_models"] = embedding_models  # type: ignore

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
