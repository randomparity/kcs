# VectorStore API Method Signatures

**Generated**: 2025-09-25
**Source**: `/home/dave/src/kcs/src/python/semantic_search/database/vector_store.py`

## Overview

The VectorStore class provides 9 public methods for managing vector embeddings and content storage with efficient similarity search using PostgreSQL and pgvector.

## Method Signatures

### 1. store_content()

```python
async def store_content(
    content_type: str,
    source_path: str,
    content: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None
) -> int
```

**Description**: Store content for indexing
**Returns**: Content ID (integer)
**Raises**:

- `ValueError`: If content is empty or source path is invalid
- `RuntimeError`: If storage fails

**Implementation Details**:

- Generates SHA256 hash for content change detection
- Checks for existing identical content before inserting
- Returns existing ID if content already exists

---

### 2. store_embedding()

```python
async def store_embedding(
    content_id: int,
    embedding: list[float],
    chunk_text: str,
    chunk_index: int = 0,
    model_name: str = "BAAI/bge-small-en-v1.5",
    model_version: str = "1.5"
) -> int
```

**Description**: Store vector embedding for content chunk
**Returns**: Embedding ID (integer)
**Raises**:

- `ValueError`: If embedding is empty or not 384 dimensions
- `RuntimeError`: If storage fails

**Implementation Details**:

- Validates embedding has exactly 384 dimensions
- Updates existing embedding if chunk_index already exists
- Supports multiple chunks per content (unique on content_id + chunk_index)

---

### 3. similarity_search()

```python
async def similarity_search(
    query_embedding: list[float],
    filters: SimilaritySearchFilter | None = None
) -> list[dict[str, Any]]
```

**Description**: Perform vector similarity search
**Returns**: List of search results with similarity scores
**Raises**:

- `ValueError`: If query embedding is invalid
- `RuntimeError`: If search fails

**Filter Options**:

- `similarity_threshold`: Minimum cosine similarity score (0.0-1.0)
- `max_results`: Maximum number of results (1-100, default: 20)
- `content_types`: Filter by content types
- `file_paths`: Filter by specific file paths
- `include_content`: Include full content in results (default: true)

---

### 4. get_content_by_id()

```python
async def get_content_by_id(
    content_id: int
) -> DBIndexedContent | None
```

**Description**: Retrieve content by ID
**Returns**: DBIndexedContent object or None if not found
**Raises**: `RuntimeError`: If retrieval fails

---

### 5. get_embedding_by_content_id()

```python
async def get_embedding_by_content_id(
    content_id: int,
    chunk_index: int = 0
) -> DBVectorEmbedding | None
```

**Description**: Retrieve embedding by content ID and chunk index
**Returns**: DBVectorEmbedding object or None if not found
**Raises**: `RuntimeError`: If retrieval fails

---

### 6. list_content()

```python
async def list_content(
    filters: ContentFilter | None = None
) -> list[DBIndexedContent]
```

**Description**: List content with optional filters
**Returns**: List of DBIndexedContent objects
**Raises**: `RuntimeError`: If listing fails

**Filter Options**:

- `content_types`: Filter by content types
- `file_paths`: Filter by specific file paths
- `path_patterns`: Path patterns for LIKE matching
- `status_filter`: Filter by indexing status (PENDING, PROCESSING, COMPLETED, FAILED)
- `max_results`: Maximum results (1-1000, default: 100)

---

### 7. update_content_status()

```python
async def update_content_status(
    content_id: int,
    status: str,
    indexed_at: datetime | None = None
) -> bool
```

**Description**: Update content indexing status
**Returns**: True if updated, False if not found
**Raises**:

- `ValueError`: If status is invalid
- `RuntimeError`: If update fails

**Valid Status Values**:

- `PENDING`: Initial state
- `PROCESSING`: Currently being indexed
- `COMPLETED`: Successfully indexed
- `FAILED`: Indexing failed

---

### 8. delete_content()

```python
async def delete_content(
    content_id: int
) -> bool
```

**Description**: Delete content and all associated embeddings
**Returns**: True if deleted, False if not found
**Raises**: `RuntimeError`: If deletion fails

**Cascade Behavior**:

- Deletes all vector_embedding records with matching content_id
- Deletes all search_result records referencing the content

---

### 9. get_storage_stats()

```python
async def get_storage_stats() -> dict[str, Any]
```

**Description**: Get vector storage statistics
**Returns**: Dictionary with storage statistics
**Raises**: `RuntimeError`: If stats retrieval fails

**Returned Statistics**:

```python
{
    "content_by_type_status": {
        "source_file": {
            "COMPLETED": 150,
            "PENDING": 5
        }
    },
    "total_content": 155,
    "total_embeddings": 450,
    "embedding_models": [
        {
            "model": "BAAI/bge-small-en-v1.5",
            "count": 450
        }
    ]
}
```

---

## Data Models

### DBIndexedContent

```python
class DBIndexedContent:
    id: int
    content_type: str
    source_path: str
    content_hash: str
    title: str | None
    content: str
    metadata: dict[str, Any]
    status: str
    indexed_at: datetime | None
    updated_at: datetime
    created_at: datetime
```

### DBVectorEmbedding

```python
class DBVectorEmbedding:
    id: int
    content_id: int
    embedding: list[float]  # 384 dimensions
    chunk_index: int
    created_at: datetime
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_version: str = "1.0"
```

### ContentFilter

```python
class ContentFilter(BaseModel):
    content_types: list[str] | None = None
    file_paths: list[str] | None = None
    path_patterns: list[str] | None = None
    status_filter: list[str] | None = None
    max_results: int = 100  # Range: 1-1000
```

### SimilaritySearchFilter

```python
class SimilaritySearchFilter(BaseModel):
    similarity_threshold: float = 0.0  # Range: 0.0-1.0
    max_results: int = 20  # Range: 1-100
    content_types: list[str] | None = None
    file_paths: list[str] | None = None
    include_content: bool = True
```

---

## Vector Configuration

| Property | Value |
|----------|-------|
| **Dimensions** | 384 |
| **Model** | BAAI/bge-small-en-v1.5 |
| **Distance Metric** | Cosine similarity |
| **Index Type** | IVFFlat (lists=100) |
| **Operator** | `<=>` (pgvector cosine distance) |

---

## Usage Notes

1. **Connection Initialization**: The VectorStore requires database connection to be initialized via `init_database_connection()` before use

2. **Multiple Chunks Support**: The schema explicitly supports multiple embeddings per content file using the `chunk_index` field

3. **Idempotent Storage**: Content storage is idempotent - storing the same content (same hash) returns the existing ID

4. **Vector Validation**: All embeddings must be exactly 384 dimensions (validated in `store_embedding()`)

5. **Cascade Deletion**: Deleting content automatically removes all associated embeddings via foreign key constraints

---

## Error Handling

All methods follow consistent error handling patterns:

- **ValueError**: Input validation failures (empty content, wrong dimensions, invalid status)
- **RuntimeError**: Database operation failures (connection errors, query failures)
- **None Returns**: Methods return None when the requested resource doesn't exist

Example error handling:

```python
try:
    content_id = await store.store_content(
        content_type="source_file",
        source_path="/src/main.py",
        content="..."
    )
except ValueError as e:
    # Handle validation error
    logger.error(f"Invalid input: {e}")
except RuntimeError as e:
    # Handle database error
    logger.error(f"Database operation failed: {e}")
```
