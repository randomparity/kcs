# VectorStore API Reference

**Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: 2025-09-25

## Overview

The VectorStore provides high-performance vector storage and similarity search operations for semantic search functionality using PostgreSQL with pgvector extension. This comprehensive reference combines all documentation for the VectorStore system.

## Quick Links

| Documentation | Description |
|--------------|-------------|
| [Setup Guide](./setup.md) | Installation, configuration, and deployment |
| [API Examples](./examples.md) | Usage examples for all 9 API methods |
| [Error Handling](./errors.md) | Error types, recovery patterns, debugging |
| [OpenAPI Spec](./api.html) | Interactive API documentation |
| [Database Schema](./schema.svg) | Entity relationship diagram |
| [Method Reference](./methods.md) | Detailed method signatures |
| [Column Reference](./columns.md) | Database column specifications |

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Application   │────▶│   VectorStore    │────▶│   PostgreSQL    │
│     Layer       │     │      API         │     │   + pgvector    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                         │
        │                       │                         │
        ▼                       ▼                         ▼
   Store Content         Manage Embeddings         Vector Search
   Text & Metadata       384-dim Vectors          Cosine Similarity
```

## Core Features

### 1. Content Storage
- Store documents with metadata
- SHA256 hash-based deduplication
- JSONB metadata for flexible attributes
- Status tracking (PENDING → PROCESSING → COMPLETED/FAILED)

### 2. Vector Embeddings
- 384-dimensional vectors (BAAI/bge-small-en-v1.5)
- Multiple chunks per document
- IVFFlat indexing for performance
- Cosine similarity search

### 3. Search Capabilities
- Similarity threshold filtering
- Content type filtering
- Path pattern matching
- Result limiting and ranking

## API Methods

### Content Management

#### `store_content()`
Store content for indexing with automatic deduplication.

```python
async def store_content(
    content_type: str,           # Type of content (e.g., "source_code")
    source_path: str,            # Unique path identifier
    content: str,                # Text content to store
    title: str | None = None,    # Optional title
    metadata: dict | None = None # Optional metadata
) -> int  # Returns content ID
```

**Key Features**:
- Automatic SHA256 hash generation
- Idempotent operation (returns existing ID for duplicates)
- JSONB metadata storage
- Automatic status management

#### `get_content_by_id()`
Retrieve specific content by ID.

```python
async def get_content_by_id(
    content_id: int
) -> DBIndexedContent | None
```

#### `list_content()`
List content with advanced filtering.

```python
async def list_content(
    filters: ContentFilter | None = None
) -> list[DBIndexedContent]
```

**Filter Options**:
- `content_types`: Filter by content type
- `file_paths`: Exact path matches
- `path_patterns`: SQL LIKE patterns
- `status_filter`: Status values
- `max_results`: Limit results (1-1000)

#### `update_content_status()`
Update content indexing status.

```python
async def update_content_status(
    content_id: int,
    status: str,  # PENDING|PROCESSING|COMPLETED|FAILED
    error_message: str | None = None
) -> bool
```

#### `delete_content()`
Delete content and cascade to embeddings.

```python
async def delete_content(
    content_id: int
) -> bool
```

### Embedding Management

#### `store_embedding()`
Store vector embedding for content chunk.

```python
async def store_embedding(
    content_id: int,                # Reference to content
    embedding: list[float],          # 384-dimensional vector
    chunk_text: str,                 # Text that was embedded
    chunk_index: int = 0,            # Chunk number
    model_name: str = "BAAI/bge-small-en-v1.5",
    model_version: str = "1.5"
) -> int  # Returns embedding ID
```

**Validation**:
- Enforces exactly 384 dimensions
- Unique constraint on (content_id, chunk_index)
- Automatic status update on parent content

#### `get_embedding_by_content_id()`
Retrieve embeddings for content.

```python
async def get_embedding_by_content_id(
    content_id: int,
    chunk_index: int | None = None  # Optional specific chunk
) -> list[DBVectorEmbedding]
```

### Search Operations

#### `similarity_search()`
Perform vector similarity search.

```python
async def similarity_search(
    query_embedding: list[float],      # 384-dim query vector
    filters: SimilaritySearchFilter | None = None
) -> list[dict[str, Any]]
```

**Filter Options**:
- `similarity_threshold`: Minimum similarity (0.0-1.0)
- `max_results`: Limit results (1-100)
- `content_types`: Filter by type
- `file_paths`: Filter by paths
- `include_content`: Include full text in results

**Result Format**:
```python
{
    "content_id": int,
    "content_type": str,
    "source_path": str,
    "title": str | None,
    "metadata": dict,
    "embedding_id": int,
    "chunk_index": int,
    "similarity_score": float,  # 0.0 to 1.0
    "content": str  # If include_content=True
}
```

### System Operations

#### `get_storage_stats()`
Get storage statistics and system health.

```python
async def get_storage_stats() -> dict[str, Any]
```

**Returns**:
```python
{
    "total_content": int,
    "total_embeddings": int,
    "content_by_type": dict[str, int],
    "content_by_status": dict[str, int],
    "avg_chunks_per_content": float,
    "total_storage_bytes": int,
    "vector_index_size": int,
    "database_size": int
}
```

## Database Schema

### Tables

#### `indexed_content`
Stores content metadata and text.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | SERIAL | PRIMARY KEY | Auto-generated ID |
| content_type | VARCHAR(50) | NOT NULL | Content classification |
| source_path | TEXT | UNIQUE, NOT NULL | Unique identifier |
| content_hash | VARCHAR(64) | NOT NULL | SHA256 hash |
| title | TEXT | | Optional title |
| content | TEXT | NOT NULL | Full text content |
| metadata | JSONB | DEFAULT '{}' | Flexible metadata |
| status | VARCHAR(20) | NOT NULL | Indexing status |
| chunk_count | INTEGER | DEFAULT 0 | Number of chunks |
| indexed_at | TIMESTAMP | | Completion timestamp |
| created_at | TIMESTAMP | DEFAULT now() | Creation time |
| updated_at | TIMESTAMP | DEFAULT now() | Last update |

#### `vector_embedding`
Stores vector embeddings for similarity search.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | SERIAL | PRIMARY KEY | Auto-generated ID |
| content_id | INTEGER | FK, NOT NULL | References indexed_content |
| embedding | VECTOR(384) | | 384-dimensional vector |
| chunk_index | INTEGER | NOT NULL | Chunk sequence number |
| chunk_text | TEXT | NOT NULL | Embedded text |
| line_start | INTEGER | | Starting line number |
| line_end | INTEGER | | Ending line number |
| metadata | JSONB | DEFAULT '{}' | Chunk metadata |
| created_at | TIMESTAMP | DEFAULT now() | Creation time |

**Constraints**:
- UNIQUE(content_id, chunk_index)
- ON DELETE CASCADE for content_id

### Indexes

#### Performance Indexes
- `idx_vector_embedding_similarity`: IVFFlat on embedding (lists=100)
- `idx_indexed_content_source_path`: BTREE for fast lookups
- `idx_indexed_content_content_fts`: GIN for full-text search
- `idx_vector_embedding_content_id`: BTREE for joins

## Configuration

### Environment Variables

```bash
# Database Connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=kcs
POSTGRES_USER=kcs
POSTGRES_PASSWORD=your_password

# Connection Pool
POSTGRES_POOL_MIN_SIZE=2
POSTGRES_POOL_MAX_SIZE=20
POSTGRES_POOL_TIMEOUT=30

# Vector Configuration
VECTOR_DIMENSIONS=384
VECTOR_MODEL=BAAI/bge-small-en-v1.5
```

### Connection Management

```python
from semantic_search.database import DatabaseConfig, init_database_connection

# Initialize connection
config = DatabaseConfig.from_env()
await init_database_connection(config)

# Use VectorStore
store = VectorStore()
results = await store.similarity_search(query_embedding)
```

## Performance Considerations

### IVFFlat Index
- Current: `lists=100` for datasets up to 1M vectors
- Provides good balance of speed vs accuracy
- Consider HNSW for larger datasets

### Batch Operations
```python
# Efficient batch processing
async def batch_index(items: list, batch_size=100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        await process_batch(batch)
```

### Connection Pooling
- Minimum connections: 2
- Maximum connections: 20
- Timeout: 30 seconds
- Monitor pool usage for optimization

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Expected 384 dimensions` | Wrong model output | Use BAAI/bge-small-en-v1.5 |
| `ConnectionError: Failed to connect` | Database down | Check PostgreSQL status |
| `UniqueViolationError` | Duplicate path | Use idempotent operations |
| `ForeignKeyViolationError` | Invalid reference | Validate content exists |

### Recovery Patterns

```python
# Retry with exponential backoff
async def retry_operation(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except (ConnectionError, RuntimeError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Testing

### Verification Script
```bash
python tests/verification/test_verify_foundation.py
```

### Test Coverage
- ✅ All 9 API methods verified
- ✅ Database schema validated
- ✅ 384-dimensional vectors confirmed
- ✅ Multiple chunks per file tested
- ✅ Constraint enforcement validated
- ✅ Error handling verified

## Monitoring

### Health Check Endpoint
```python
async def health_check() -> dict:
    return {
        "status": "healthy",
        "database": await check_database(),
        "pgvector": await check_extension(),
        "stats": await store.get_storage_stats()
    }
```

### Metrics to Track
- Query latency (p50, p95, p99)
- Error rates by operation
- Connection pool utilization
- Index performance
- Storage growth rate

## Migration Guide

For migration instructions and schema updates, see [migration.md](./migration.md).

## Performance Tuning

For performance optimization and index selection, see [performance.md](./performance.md).

## Support

### Documentation
- [Setup Guide](./setup.md) - Installation and configuration
- [Examples](./examples.md) - Code examples for all operations
- [Error Handling](./errors.md) - Debugging and recovery
- [OpenAPI Spec](./api.html) - Interactive API documentation

### Troubleshooting

1. **Connection Issues**: Check environment variables and PostgreSQL status
2. **Dimension Errors**: Verify using correct embedding model
3. **Performance Issues**: Review index configuration and query patterns
4. **Memory Issues**: Adjust connection pool and batch sizes

## Version History

### v1.0.0 (2025-09-25)
- Initial production release
- 384-dimensional vector support
- IVFFlat indexing
- Complete API implementation
- Comprehensive documentation

---

*This API reference combines all VectorStore documentation. For specific topics, refer to the individual documentation files linked above.*