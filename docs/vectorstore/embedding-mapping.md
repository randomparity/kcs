# DBVectorEmbedding Field to Database Column Mapping

**Generated**: 2025-09-25
**Source**: `/home/dave/src/kcs/src/python/semantic_search/database/vector_store.py`
**Database**: PostgreSQL table `vector_embedding`

## Overview

This document maps the Python `DBVectorEmbedding` class fields to their corresponding database columns in the `vector_embedding` table, highlighting discrepancies between the Python model and actual database schema.

## Field Mapping Table

| Python Field | Python Type | Database Column | Database Type | Constraints | Notes |
|--------------|-------------|-----------------|---------------|-------------|-------|
| `id` | `int` | `id` | `INTEGER` | PRIMARY KEY, NOT NULL | Auto-generated via `SERIAL` |
| `content_id` | `int` | `content_id` | `INTEGER` | NOT NULL, FK | References `indexed_content(id)` |
| `embedding` | `list[float]` | `embedding` | `VECTOR(384)` | NULL | 384-dimensional float array |
| `chunk_index` | `int` | `chunk_index` | `INTEGER` | NOT NULL | Sequential chunk number |
| `created_at` | `datetime` | `created_at` | `TIMESTAMP WITH TIME ZONE` | DEFAULT now() | Record creation time |
| `model_name` | `str` | **NOT STORED** | - | - | Default: "BAAI/bge-small-en-v1.5" |
| `model_version` | `str` | **NOT STORED** | - | - | Default: "1.0" |

## Database Columns Not in Python Model

| Database Column | Database Type | Purpose | Why Not in Python Model |
|-----------------|---------------|---------|------------------------|
| `chunk_text` | `TEXT` | Stores actual text that was embedded | Passed as parameter, not exposed in model |
| `line_start` | `INTEGER` | Starting line number in source | Not currently used by Python code |
| `line_end` | `INTEGER` | Ending line number in source | Not currently used by Python code |
| `metadata` | `JSONB` | Additional metadata storage | Not exposed in Python model |

## Critical Discrepancies

### 1. Model Metadata Not Persisted

- **Python Model**: Has `model_name` and `model_version` fields with defaults
- **Database**: No columns for storing this information
- **Impact**: Cannot track which model generated embeddings
- **Workaround**: Could store in `metadata` JSONB field

### 2. Chunk Text Storage Mismatch

- **store_embedding() method**: Accepts `chunk_text` parameter
- **Database**: Stores in `chunk_text` column
- **Python Model**: No `chunk_text` field
- **Impact**: Text is stored but not retrievable via model

## Data Type Conversions

### Python → Database

1. **Vector Embedding** (`list[float]`)
   - Python list of 384 floats
   - Converted to PostgreSQL `VECTOR(384)` type
   - Stored as string representation: `[0.1, 0.2, ...]`
   - Must be exactly 384 dimensions (validated)

2. **Foreign Key** (`int`)
   - `content_id` references `indexed_content(id)`
   - Cascade deletion enabled

### Database → Python

1. **Vector Retrieval**
   - PostgreSQL `VECTOR` → Python `list[float]`
   - Automatic conversion handled by pgvector

2. **NULL Handling**
   - Database allows NULL embedding (during processing)
   - Python expects list, not None

## Code Implementation

### Python Class Definition

```python
class DBVectorEmbedding:
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
        self.model_name = model_name  # NOT STORED IN DB
        self.model_version = model_version  # NOT STORED IN DB
        self.chunk_index = chunk_index
        self.created_at = created_at
```

### Database Insert Implementation

```python
async def store_embedding(
    self,
    content_id: int,
    embedding: list[float],
    chunk_text: str,  # Stored but not in model
    chunk_index: int = 0,
    model_name: str = "BAAI/bge-small-en-v1.5",  # Ignored
    model_version: str = "1.5",  # Ignored
) -> int:
    # Validate 384 dimensions
    if len(embedding) != 384:
        raise ValueError(f"Expected 384 dimensions, got {len(embedding)}")

    # Insert or update
    embedding_id = await self._db.fetch_val(
        """
        INSERT INTO vector_embedding (
            content_id, embedding, chunk_text, chunk_index
        ) VALUES ($1, $2::vector, $3, $4)
        ON CONFLICT (content_id, chunk_index)
        DO UPDATE SET embedding = $2::vector, chunk_text = $3
        RETURNING id
        """,
        content_id,
        str(embedding),  # Convert to string for pgvector
        chunk_text,
        chunk_index,
    )
    return int(embedding_id)
```

## Validation Rules

### Python-Level Validation

- Embedding must be exactly 384 dimensions
- Embedding cannot be empty
- `chunk_index` must be >= 0

### Database-Level Constraints

- UNIQUE constraint on `(content_id, chunk_index)` pair
- Foreign key constraint on `content_id`
- Cascade deletion when parent content deleted

## Indexing Strategy

| Index Name | Column(s) | Type | Purpose |
|------------|-----------|------|---------|
| `idx_vector_embedding_similarity` | `embedding` | IVFFlat | Similarity search (lists=100) |
| `idx_vector_embedding_content_id` | `content_id` | BTREE | Fast lookups by content |
| `idx_vector_embedding_chunk_index` | `chunk_index` | BTREE | Ordered chunk retrieval |
| `idx_vector_embedding_lines` | `line_start, line_end` | BTREE | Line range queries |
| `idx_vector_embedding_created_at` | `created_at DESC` | BTREE | Recent embeddings first |

## Vector Operations

### Similarity Search

```sql
-- Find similar embeddings using cosine distance
SELECT id, content_id, chunk_index,
       1 - (embedding <=> $1::vector) AS similarity
FROM vector_embedding
WHERE embedding <=> $1::vector < 0.3  -- Distance threshold
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

### Distance Metrics

- `<=>`: Cosine distance (default)
- `<->`: Euclidean distance
- `<#>`: Inner product

## Usage Notes

1. **Multiple Chunks**: Supports multiple embeddings per content via `chunk_index`
2. **Idempotent Updates**: Conflicts on `(content_id, chunk_index)` result in updates
3. **384 Dimensions**: Strictly enforced, matches BAAI/bge-small-en-v1.5 model output
4. **Model Tracking Gap**: Model version not stored - consider adding to metadata
5. **Text Storage**: `chunk_text` stored but not exposed in Python model

## Recommendations

1. **Add Model Tracking**: Store `model_name` and `model_version` in `metadata` JSONB
2. **Expose Chunk Text**: Add `chunk_text` field to Python model for retrieval
3. **Use Line Numbers**: Implement line number tracking for better context
4. **Metadata Usage**: Leverage JSONB `metadata` field for additional properties
