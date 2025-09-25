# DBIndexedContent Field to Database Column Mapping

**Generated**: 2025-09-25
**Source**: `/home/dave/src/kcs/src/python/semantic_search/database/vector_store.py`
**Database**: PostgreSQL table `indexed_content`

## Overview

This document maps the Python `DBIndexedContent` class fields to their corresponding database columns in the `indexed_content` table, including data type conversions and constraints.

## Field Mapping Table

| Python Field | Python Type | Database Column | Database Type | Constraints | Notes |
|--------------|-------------|-----------------|---------------|-------------|-------|
| `id` | `int` | `id` | `INTEGER` | PRIMARY KEY, NOT NULL | Auto-generated via `SERIAL` |
| `content_type` | `str` | `content_type` | `VARCHAR(50)` | NOT NULL | Type of content (source_file, documentation) |
| `source_path` | `str` | `source_path` | `TEXT` | NOT NULL, UNIQUE | Filesystem or logical path |
| `content_hash` | `str` | `content_hash` | `VARCHAR(64)` | NOT NULL | SHA256 hash (64 hex chars) |
| `title` | `str \| None` | `title` | `TEXT` | NULL | Optional descriptive title |
| `content` | `str` | `content` | `TEXT` | NOT NULL | Full text content |
| `metadata` | `dict[str, Any]` | `metadata` | `JSONB` | DEFAULT '{}' | Flexible key-value storage |
| `status` | `str` | `status` | `VARCHAR(20)` | NOT NULL, DEFAULT 'PENDING' | PENDING/PROCESSING/COMPLETED/FAILED |
| `indexed_at` | `datetime \| None` | `indexed_at` | `TIMESTAMP WITH TIME ZONE` | NULL | When indexing completed |
| `updated_at` | `datetime` | `updated_at` | `TIMESTAMP WITH TIME ZONE` | DEFAULT now() | Auto-updated via trigger |
| `created_at` | `datetime` | `created_at` | `TIMESTAMP WITH TIME ZONE` | DEFAULT now() | Record creation time |

## Additional Database Columns (Not in Python Model)

| Database Column | Database Type | Purpose | Why Not in Python Model |
|-----------------|---------------|---------|------------------------|
| `chunk_count` | `INTEGER` | Tracks number of embeddings | Computed from vector_embedding table |

## Data Type Conversions

### Python → Database

1. **String Fields** (`str`)
   - Directly stored as `TEXT` or `VARCHAR`
   - Unicode supported via PostgreSQL UTF-8 encoding

2. **Integer Fields** (`int`)
   - Stored as `INTEGER` (32-bit)
   - Auto-increment handled by `SERIAL` type

3. **DateTime Fields** (`datetime`)
   - Converted to `TIMESTAMP WITH TIME ZONE`
   - Timezone-aware storage and retrieval
   - Python `datetime` objects ↔ PostgreSQL timestamps

4. **Dictionary/JSON Fields** (`dict[str, Any]`)
   - Serialized to `JSONB`
   - Supports nested structures
   - Queryable via PostgreSQL JSON operators

### Database → Python

1. **NULL Handling**
   - Database `NULL` → Python `None`
   - Python `None` → Database `NULL`

2. **Default Values**
   - `status`: Database default 'PENDING' if not provided
   - `metadata`: Database default '{}' (empty JSON) if not provided
   - `created_at`/`updated_at`: Database `now()` function

## Code Implementation

### Python Class Definition

```python
class DBIndexedContent:
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
```

### Database Insert Example

```python
async def store_content(
    self,
    content_type: str,
    source_path: str,
    content: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    metadata_json = json.dumps(metadata or {})
    content_id = await self._db.fetch_val(
        """
        INSERT INTO indexed_content (
            content_type, source_path, content_hash,
            title, content, metadata
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
    return int(content_id)
```

## Validation Rules

### Python-Level Validation

- `content` must not be empty (checked in `store_content()`)
- `source_path` must not be empty (checked in `store_content()`)
- `content_hash` generated as 64-character SHA256 hex string

### Database-Level Constraints

- `source_path` must be unique (UNIQUE constraint)
- `status` limited to 20 characters
- `content_type` limited to 50 characters
- Foreign key relationships cascade on delete

## Indexing Strategy

The following indexes optimize query performance:

| Index Name | Column(s) | Type | Purpose |
|------------|-----------|------|---------|
| `idx_indexed_content_source_path` | `source_path` | BTREE | Fast lookups by path |
| `idx_indexed_content_content_type` | `content_type` | BTREE | Filter by type |
| `idx_indexed_content_status` | `status` | BTREE | Filter by status |
| `idx_indexed_content_hash` | `content_hash` | BTREE | Duplicate detection |
| `idx_indexed_content_metadata` | `metadata` | GIN | JSON queries |
| `idx_indexed_content_content_fts` | `content` | GIN | Full-text search |

## Usage Notes

1. **Hash Calculation**: Content hash is computed before storage to detect duplicates
2. **Idempotent Storage**: Same content (same hash) returns existing ID without creating duplicate
3. **Status Workflow**: PENDING → PROCESSING → COMPLETED/FAILED
4. **Metadata Flexibility**: JSONB allows schema-less additional data
5. **Auto-update Trigger**: `updated_at` automatically updated on any row change
