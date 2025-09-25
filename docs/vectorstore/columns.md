# VectorStore Database Column Reference

**Generated**: 2025-09-25
**Source**: PostgreSQL Migration 014_semantic_search_core.sql (Production Schema)
**Verified**: Via live database introspection

## Table: indexed_content

Stores metadata about content indexed for semantic search.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| **id** | INTEGER | NOT NULL | nextval('indexed_content_id_seq') | Primary key, auto-incrementing |
| **content_type** | VARCHAR(50) | NOT NULL | - | Type of content (source_file, documentation, etc.) |
| **source_path** | TEXT | NOT NULL | - | Path to source file (UNIQUE constraint) |
| **content_hash** | VARCHAR(64) | NOT NULL | - | SHA256 hash for change detection |
| **title** | TEXT | NULL | - | Optional title for content |
| **content** | TEXT | NOT NULL | - | Full text content |
| **metadata** | JSONB | NULL | '{}' | Additional metadata as JSON |
| **status** | VARCHAR(20) | NOT NULL | 'PENDING' | Status: PENDING, PROCESSING, COMPLETED, FAILED |
| **chunk_count** | INTEGER | NULL | 0 | Number of embedding chunks created |
| **indexed_at** | TIMESTAMP WITH TIME ZONE | NULL | - | When indexing completed |
| **created_at** | TIMESTAMP WITH TIME ZONE | NULL | now() | Record creation timestamp |
| **updated_at** | TIMESTAMP WITH TIME ZONE | NULL | now() | Last update timestamp (auto-updated via trigger) |

### Constraints

- **PRIMARY KEY**: id (indexed_content_pkey)
- **UNIQUE**: source_path (indexed_content_source_path_key)

### Indexes

- **idx_indexed_content_source_path**: BTREE on source_path
- **idx_indexed_content_content_type**: BTREE on content_type
- **idx_indexed_content_status**: BTREE on status
- **idx_indexed_content_hash**: BTREE on content_hash
- **idx_indexed_content_indexed_at**: BTREE on indexed_at DESC
- **idx_indexed_content_updated_at**: BTREE on updated_at DESC
- **idx_indexed_content_metadata**: GIN on metadata
- **idx_indexed_content_content_fts**: GIN full-text search on content

---

## Table: vector_embedding

Stores 384-dimensional vector embeddings for semantic similarity search.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| **id** | INTEGER | NOT NULL | nextval('vector_embedding_id_seq') | Primary key, auto-incrementing |
| **content_id** | INTEGER | NOT NULL | - | Foreign key to indexed_content(id) |
| **embedding** | VECTOR(384) | NULL | - | 384-dimensional vector from BAAI/bge-small-en-v1.5 |
| **chunk_index** | INTEGER | NOT NULL | - | Sequential index within parent content |
| **chunk_text** | TEXT | NOT NULL | - | Text content that was embedded |
| **line_start** | INTEGER | NULL | - | Starting line number in source |
| **line_end** | INTEGER | NULL | - | Ending line number in source |
| **metadata** | JSONB | NULL | '{}' | Additional metadata as JSON |
| **created_at** | TIMESTAMP WITH TIME ZONE | NULL | now() | Record creation timestamp |

### Constraints

- **PRIMARY KEY**: id (vector_embedding_pkey)
- **FOREIGN KEY**: content_id → indexed_content(id) ON DELETE CASCADE
- **UNIQUE**: (content_id, chunk_index) - Allows multiple chunks per content

### Indexes

- **idx_vector_embedding_content_id**: BTREE on content_id
- **idx_vector_embedding_chunk_index**: BTREE on chunk_index
- **idx_vector_embedding_lines**: BTREE on (line_start, line_end)
- **idx_vector_embedding_created_at**: BTREE on created_at DESC
- **idx_vector_embedding_similarity**: IVFFlat on embedding vector_cosine_ops WITH (lists = 100)

---

## Table: search_query

Stores search queries and processing metadata for analytics and optimization.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| **id** | INTEGER | NOT NULL | nextval('search_query_id_seq') | Primary key, auto-incrementing |
| **query_text** | TEXT | NOT NULL | - | Original query text |
| **query_embedding** | VECTOR(384) | NULL | - | Vector embedding of query |
| **preprocessing_config** | JSONB | NULL | '{}' | Query preprocessing settings |
| **created_at** | TIMESTAMP WITH TIME ZONE | NULL | now() | Query submission timestamp |
| **processed_at** | TIMESTAMP WITH TIME ZONE | NULL | - | Processing completion timestamp |
| **processing_time_ms** | INTEGER | NULL | - | Processing duration in milliseconds |
| **status** | VARCHAR(20) | NOT NULL | 'PENDING' | Status: PENDING, PROCESSING, COMPLETED, FAILED |

### Constraints

- **PRIMARY KEY**: id (search_query_pkey)

### Indexes

- **idx_search_query_text**: BTREE on query_text
- **idx_search_query_status**: BTREE on status
- **idx_search_query_created_at**: BTREE on created_at DESC
- **idx_search_query_processing_time**: BTREE on processing_time_ms
- **idx_search_query_text_fts**: GIN full-text search on query_text
- **idx_search_query_embedding**: IVFFlat on query_embedding

---

## Table: search_result

Stores search results with scoring and ranking for result analysis.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| **id** | INTEGER | NOT NULL | nextval('search_result_id_seq') | Primary key, auto-incrementing |
| **query_id** | INTEGER | NOT NULL | - | Foreign key to search_query(id) |
| **content_id** | INTEGER | NOT NULL | - | Foreign key to indexed_content(id) |
| **embedding_id** | INTEGER | NOT NULL | - | Foreign key to vector_embedding(id) |
| **similarity_score** | DOUBLE PRECISION | NOT NULL | - | Cosine similarity score (0.0-1.0) |
| **bm25_score** | DOUBLE PRECISION | NULL | - | BM25 text similarity score |
| **combined_score** | DOUBLE PRECISION | NOT NULL | - | Hybrid score combining similarity and BM25 |
| **result_rank** | INTEGER | NOT NULL | - | Result ranking position (1-based) |
| **context_lines** | TEXT[] | NULL | - | Array of surrounding context lines |
| **explanation** | TEXT | NULL | - | Result relevance explanation |
| **metadata** | JSONB | NULL | '{}' | Additional metadata as JSON |
| **created_at** | TIMESTAMP WITH TIME ZONE | NULL | now() | Record creation timestamp |

### Constraints

- **PRIMARY KEY**: id (search_result_pkey)
- **FOREIGN KEY**: query_id → search_query(id) ON DELETE CASCADE
- **FOREIGN KEY**: content_id → indexed_content(id) ON DELETE CASCADE
- **FOREIGN KEY**: embedding_id → vector_embedding(id) ON DELETE CASCADE
- **UNIQUE**: (query_id, result_rank) - One rank per result per query

### Indexes

- **idx_search_result_query_id**: BTREE on query_id
- **idx_search_result_content_id**: BTREE on content_id
- **idx_search_result_embedding_id**: BTREE on embedding_id
- **idx_search_result_similarity_score**: BTREE on similarity_score DESC
- **idx_search_result_combined_score**: BTREE on combined_score DESC
- **idx_search_result_rank**: BTREE on result_rank
- **idx_search_result_created_at**: BTREE on created_at DESC

---

## Data Types Reference

### PostgreSQL Native Types

- **INTEGER**: 32-bit signed integer (-2,147,483,648 to 2,147,483,647)
- **TEXT**: Variable-length character string (unlimited)
- **VARCHAR(n)**: Variable character with limit n
- **DOUBLE PRECISION**: 64-bit floating point
- **TIMESTAMP WITH TIME ZONE**: Date and time with timezone
- **JSONB**: Binary JSON data (supports indexing)
- **TEXT[]**: Array of text values

### pgvector Extension Types

- **VECTOR(384)**: 384-dimensional floating-point vector
  - Used for semantic embeddings
  - Supports cosine similarity operations
  - Indexed with IVFFlat for efficient similarity search

---

## Database Functions

### Auto-update Trigger

```sql
CREATE TRIGGER update_indexed_content_updated_at
    BEFORE UPDATE ON indexed_content
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### Helper Functions Available

- **update_updated_at_column()**: Auto-updates updated_at timestamp
- **vector_cosine_similarity(a, b)**: Compute cosine similarity between vectors
- **find_similar_embeddings()**: Similarity search with filters
- **cleanup_old_search_results()**: Retention management for search results
- **reindex_content()**: Mark content for reprocessing

---

## Status Enumerations

### indexed_content.status / search_query.status

- **PENDING**: Initial state, waiting to be processed
- **PROCESSING**: Currently being processed
- **COMPLETED**: Successfully processed
- **FAILED**: Processing failed with error

---

## Index Performance Notes

### IVFFlat Index Configuration

- **lists = 100**: Number of inverted lists for IVFFlat index
- Provides good balance between build time and query performance
- Suitable for datasets with 100K-1M vectors

### GIN Indexes

- Used for JSONB metadata and full-text search
- Supports containment queries on JSON fields
- Enables fast text search with PostgreSQL full-text capabilities

### BTREE Indexes

- Standard indexes for equality and range queries
- Used for primary keys, foreign keys, and filtering columns
- DESC ordering on timestamp columns for recent-first queries

---

## Storage Considerations

### Estimated Storage per Record

**indexed_content**:

- Base record: ~100 bytes
- Content text: Variable (typically 1-50 KB)
- Metadata JSONB: ~200 bytes average
- **Total**: 1-50 KB per record

**vector_embedding**:

- Base record: ~50 bytes
- Embedding vector: 384 * 4 bytes = 1,536 bytes
- Chunk text: Variable (typically 500-2000 bytes)
- **Total**: ~3-4 KB per embedding

**search_query**:

- Base record: ~100 bytes
- Query embedding: 1,536 bytes
- **Total**: ~2 KB per query

**search_result**:

- Base record: ~150 bytes
- Context lines: Variable (typically 500-1000 bytes)
- **Total**: ~1-2 KB per result

---

## Migration Notes

This schema is from production migration 014_semantic_search_core.sql. Key features:

- Supports multiple embeddings per content via chunk_index
- Uses VARCHAR for status fields (more flexible than ENUM)
- IVFFlat indexing for production performance
- Full cascade deletion support via foreign keys
- Automatic timestamp management via triggers
