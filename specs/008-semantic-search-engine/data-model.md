# Data Model: Semantic Search Engine

## Core Entities

### SearchQuery

**Purpose**: Represents a user's natural language search request

**Fields**:

- `query_text` (string): The original user query
- `processed_query` (string): Normalized and enhanced query text
- `embedding` (vector[384]): Query vector representation
- `timestamp` (datetime): When the query was submitted
- `user_id` (string, optional): Authenticated user identifier
- `config_context` (list[string]): Kernel configuration filters

**Validation Rules**:

- `query_text` must be 1-1000 characters
- `processed_query` generated automatically from query_text
- `embedding` generated via BAAI/bge-small-en-v1.5 model
- `timestamp` set automatically on creation

### VectorEmbedding

**Purpose**: Stores vector representations of indexed content

**Fields**:

- `content_id` (string): Unique identifier for source content
- `file_path` (string): Absolute path to source file
- `content_type` (enum): SOURCE_CODE, DOCUMENTATION, HEADER, COMMENT
- `content_text` (string): Original text content
- `embedding` (vector[384]): Vector representation
- `line_start` (integer): Starting line number
- `line_end` (integer): Ending line number
- `metadata` (json): Additional context (function names, symbols, etc.)
- `created_at` (datetime): When embedding was generated
- `config_guards` (list[string]): Configuration dependencies

**Validation Rules**:

- `file_path` must be valid absolute path
- `content_type` must be valid enum value
- `embedding` must be 384-dimensional vector
- `line_start` ≤ `line_end`
- `metadata` follows structured schema

**Relationships**:

- One-to-many with SearchResult (through similarity matching)

### SearchResult

**Purpose**: Individual result returned from semantic search

**Fields**:

- `query_id` (string): Reference to originating SearchQuery
- `content_id` (string): Reference to VectorEmbedding
- `similarity_score` (float): Semantic similarity (0.0-1.0)
- `bm25_score` (float): Keyword matching score
- `combined_score` (float): Final ranking score
- `confidence` (float): Result confidence (0.0-1.0)
- `context_lines` (list[string]): Surrounding code lines
- `explanation` (string): Why this result matches the query

**Validation Rules**:

- All scores must be in range 0.0-1.0
- `combined_score` = (0.7 *similarity_score) + (0.3* bm25_score)
- `confidence` derived from distance metrics and context
- `context_lines` limited to 10 lines maximum

**Relationships**:

- Many-to-one with SearchQuery
- Many-to-one with VectorEmbedding

### IndexedContent

**Purpose**: Metadata about content available for search

**Fields**:

- `file_path` (string): Absolute path to source file
- `file_type` (enum): C_SOURCE, C_HEADER, DOCUMENTATION, MAKEFILE
- `file_size` (integer): Size in bytes
- `last_modified` (datetime): File modification timestamp
- `indexed_at` (datetime): When file was processed
- `chunk_count` (integer): Number of embedding chunks created
- `processing_status` (enum): PENDING, PROCESSING, COMPLETED, FAILED
- `error_message` (string, optional): Processing error details

**Validation Rules**:

- `file_path` must exist and be readable
- `file_size` must be positive
- `chunk_count` must be non-negative
- `error_message` required when status is FAILED

**Relationships**:

- One-to-many with VectorEmbedding

## State Transitions

### IndexedContent Processing

```
PENDING → PROCESSING → COMPLETED
              ↓
            FAILED
```

**Triggers**:

- PENDING → PROCESSING: File queued for indexing
- PROCESSING → COMPLETED: All chunks successfully embedded
- PROCESSING → FAILED: Error during processing
- FAILED → PENDING: Manual retry requested

### SearchQuery Lifecycle

```
SUBMITTED → PROCESSING → COMPLETED
              ↓
            TIMEOUT
```

**Triggers**:

- SUBMITTED: Query received via MCP endpoint
- PROCESSING: Embedding generation and similarity search
- COMPLETED: Results returned to user
- TIMEOUT: Query exceeded 600ms limit

## Indexes and Performance

### Database Indexes

**Primary Indexes**:

- `VectorEmbedding.embedding` using pgvector HNSW
- `VectorEmbedding.file_path` for path-based filtering
- `SearchResult.combined_score` for ranking
- `IndexedContent.last_modified` for incremental updates

**Composite Indexes**:

- `(content_type, config_guards)` for filtered searches
- `(file_path, line_start, line_end)` for context retrieval

### Vector Index Configuration

```sql
CREATE INDEX ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Parameters**:

- `m = 16`: Good balance of speed and recall
- `ef_construction = 64`: Build quality for accuracy
- `vector_cosine_ops`: Cosine distance for semantic similarity

## Schema Constraints

### Data Retention (per FR-012)

- **VectorEmbedding**: Retain for 1 year
- **SearchQuery**: Raw queries for 7 days, anonymized analytics for 90 days
- **SearchResult**: Linked to query retention policy
- **IndexedContent**: Retain indefinitely, purge on file deletion

### Configuration Awareness

All entities support `config_context` filtering to respect kernel build configurations:

```json
{
  "config_guards": ["CONFIG_NET", "CONFIG_INET", "!CONFIG_EMBEDDED"],
  "applies_when": "all_conditions_met"
}
```

### Audit Trail

All search operations logged with:

- Query text (anonymized after 7 days)
- Result count and quality metrics
- Response time and performance data
- User context (if authenticated)

## Error Handling

### Embedding Generation Failures

- **Cause**: Model loading, text encoding, or memory issues
- **Response**: Log error, return cached results if available
- **Recovery**: Retry with simplified query or fallback to keyword search

### Vector Search Failures

- **Cause**: Database connectivity, index corruption, or timeout
- **Response**: Return error with suggested retry time
- **Recovery**: Automatic failover to backup search methods

### Data Consistency

- **Stale Embeddings**: Detected via file modification timestamps
- **Orphaned Results**: Cleaned up via background job
- **Index Corruption**: Detected via health checks, triggers reindex
