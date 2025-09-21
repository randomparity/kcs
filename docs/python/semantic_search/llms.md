# Semantic Search Engine (kcs-search) - llms.txt Documentation

## Overview

The semantic search engine provides fast, accurate search capabilities for kernel code and
documentation using hybrid BM25+semantic similarity scoring. It indexes source files,
documentation, and configuration data to enable natural language queries that understand both
keyword matches and semantic meaning.

## Architecture

### Core Components

1. **Query Processing Pipeline**
   - QueryPreprocessor: Multi-stage text normalization with code-aware tokenization
   - EmbeddingService: Generates 384-dimensional vectors using BAAI/bge-small-en-v1.5
   - VectorSearchService: Performs similarity search using pgvector operations

2. **Hybrid Ranking System**
   - BM25Calculator: Keyword-based scoring with TF-IDF weighting
   - RankingService: Combines semantic similarity (70%) + BM25 scores (30%)
   - Confidence scoring based on score agreement and context quality

3. **Vector Storage Layer**
   - VectorStore: PostgreSQL + pgvector for efficient similarity search
   - IndexedContent: Tracks content metadata, status, and change detection
   - VectorEmbedding: Stores normalized 384-dimensional embeddings

4. **MCP Interface**
   - semantic_search: Query content with natural language
   - index_content: Add/update content in the search index
   - get_index_status: Monitor indexing progress and statistics

### Data Models

#### SearchQuery

Represents user search queries with preprocessing and embedding.

```python
class SearchQuery:
    query_text: str          # Original user query
    processed_text: str      # Preprocessed query text
    embedding: list[float]   # 384-dimensional query vector
    max_results: int         # Result limit (default: 20)
    similarity_threshold: float  # Minimum similarity (default: 0.0)
```

#### VectorEmbedding

Stores embeddings for indexed content chunks.

```python
class VectorEmbedding:
    content_id: str         # Reference to indexed content
    embedding: list[float]  # 384-dimensional vector
    model_name: str         # "BAAI/bge-small-en-v1.5"
    model_version: str      # Model version
    chunk_index: int        # For split content (default: 0)
```

#### SearchResult

Contains ranked search results with explanations.

```python
class SearchResult:
    query_id: str           # Search query identifier
    content_id: str         # Matched content ID
    similarity_score: float # Semantic similarity (0.0-1.0)
    bm25_score: float      # Keyword matching score (0.0-1.0)
    combined_score: float   # Weighted hybrid score
    confidence: float       # Result confidence (0.0-1.0)
    context_lines: list[str] # Relevant content excerpts
    explanation: str        # Human-readable match explanation
```

#### IndexedContent

Tracks content metadata and indexing status.

```python
class IndexedContent:
    content_id: str         # Unique content identifier
    content_type: str       # source_file, documentation, config
    source_path: str        # Original file path
    content_hash: str       # SHA-256 for change detection
    title: str             # Display title
    content: str           # Full text content
    metadata: dict         # Additional attributes
    status: str            # pending, indexing, completed, failed, stale
    indexed_at: datetime   # Last successful indexing
```

## Query Processing

### Multi-Stage Preprocessing Pipeline

The QueryPreprocessor implements a sophisticated pipeline that preserves code semantics while
enhancing natural language understanding:

1. **Code Identifier Detection**
   - Detects snake_case, CamelCase, CONSTANT_CASE patterns
   - Preserves function calls, hex values, version numbers
   - Maintains kernel configuration options (CONFIG_*)

2. **Technical Abbreviation Expansion**
   - Expands domain-specific abbreviations (mem → memory, alloc → allocation)
   - Includes both original and expanded forms for better recall
   - Covers kernel, networking, and hardware terminology

3. **Case Normalization with Semantic Preservation**
   - Lowercases general text while preserving meaningful code patterns
   - Maintains CamelCase and CONSTANT_CASE identifiers
   - Detects emphasis vs. actual code constants

4. **Domain-Specific Synonym Enrichment**
   - Adds relevant synonyms for kernel concepts (lock → mutex, semaphore)
   - Includes security terminology (vulnerability → exploit, CVE)
   - Expands performance and hardware terms

### Example Query Transformations

```
Input: "memory allocation ERROR in CONFIG_NUMA"
Code Detection: True (CONFIG_NUMA detected)
Preserved: CONFIG_NUMA
Expanded: "memory allocation allocation allocate ERROR bug fault in CONFIG_NUMA"
Final: "memory heap stack allocation allocation allocate error bug fault in CONFIG_NUMA"

Input: "buffer overflow security issue"
Code Detection: False
Expanded: "buffer overflow security vulnerability exploit cve issue"
Final: "buffer overflow security vulnerability exploit cve issue"
```

## Embedding Model

### BAAI/bge-small-en-v1.5 Configuration

- **Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **Device**: CPU-only for constitutional compliance
- **Normalization**: L2 normalized vectors for cosine similarity
- **Thread Safety**: Cached model instances with lock-based loading
- **Performance**: Optimized for batch processing with progress disabled

### Model Characteristics

- Specialized for semantic similarity tasks
- Balanced performance vs. resource usage
- Strong performance on technical documentation
- Efficient 384-dimensional representation
- Fast inference suitable for real-time search

## Hybrid Ranking Algorithm

### Score Combination

The RankingService combines two complementary scoring methods:

1. **Semantic Similarity (70% weight)**
   - Cosine similarity between query and content embeddings
   - Captures conceptual relationships and synonyms
   - Handles paraphrased queries and conceptual matches

2. **BM25 Keyword Scoring (30% weight)**
   - Okapi BM25 with k1=1.2, b=0.75 parameters
   - Term frequency with saturation and length normalization
   - IDF weighting for rare term importance
   - Ensures exact keyword matches are highly ranked

### Confidence Calculation

Confidence scores incorporate multiple factors:

```python
confidence = combined_score + score_agreement_boost + context_quality - variance_penalty
```

- **Score Agreement**: Boost when both semantic and BM25 scores are high
- **Context Quality**: Factor based on surrounding content relevance
- **Variance Penalty**: Reduced confidence for inconsistent results

### Result Ranking

Results are sorted by combined_score with additional context extraction:

1. **Context Line Extraction**: Identifies most relevant lines containing query terms
2. **Explanation Generation**: Human-readable match reasoning
3. **Score Transparency**: All component scores are preserved and exposed

## Vector Storage

### PostgreSQL + pgvector Integration

The VectorStore uses PostgreSQL with the pgvector extension for efficient similarity search:

```sql
-- Core tables
CREATE TABLE indexed_content (
    id SERIAL PRIMARY KEY,
    content_type VARCHAR(50),
    source_path TEXT UNIQUE,
    content_hash VARCHAR(64),
    title TEXT,
    content TEXT,
    metadata JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    indexed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE vector_embedding (
    id SERIAL PRIMARY KEY,
    content_id INTEGER REFERENCES indexed_content(id),
    embedding vector(384),
    model_name VARCHAR(100),
    model_version VARCHAR(20),
    chunk_index INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Optimized indexes
CREATE INDEX idx_vector_embedding_content_id ON vector_embedding(content_id);
CREATE INDEX idx_vector_embedding_cosine ON vector_embedding USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_indexed_content_status ON indexed_content(status);
CREATE INDEX idx_indexed_content_type ON indexed_content(content_type);
```

### Similarity Search Optimization

- **Distance Function**: Cosine distance (1 - cosine_similarity)
- **Index Type**: IVFFlat for approximate nearest neighbor search
- **Query Optimization**: Dynamic WHERE clause construction with parameter binding
- **Result Filtering**: Content type, file path, and similarity threshold filters

## MCP Tool Interface

### semantic_search Tool

Performs natural language search across indexed content.

**Input Schema:**

```json
{
  "query": "string (required) - Natural language search query",
  "max_results": "integer (optional, default: 20) - Maximum results to return",
  "similarity_threshold": "number (optional, default: 0.0) - Minimum similarity score",
  "content_types": "array (optional) - Filter by content types",
  "file_paths": "array (optional) - Filter by specific file paths"
}
```

**Output Schema:**

```json
{
  "results": [
    {
      "content_id": "string - Unique content identifier",
      "title": "string - Content title or filename",
      "source_path": "string - Original file path",
      "content_type": "string - Type of content",
      "similarity_score": "number - Semantic similarity (0.0-1.0)",
      "bm25_score": "number - Keyword matching score (0.0-1.0)",
      "combined_score": "number - Final ranking score (0.0-1.0)",
      "confidence": "number - Result confidence (0.0-1.0)",
      "context_lines": ["string"] - Relevant content excerpts",
      "explanation": "string - Human-readable match explanation"
    }
  ],
  "query_stats": {
    "total_results": "integer - Total matches found",
    "processing_time": "number - Query processing time in seconds",
    "query_embedding_time": "number - Time to generate query embedding",
    "search_time": "number - Vector search time",
    "ranking_time": "number - Result ranking time"
  }
}
```

### index_content Tool

Adds or updates content in the search index.

**Input Schema:**

```json
{
  "content_type": "string (required) - Type of content (source_file, documentation, config)",
  "source_path": "string (required) - Path to source file",
  "content": "string (required) - Full text content to index",
  "title": "string (optional) - Display title for content",
  "metadata": "object (optional) - Additional content metadata",
  "force_reindex": "boolean (optional, default: false) - Force reindexing even if unchanged"
}
```

**Output Schema:**

```json
{
  "content_id": "string - Assigned content identifier",
  "status": "string - Indexing status (pending, completed, failed)",
  "embedding_id": "string - Generated embedding identifier",
  "previous_content_id": "string - Previous version if updated",
  "indexing_stats": {
    "content_hash": "string - SHA-256 content hash",
    "content_length": "integer - Character count",
    "embedding_time": "number - Time to generate embedding",
    "storage_time": "number - Time to store in database"
  }
}
```

### get_index_status Tool

Retrieves indexing status and system statistics.

**Input Schema:**

```json
{
  "content_id": "string (optional) - Specific content to check",
  "include_stats": "boolean (optional, default: true) - Include system statistics"
}
```

**Output Schema:**

```json
{
  "content_status": {
    "content_id": "string - Content identifier",
    "status": "string - Current status",
    "indexed_at": "string - ISO timestamp of last indexing",
    "error_message": "string - Error details if failed"
  },
  "system_stats": {
    "total_content": "integer - Total indexed content items",
    "total_embeddings": "integer - Total stored embeddings",
    "content_by_type": {
      "source_file": "integer",
      "documentation": "integer",
      "config": "integer"
    },
    "status_breakdown": {
      "completed": "integer",
      "pending": "integer",
      "failed": "integer",
      "stale": "integer"
    },
    "embedding_models": [
      {
        "model": "string - Model name and version",
        "count": "integer - Number of embeddings"
      }
    ]
  }
}
```

## Performance Characteristics

### Query Response Times

- **Target p95 latency**: ≤ 600ms for typical queries
- **Query preprocessing**: 5-15ms for complex code-aware queries
- **Embedding generation**: 50-150ms depending on query length
- **Vector search**: 10-100ms depending on corpus size and filters
- **Result ranking**: 5-20ms for BM25 calculation and context extraction

### Scalability Metrics

- **Corpus size**: Optimized for 100K-1M documents
- **Concurrent queries**: Supports 10-50 concurrent users
- **Index update rate**: 100-500 documents per minute
- **Memory usage**: ~500MB baseline + ~1KB per indexed document
- **Storage**: ~2KB per document (content + embedding + metadata)

### Optimization Strategies

1. **Query Result Caching**: LRU cache for frequent queries
2. **Vector Index Tuning**: IVFFlat parameters optimized for corpus size
3. **Connection Pooling**: Database connection management for concurrency
4. **Batch Processing**: Efficient bulk indexing operations
5. **Incremental Updates**: Change detection via content hashing

## Constitutional Requirements

### Read-Only Operation

The semantic search system operates as a read-only interface:

- **No file modification**: Only reads and indexes existing files
- **No code execution**: Static analysis and text processing only
- **No system changes**: Cannot modify configuration or system state
- **Audit trail**: All operations are logged for transparency

### Citation and Attribution

All search results include proper attribution:

- **Source path**: Original file location for every result
- **Content excerpts**: Relevant lines with preserved formatting
- **Metadata preservation**: Author, timestamp, and version information
- **Link generation**: Direct references to original content

### Privacy and Security

- **No credential storage**: No authentication data in index
- **Local processing**: All embeddings generated locally
- **Content isolation**: Proper access control inheritance
- **Secure queries**: Input validation and sanitization

## Usage Examples

### Basic Semantic Search

```bash
# Natural language query
kcs-search query "how to allocate memory safely"

# Returns ranked results with explanations:
# 1. kmalloc.c:142 - Safe memory allocation patterns (0.89 confidence)
# 2. memory.md:67 - Memory safety guidelines (0.82 confidence)
# 3. alloc.h:23 - Allocation function declarations (0.74 confidence)
```

### Code-Aware Search

```bash
# Preserves code identifiers
kcs-search query "CONFIG_MEMORY_HOTPLUG initialization"

# Finds exact matches plus semantic alternatives:
# 1. config.c:89 - CONFIG_MEMORY_HOTPLUG setup code
# 2. hotplug.c:156 - Memory hotplug initialization sequence
# 3. docs/memory.md:234 - Memory hotplug configuration guide
```

### Filtered Search

```bash
# Search within specific content types
kcs-search query "network protocol implementation" --content-types source_file

# File path pattern matching
kcs-search query "buffer overflow" --paths "drivers/net/*"

# Similarity threshold filtering
kcs-search query "concurrency issues" --min-similarity 0.7
```

### Content Indexing

```bash
# Index a new source file
kcs-search index --type source_file --path "drivers/new_driver.c" --title "New Network Driver"

# Update documentation
kcs-search index --type documentation --path "docs/api.md" --force-reindex

# Bulk indexing with metadata
kcs-search index-directory --path "drivers/" --type source_file --recursive
```

### Status Monitoring

```bash
# Overall system status
kcs-search status

# Specific content status
kcs-search status --content-id "drivers/ethernet.c"

# Indexing progress
kcs-search status --include-progress --format json
```

## Integration Points

### KCS Server Integration

The semantic search engine integrates with the existing KCS architecture:

1. **Startup Integration**: Automatic initialization with KCS server
2. **Logging Integration**: Uses KCS logging infrastructure and formats
3. **Configuration Management**: Shares KCS configuration system
4. **Database Integration**: Uses existing PostgreSQL connection pool
5. **MCP Registration**: Automatic tool registration with KCS MCP server

### Background Processing

Content indexing operates as background workers:

1. **File Change Detection**: Monitors file system changes via inotify
2. **Incremental Updates**: Only reindexes changed content
3. **Batch Processing**: Efficient bulk operations for large updates
4. **Error Recovery**: Automatic retry with exponential backoff
5. **Resource Throttling**: CPU and memory usage limits

### API Integration

The semantic search can be accessed through multiple interfaces:

1. **MCP Tools**: Primary interface for LLM interactions
2. **CLI Commands**: Direct command-line access for administration
3. **REST API**: HTTP interface for web applications
4. **Python API**: Direct library access for custom integrations
5. **WebSocket**: Real-time search for interactive applications

## Troubleshooting

### Common Issues

1. **Slow Query Performance**
   - Check vector index statistics: `ANALYZE vector_embedding;`
   - Verify index usage: `EXPLAIN ANALYZE SELECT ... ORDER BY embedding <=> $1`
   - Consider index rebuilding: `REINDEX INDEX idx_vector_embedding_cosine;`

2. **Embedding Generation Failures**
   - Check model availability: `kcs-search status --check-model`
   - Verify disk space: Model cache requires ~500MB
   - Check memory usage: Large texts may exceed limits

3. **Indexing Errors**
   - Review content validation: Empty or binary files may be rejected
   - Check database connections: Connection pool exhaustion
   - Verify permissions: Read access to source files required

4. **Poor Search Results**
   - Adjust ranking weights: Increase BM25 weight for keyword-heavy queries
   - Review query preprocessing: Check for over-expansion or term loss
   - Validate content quality: Ensure indexed content is relevant and complete

### Diagnostic Commands

```bash
# Check system health
kcs-search diagnose --full

# Performance profiling
kcs-search profile --query "test query" --iterations 100

# Index integrity check
kcs-search verify-index --check-embeddings --check-content

# Database optimization
kcs-search optimize --vacuum --reindex --analyze
```

### Monitoring and Metrics

Key metrics to monitor for production deployment:

1. **Query Latency**: p50, p95, p99 response times
2. **Throughput**: Queries per second, concurrent users
3. **Index Health**: Completion rate, error rate, staleness
4. **Resource Usage**: CPU, memory, disk space, database connections
5. **Quality Metrics**: User satisfaction, result relevance scores

## Development and Extension

### Adding New Content Types

To support new content types (e.g., configuration files, test data):

1. **Define Content Type**: Add to `ContentType` enum
2. **Update Validation**: Extend content validation logic
3. **Customize Preprocessing**: Add type-specific text processing
4. **Test Integration**: Ensure proper indexing and search behavior

### Extending Query Processing

The query preprocessing pipeline can be extended:

1. **Custom Tokenizers**: Add domain-specific tokenization rules
2. **Additional Abbreviations**: Expand the abbreviation dictionary
3. **New Synonyms**: Include project-specific terminology
4. **Language Support**: Add multi-language processing capabilities

### Model Upgrades

To upgrade the embedding model:

1. **Model Compatibility**: Ensure consistent vector dimensions
2. **Migration Strategy**: Plan for reindexing existing content
3. **Performance Testing**: Validate latency and quality metrics
4. **Rollback Plan**: Maintain ability to revert to previous model

This documentation provides comprehensive guidance for understanding, deploying, and maintaining
the semantic search engine within the KCS ecosystem.
