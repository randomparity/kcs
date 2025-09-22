# Quickstart: Semantic Search Engine

## Prerequisites

- Python 3.11+ with pip
- PostgreSQL 14+ with pgvector extension
- Git access to kernel source repository
- 4GB+ RAM for model and vector storage

## Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install sentence-transformers==2.7.0 pgvector==0.3.0 psycopg2-binary==2.9.9

# Install PostgreSQL with pgvector (Ubuntu/Debian)
sudo apt install postgresql-14 postgresql-14-pgvector

# Or build pgvector from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```

### 2. Database Setup

```bash
# Create database and enable pgvector
sudo -u postgres psql
CREATE DATABASE kcs_search;
\c kcs_search
CREATE EXTENSION vector;
\q
```

### 3. Configure Environment

```bash
# Set database connection
export DATABASE_URL="postgresql://kcs:password@localhost/kcs_search"

# Download embedding model (one-time setup)
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Model downloaded successfully')
"
```

## Basic Usage

### 1. Index Content

```bash
# Index kernel source files
kcs-search index --path /path/to/linux/kernel \
  --include "*.c,*.h" \
  --exclude "*.o,*.so,*.a" \
  --chunk-size 500

# Check indexing progress
kcs-search status
```

### 2. Search Examples

```bash
# Basic semantic search
kcs-search query "memory allocation functions that can fail"

# Search with confidence threshold
kcs-search query "buffer overflow in network drivers" --min-confidence 0.7

# Filter by file patterns
kcs-search query "mutex locking" --files "kernel/locking/*,fs/*"

# Configuration-aware search
kcs-search query "DMA mapping" --config "CONFIG_DMA_API"
```

### 3. MCP Integration

```python
# Example MCP client usage
import mcp_client

client = mcp_client.Client("http://localhost:8000")

# Semantic search via MCP
result = client.call_tool("semantic_search", {
    "query": "race conditions in file operations",
    "max_results": 5,
    "content_types": ["SOURCE_CODE", "HEADER"],
    "config_context": ["CONFIG_SMP"]
})

print(f"Found {len(result['results'])} matches")
for match in result['results']:
    print(f"{match['file_path']}:{match['line_start']} (confidence: {match['confidence']:.2f})")
```

## Validation Tests

### Test 1: Basic Search Functionality

**Objective**: Verify semantic search returns relevant results

```bash
# Expected: Returns kmalloc, vmalloc, kzalloc functions
kcs-search query "allocate memory that might fail" --max-results 3
```

**Success Criteria**:

- At least 3 results returned
- All results have confidence > 0.5
- Results include memory allocation functions
- Response time < 600ms

### Test 2: Configuration Awareness

**Objective**: Verify configuration filtering works

```bash
# Search for network-specific code
kcs-search query "socket operations" --config "CONFIG_NET"

# Exclude embedded-specific code
kcs-search query "power management" --config "!CONFIG_EMBEDDED"
```

**Success Criteria**:

- Results respect configuration filters
- No results from disabled configuration paths
- Metadata includes config_guards information

### Test 3: MCP Endpoint Integration

**Objective**: Verify MCP tools work correctly

```python
# Test MCP semantic_search tool
result = client.call_tool("semantic_search", {
    "query": "filesystem read operations",
    "max_results": 5
})

assert result["search_stats"]["search_time_ms"] < 600
assert len(result["results"]) > 0
assert all(r["confidence"] >= 0.5 for r in result["results"])
```

**Success Criteria**:

- MCP tool returns valid JSON response
- Response includes required fields per contract
- Error handling works for invalid inputs

### Test 4: Performance Validation

**Objective**: Verify system meets performance requirements

```bash
# Performance test script
for i in {1..10}; do
  time kcs-search query "kernel synchronization primitives" >/dev/null
done
```

**Success Criteria**:

- 95th percentile query time ≤ 600ms
- System handles 10 concurrent queries
- Memory usage stays under 500MB per process

### Test 5: Content Indexing

**Objective**: Verify content can be indexed correctly

```bash
# Index a small test directory
kcs-search index --path ./test-kernel-src --force-reindex

# Verify indexing results
kcs-search status --pattern "./test-kernel-src/*"
```

**Success Criteria**:

- All .c and .h files successfully indexed
- Chunk count matches expected file content
- No indexing errors for valid files
- Index status shows COMPLETED

## Expected Results

### Query: "memory allocation functions that can fail"

```
Results (5 matches found):
1. mm/slab.c:3421-3425 (confidence: 0.89)
   kmalloc - allocate memory with GFP flags

2. mm/vmalloc.c:2567-2572 (confidence: 0.85)
   vmalloc - allocate virtually contiguous memory

3. mm/page_alloc.c:5123-5128 (confidence: 0.82)
   alloc_pages - allocate pages with specified order
```

### Query: "buffer overflow vulnerabilities in network drivers"

```
Results (3 matches found):
1. drivers/net/ethernet/realtek/r8169.c:1234-1240 (confidence: 0.76)
   Buffer bounds checking in packet processing

2. drivers/net/wireless/ath/ath9k/recv.c:567-573 (confidence: 0.72)
   SKB buffer length validation
```

## Troubleshooting

### Common Issues

**"Model not found" error**:

```bash
# Re-download the embedding model
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='/tmp/models')
"
```

**"pgvector extension not found"**:

```bash
# Ensure pgvector is installed and enabled
sudo -u postgres psql -d kcs_search -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Search timeout errors**:

- Check database connection pool settings
- Verify vector index is properly built
- Monitor system memory usage

**Low confidence scores**:

- Try more specific queries
- Check if content is properly indexed
- Verify query preprocessing is working

### Performance Tuning

**Vector Index Optimization**:

```sql
-- Adjust HNSW parameters for your dataset size
CREATE INDEX CONCURRENTLY vec_idx ON embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);
```

**Query Caching**:

```bash
# Enable Redis for query result caching
export REDIS_URL="redis://localhost:6379/0"
kcs-search config set cache.enabled true
```

## Integration Examples

### With IDE/Editor

```python
# VS Code extension integration
def semantic_search_handler(query):
    result = mcp_client.call_tool("semantic_search", {
        "query": query,
        "max_results": 10,
        "file_patterns": [workspace.root_path + "/*"]
    })
    return format_results_for_editor(result)
```

### With Analysis Pipeline

```bash
# Automated vulnerability scanning
kcs-search query "unchecked user input" --format json | \
  jq '.results[] | select(.confidence > 0.8)' | \
  vulnerability-analyzer --stdin
```

### With Documentation

```python
# Generate documentation from semantic clusters
clusters = semantic_clustering_tool.cluster_by_functionality(
    mcp_client.call_tool("semantic_search", {
        "query": "file system operations",
        "max_results": 100
    })
)
```

This quickstart validates all major user scenarios from the feature specification and ensures the semantic search engine meets both functional requirements and constitutional performance standards.
