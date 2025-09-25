# VectorStore Error Handling Guide

**Generated**: 2025-09-25
**Source Code**: `/home/dave/src/kcs/src/python/semantic_search/database/`
**Status**: Production Ready

## Overview

Comprehensive guide to error handling in the VectorStore system, covering all error types, recovery patterns, and production best practices.

## Error Types and Exceptions

### Python Exceptions Used

| Exception Type | When Used | Recovery Strategy |
|---------------|-----------|-------------------|
| `ValueError` | Invalid input parameters | Validate inputs before retry |
| `RuntimeError` | Operation failures | Check system state, retry with backoff |
| `ConnectionError` | Database connection issues | Reconnect with exponential backoff |
| `asyncpg.PostgresError` | Database-specific errors | Handle based on error code |
| `asyncpg.InvalidPasswordError` | Authentication failures | Check credentials, reload config |
| `asyncpg.UndefinedTableError` | Missing tables | Run migrations |
| `asyncpg.ForeignKeyViolationError` | Invalid references | Validate relationships |
| `asyncpg.UniqueViolationError` | Duplicate key | Return existing or update |

### Custom Error Classes

```python
class VectorStoreError(Exception):
    """Base exception for VectorStore operations."""
    pass

class DimensionMismatchError(VectorStoreError):
    """Raised when vector dimensions don't match expected size."""
    pass

class ContentNotFoundError(VectorStoreError):
    """Raised when requested content doesn't exist."""
    pass

class IndexingError(VectorStoreError):
    """Raised when indexing operations fail."""
    pass
```

## Common Error Scenarios

### 1. Connection Errors

**Scenario**: Database connection failures

```python
# Error manifestation
ConnectionError: Failed to connect to database: password authentication failed for user "kcs"

# Root causes
- Missing environment variables
- Incorrect credentials
- Database not running
- Network issues
- Connection pool exhausted
```

**Recovery Pattern**:

```python
async def connect_with_retry(max_retries=3, backoff_factor=2):
    """Connect to database with exponential backoff."""
    for attempt in range(max_retries):
        try:
            config = DatabaseConfig.from_env()
            await init_database_connection(config)
            return
        except ConnectionError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_factor ** attempt
            logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
```

### 2. Dimension Validation Errors

**Scenario**: Wrong vector dimensions

```python
# Error from store_embedding()
ValueError: Expected 384 dimensions, got 768

# Error from similarity_search()
ValueError: Expected 384 dimensions, got 512
```

**Recovery Pattern**:

```python
def validate_embedding(embedding: list[float], expected_dim: int = 384):
    """Validate embedding dimensions before storage."""
    if not embedding:
        raise ValueError("Embedding cannot be empty")
    
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Dimension mismatch: expected {expected_dim}, got {len(embedding)}. "
            f"Ensure you're using BAAI/bge-small-en-v1.5 model."
        )
    
    # Validate values are finite
    if not all(math.isfinite(x) for x in embedding):
        raise ValueError("Embedding contains non-finite values (NaN or Inf)")
```

### 3. Unique Constraint Violations

**Scenario**: Duplicate content or embeddings

```python
# Database error
asyncpg.exceptions.UniqueViolationError: duplicate key value violates unique constraint "indexed_content_source_path_key"
```

**Recovery Pattern**:

```python
async def store_content_idempotent(store: VectorStore, **kwargs):
    """Store content idempotently, returning existing if duplicate."""
    try:
        return await store.store_content(**kwargs)
    except RuntimeError as e:
        if "unique constraint" in str(e).lower():
            # Content already exists, fetch and return existing
            results = await store.list_content(
                filters=ContentFilter(
                    file_paths=[kwargs["source_path"]]
                )
            )
            if results:
                logger.info(f"Content already exists: {results[0].id}")
                return results[0].id
        raise
```

### 4. Foreign Key Violations

**Scenario**: Invalid content_id reference

```python
# When storing embedding for non-existent content
asyncpg.exceptions.ForeignKeyViolationError: insert or update on table "vector_embedding" violates foreign key constraint
```

**Recovery Pattern**:

```python
async def store_embedding_safe(store: VectorStore, content_id: int, **kwargs):
    """Store embedding with content validation."""
    # Verify content exists
    content = await store.get_content_by_id(content_id)
    if not content:
        raise ContentNotFoundError(f"Content {content_id} not found")
    
    try:
        return await store.store_embedding(content_id=content_id, **kwargs)
    except RuntimeError as e:
        if "foreign key" in str(e).lower():
            # Content was deleted between check and insert
            raise ContentNotFoundError(f"Content {content_id} was deleted")
        raise
```

### 5. Status Validation Errors

**Scenario**: Invalid status value

```python
# From update_content_status()
ValueError: Invalid status 'PROCESSING'. Must be one of: ['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED']
```

**Recovery Pattern**:

```python
class ContentStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

async def update_status_typed(store: VectorStore, content_id: int, status: ContentStatus):
    """Update status with type safety."""
    return await store.update_content_status(
        content_id=content_id,
        status=status.value
    )
```

## Error Handling Strategies

### 1. Defensive Programming

```python
# Input validation
if not content.strip():
    raise ValueError("Content cannot be empty")

if not source_path.strip():
    raise ValueError("Source path cannot be empty")

# Type checking
if not isinstance(embedding, list):
    raise TypeError(f"Expected list, got {type(embedding)}")

# Range validation
if not 0.0 <= similarity_threshold <= 1.0:
    raise ValueError(f"Similarity threshold must be between 0 and 1")
```

### 2. Try-Except Patterns

```python
# Specific exception handling
try:
    result = await store.similarity_search(query_embedding)
except ValueError as e:
    logger.error(f"Invalid query: {e}")
    return {"error": "Invalid query parameters"}
except RuntimeError as e:
    logger.error(f"Search failed: {e}")
    return {"error": "Search temporarily unavailable"}
except Exception as e:
    logger.exception("Unexpected error in similarity search")
    raise

# Context managers for cleanup
async def process_with_status(store: VectorStore, content_id: int):
    """Process content with automatic status management."""
    try:
        await store.update_content_status(content_id, "PROCESSING")
        # Processing logic here
        await store.update_content_status(content_id, "COMPLETED")
    except Exception as e:
        await store.update_content_status(
            content_id, 
            "FAILED",
            error_message=str(e)
        )
        raise
```

### 3. Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Circuit breaker for database operations."""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise RuntimeError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise
```

## Recovery Patterns

### 1. Retry with Exponential Backoff

```python
async def retry_with_backoff(
    func,
    max_retries=3,
    base_delay=1,
    max_delay=60,
    exponential_base=2
):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except (ConnectionError, RuntimeError) as e:
            if attempt == max_retries - 1:
                raise
            
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s")
            await asyncio.sleep(delay)
```

### 2. Graceful Degradation

```python
async def search_with_fallback(store: VectorStore, query: str):
    """Search with fallback to text search if vector search fails."""
    try:
        # Try vector similarity search
        embedding = await generate_embedding(query)
        return await store.similarity_search(embedding)
    except Exception as e:
        logger.warning(f"Vector search failed: {e}, falling back to text search")
        # Fallback to text pattern matching
        return await store.list_content(
            filters=ContentFilter(
                path_patterns=[f"*{query}*"]
            )
        )
```

### 3. Transaction Rollback

```python
async def atomic_multi_store(store: VectorStore, items: list):
    """Store multiple items atomically."""
    stored_ids = []
    
    try:
        for item in items:
            content_id = await store.store_content(**item["content"])
            stored_ids.append(content_id)
            
            for embedding in item["embeddings"]:
                await store.store_embedding(
                    content_id=content_id,
                    **embedding
                )
    except Exception as e:
        # Rollback: delete all stored content
        logger.error(f"Transaction failed, rolling back {len(stored_ids)} items")
        for content_id in stored_ids:
            try:
                await store.delete_content(content_id)
            except Exception:
                pass  # Best effort cleanup
        raise
```

## Logging and Monitoring

### 1. Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "content_stored",
    content_id=content_id,
    source_path=source_path,
    content_size=len(content),
    duration_ms=elapsed_ms
)

# Log errors with full context
logger.error(
    "embedding_storage_failed",
    content_id=content_id,
    dimension=len(embedding),
    error_type=type(e).__name__,
    error_message=str(e),
    traceback=traceback.format_exc()
)
```

### 2. Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
operation_errors = Counter(
    'vectorstore_errors_total',
    'Total number of VectorStore errors',
    ['operation', 'error_type']
)

operation_duration = Histogram(
    'vectorstore_operation_duration_seconds',
    'Duration of VectorStore operations',
    ['operation']
)

connection_pool_size = Gauge(
    'vectorstore_connection_pool_size',
    'Current connection pool size'
)

# Use in code
@track_metrics
async def store_content_monitored(store: VectorStore, **kwargs):
    with operation_duration.labels(operation='store_content').time():
        try:
            return await store.store_content(**kwargs)
        except Exception as e:
            operation_errors.labels(
                operation='store_content',
                error_type=type(e).__name__
            ).inc()
            raise
```

### 3. Health Checks

```python
async def health_check(store: VectorStore) -> dict:
    """Comprehensive health check for VectorStore."""
    health = {
        "status": "healthy",
        "checks": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check database connection
    try:
        await store._db.execute("SELECT 1")
        health["checks"]["database"] = "ok"
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["database"] = str(e)
    
    # Check pgvector extension
    try:
        result = await store._db.fetch_one(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        health["checks"]["pgvector"] = f"ok (v{result['extversion']})"
    except Exception as e:
        health["status"] = "degraded"
        health["checks"]["pgvector"] = str(e)
    
    # Check storage stats
    try:
        stats = await store.get_storage_stats()
        health["checks"]["storage"] = {
            "total_content": stats["total_content"],
            "total_embeddings": stats["total_embeddings"]
        }
    except Exception as e:
        health["status"] = "degraded"
        health["checks"]["storage"] = str(e)
    
    return health
```

## Debugging Techniques

### 1. Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('asyncpg').setLevel(logging.DEBUG)

# Log SQL queries
class DebugConnection:
    def __init__(self, pool):
        self.pool = pool
    
    async def fetch_all(self, query, *args):
        logger.debug(f"SQL: {query}")
        logger.debug(f"Args: {args}")
        start = time.time()
        try:
            result = await self.pool.fetch(query, *args)
            logger.debug(f"Result: {len(result)} rows in {time.time() - start:.3f}s")
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
```

### 2. Connection Diagnostics

```python
async def diagnose_connection():
    """Diagnose connection issues."""
    print("Checking environment variables...")
    for var in ['POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER']:
        value = os.getenv(var)
        if value:
            print(f"  {var}: {value}")
        else:
            print(f"  {var}: NOT SET")
    
    print("\nTesting connection...")
    try:
        config = DatabaseConfig.from_env()
        print(f"  URL: {config.to_url()}")
        
        conn = await asyncpg.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password
        )
        print("  Connection: SUCCESS")
        
        version = await conn.fetchval("SELECT version()")
        print(f"  PostgreSQL: {version.split(',')[0]}")
        
        pgvector = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        print(f"  pgvector: v{pgvector}")
        
        await conn.close()
    except Exception as e:
        print(f"  Connection: FAILED - {e}")
```

### 3. Query Performance Analysis

```python
async def analyze_query_performance(store: VectorStore, query_embedding: list[float]):
    """Analyze similarity search performance."""
    # Explain query plan
    query = """
    EXPLAIN ANALYZE
    SELECT ic.id, (1 - (ve.embedding <=> $1::vector)) as similarity
    FROM vector_embedding ve
    JOIN indexed_content ic ON ve.content_id = ic.id
    WHERE ve.embedding IS NOT NULL
    ORDER BY ve.embedding <=> $1::vector
    LIMIT 10
    """
    
    plan = await store._db.fetch_all(query, str(query_embedding))
    for row in plan:
        print(row['QUERY PLAN'])
```

## Production Error Handling

### 1. Error Response Format

```python
class ErrorResponse:
    """Standardized error response."""
    
    def __init__(self, error: Exception, request_id: str):
        self.error_code = self._get_error_code(error)
        self.message = self._get_user_message(error)
        self.details = self._get_details(error)
        self.request_id = request_id
        self.timestamp = datetime.utcnow().isoformat()
    
    def _get_error_code(self, error):
        error_codes = {
            ValueError: "INVALID_INPUT",
            ConnectionError: "CONNECTION_ERROR",
            ContentNotFoundError: "NOT_FOUND",
            DimensionMismatchError: "DIMENSION_ERROR",
        }
        return error_codes.get(type(error), "INTERNAL_ERROR")
    
    def _get_user_message(self, error):
        if isinstance(error, ValueError):
            return str(error)
        elif isinstance(error, ConnectionError):
            return "Service temporarily unavailable"
        else:
            return "An error occurred processing your request"
    
    def to_dict(self):
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
                "request_id": self.request_id,
                "timestamp": self.timestamp
            }
        }
```

### 2. Global Exception Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(VectorStoreError)
async def vectorstore_exception_handler(request: Request, exc: VectorStoreError):
    error_response = ErrorResponse(exc, request.state.request_id)
    
    # Log error with context
    logger.error(
        "vectorstore_error",
        error_code=error_response.error_code,
        request_id=request.state.request_id,
        path=request.url.path,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=400 if isinstance(exc, ValueError) else 500,
        content=error_response.to_dict()
    )
```

### 3. Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: vectorstore
    rules:
      - alert: HighErrorRate
        expr: rate(vectorstore_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate in VectorStore"
          description: "Error rate is {{ $value }} errors/sec"
      
      - alert: ConnectionPoolExhausted
        expr: vectorstore_connection_pool_size >= 20
        for: 1m
        annotations:
          summary: "Connection pool near capacity"
          description: "Pool size is {{ $value }}/20"
      
      - alert: SlowQueries
        expr: histogram_quantile(0.99, vectorstore_operation_duration_seconds) > 1
        for: 5m
        annotations:
          summary: "Slow VectorStore operations"
          description: "P99 latency is {{ $value }}s"
```

## Best Practices

### 1. Input Validation

- Always validate inputs at the boundary
- Use type hints and runtime validation
- Provide clear error messages
- Sanitize user input

### 2. Error Messages

- Be specific but don't leak sensitive information
- Include actionable information
- Use consistent format
- Log detailed errors, return sanitized messages

### 3. Recovery Strategies

- Implement retry logic for transient failures
- Use circuit breakers for cascading failures
- Provide fallback mechanisms
- Clean up resources on failure

### 4. Monitoring

- Track error rates by type and operation
- Monitor latency percentiles
- Set up alerting thresholds
- Keep audit logs

### 5. Testing Error Paths

```python
@pytest.mark.asyncio
async def test_dimension_validation():
    """Test dimension validation error handling."""
    store = VectorStore()
    
    # Test wrong dimensions
    with pytest.raises(ValueError, match="Expected 384 dimensions"):
        await store.store_embedding(
            content_id=1,
            embedding=[0.1] * 768,  # Wrong size
            chunk_text="test"
        )
    
    # Test empty embedding
    with pytest.raises(ValueError, match="cannot be empty"):
        await store.store_embedding(
            content_id=1,
            embedding=[],
            chunk_text="test"
        )

@pytest.mark.asyncio
async def test_connection_retry():
    """Test connection retry logic."""
    with patch('asyncpg.create_pool') as mock_pool:
        mock_pool.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"),
            MagicMock()  # Success on third try
        ]
        
        await connect_with_retry(max_retries=3)
        assert mock_pool.call_count == 3
```

## Conclusion

Proper error handling is crucial for a production VectorStore system. This guide provides:

1. **Comprehensive error taxonomy** - All error types and their handling
2. **Recovery patterns** - Proven strategies for resilience
3. **Production practices** - Logging, monitoring, and alerting
4. **Debugging tools** - Techniques for troubleshooting
5. **Code examples** - Ready-to-use implementations

Following these patterns ensures a robust, maintainable, and observable VectorStore system.

---

*Error handling guide completed as part of T023 - Document error handling patterns*
