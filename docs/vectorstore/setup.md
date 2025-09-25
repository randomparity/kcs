# VectorStore Setup and Connection Guide

This guide provides comprehensive documentation for setting up and configuring the VectorStore system, including database connections, environment configuration, and deployment scenarios.

## Table of Contents

1. [Overview](#overview)
2. [Environment Configuration](#environment-configuration)
3. [Database Connection Setup](#database-connection-setup)
4. [Connection Pool Configuration](#connection-pool-configuration)
5. [pgvector Extension Requirements](#pgvector-extension-requirements)
6. [Docker Container Setup](#docker-container-setup)
7. [Error Handling](#error-handling)
8. [Production Deployment Best Practices](#production-deployment-best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The KCS VectorStore system provides semantic search capabilities through PostgreSQL with the pgvector extension. It manages vector embeddings for content indexing and similarity search operations with high-performance connection pooling and robust error handling.

### Key Components

- **DatabaseConnection**: Connection pool manager with transaction support
- **VectorStore**: High-level interface for vector operations
- **DatabaseConfig**: Configuration management with environment variable support

## Environment Configuration

### Using python-dotenv

The system automatically loads environment variables from a `.env` file using python-dotenv:

```python
from dotenv import load_dotenv

# Automatically loaded at module import
load_dotenv()
```

### Required Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# PostgreSQL Database Settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=kcs
POSTGRES_USER=kcs
POSTGRES_PASSWORD=your_secure_password

# Optional: Full database URL (overrides individual settings)
DATABASE_URL=postgresql://kcs:password@localhost:5432/kcs
```

### Configuration Examples

#### Development Environment

```bash
# .env.development
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=kcs_dev
POSTGRES_USER=kcs_dev
POSTGRES_PASSWORD=dev_password
```

#### Production Environment

```bash
# .env.production
POSTGRES_HOST=prod-db.example.com
POSTGRES_PORT=5432
POSTGRES_DB=kcs
POSTGRES_USER=kcs_app
POSTGRES_PASSWORD=secure_production_password
```

#### Docker Environment

```bash
# .env.docker
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=kcs
POSTGRES_USER=kcs
POSTGRES_PASSWORD=kcs_dev_password
```

## Database Connection Setup

### Basic Connection Initialization

```python
from semantic_search.database.connection import (
    DatabaseConfig,
    init_database_connection,
    get_database_connection
)

# Method 1: From environment variables
config = DatabaseConfig.from_env()
db = await init_database_connection(config)

# Method 2: From database URL
db = await init_database_connection(
    database_url="postgresql://user:pass@host:port/database"
)

# Method 3: Manual configuration
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="kcs",
    username="kcs",
    password="password"
)
db = await init_database_connection(config)
```

### Connection Usage Patterns

#### Basic Query Execution

```python
# Get the global connection instance
db = get_database_connection()

# Execute queries
result = await db.fetch_one("SELECT * FROM indexed_content WHERE id = $1", 1)
results = await db.fetch_all("SELECT * FROM indexed_content LIMIT 10")
value = await db.fetch_val("SELECT COUNT(*) FROM indexed_content")
await db.execute("UPDATE indexed_content SET status = $1 WHERE id = $2", "completed", 1)
```

#### Using Connection Context Manager

```python
async with db.acquire() as conn:
    result = await conn.fetchrow("SELECT * FROM indexed_content WHERE id = $1", 1)
    await conn.execute("UPDATE indexed_content SET updated_at = NOW() WHERE id = $1", 1)
```

#### Transaction Support

```python
async with db.transaction() as conn:
    # All operations within this block are atomic
    content_id = await conn.fetchval(
        "INSERT INTO indexed_content (...) VALUES (...) RETURNING id"
    )
    await conn.execute(
        "INSERT INTO vector_embedding (content_id, embedding) VALUES ($1, $2)",
        content_id, embedding_vector
    )
    # Transaction automatically commits on success or rolls back on exception
```

## Connection Pool Configuration

### Default Pool Settings

```python
class DatabaseConfig(BaseModel):
    min_pool_size: int = Field(default=2, ge=1)
    max_pool_size: int = Field(default=10, ge=1)
    command_timeout: int = Field(default=30, ge=1)
```

### Custom Pool Configuration

```python
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="kcs",
    username="kcs",
    password="password",
    min_pool_size=5,      # Minimum connections kept open
    max_pool_size=20,     # Maximum concurrent connections
    command_timeout=60    # Query timeout in seconds
)

db = await init_database_connection(config)
```

### Production Pool Settings

For production environments, consider these settings based on your load:

```python
# High-load production settings
config = DatabaseConfig(
    min_pool_size=10,
    max_pool_size=50,
    command_timeout=30
)

# Memory-constrained environments
config = DatabaseConfig(
    min_pool_size=2,
    max_pool_size=10,
    command_timeout=60
)
```

### Monitoring Pool Health

```python
# Get pool statistics
stats = await db.get_pool_stats()
print(f"Pool size: {stats['size']}/{stats['max_size']}")
print(f"Idle connections: {stats['idle_size']}")

# Health check
health = await db.health_check()
if health['healthy']:
    print(f"Database healthy, response time: {health['response_time_ms']}ms")
else:
    print(f"Database unhealthy: {health['error']}")
```

## pgvector Extension Requirements

### Installation

The pgvector extension must be installed in your PostgreSQL database:

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### Docker Setup with pgvector

Use the official pgvector Docker image:

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: kcs
      POSTGRES_USER: kcs
      POSTGRES_PASSWORD: password
```

### Verification

The connection manager automatically verifies pgvector availability:

```python
async with db.acquire() as conn:
    has_vector = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
    )
    if not has_vector:
        print("WARNING: pgvector extension not found")
```

### Required Schema Tables

The system expects these tables to exist:

- `indexed_content` - Content storage
- `vector_embedding` - Vector embeddings with pgvector support
- `search_query` - Query history
- `search_result` - Search results

## Docker Container Setup

### Complete Docker Compose Configuration

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-kcs}
      POSTGRES_USER: ${POSTGRES_USER:-kcs}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-kcs_dev_password}
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "${POSTGRES_EXTERNAL_PORT:-5432}:5432"
    volumes:
      - ${POSTGRES_DATA_DIR:-./data/postgres}:/var/lib/postgresql/data
      - ${POSTGRES_LOG_DIR:-./data/logs/postgres}:/var/log/postgresql
      - ./src/sql/migrations:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-kcs} -d ${POSTGRES_DB:-kcs}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - kcs-network
    deploy:
      resources:
        limits:
          memory: ${POSTGRES_MEMORY_LIMIT:-2g}
          cpus: "${POSTGRES_CPU_LIMIT:-2.0}"

networks:
  kcs-network:
    driver: bridge
```

### Local Development Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd kcs

# 2. Copy environment configuration
cp .env.example .env

# 3. Start PostgreSQL with pgvector
docker compose up postgres -d

# 4. Wait for database to be ready
docker compose logs -f postgres

# 5. Run database migrations
python -m alembic upgrade head

# 6. Verify setup
python -c "
from semantic_search.database.connection import init_database_connection
import asyncio

async def test():
    db = await init_database_connection()
    health = await db.health_check()
    print('Database health:', health)

asyncio.run(test())
"
```

### Production Docker Setup

```bash
# 1. Set production environment
export ENVIRONMENT=production

# 2. Configure secure passwords
export POSTGRES_PASSWORD=$(openssl rand -base64 32)

# 3. Start with resource limits
docker compose up -d

# 4. Monitor startup
docker compose logs -f postgres
```

## Error Handling

### Connection Error Handling

```python
from semantic_search.database.connection import init_database_connection
import logging

logger = logging.getLogger(__name__)

async def initialize_database():
    try:
        db = await init_database_connection()
        logger.info("Database connection established successfully")
        return db
    except ConnectionError as e:
        logger.error(f"Failed to connect to database: {e}")
        # Implement retry logic or fallback
        raise
    except ValueError as e:
        logger.error(f"Database configuration error: {e}")
        raise
```

### Graceful Shutdown

```python
from semantic_search.database.connection import close_database_connection
import signal
import asyncio

async def shutdown_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully")
    await close_database_connection()
    logger.info("Database connections closed")

# Register shutdown handlers
signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(shutdown_handler(s, f)))
signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(shutdown_handler(s, f)))
```

### VectorStore Error Handling

```python
from semantic_search.database.vector_store import VectorStore

async def safe_vector_operations():
    store = VectorStore()

    try:
        # Store content with validation
        content_id = await store.store_content(
            content_type="source_file",
            source_path="/path/to/file.py",
            content="file contents",
            title="Example File"
        )

        # Store embedding with dimension validation
        embedding_id = await store.store_embedding(
            content_id=content_id,
            embedding=[0.1] * 384,  # 384 dimensions for BAAI/bge-small-en-v1.5
            chunk_text="chunk content",
            chunk_index=0
        )

        logger.info(f"Successfully stored content {content_id} and embedding {embedding_id}")

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Storage operation failed: {e}")
        raise
```

### Network and Timeout Handling

```python
import asyncio
from asyncpg.exceptions import ConnectionDoesNotExistError, InterfaceError

async def robust_database_operation(db, query, *args):
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            return await db.fetch_one(query, *args)
        except (ConnectionDoesNotExistError, InterfaceError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Database connection failed after {max_retries} attempts")
                raise

            logger.warning(f"Connection error on attempt {attempt + 1}, retrying in {retry_delay}s")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except asyncio.TimeoutError as e:
            logger.error(f"Query timeout: {e}")
            raise
```

## Production Deployment Best Practices

### Security Configuration

```python
# Use environment-specific configurations
config = DatabaseConfig(
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("POSTGRES_DB"),
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    # Production pool settings
    min_pool_size=10,
    max_pool_size=50,
    command_timeout=30
)
```

### SSL Configuration

For production deployments with SSL:

```python
# Add SSL parameters to connection URL
def get_production_database_url():
    base_url = DatabaseConfig.from_env().to_url()
    return f"{base_url}?sslmode=require&sslcert=/path/to/client.crt&sslkey=/path/to/client.key"

db = await init_database_connection(database_url=get_production_database_url())
```

### Monitoring and Logging

```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Log database operations
async def logged_database_operation(operation_name, func, *args, **kwargs):
    logger.info("database_operation_start", operation=operation_name)
    start_time = time.time()

    try:
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(
            "database_operation_success",
            operation=operation_name,
            duration_ms=round(duration * 1000, 2)
        )
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "database_operation_error",
            operation=operation_name,
            error=str(e),
            duration_ms=round(duration * 1000, 2)
        )
        raise
```

### Performance Optimization

```python
# Connection pool tuning based on load
def get_pool_config_for_load(expected_concurrent_users):
    """Calculate optimal pool settings based on expected load."""
    base_connections = max(2, expected_concurrent_users // 10)
    max_connections = min(100, expected_concurrent_users)

    return DatabaseConfig(
        min_pool_size=base_connections,
        max_pool_size=max_connections,
        command_timeout=30
    )

# Example usage
config = get_pool_config_for_load(expected_concurrent_users=500)
```

### Health Check Endpoints

```python
from fastapi import HTTPException

async def database_health_endpoint():
    """Health check endpoint for load balancers."""
    try:
        db = get_database_connection()
        health = await db.health_check()

        if not health['healthy']:
            raise HTTPException(status_code=503, detail=health['error'])

        return {
            "status": "healthy",
            "database": health['database'],
            "response_time_ms": health['response_time_ms'],
            "pool_stats": health['pool']
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database health check failed: {e}")
```

### Backup and Recovery

```bash
# Database backup script
#!/bin/bash
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB \
        --verbose --clean --create --format=custom \
        --file="$BACKUP_DIR/kcs_backup_$DATE.dump"

# Compress backup
gzip "$BACKUP_DIR/kcs_backup_$DATE.dump"

# Clean old backups (keep 7 days)
find "$BACKUP_DIR" -name "kcs_backup_*.dump.gz" -mtime +7 -delete
```

## Troubleshooting

### Common Issues and Solutions

#### Connection Refused

```
Error: connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused
```

**Solution:**

1. Verify PostgreSQL is running: `docker compose ps postgres`
2. Check port binding: `netstat -tlnp | grep 5432`
3. Verify environment variables in `.env`

#### pgvector Extension Missing

```
WARNING: pgvector extension not found - vector operations will fail
```

**Solution:**

```sql
-- Connect to database and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Pool Exhaustion

```
Error: pool is exhausted, please retry later
```

**Solution:**

1. Increase `max_pool_size` in configuration
2. Investigate long-running queries
3. Add connection monitoring

```python
# Monitor pool usage
stats = await db.get_pool_stats()
if stats['idle_size'] == 0:
    logger.warning("Connection pool exhausted", pool_stats=stats)
```

#### Schema Migration Issues

```
ERROR: Required table 'indexed_content' not found
```

**Solution:**

1. Run database migrations: `alembic upgrade head`
2. Verify migration files in `/src/sql/migrations/`
3. Check database initialization logs

#### Memory Issues

```
Error: out of memory for query result
```

**Solution:**

1. Use pagination for large result sets
2. Increase PostgreSQL memory settings
3. Use streaming queries for large datasets

```python
# Use pagination instead of fetch_all for large results
async def paginated_fetch(db, query, page_size=1000):
    offset = 0
    while True:
        results = await db.fetch_all(
            f"{query} LIMIT $1 OFFSET $2",
            page_size, offset
        )
        if not results:
            break
        yield results
        offset += page_size
```

### Debug Mode

Enable debug logging for detailed connection information:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('asyncpg').setLevel(logging.DEBUG)

# This will log all SQL queries and connection events
db = await init_database_connection()
```

### Performance Monitoring

```python
async def monitor_database_performance():
    """Monitor database performance metrics."""
    db = get_database_connection()

    # Check pool utilization
    stats = await db.get_pool_stats()
    utilization = (stats['size'] - stats['idle_size']) / stats['max_size']

    if utilization > 0.8:
        logger.warning(f"High pool utilization: {utilization:.2%}")

    # Check query performance
    slow_queries = await db.fetch_all("""
        SELECT query, mean_time, calls
        FROM pg_stat_statements
        WHERE mean_time > 1000
        ORDER BY mean_time DESC
        LIMIT 10
    """)

    for query in slow_queries:
        logger.warning(
            "Slow query detected",
            query=query['query'][:100],
            mean_time=query['mean_time'],
            calls=query['calls']
        )
```

This comprehensive guide covers all aspects of VectorStore setup and connection management. For additional support, refer to the PostgreSQL and pgvector documentation, or check the project's troubleshooting guide.
