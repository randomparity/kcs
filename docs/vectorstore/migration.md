# VectorStore Schema Migration Guide

**Generated**: 2025-09-25
**Purpose**: Provide complete migration path for schema discrepancies and upgrades

## Overview

This guide addresses the schema discrepancies identified between the current production schema (014_semantic_search_core.sql) and the intended template design (schema.sql), as well as providing migration strategies for common upgrades including dimensionality changes, index optimization, and missing functionality.

## Current State Analysis

### Production Schema (Current - Migration 014)
- **Location**: `/src/sql/migrations/014_semantic_search_core.sql`
- **Status**: Active in production
- **Vector Dimensions**: 384 (BAAI/bge-small-en-v1.5)
- **Index Type**: IVFFlat (lists=100)
- **Status Fields**: VARCHAR(20) with CHECK constraints
- **Content Fields**: VARCHAR(50) with CHECK constraints
- **Extra Columns**: chunk_text, line_start, line_end

### Template Schema (Target Design)
- **Location**: `/src/python/semantic_search/schema.sql`
- **Status**: Design template only
- **Vector Dimensions**: 384 (same)
- **Index Type**: HNSW (m=16, ef_construction=64)
- **Status Fields**: ENUM types
- **Content Fields**: ENUM types
- **Extra Columns**: model_name, model_version

---

## Schema Discrepancies Summary

| Component | Production (014) | Template Design | Impact |
|-----------|------------------|-----------------|---------|
| **Status Field** | VARCHAR(20) | ENUM type | Flexibility vs Type Safety |
| **Content Type** | VARCHAR(50) | ENUM type | Flexibility vs Type Safety |
| **Index Strategy** | IVFFlat | HNSW | Performance characteristics |
| **Model Tracking** | Via metadata JSONB | Dedicated columns | Data structure vs normalization |
| **Chunk Context** | line_start, line_end | Not present | Debugging vs simplicity |

---

## Migration Strategies

### 1. Template Schema to Production Schema

**Use Case**: Aligning new installations with current production
**Complexity**: Low
**Data Loss Risk**: None (new installation)

#### Migration Script

```sql
-- 001_template_to_production.sql
-- Convert template schema to production-compatible schema

BEGIN;

-- Drop ENUM types if they exist (template)
DROP TYPE IF EXISTS index_status CASCADE;
DROP TYPE IF EXISTS content_type CASCADE;

-- Modify indexed_content table
ALTER TABLE indexed_content
    ALTER COLUMN status TYPE VARCHAR(20),
    ALTER COLUMN content_type TYPE VARCHAR(50);

-- Add production-specific columns to vector_embedding
ALTER TABLE vector_embedding
    ADD COLUMN chunk_text TEXT,
    ADD COLUMN line_start INTEGER,
    ADD COLUMN line_end INTEGER,
    DROP COLUMN IF EXISTS model_name,
    DROP COLUMN IF EXISTS model_version;

-- Convert HNSW index to IVFFlat
DROP INDEX IF EXISTS idx_vector_embedding_hnsw;
CREATE INDEX idx_vector_embedding_similarity ON vector_embedding
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Add CHECK constraints for status validation
ALTER TABLE indexed_content
    ADD CONSTRAINT indexed_content_status_check
    CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'));

ALTER TABLE search_query
    ADD CONSTRAINT search_query_status_check
    CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'));

-- Add missing production columns and indexes
ALTER TABLE vector_embedding
    ADD CONSTRAINT vector_embedding_not_null_chunk_text
    CHECK (chunk_text IS NOT NULL);

-- Add production-specific indexes
CREATE INDEX idx_vector_embedding_lines ON vector_embedding (line_start, line_end);

COMMIT;
```

---

### 2. Production Schema to Template Schema

**Use Case**: Upgrading production to improved template design
**Complexity**: Medium
**Data Loss Risk**: Low (model metadata will be lost)

#### Migration Script

```sql
-- 002_production_to_template.sql
-- Upgrade production schema to template design

BEGIN;

-- Create ENUM types
CREATE TYPE index_status AS ENUM (
    'pending', 'indexing', 'completed', 'failed', 'stale'
);

CREATE TYPE content_type AS ENUM (
    'source_file', 'documentation', 'comment_block',
    'function_definition', 'struct_definition'
);

-- Backup existing data mappings
CREATE TEMPORARY TABLE status_mapping AS
SELECT DISTINCT
    status as old_status,
    CASE
        WHEN status = 'PENDING' THEN 'pending'::index_status
        WHEN status = 'PROCESSING' THEN 'indexing'::index_status
        WHEN status = 'COMPLETED' THEN 'completed'::index_status
        WHEN status = 'FAILED' THEN 'failed'::index_status
        ELSE 'pending'::index_status
    END as new_status
FROM indexed_content;

CREATE TEMPORARY TABLE content_type_mapping AS
SELECT DISTINCT
    content_type as old_type,
    CASE
        WHEN content_type ILIKE '%source%' THEN 'source_file'::content_type
        WHEN content_type ILIKE '%doc%' THEN 'documentation'::content_type
        WHEN content_type ILIKE '%comment%' THEN 'comment_block'::content_type
        WHEN content_type ILIKE '%function%' THEN 'function_definition'::content_type
        WHEN content_type ILIKE '%struct%' THEN 'struct_definition'::content_type
        ELSE 'source_file'::content_type
    END as new_type
FROM indexed_content;

-- Migrate indexed_content table
ALTER TABLE indexed_content
    ADD COLUMN new_status index_status,
    ADD COLUMN new_content_type content_type;

UPDATE indexed_content
SET new_status = sm.new_status
FROM status_mapping sm
WHERE indexed_content.status = sm.old_status;

UPDATE indexed_content
SET new_content_type = ctm.new_type
FROM content_type_mapping ctm
WHERE indexed_content.content_type = ctm.old_type;

-- Drop old columns and rename new ones
ALTER TABLE indexed_content
    DROP CONSTRAINT indexed_content_status_check,
    DROP COLUMN status,
    DROP COLUMN content_type;

ALTER TABLE indexed_content
    RENAME COLUMN new_status TO status;
ALTER TABLE indexed_content
    RENAME COLUMN new_content_type TO content_type;

-- Migrate search_query table
ALTER TABLE search_query
    ADD COLUMN new_status index_status;

UPDATE search_query
SET new_status =
    CASE
        WHEN status = 'PENDING' THEN 'pending'::index_status
        WHEN status = 'PROCESSING' THEN 'indexing'::index_status
        WHEN status = 'COMPLETED' THEN 'completed'::index_status
        WHEN status = 'FAILED' THEN 'failed'::index_status
        ELSE 'pending'::index_status
    END;

ALTER TABLE search_query
    DROP CONSTRAINT search_query_status_check,
    DROP COLUMN status;
ALTER TABLE search_query
    RENAME COLUMN new_status TO status;

-- Add model tracking columns to vector_embedding
ALTER TABLE vector_embedding
    ADD COLUMN model_name TEXT NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5',
    ADD COLUMN model_version TEXT NOT NULL DEFAULT '1.5';

-- Migrate existing embeddings (extract from metadata if available)
UPDATE vector_embedding
SET
    model_name = COALESCE(metadata->>'model_name', 'BAAI/bge-small-en-v1.5'),
    model_version = COALESCE(metadata->>'model_version', '1.5');

-- Convert IVFFlat index to HNSW
DROP INDEX IF EXISTS idx_vector_embedding_similarity;
CREATE INDEX idx_vector_embedding_hnsw ON vector_embedding
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Add template-specific indexes
CREATE INDEX idx_vector_embedding_model ON vector_embedding(model_name, model_version);

-- Note: line_start/line_end columns are preserved for backward compatibility

COMMIT;
```

---

### 3. Model Metadata Migration

**Use Case**: Adding persistent model tracking to production
**Complexity**: Low
**Data Loss Risk**: None

#### Migration Script

```sql
-- 003_add_model_metadata.sql
-- Add model metadata tracking without breaking changes

BEGIN;

-- Add model tracking columns
ALTER TABLE vector_embedding
    ADD COLUMN model_name TEXT DEFAULT 'BAAI/bge-small-en-v1.5',
    ADD COLUMN model_version TEXT DEFAULT '1.5';

-- Populate from metadata JSONB where available
UPDATE vector_embedding
SET
    model_name = COALESCE(metadata->>'model_name', 'BAAI/bge-small-en-v1.5'),
    model_version = COALESCE(metadata->>'model_version', '1.5')
WHERE metadata IS NOT NULL;

-- Make columns NOT NULL after population
ALTER TABLE vector_embedding
    ALTER COLUMN model_name SET NOT NULL,
    ALTER COLUMN model_version SET NOT NULL;

-- Add index for model-based queries
CREATE INDEX idx_vector_embedding_model ON vector_embedding(model_name, model_version);

-- Add helpful function for model-specific queries
CREATE OR REPLACE FUNCTION find_embeddings_by_model(
    target_model_name text,
    target_model_version text DEFAULT NULL
)
RETURNS TABLE (
    embedding_id integer,
    content_id integer,
    chunk_index integer,
    model_name text,
    model_version text
) AS $$
BEGIN
    RETURN QUERY
    SELECT ve.id, ve.content_id, ve.chunk_index, ve.model_name, ve.model_version
    FROM vector_embedding ve
    WHERE ve.model_name = target_model_name
      AND (target_model_version IS NULL OR ve.model_version = target_model_version);
END;
$$ LANGUAGE plpgsql;

COMMIT;
```

---

### 4. Dimension Migration (768 to 384)

**Use Case**: Migrating from larger to smaller embedding dimensions
**Complexity**: High
**Data Loss Risk**: High (requires re-embedding)

#### Migration Strategy

```sql
-- 004_dimension_migration_768_to_384.sql
-- Migrate from 768 to 384 dimensions (requires re-embedding)

BEGIN;

-- Create backup table for 768-dimensional embeddings
CREATE TABLE vector_embedding_768_backup AS
SELECT * FROM vector_embedding;

-- Create new 384-dimensional embedding table
CREATE TABLE vector_embedding_new (
    id SERIAL PRIMARY KEY,
    content_id INTEGER NOT NULL REFERENCES indexed_content(id) ON DELETE CASCADE,
    embedding VECTOR(384), -- New 384 dimensions
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    model_name TEXT NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5',
    model_version TEXT NOT NULL DEFAULT '1.5',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE (content_id, chunk_index)
);

-- Mark all content as PENDING for re-embedding
UPDATE indexed_content
SET status = 'PENDING',
    indexed_at = NULL,
    updated_at = now();

-- Preserve metadata and chunk information
INSERT INTO vector_embedding_new (
    content_id, chunk_index, chunk_text, line_start, line_end,
    model_name, model_version, metadata, created_at
)
SELECT
    content_id, chunk_index, chunk_text, line_start, line_end,
    'BAAI/bge-small-en-v1.5', '1.5', metadata, created_at
FROM vector_embedding_768_backup;

-- Drop old table and rename new one
DROP TABLE vector_embedding CASCADE;
ALTER TABLE vector_embedding_new RENAME TO vector_embedding;

-- Recreate indexes
CREATE INDEX idx_vector_embedding_content_id ON vector_embedding (content_id);
CREATE INDEX idx_vector_embedding_chunk_index ON vector_embedding (chunk_index);
CREATE INDEX idx_vector_embedding_lines ON vector_embedding (line_start, line_end);
CREATE INDEX idx_vector_embedding_created_at ON vector_embedding (created_at DESC);
CREATE INDEX idx_vector_embedding_model ON vector_embedding(model_name, model_version);

-- Vector similarity index will be created after re-embedding
-- CREATE INDEX idx_vector_embedding_similarity ON vector_embedding
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Update sequence
SELECT setval('vector_embedding_id_seq', (SELECT MAX(id) FROM vector_embedding));

COMMIT;

-- Note: After this migration, you must:
-- 1. Re-run embedding generation for all content
-- 2. Create the vector similarity index after embeddings are populated
-- 3. Verify all embeddings have 384 dimensions
```

#### Post-Migration Steps

```python
# Python script for re-embedding after dimension migration
async def re_embed_all_content():
    """Re-embed all content after dimension migration."""
    vector_store = VectorStore()

    # Get all pending content
    pending_content = await vector_store.list_content(
        ContentFilter(status_filter=['PENDING'])
    )

    for content in pending_content:
        try:
            # Update status to processing
            await vector_store.update_content_status(
                content.id, 'PROCESSING'
            )

            # Generate new 384-dimensional embeddings
            # (Implementation depends on your embedding service)
            new_embeddings = await generate_embeddings_384(content.content)

            # Store new embeddings
            for i, embedding in enumerate(new_embeddings):
                await vector_store.store_embedding(
                    content_id=content.id,
                    embedding=embedding.values,
                    chunk_text=embedding.text,
                    chunk_index=i,
                    model_name='BAAI/bge-small-en-v1.5',
                    model_version='1.5'
                )

        except Exception as e:
            logger.error(f"Failed to re-embed content {content.id}: {e}")
            await vector_store.update_content_status(content.id, 'FAILED')
```

---

### 5. Index Migration (IVFFlat to HNSW)

**Use Case**: Upgrading to HNSW for better performance at scale
**Complexity**: Medium
**Data Loss Risk**: None (index only)

#### Migration Script

```sql
-- 005_index_migration_ivfflat_to_hnsw.sql
-- Migrate from IVFFlat to HNSW index

BEGIN;

-- Check current index type and drop if IVFFlat
DO $$
DECLARE
    index_exists boolean;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_vector_embedding_similarity'
    ) INTO index_exists;

    IF index_exists THEN
        DROP INDEX idx_vector_embedding_similarity;
        RAISE NOTICE 'Dropped existing IVFFlat index';
    END IF;
END $$;

-- Create HNSW index with optimized parameters
-- Parameters chosen based on dataset size and performance requirements
CREATE INDEX idx_vector_embedding_hnsw ON vector_embedding
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Update search_query index as well if it exists
DO $$
DECLARE
    query_index_exists boolean;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_search_query_embedding'
    ) INTO query_index_exists;

    IF query_index_exists THEN
        DROP INDEX idx_search_query_embedding;
        CREATE INDEX idx_search_query_embedding_hnsw ON search_query
            USING hnsw (query_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        RAISE NOTICE 'Migrated search_query index to HNSW';
    END IF;
END $$;

-- Add function to tune HNSW parameters dynamically
CREATE OR REPLACE FUNCTION tune_hnsw_search_parameters(
    ef_search integer DEFAULT 100
)
RETURNS void AS $$
BEGIN
    -- Set search parameters for current session
    EXECUTE format('SET hnsw.ef_search = %s', ef_search);
    RAISE NOTICE 'Set HNSW ef_search to %', ef_search;
END;
$$ LANGUAGE plpgsql;

COMMIT;

-- Performance tuning recommendations
COMMENT ON INDEX idx_vector_embedding_hnsw IS
'HNSW index parameters: m=16 (connections per node), ef_construction=64 (build quality).
 For search tuning, use tune_hnsw_search_parameters(ef_search) where:
 - ef_search=40-100 for speed
 - ef_search=100-200 for accuracy
 - ef_search should be >= number of desired results';
```

#### HNSW Tuning Guide

```sql
-- Tuning queries for different use cases

-- High-speed searches (lower accuracy)
SELECT tune_hnsw_search_parameters(40);

-- Balanced searches (default)
SELECT tune_hnsw_search_parameters(100);

-- High-accuracy searches (slower)
SELECT tune_hnsw_search_parameters(200);

-- Monitor index performance
SELECT
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%hnsw%';
```

---

### 6. Adding Missing Columns (chunk_text exposure)

**Use Case**: Exposing chunk_text in API responses
**Complexity**: Low
**Data Loss Risk**: None

#### Python Model Updates

```python
# Update DBVectorEmbedding model to include chunk_text
class DBVectorEmbedding:
    """Database representation of vector embedding."""

    def __init__(
        self,
        id: int,
        content_id: int,
        embedding: list[float],
        chunk_index: int,
        created_at: datetime,
        chunk_text: str = "",  # Add chunk_text field
        line_start: int | None = None,  # Add line tracking
        line_end: int | None = None,    # Add line tracking
        model_name: str = "BAAI/bge-small-en-v1.5",
        model_version: str = "1.0",
    ):
        self.id = id
        self.content_id = content_id
        self.embedding = embedding
        self.chunk_text = chunk_text  # Store chunk text
        self.line_start = line_start  # Store line numbers
        self.line_end = line_end
        self.model_name = model_name
        self.model_version = model_version
        self.chunk_index = chunk_index
        self.created_at = created_at
```

#### API Response Updates

```python
# Update similarity_search to include chunk_text in results
async def similarity_search(
    self,
    query_embedding: list[float],
    filters: SimilaritySearchFilter | None = None,
) -> list[dict[str, Any]]:
    """Enhanced similarity search with chunk_text exposure."""

    # Updated select fields to include chunk_text
    select_fields = [
        "ic.id as content_id",
        "ic.content_type",
        "ic.source_path",
        "ic.title",
        "ic.metadata",
        "ve.id as embedding_id",
        "ve.chunk_index",
        "ve.chunk_text",  # Now included in response
        "ve.line_start",  # Include line numbers
        "ve.line_end",    # Include line numbers
        "(1 - (ve.embedding <=> $1::vector)) as similarity_score",
    ]

    # Rest of method unchanged...

    # Updated result building to include new fields
    for row in results:
        result = {
            "content_id": row["content_id"],
            "content_type": row["content_type"],
            "source_path": row["source_path"],
            "title": row["title"],
            "metadata": row["metadata"] if row["metadata"] else {},
            "embedding_id": row["embedding_id"],
            "chunk_index": row["chunk_index"],
            "chunk_text": row["chunk_text"],  # Now exposed
            "line_start": row["line_start"],    # Now exposed
            "line_end": row["line_end"],        # Now exposed
            "similarity_score": float(row["similarity_score"]),
        }
        search_results.append(result)
```

---

## Backward Compatibility Strategies

### 1. Gradual Migration Approach

```sql
-- Create migration tracking table
CREATE TABLE IF NOT EXISTS schema_migration_log (
    id SERIAL PRIMARY KEY,
    migration_name TEXT NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    rollback_sql TEXT,
    notes TEXT
);

-- Function to check if migration was applied
CREATE OR REPLACE FUNCTION migration_applied(migration_name text)
RETURNS boolean AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM schema_migration_log
        WHERE migration_name = $1
    );
END;
$$ LANGUAGE plpgsql;
```

### 2. Feature Flags for API Changes

```python
# Configuration-driven API response fields
class VectorStoreConfig:
    """Configuration for VectorStore behavior."""

    def __init__(self):
        self.expose_chunk_text = os.getenv('VECTOR_STORE_EXPOSE_CHUNK_TEXT', 'false').lower() == 'true'
        self.expose_line_numbers = os.getenv('VECTOR_STORE_EXPOSE_LINE_NUMBERS', 'false').lower() == 'true'
        self.include_model_metadata = os.getenv('VECTOR_STORE_INCLUDE_MODEL_METADATA', 'false').lower() == 'true'

# Use in similarity_search method
config = VectorStoreConfig()

result = {
    "content_id": row["content_id"],
    "content_type": row["content_type"],
    "source_path": row["source_path"],
    # ... base fields ...
}

if config.expose_chunk_text:
    result["chunk_text"] = row["chunk_text"]

if config.expose_line_numbers:
    result["line_start"] = row["line_start"]
    result["line_end"] = row["line_end"]

if config.include_model_metadata:
    result["model_name"] = row.get("model_name")
    result["model_version"] = row.get("model_version")
```

---

## Step-by-Step Migration Scripts

### Complete Production Upgrade Script

```bash
#!/bin/bash
# migrate_vectorstore_production.sh
# Complete migration from current production to enhanced schema

set -e  # Exit on error

DATABASE_URL="${DATABASE_URL:-}"
BACKUP_FILE="vectorstore_backup_$(date +%Y%m%d_%H%M%S).sql"

if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL environment variable not set"
    exit 1
fi

echo "Starting VectorStore migration..."

# Step 1: Create backup
echo "Creating backup..."
pg_dump "$DATABASE_URL" > "$BACKUP_FILE"
echo "Backup created: $BACKUP_FILE"

# Step 2: Apply model metadata migration
echo "Adding model metadata tracking..."
psql "$DATABASE_URL" -f migrations/003_add_model_metadata.sql

# Step 3: Apply index migration (optional, based on data size)
EMBEDDING_COUNT=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM vector_embedding;")
echo "Current embedding count: $EMBEDDING_COUNT"

if [ "$EMBEDDING_COUNT" -gt 1000000 ]; then
    echo "Large dataset detected. Migrating to HNSW index..."
    psql "$DATABASE_URL" -f migrations/005_index_migration_ivfflat_to_hnsw.sql
else
    echo "Dataset size appropriate for IVFFlat. Keeping current index."
fi

# Step 4: Update Python configuration
echo "Enabling enhanced API features..."
export VECTOR_STORE_EXPOSE_CHUNK_TEXT=true
export VECTOR_STORE_EXPOSE_LINE_NUMBERS=true
export VECTOR_STORE_INCLUDE_MODEL_METADATA=true

# Step 5: Restart application services
echo "Restarting services..."
systemctl restart kcs-semantic-search || echo "Warning: Could not restart service"

# Step 6: Verify migration
echo "Verifying migration..."
python3 -c "
import asyncio
from src.python.semantic_search.database.vector_store import VectorStore

async def verify():
    vs = VectorStore()
    stats = await vs.get_storage_stats()
    print('Migration verification:')
    print(f'  Total content: {stats.get(\"total_content\", 0)}')
    print(f'  Total embeddings: {stats.get(\"total_embeddings\", 0)}')
    print(f'  Embedding models: {stats.get(\"embedding_models\", [])}')

asyncio.run(verify())
"

echo "Migration completed successfully!"
echo "Backup available at: $BACKUP_FILE"
```

---

## Rollback Procedures

### 1. Model Metadata Rollback

```sql
-- rollback_003_model_metadata.sql
-- Rollback model metadata additions

BEGIN;

-- Remove model tracking columns
ALTER TABLE vector_embedding
    DROP COLUMN IF EXISTS model_name,
    DROP COLUMN IF EXISTS model_version;

-- Drop model-specific index
DROP INDEX IF EXISTS idx_vector_embedding_model;

-- Drop helper function
DROP FUNCTION IF EXISTS find_embeddings_by_model;

-- Log rollback
INSERT INTO schema_migration_log (migration_name, notes)
VALUES ('rollback_003_model_metadata', 'Removed model tracking columns');

COMMIT;
```

### 2. Index Rollback (HNSW to IVFFlat)

```sql
-- rollback_005_index_migration.sql
-- Rollback from HNSW to IVFFlat index

BEGIN;

-- Drop HNSW indexes
DROP INDEX IF EXISTS idx_vector_embedding_hnsw;
DROP INDEX IF EXISTS idx_search_query_embedding_hnsw;

-- Recreate IVFFlat indexes
CREATE INDEX idx_vector_embedding_similarity ON vector_embedding
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_search_query_embedding ON search_query
    USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);

-- Remove HNSW tuning function
DROP FUNCTION IF EXISTS tune_hnsw_search_parameters;

-- Log rollback
INSERT INTO schema_migration_log (migration_name, notes)
VALUES ('rollback_005_index_migration', 'Reverted to IVFFlat indexes');

COMMIT;
```

### 3. Complete Rollback Script

```bash
#!/bin/bash
# rollback_vectorstore.sh
# Complete rollback to baseline production schema

set -e

DATABASE_URL="${DATABASE_URL:-}"
ROLLBACK_TO_BACKUP="${1:-}"

if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL environment variable not set"
    exit 1
fi

echo "Starting VectorStore rollback..."

if [ -n "$ROLLBACK_TO_BACKUP" ] && [ -f "$ROLLBACK_TO_BACKUP" ]; then
    echo "Restoring from backup: $ROLLBACK_TO_BACKUP"

    # Create safety backup before rollback
    SAFETY_BACKUP="pre_rollback_backup_$(date +%Y%m%d_%H%M%S).sql"
    pg_dump "$DATABASE_URL" > "$SAFETY_BACKUP"

    # Restore from backup
    psql "$DATABASE_URL" < "$ROLLBACK_TO_BACKUP"
    echo "Restored from backup successfully"

else
    echo "Performing step-by-step rollback..."

    # Rollback in reverse order
    psql "$DATABASE_URL" -f rollbacks/rollback_005_index_migration.sql
    psql "$DATABASE_URL" -f rollbacks/rollback_003_model_metadata.sql

    echo "Step-by-step rollback completed"
fi

# Reset configuration
unset VECTOR_STORE_EXPOSE_CHUNK_TEXT
unset VECTOR_STORE_EXPOSE_LINE_NUMBERS
unset VECTOR_STORE_INCLUDE_MODEL_METADATA

echo "Rollback completed successfully!"
```

---

## Data Validation After Migration

### 1. Schema Validation

```sql
-- validate_schema.sql
-- Comprehensive schema validation after migration

-- Validate table structure
SELECT
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name IN ('indexed_content', 'vector_embedding', 'search_query', 'search_result')
ORDER BY table_name, ordinal_position;

-- Validate indexes
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('indexed_content', 'vector_embedding', 'search_query', 'search_result')
ORDER BY tablename, indexname;

-- Validate constraints
SELECT
    table_name,
    constraint_name,
    constraint_type
FROM information_schema.table_constraints
WHERE table_name IN ('indexed_content', 'vector_embedding', 'search_query', 'search_result')
ORDER BY table_name, constraint_type;

-- Validate functions
SELECT
    routine_name,
    routine_type,
    data_type
FROM information_schema.routines
WHERE routine_name LIKE '%embedding%' OR routine_name LIKE '%search%'
ORDER BY routine_name;
```

### 2. Data Integrity Validation

```sql
-- validate_data_integrity.sql
-- Validate data integrity after migration

-- Check for orphaned embeddings
SELECT COUNT(*) as orphaned_embeddings
FROM vector_embedding ve
LEFT JOIN indexed_content ic ON ve.content_id = ic.id
WHERE ic.id IS NULL;

-- Check embedding dimensions
SELECT
    model_name,
    model_version,
    COUNT(*) as count,
    MIN(array_length(embedding, 1)) as min_dimensions,
    MAX(array_length(embedding, 1)) as max_dimensions
FROM vector_embedding
WHERE embedding IS NOT NULL
GROUP BY model_name, model_version;

-- Check status consistency
SELECT
    status,
    COUNT(*) as count,
    COUNT(CASE WHEN indexed_at IS NULL THEN 1 END) as missing_indexed_at
FROM indexed_content
GROUP BY status;

-- Check duplicate embeddings
SELECT
    content_id,
    chunk_index,
    COUNT(*) as duplicates
FROM vector_embedding
GROUP BY content_id, chunk_index
HAVING COUNT(*) > 1;
```

### 3. Performance Validation

```python
# validate_performance.py
# Performance validation after migration

import asyncio
import time
from src.python.semantic_search.database.vector_store import VectorStore

async def validate_performance():
    """Validate performance after migration."""
    vs = VectorStore()

    # Test embedding storage
    start_time = time.time()
    test_embedding = [0.1] * 384  # 384-dimensional test embedding

    content_id = await vs.store_content(
        content_type="test",
        source_path="/test/performance_test.txt",
        content="Test content for performance validation"
    )

    embedding_id = await vs.store_embedding(
        content_id=content_id,
        embedding=test_embedding,
        chunk_text="Test chunk",
        model_name="BAAI/bge-small-en-v1.5",
        model_version="1.5"
    )

    storage_time = time.time() - start_time
    print(f"Storage performance: {storage_time:.3f}s")

    # Test similarity search
    start_time = time.time()
    results = await vs.similarity_search(test_embedding)
    search_time = time.time() - start_time
    print(f"Search performance: {search_time:.3f}s")
    print(f"Results returned: {len(results)}")

    # Cleanup test data
    await vs.delete_content(content_id)

    # Validate expected fields in results
    if results:
        result = results[0]
        expected_fields = [
            'content_id', 'content_type', 'source_path',
            'embedding_id', 'chunk_index', 'similarity_score'
        ]

        missing_fields = [f for f in expected_fields if f not in result]
        if missing_fields:
            print(f"Warning: Missing fields in results: {missing_fields}")
        else:
            print("All expected fields present in results")

if __name__ == "__main__":
    asyncio.run(validate_performance())
```

---

## Migration Checklist

### Pre-Migration Checklist

- [ ] **Backup created and verified**
- [ ] **Database connection tested**
- [ ] **pgvector extension available**
- [ ] **Current schema version identified**
- [ ] **Migration scripts reviewed and tested**
- [ ] **Rollback procedure prepared**
- [ ] **Application downtime scheduled**

### Migration Execution Checklist

- [ ] **Application services stopped**
- [ ] **Database backup completed**
- [ ] **Migration scripts applied in order**
- [ ] **Schema validation passed**
- [ ] **Data integrity validation passed**
- [ ] **Performance validation passed**
- [ ] **Application configuration updated**
- [ ] **Services restarted successfully**

### Post-Migration Checklist

- [ ] **All API endpoints responding**
- [ ] **Search functionality working**
- [ ] **New features enabled (if applicable)**
- [ ] **Performance metrics within acceptable range**
- [ ] **Error logs reviewed**
- [ ] **Monitoring alerts configured**
- [ ] **Team notified of completion**

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Index Creation Timeouts

**Problem**: HNSW index creation takes too long
**Solution**:
```sql
-- Create index concurrently to avoid blocking
CREATE INDEX CONCURRENTLY idx_vector_embedding_hnsw ON vector_embedding
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

#### 2. Memory Issues During Migration

**Problem**: Out of memory errors during large migrations
**Solution**:
```sql
-- Increase work_mem for migration session
SET work_mem = '1GB';
SET maintenance_work_mem = '2GB';

-- Process in smaller batches
UPDATE vector_embedding
SET model_name = 'BAAI/bge-small-en-v1.5'
WHERE id BETWEEN 1 AND 10000;
-- Repeat for subsequent ranges
```

#### 3. Constraint Violations

**Problem**: CHECK constraint violations during status migration
**Solution**:
```sql
-- Identify invalid values
SELECT DISTINCT status FROM indexed_content
WHERE status NOT IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- Fix invalid values before migration
UPDATE indexed_content
SET status = 'PENDING'
WHERE status NOT IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');
```

#### 4. Python Model Sync Issues

**Problem**: Python models out of sync with database schema
**Solution**:
```python
# Update get_embedding_by_content_id to handle missing columns gracefully
async def get_embedding_by_content_id(
    self, content_id: int, chunk_index: int = 0
) -> DBVectorEmbedding | None:
    try:
        # Query with column existence checks
        row = await self._db.fetch_one(
            """
            SELECT
                id, content_id, embedding, chunk_index, created_at,
                COALESCE(chunk_text, '') as chunk_text,
                line_start, line_end,
                COALESCE(model_name, 'BAAI/bge-small-en-v1.5') as model_name,
                COALESCE(model_version, '1.5') as model_version
            FROM vector_embedding
            WHERE content_id = $1 AND chunk_index = $2
            """,
            content_id, chunk_index
        )

        if not row:
            return None

        return DBVectorEmbedding(
            id=row["id"],
            content_id=row["content_id"],
            embedding=list(row["embedding"]) if row["embedding"] else [],
            chunk_index=row["chunk_index"],
            chunk_text=row["chunk_text"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            created_at=row["created_at"],
        )

    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        return None
```

---

## Summary

This migration guide provides comprehensive strategies for addressing all identified schema discrepancies and common upgrade scenarios. The modular approach allows for selective application of migrations based on specific needs and risk tolerance.

**Key Recommendations:**

1. **Start with Model Metadata Migration** - Low risk, high value
2. **Consider Index Migration** - Based on dataset size and performance requirements
3. **Plan Dimension Migrations Carefully** - Requires full re-embedding
4. **Use Feature Flags** - For gradual API changes
5. **Always Test in Staging** - Before production deployment

Each migration includes rollback procedures and comprehensive validation to ensure data integrity and system stability.