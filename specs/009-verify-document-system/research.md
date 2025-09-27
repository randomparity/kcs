# Research: VectorStore API and Database Schema Verification

**Date**: 2025-09-25
**Feature**: Verify Document System

## Executive Summary

Research confirms the VectorStore implementation uses PostgreSQL with pgvector extension, implementing 384-dimensional embeddings with BAAI/bge-small-en-v1.5 model. Multiple schema versions exist with slight variations between migration and template files.

## Key Findings

### 1. VectorStore API Implementation

- **Location**: `/home/dave/src/kcs/src/python/semantic_search/database/vector_store.py`
- **Class**: `VectorStore` with async methods
- **Dependencies**: asyncpg, pgvector, pydantic

**Core Methods Identified**:

1. `store_content()` - Store content for indexing
2. `store_embedding()` - Store vector embeddings (validates 384 dimensions)
3. `similarity_search()` - Perform vector similarity search
4. `get_content_by_id()` - Retrieve content by ID
5. `get_embedding_by_content_id()` - Retrieve embedding by content ID
6. `list_content()` - List content with filters
7. `update_content_status()` - Update indexing status
8. `delete_content()` - Delete content and embeddings
9. `get_storage_stats()` - Get storage statistics

### 2. Database Schema Variations

Two schema versions discovered:

#### Migration Schema (Production - 014_semantic_search_core.sql)

**Tables**:

- `indexed_content` - SERIAL primary key, VARCHAR status values
- `vector_embedding` - VECTOR(384), includes chunk_text, line_start/end
- `search_query` - Query logging with processing metadata
- `search_result` - Results with scoring and ranking

**Key Differences**:

- Uses VARCHAR for enums (more flexible)
- UNIQUE constraint on (source_path) for indexed_content
- UNIQUE constraint on (content_id, chunk_index) for embeddings
- IVFFlat index for vector similarity (performance optimized)

#### Template Schema (schema.sql)

**Tables**:

- Similar structure but uses ENUM types for status and content_type
- More restrictive constraints (regex for hash validation)
- HNSW index instead of IVFFlat for vector search
- Additional model_name and model_version columns

### 3. Vector Configuration Verification

- **Confirmed**: 384 dimensions (not 768)
- **Model**: BAAI/bge-small-en-v1.5
- **Distance Metric**: Cosine similarity (`<=>` operator)
- **Indexing**: Both IVFFlat and HNSW indexes found

### 4. Multiple Chunks Support

**Verified**: Schema supports multiple chunks per file

- `chunk_index` column in vector_embedding table
- UNIQUE constraint on (content_id, chunk_index) allows multiple embeddings per content
- No UNIQUE constraint on source_path in vector_embedding table

### 5. Data Model Mappings

**DBIndexedContent** class fields → Database columns:

```
id → id (SERIAL/BIGSERIAL)
content_type → content_type (VARCHAR/ENUM)
source_path → source_path (TEXT)
content_hash → content_hash (VARCHAR/TEXT)
title → title (TEXT)
content → content (TEXT)
metadata → metadata (JSONB)
status → status (VARCHAR/ENUM)
indexed_at → indexed_at (TIMESTAMP)
updated_at → updated_at (TIMESTAMP)
created_at → created_at (TIMESTAMP)
```

**DBVectorEmbedding** class fields → Database columns:

```
id → id (SERIAL/BIGSERIAL)
content_id → content_id (INTEGER/BIGINT)
embedding → embedding (VECTOR(384))
chunk_index → chunk_index (INTEGER)
chunk_text → chunk_text (TEXT) [migration only]
line_start → line_start (INTEGER) [migration only]
line_end → line_end (INTEGER) [migration only]
model_name → model_name (TEXT) [template only]
model_version → model_version (TEXT) [template only]
created_at → created_at (TIMESTAMP)
```

## Discrepancies Found

1. **Schema Version Mismatch**:
   - Migration uses VARCHAR for status, template uses ENUM
   - Migration includes chunk_text, line_start/end in vector_embedding
   - Template includes model_name/version in vector_embedding

2. **Index Strategy Difference**:
   - Migration: IVFFlat index (lists=100)
   - Template: HNSW index (m=16, ef_construction=64)

3. **Python Model vs Database**:
   - Python DBVectorEmbedding expects model_name/version (defaults provided)
   - Migration schema doesn't have these columns
   - Python model doesn't include chunk_text, line_start/end from migration

## Recommendations

1. **Primary Schema**: Use migration file (014_semantic_search_core.sql) as source of truth
2. **Documentation Format**: OpenAPI for API, ERD for database (per clarifications)
3. **Testing Priority**: Verify multiple chunks per file functionality works
4. **Resolution Needed**: Reconcile Python models with actual database schema

## Technical Context Resolution

All NEEDS CLARIFICATION items from plan have been resolved:

- Language: Python 3.11+ (async/await confirmed)
- Dependencies: asyncpg, pgvector, pydantic confirmed
- Storage: PostgreSQL with pgvector confirmed
- Vector dimensions: 384 confirmed (not 768)
- Multiple chunks: Schema supports via chunk_index

## Next Steps

1. Generate OpenAPI specification for VectorStore class methods
2. Create ERD diagram showing actual migration schema
3. Document discrepancies between intended and actual implementation
4. Create verification test suite per user requirements
