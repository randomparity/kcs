# VectorStore Implementation Discrepancies

**Generated**: 2025-09-25
**Purpose**: Document differences between intended design and actual implementation

## Overview

This document identifies and explains discrepancies found between different schema versions, Python models, and the actual database implementation. These discrepancies were discovered during the verification process.

## Schema Version Discrepancies

### 1. Migration vs Template Schema

Two different schema versions exist in the codebase:

| Aspect | Migration (014_semantic_search_core.sql) | Template (schema.sql) |
|--------|-------------------------------------------|----------------------|
| **Location** | `/src/sql/migrations/` | `/src/sql/templates/` |
| **Status** | **PRODUCTION** (Active) | Design template |
| **Status Field Type** | VARCHAR(20) | ENUM type |
| **Content Type Field** | VARCHAR(50) | ENUM type |
| **Index Strategy** | IVFFlat (lists=100) | HNSW (m=16, ef_construction=64) |
| **Extra Columns** | chunk_text, line_start, line_end | model_name, model_version |

**Impact**: The production migration is the actual schema in use. The template represents a newer design that hasn't been migrated yet.

---

## Python Model vs Database Discrepancies

### 2. DBVectorEmbedding Model Mismatch

The Python model in `vector_store.py` doesn't fully align with the database schema:

**Python Model Has**:

```python
- model_name: str = "BAAI/bge-small-en-v1.5"
- model_version: str = "1.0"
```

**Database Has**:

```sql
- chunk_text: TEXT NOT NULL
- line_start: INTEGER NULL
- line_end: INTEGER NULL
```

**Resolution**: The Python code provides default values for model_name/version but doesn't use them. The database stores chunk_text which the Python model doesn't expose.

### 3. Method Parameter vs Database Storage

**store_embedding() accepts**:

- `chunk_text` parameter (stored in database)
- `model_name` parameter (NOT stored in database)
- `model_version` parameter (NOT stored in database)

**Impact**: Model metadata is accepted but discarded. This could cause confusion when tracking which model generated embeddings.

---

## API Contract vs Implementation Discrepancies

### 4. Missing Model Metadata Persistence

**OpenAPI Spec Defines**:

```yaml
StoreEmbeddingRequest:
  model_name: string
  model_version: string
```

**Implementation Reality**:

- Parameters are accepted
- Values are NOT stored in database
- No error or warning is given

**Recommendation**: Either remove from API or add columns to database.

### 5. Incomplete Field Mapping

**VectorEmbedding Response** in OpenAPI doesn't include:

- chunk_text
- line_start
- line_end
- metadata

These fields exist in the database but aren't exposed in the API response schema.

---

## Index Strategy Discrepancy

### 6. IVFFlat vs HNSW

**Current Production**: IVFFlat index

```sql
CREATE INDEX idx_vector_embedding_similarity
ON vector_embedding
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Template Design**: HNSW index

```sql
CREATE INDEX idx_vector_embedding_hnsw
ON vector_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Performance Implications**:

- **IVFFlat**: Faster build time, good for 100K-1M vectors
- **HNSW**: Better recall, slower build, better for 1M+ vectors

**Current Choice Rationale**: IVFFlat is appropriate for current scale.

---

## Data Type Discrepancies

### 7. Flexible VARCHAR vs Strict ENUM

**Production Uses VARCHAR**:

- Allows any string value
- More flexible for additions
- No migration needed for new values

**Template Uses ENUM**:

- Type safety
- Database-enforced validation
- Requires migration for new values

**Current Implementation**: VARCHAR provides needed flexibility during development.

---

## Missing Functionality

### 8. Model Version Tracking

**Issue**: No way to track which embedding model version was used

**Impact**:

- Can't identify outdated embeddings
- Can't selectively re-index with new models
- No audit trail for model upgrades

**Workaround**: Currently assumes all embeddings use BAAI/bge-small-en-v1.5

### 9. Chunk Metadata

**Database Has**: line_start, line_end columns
**Python API**: Doesn't accept or return these values

**Impact**: Can't track source line numbers for chunks, limiting debugging and context display.

---

## Validation Discrepancies

### 10. Status Value Validation

**Python Code**: No validation of status values
**Database**: Accepts any VARCHAR(20) string
**OpenAPI**: Defines enum [PENDING, PROCESSING, COMPLETED, FAILED]

**Risk**: Invalid status values could be stored, breaking queries that filter by status.

---

## Resolution Priority

### High Priority (Blocking Issues)

1. ✅ **Vector dimensions** - VERIFIED as 384 (correct)
2. ✅ **Multiple chunks support** - VERIFIED working
3. ✅ **Unique constraints** - VERIFIED correct

### Medium Priority (Functionality Gaps)

1. Model version tracking - Add to metadata JSONB
2. Line number tracking - Expose in Python API
3. Status validation - Add Python validation

### Low Priority (Future Improvements)

1. HNSW index migration - When scale requires
2. ENUM type migration - During schema v2
3. Model metadata columns - Next major version

---

## Recommended Actions

### Immediate (No Code Changes)

1. **Document** model version in metadata JSONB field
2. **Use** existing metadata field for model tracking
3. **Validate** status in Python before database insert

### Short Term (Minor Changes)

1. **Add** validation to VectorStore class
2. **Expose** chunk_text in API responses
3. **Document** the model version assumption

### Long Term (Migration Required)

1. **Plan** migration to HNSW when approaching 1M vectors
2. **Design** v2 schema with proper model tracking
3. **Consider** ENUM types for production stability

---

## Validation Checklist

- [x] All 9 VectorStore methods documented
- [x] All 4 database tables documented
- [x] 384-dimensional vectors confirmed
- [x] Multiple chunks per file verified
- [x] IVFFlat index confirmed in production
- [x] Status as VARCHAR confirmed
- [x] Cascade deletion verified

---

## Notes

1. **Source of Truth**: Migration 014_semantic_search_core.sql is the production schema
2. **Template Schema**: Represents future design direction, not current state
3. **Python Models**: Simplified for current needs, can be extended
4. **No Data Loss**: All discrepancies are additive; no data corruption risk
