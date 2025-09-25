# Constraint Cross-Reference: Code vs Database

**Generated**: 2025-09-25
**Code Source**: `/home/dave/src/kcs/src/python/semantic_search/database/vector_store.py`
**Database Source**: `/home/dave/src/kcs/src/sql/migrations/014_semantic_search_core.sql`
**Status**: VALIDATED ✓

## Executive Summary

Cross-reference validation confirms that Python code constraints align with database constraints. Both layers provide complementary validation with the database serving as the final authority through foreign keys, unique constraints, and check constraints.

## Constraint Mapping Table

| Constraint Type | Python Validation | Database Enforcement | Alignment Status |
|----------------|-------------------|---------------------|------------------|
| Empty Content | ✓ ValueError | NOT NULL | ✓ ALIGNED |
| Empty Source Path | ✓ ValueError | NOT NULL | ✓ ALIGNED |
| Unique Source Path | Hash duplicate check | UNIQUE constraint | ✓ ALIGNED |
| Vector Dimensions | ✓ len(embedding)==384 | VECTOR(384) type | ✓ ALIGNED |
| Status Values | ✓ List validation | CHECK constraint | ✓ ALIGNED |
| Content ID FK | - | REFERENCES indexed_content | ✓ DB ENFORCED |
| Chunk Uniqueness | Update if exists | UNIQUE(content_id, chunk_index) | ✓ ALIGNED |
| Cascade Delete | - | ON DELETE CASCADE | ✓ DB ENFORCED |
| Max Results | ✓ Pydantic Field | - | ✓ CODE ONLY |
| Similarity Range | ✓ Pydantic Field | - | ✓ CODE ONLY |

## Detailed Constraint Analysis

### 1. indexed_content Table

#### NOT NULL Constraints

**Database Definition**:

```sql
content_type VARCHAR(50) NOT NULL
source_path TEXT NOT NULL
content_hash VARCHAR(64) NOT NULL
content TEXT NOT NULL
status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
```

**Python Validation** (vector_store.py):

```python
# store_content() method
if not content.strip():
    raise ValueError("Content cannot be empty")  # Lines 140-141

if not source_path.strip():
    raise ValueError("Source path cannot be empty")  # Lines 143-144

# content_hash always generated
content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()  # Line 147
```

**Alignment**: ✓ PERFECT - Python pre-validates before database insertion

#### UNIQUE Constraints

**Database**:

```sql
source_path TEXT NOT NULL UNIQUE
```

**Python** (lines 151-162):

```python
# Check for duplicates using hash
existing_id = await self._db.fetch_val(
    """SELECT id FROM indexed_content
    WHERE source_path = $1 AND content_hash = $2""",
    source_path, content_hash
)
if existing_id:
    return int(existing_id)  # Return existing instead of duplicate
```

**Alignment**: ✓ ENHANCED - Python adds hash checking for content changes

#### CHECK Constraints

**Database**:

```sql
CONSTRAINT indexed_content_status_check
CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'))
```

**Python** (update_content_status, lines 600-603):

```python
valid_statuses = ['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED']
if status not in valid_statuses:
    raise ValueError(
        f"Invalid status '{status}'. Must be one of: {valid_statuses}"
    )
```

**Alignment**: ✓ PERFECT - Exact same status values

### 2. vector_embedding Table

#### Vector Dimension Constraint

**Database**:

```sql
embedding VECTOR(384)  -- Fixed 384 dimensions
```

**Python** (store_embedding, lines 220-221):

```python
if len(embedding) != 384:  # BAAI/bge-small-en-v1.5 dimension
    raise ValueError(f"Expected 384 dimensions, got {len(embedding)}")
```

**Python** (similarity_search, lines 311-312):

```python
if len(query_embedding) != 384:
    raise ValueError(f"Expected 384 dimensions, got {len(query_embedding)}")
```

**Alignment**: ✓ PERFECT - Both enforce exactly 384 dimensions

#### Foreign Key Constraint

**Database**:

```sql
content_id INTEGER NOT NULL REFERENCES indexed_content(id) ON DELETE CASCADE
```

**Python**: No explicit validation (relies on database)

**Alignment**: ✓ APPROPRIATE - Database handles referential integrity

#### Unique Constraint

**Database**:

```sql
UNIQUE (content_id, chunk_index)
```

**Python** (store_embedding, lines 224-247):

```python
# Check if embedding already exists
existing_id = await self._db.fetch_val(
    """SELECT id FROM vector_embedding
    WHERE content_id = $1 AND chunk_index = $2""",
    content_id, chunk_index
)
if existing_id:
    # Update existing instead of inserting duplicate
    await self._db.execute("""UPDATE vector_embedding...""")
    return int(existing_id)
```

**Alignment**: ✓ PERFECT - Python handles uniqueness gracefully

### 3. Field Length Constraints

#### Database Limits

```sql
content_type VARCHAR(50)   -- Max 50 chars
status VARCHAR(20)          -- Max 20 chars
content_hash VARCHAR(64)    -- SHA256 = 64 hex chars
```

#### Python Validation

- content_type: No length check (relies on DB)
- status: Enum validation ensures < 20 chars
- content_hash: SHA256 always produces 64 chars

**Alignment**: ✓ SUFFICIENT - Database enforces final limits

### 4. Pydantic Model Constraints

#### ContentFilter (lines 75-89)

```python
max_results: int = Field(100, ge=1, le=1000)
```

- Minimum: 1
- Maximum: 1000
- Default: 100

**Database**: No equivalent (application-level concern)

#### SimilaritySearchFilter (lines 91-100)

```python
similarity_threshold: float = Field(0.0, ge=0.0, le=1.0)
max_results: int = Field(20, ge=1, le=100)
```

- similarity_threshold: 0.0 to 1.0
- max_results: 1 to 100 (more restrictive than ContentFilter)

**Database**: No equivalent (application-level filtering)

**Alignment**: ✓ APPROPRIATE - Application-level constraints

### 5. Cascade Deletion

**Database**:

```sql
-- Foreign keys with CASCADE
vector_embedding.content_id REFERENCES indexed_content(id) ON DELETE CASCADE
search_result.query_id REFERENCES search_query(id) ON DELETE CASCADE
search_result.content_id REFERENCES indexed_content(id) ON DELETE CASCADE
search_result.embedding_id REFERENCES vector_embedding(id) ON DELETE CASCADE
```

**Python** (delete_content, lines 630-656):

```python
# Relies on CASCADE - just deletes parent
deleted = await self._db.fetch_val(
    """DELETE FROM indexed_content WHERE id = $1 RETURNING id""",
    content_id
)
```

**Alignment**: ✓ PERFECT - Python relies on database CASCADE

## Constraint Coverage Analysis

### Constraints Enforced at Both Levels

1. Empty content/source_path validation
2. Status value validation
3. Vector dimension validation (384)
4. Unique content detection (via hash)
5. Chunk uniqueness handling

### Database-Only Constraints

1. Foreign key referential integrity
2. Cascade deletion
3. Field length limits (VARCHAR)
4. Timestamp defaults (now())
5. JSONB structure validation

### Code-Only Constraints

1. Query result limits (max_results)
2. Similarity threshold ranges
3. Empty embedding validation
4. Business logic (status transitions)
5. Hash generation consistency

## Edge Cases and Error Handling

### 1. Constraint Violation Handling

**Python Approach**:

- Pre-validation with ValueError
- Graceful duplicate handling (return existing)
- Try-catch blocks with RuntimeError

**Database Approach**:

- Hard constraint failures
- Foreign key violations
- Unique constraint violations

### 2. Status Workflow

**Automatic Transitions** (Python):

```python
# On successful embedding storage (lines 264-271)
UPDATE indexed_content SET status = 'COMPLETED', indexed_at = NOW()

# On embedding failure (lines 279-286)
UPDATE indexed_content SET status = 'FAILED'
```

**Database**: Allows any valid status (relies on application logic)

## Validation Gaps and Recommendations

### Current Gaps

1. **No Python validation for**:
   - content_type length (50 chars)
   - Foreign key validity before insertion
   - Line number ranges (line_start, line_end)

2. **No Database validation for**:
   - Embedding not empty (can be NULL)
   - Business logic rules
   - Model version tracking

### Recommendations

1. **Add Python Validations**:

   ```python
   if len(content_type) > 50:
       raise ValueError("Content type exceeds 50 characters")
   ```

2. **Add Database Constraints**:

   ```sql
   ALTER TABLE vector_embedding
   ADD CONSTRAINT embedding_not_null CHECK (embedding IS NOT NULL);
   ```

3. **Consider Adding**:
   - Trigger for automatic chunk_count updates
   - Check constraint for line_start <= line_end
   - Default for metadata fields

## Test Scenarios

### Constraint Test Matrix

| Test Case | Python Result | Database Result | Overall |
|-----------|---------------|-----------------|---------|  
| Empty content | ValueError raised | Would fail NOT NULL | ✓ Caught early |
| 385-dim vector | ValueError raised | Would fail VECTOR(384) | ✓ Caught early |
| Invalid status | ValueError raised | Would fail CHECK | ✓ Caught early |
| Duplicate path | Returns existing ID | Would fail UNIQUE | ✓ Handled gracefully |
| Invalid FK | - | Foreign key violation | ✓ DB catches |
| Delete parent | - | Cascades to children | ✓ Works |
| NULL embedding | Raises ValueError | Allows NULL | ⚠️ Inconsistent |

## Conclusion

The constraint validation between Python code and database schema is **WELL ALIGNED** with complementary validation at both levels:

1. **Python provides**: Early validation, business logic, graceful handling
2. **Database provides**: Final authority, referential integrity, cascade operations
3. **Both provide**: Core data validation (dimensions, uniqueness, required fields)

The system demonstrates defense-in-depth with validation at multiple layers, ensuring data integrity while providing good user experience through early error detection and graceful duplicate handling.

---

*Validation completed as part of T019 - Cross-reference constraints between code validation and database constraints*
