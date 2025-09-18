# Research: Multi-File JSON Output Strategy

**Date**: 2025-01-18
**Feature**: 006-multi-file-json

## Executive Summary

Research findings for implementing chunked JSON output to replace single 2.8GB file with multiple 50MB chunks, enabling parallel processing and memory-efficient database loading.

## Technical Decisions

### 1. Chunk Size Strategy

**Decision**: Fixed 50MB chunk size with configurable override
**Rationale**:
- Balances memory usage (250MB max per chunk at 5x multiplier)
- Generates ~60 chunks for typical kernel (manageable file count)
- Fits in typical system buffers and caches
**Alternatives considered**:
- 10MB: Too many files (300+), excessive filesystem overhead
- 100MB: Risk of memory issues on constrained systems
- Dynamic sizing: Complex implementation, harder to predict behavior

### 2. Chunking Algorithm

**Decision**: Streaming JSON writer with size-based cutoff
**Rationale**:
- No need to load entire dataset in memory
- Clean JSON array boundaries at chunk points
- Compatible with existing serde_json serialization
**Alternatives considered**:
- Memory-mapped files: Complex for write operations, platform-specific
- Line-based splitting: Could break JSON structure mid-object
- Tree-based splitting: Would require full parse tree in memory

### 3. Checksum Algorithm

**Decision**: SHA256 for chunk integrity validation
**Rationale**:
- Cryptographically secure against tampering
- Standard library support in Rust and Python
- Fast enough for 50MB chunks (~100ms per chunk)
**Alternatives considered**:
- MD5: Deprecated for security, only marginally faster
- CRC32: Too weak for large files, collision risk
- BLAKE3: Better performance but less standard library support

### 4. Chunk Naming Convention

**Decision**: `{subsystem}_{sequence}.json` pattern
**Rationale**:
- Human-readable subsystem identification
- Natural sort order with zero-padded sequence
- Supports parallel processing by subsystem
**Example**: `kernel_001.json`, `drivers_net_001.json`
**Alternatives considered**:
- UUID names: Not human-friendly, no natural order
- Timestamp-based: Could conflict with parallel generation
- Hash-based: No semantic meaning, hard to debug

### 5. Manifest Format

**Decision**: JSON manifest with chunk metadata and dependencies
**Rationale**:
- Consistent with chunk format (all JSON)
- Easy to parse and validate
- Supports incremental updates via version field
**Schema**:
```json
{
  "version": "1.0.0",
  "created": "2025-01-18T10:00:00Z",
  "kernel_version": "6.7.0",
  "total_chunks": 60,
  "chunks": [
    {
      "id": "kernel_001",
      "file": "kernel_001.json",
      "subsystem": "kernel",
      "size_bytes": 52428800,
      "checksum_sha256": "abc123...",
      "symbol_count": 15000,
      "entry_point_count": 50
    }
  ]
}
```
**Alternatives considered**:
- SQLite database: Overkill for metadata, adds dependency
- CSV format: Limited structure for nested data
- YAML: Additional parser dependency

### 6. Parallel Processing Strategy

**Decision**: Worker pool with configurable parallelism (default 4)
**Rationale**:
- Matches typical CPU core count
- Prevents database connection exhaustion
- Allows system resource tuning
**Implementation**:
- Rust: rayon parallel iterator with chunk size
- Python: asyncio with semaphore for concurrency control
**Alternatives considered**:
- Thread-per-chunk: Could overwhelm system
- Sequential only: Wastes multi-core potential
- Fixed parallelism: Not flexible for different systems

### 7. Resume/Recovery Mechanism

**Decision**: Database table tracking processed chunks
**Rationale**:
- Survives process crashes
- Enables cluster-wide coordination
- Simple SQL queries for status
**Schema**:
```sql
CREATE TABLE chunk_processing (
    chunk_id VARCHAR(255) PRIMARY KEY,
    manifest_version VARCHAR(50),
    status VARCHAR(20), -- pending, processing, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INT DEFAULT 0
);
```
**Alternatives considered**:
- File-based markers: Could get out of sync
- Memory-only tracking: Lost on restart
- Redis/cache: Additional infrastructure dependency

### 8. Transaction Boundaries

**Decision**: One transaction per chunk with savepoints
**Rationale**:
- Atomic chunk processing (all or nothing)
- Allows partial rollback on errors
- Bounded transaction size prevents lock escalation
**Alternatives considered**:
- Single transaction for all: Too large, lock issues
- No transactions: Risk of partial data on failure
- Row-level commits: Inconsistent chunk state

## Implementation Recommendations

### Critical Path

1. Implement Rust chunk writer in kcs-serializer
2. Create chunk_processing database table
3. Build Python chunk loader with resume logic
4. Update index_kernel.sh with chunking flags
5. Add integration tests for multi-chunk scenarios

### Performance Targets

- Chunk generation: ≤500ms per 50MB chunk
- Chunk loading: ≤30s per chunk (including DB inserts)
- Full kernel processing: ≤30 minutes (60 chunks parallel)
- Memory per chunk: ≤250MB (5x chunk size)

### Error Handling

- Retry failed chunks up to 3 times with exponential backoff
- Log detailed errors with chunk ID and position
- Support manual retry via CLI flag
- Alert on >10% chunk failure rate

### Monitoring Points

- Chunk generation rate (chunks/minute)
- Database insert rate (rows/second)
- Memory usage per chunk processor
- Chunk failure rate and reasons
- Resume operations count

## Risks and Mitigations

### Risk 1: Chunk Boundary Corruption

**Risk**: JSON structure broken at chunk boundaries
**Mitigation**: Write complete JSON arrays per chunk, validate with jq

### Risk 2: Memory Spike During Transitions

**Risk**: Loading old and new chunks simultaneously
**Mitigation**: Strict memory limits, sequential chunk unloading

### Risk 3: Manifest Desync

**Risk**: Manifest doesn't match actual chunk files
**Mitigation**: Validate all chunks present before processing, checksums

### Risk 4: Database Connection Pool Exhaustion

**Risk**: Parallel chunks consume all connections
**Mitigation**: Semaphore limiting concurrent chunk processors

## Next Steps

1. Create detailed data model for chunks and manifest
2. Define OpenAPI contracts for chunk management endpoints
3. Write contract tests that will fail without implementation
4. Generate quickstart guide for chunked processing workflow

---
*Research completed for Phase 0 of plan.md*