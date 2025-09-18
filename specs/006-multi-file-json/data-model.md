# Data Model: Multi-File JSON Output

**Feature**: 006-multi-file-json
**Date**: 2025-01-18

## Entity Definitions

### ChunkManifest

Metadata file describing all chunks generated from a kernel indexing operation.

**Fields**:
- `version`: String - Manifest schema version (e.g., "1.0.0")
- `created`: DateTime - When manifest was generated
- `kernel_version`: String - Linux kernel version indexed
- `kernel_path`: String - Absolute path to kernel source
- `config`: String - Kernel configuration (e.g., "x86_64:defconfig")
- `total_chunks`: Integer - Number of chunk files generated
- `total_size_bytes`: Integer - Combined size of all chunks
- `chunks`: Array[ChunkMetadata] - List of all chunks

**Validation**:
- version must match supported schema versions
- created must be valid ISO 8601 timestamp
- total_chunks must equal length of chunks array
- kernel_path must be absolute path

### ChunkMetadata

Metadata for an individual chunk file within the manifest.

**Fields**:
- `id`: String - Unique chunk identifier (e.g., "kernel_001")
- `sequence`: Integer - Order within subsystem (1-based)
- `file`: String - Relative path to chunk file
- `subsystem`: String - Kernel subsystem (e.g., "kernel", "drivers/net")
- `size_bytes`: Integer - File size in bytes
- `checksum_sha256`: String - SHA256 hash of file contents
- `symbol_count`: Integer - Number of symbols in chunk
- `entry_point_count`: Integer - Number of entry points
- `file_count`: Integer - Number of source files represented

**Validation**:
- id must be unique within manifest
- size_bytes must not exceed configured chunk_size limit
- checksum must be 64 character hex string
- counts must be non-negative

### ChunkFile

The actual JSON file containing parsed kernel data.

**Fields**:
- `manifest_version`: String - Version of manifest this belongs to
- `chunk_id`: String - Matches id in manifest
- `subsystem`: String - Kernel subsystem
- `symbols`: Array[Symbol] - Parsed symbols (existing format)
- `entry_points`: Array[EntryPoint] - Entry points (existing format)
- `call_graph`: Array[Edge] - Call relationships (if present)

**Validation**:
- manifest_version must match parent manifest
- chunk_id must exist in manifest
- JSON must be valid and complete

### ProcessingStatus

Database record tracking chunk processing state.

**Fields**:
- `chunk_id`: String (Primary Key) - Chunk identifier
- `manifest_version`: String - Version of manifest
- `status`: Enum - One of: pending, processing, completed, failed
- `started_at`: DateTime - When processing began
- `completed_at`: DateTime - When processing finished (nullable)
- `error_message`: Text - Error details if failed (nullable)
- `retry_count`: Integer - Number of retry attempts
- `symbols_processed`: Integer - Count of symbols inserted
- `checksum_verified`: Boolean - Whether checksum matched

**Validation**:
- status transitions: pending → processing → completed/failed
- completed_at required when status is completed/failed
- error_message required when status is failed
- retry_count must not exceed max_retries (3)

## Relationships

```
ChunkManifest (1) ─── contains ──→ (*) ChunkMetadata
                                            │
                                            │ references
                                            ↓
ChunkMetadata (1) ─── describes ──→ (1) ChunkFile
                                            │
                                            │ tracks
                                            ↓
ChunkFile (1) ←─── processes ─── (1) ProcessingStatus
```

## State Transitions

### ProcessingStatus States

```
    ┌─────────┐
    │ pending │ ← (initial state)
    └────┬────┘
         │ start processing
         ↓
    ┌────────────┐
    │ processing │
    └─────┬──────┘
          │
     ┌────┴────┐
     ↓         ↓
┌──────────┐  ┌────────┐
│completed │  │ failed │
└──────────┘  └───┬────┘
                  │ retry
                  ↓
              [pending]
```

## Constraints

### Business Rules

1. **Chunk Size Limit**: No chunk file may exceed configured max_chunk_size (default 50MB)
2. **Subsystem Grouping**: Symbols from same subsystem should be in same chunk when possible
3. **Atomic Processing**: Each chunk must be processed completely or not at all
4. **Checksum Validation**: Chunk processing must verify SHA256 before insertion
5. **Retry Limit**: Failed chunks may retry up to 3 times before permanent failure
6. **Manifest Immutability**: Once created, manifest version is immutable

### Data Integrity

1. **Foreign Keys**:
   - ProcessingStatus.chunk_id must exist in current manifest
   - ChunkFile.chunk_id must match ChunkMetadata.id

2. **Uniqueness**:
   - ChunkMetadata.id unique within manifest
   - ProcessingStatus.chunk_id unique in database

3. **Completeness**:
   - All chunks in manifest must have corresponding files
   - Sum of all chunk sizes must equal total_size_bytes

## Database Schema Updates

```sql
-- Migration 012: Add chunk processing tracking
CREATE TABLE IF NOT EXISTS chunk_processing (
    chunk_id VARCHAR(255) PRIMARY KEY,
    manifest_version VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    symbols_processed INTEGER DEFAULT 0,
    checksum_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT status_check CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);

CREATE INDEX idx_chunk_processing_status ON chunk_processing(status);
CREATE INDEX idx_chunk_processing_manifest ON chunk_processing(manifest_version);

-- Add manifest tracking
CREATE TABLE IF NOT EXISTS indexing_manifest (
    version VARCHAR(50) PRIMARY KEY,
    created TIMESTAMP NOT NULL,
    kernel_version VARCHAR(100),
    kernel_path TEXT,
    config VARCHAR(100),
    total_chunks INTEGER NOT NULL,
    total_size_bytes BIGINT,
    manifest_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## JSON Schemas

### Manifest Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "created", "total_chunks", "chunks"],
  "properties": {
    "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
    "created": {"type": "string", "format": "date-time"},
    "kernel_version": {"type": "string"},
    "total_chunks": {"type": "integer", "minimum": 1},
    "chunks": {
      "type": "array",
      "items": {"$ref": "#/definitions/chunkMetadata"}
    }
  }
}
```

---
*Data model defined for Phase 1 of plan.md*