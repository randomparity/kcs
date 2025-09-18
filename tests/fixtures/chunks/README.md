# Chunk Test Fixtures

This directory contains sample chunk files for testing the multi-file JSON output feature.

## File Structure

- `sample_manifest.json` - Example manifest file with chunk metadata
- `sample_kernel_001.json` - Sample chunk from kernel subsystem
- `sample_drivers_net_001.json` - Sample chunk from drivers/net subsystem
- `sample_corrupt.json` - Intentionally malformed chunk for error testing

## Usage

These fixtures are used by:

- Contract tests for chunk API endpoints
- Integration tests for chunk processing
- Unit tests for chunk validation

## Chunk Format

Each chunk file follows the schema:

```json
{
  "manifest_version": "1.0.0",
  "chunk_id": "kernel_001",
  "subsystem": "kernel",
  "symbols": [...],
  "entry_points": [...],
  "call_graph": [...]
}
```

## Manifest Format

The manifest file contains metadata about all chunks:

```json
{
  "version": "1.0.0",
  "created": "2025-01-18T10:00:00Z",
  "kernel_version": "6.7.0",
  "total_chunks": 3,
  "chunks": [...]
}
```
