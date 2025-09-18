# Quickstart: Multi-File JSON Chunking

**Feature**: 006-multi-file-json
**Purpose**: Process large kernel indexes as manageable chunks

## Prerequisites

- KCS installed and configured
- PostgreSQL database running
- ~3GB free disk space for chunks
- Linux kernel source available

## Quick Test Scenario

### 1. Generate Chunked Output (5 minutes)

```bash
# Index a kernel with chunking enabled
tools/index_kernel.sh \
  --chunk-size 50MB \
  --output-dir /tmp/kcs-chunks \
  ~/src/linux

# Expected output:
# ✓ Parsing kernel source...
# ✓ Generated 60 chunks in /tmp/kcs-chunks/
# ✓ Manifest: /tmp/kcs-chunks/manifest.json
```

### 2. Verify Chunk Structure (1 minute)

```bash
# Check manifest was created
cat /tmp/kcs-chunks/manifest.json | jq '.total_chunks'
# Expected: 60

# Verify chunk files exist
ls -lh /tmp/kcs-chunks/*.json | head -5
# Expected: Multiple ~50MB JSON files

# Validate a chunk's JSON structure
jq '.chunk_id' /tmp/kcs-chunks/kernel_001.json
# Expected: "kernel_001"
```

### 3. Process Chunks into Database (10 minutes)

```bash
# Load all chunks with parallel processing
tools/process_chunks.py \
  --manifest /tmp/kcs-chunks/manifest.json \
  --parallel 4

# Expected output:
# Processing 60 chunks with parallelism=4
# [####------] 15/60 chunks complete (25%)
# ✓ All chunks processed successfully
```

### 4. Test Resume After Failure (2 minutes)

```bash
# Simulate failure at chunk 15
tools/process_chunks.py \
  --manifest /tmp/kcs-chunks/manifest.json \
  --fail-at-chunk 15

# Expected: Error after chunk 15

# Resume from failure point
tools/process_chunks.py \
  --manifest /tmp/kcs-chunks/manifest.json \
  --resume

# Expected output:
# Resuming from chunk 15 of 60
# ✓ Completed remaining 45 chunks
```

### 5. Verify Database Population (1 minute)

```bash
# Check chunk processing status
curl http://localhost:8080/mcp/chunks/manifest | jq '.total_chunks'
# Expected: 60

# Query a specific chunk status
curl http://localhost:8080/mcp/chunks/kernel_001/status | jq '.status'
# Expected: "completed"

# Verify symbols were loaded
psql -d kcs -c "SELECT COUNT(*) FROM symbol;"
# Expected: ~500,000 rows
```

## Full Workflow Test

### Complete End-to-End Test (20 minutes)

```bash
# Clean previous data
rm -rf /tmp/kcs-chunks
psql -d kcs -c "TRUNCATE symbol, entry_point, chunk_processing;"

# Step 1: Generate chunks with specific configuration
tools/index_kernel.sh \
  --chunk-size 50MB \
  --parallel-chunks 4 \
  --subsystem-split \
  --config x86_64:defconfig \
  ~/src/linux

# Step 2: Validate manifest
python3 -c "
import json
manifest = json.load(open('/tmp/kcs-chunks/manifest.json'))
assert manifest['total_chunks'] > 0
assert all('checksum_sha256' in c for c in manifest['chunks'])
print(f'✓ Manifest valid with {manifest[\"total_chunks\"]} chunks')
"

# Step 3: Process chunks with monitoring
tools/process_chunks.py \
  --manifest /tmp/kcs-chunks/manifest.json \
  --parallel 4 \
  --verbose \
  --progress

# Step 4: Run MCP query to verify
curl -X POST http://localhost:8080/mcp/tools/who_calls \
  -H "Content-Type: application/json" \
  -d '{"function_name": "vfs_read"}' | jq '.callers | length'
# Expected: > 50 callers

# Step 5: Test incremental update
echo "// test change" >> ~/src/linux/fs/read_write.c
tools/index_kernel.sh \
  --incremental \
  --manifest /tmp/kcs-chunks/manifest.json \
  ~/src/linux

# Expected: Only fs/ chunks regenerated
```

## Troubleshooting

### Issue: Out of Memory During Processing

```bash
# Reduce parallelism
tools/process_chunks.py --manifest manifest.json --parallel 1

# Or reduce chunk size for next run
tools/index_kernel.sh --chunk-size 25MB ~/src/linux
```

### Issue: Chunk Processing Fails Repeatedly

```bash
# Check specific chunk error
curl http://localhost:8080/mcp/chunks/kernel_042/status | jq '.error_message'

# Validate chunk file integrity
sha256sum /tmp/kcs-chunks/kernel_042.json
# Compare with manifest checksum

# Force reprocess single chunk
curl -X POST http://localhost:8080/mcp/chunks/kernel_042/process \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

### Issue: Slow Chunk Generation

```bash
# Profile chunk generation
tools/index_kernel.sh \
  --profile \
  --chunk-size 50MB \
  ~/src/linux 2> profile.log

# Check bottlenecks
grep "Chunk write time" profile.log | sort -n
```

## Performance Validation

### Metrics to Verify

1. **Chunk Generation**: Should complete in < 10 minutes
2. **Memory Usage**: Should stay under 500MB per process
3. **Database Loading**: ~30 seconds per chunk
4. **Query Performance**: < 600ms p95 after loading

### Benchmark Commands

```bash
# Measure chunk generation time
time tools/index_kernel.sh --chunk-size 50MB ~/src/linux

# Monitor memory during processing
tools/process_chunks.py --manifest manifest.json &
PID=$!
while ps -p $PID > /dev/null; do
  ps -o rss= -p $PID | awk '{print $1/1024 " MB"}'
  sleep 5
done

# Test query performance
for i in {1..100}; do
  time curl -s http://localhost:8080/mcp/tools/who_calls \
    -d '{"function_name": "vfs_read"}' > /dev/null
done | awk '{sum+=$1; count++} END {print "Avg:", sum/count}'
```

## Success Criteria

✓ All chunks generate successfully
✓ Manifest contains correct metadata
✓ Chunks process without memory errors
✓ Resume works after simulated failure
✓ Database queries return correct results
✓ Performance meets targets (< 30 min total)

---
*Complete this quickstart to validate the chunking implementation*
