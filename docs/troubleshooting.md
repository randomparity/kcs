# KCS Troubleshooting Guide

This guide covers common issues and solutions for Kernel Context Server (KCS),
with special focus on chunk processing failures and multi-file JSON workflows.

## Table of Contents

- [Chunk Processing Issues](#chunk-processing-issues)
- [Database Connection Problems](#database-connection-problems)
- [Memory and Performance Issues](#memory-and-performance-issues)
- [Indexing Failures](#indexing-failures)
- [MCP Server Issues](#mcp-server-issues)
- [Development Environment Problems](#development-environment-problems)

## Chunk Processing Issues

### Chunk Generation Failures

**Problem**: `index_kernel.sh` fails to generate chunks or produces incomplete chunks.

**Common Causes and Solutions**:

1. **Insufficient Disk Space**

   ```bash
   # Check available space
   df -h /tmp/kcs-chunks

   # Solution: Use different output directory with more space
   ./tools/index_kernel.sh \
     --chunk-size 50MB \
     --output-dir /var/cache/kcs \
     ~/src/linux
   ```

2. **Memory Exhaustion During Parsing**

   ```bash
   # Symptoms: Process killed by OOM killer
   dmesg | grep -i "killed process"

   # Solution: Use smaller chunk size
   ./tools/index_kernel.sh \
     --chunk-size 25MB \
     --parallel-chunks 2 \
     ~/src/linux
   ```

3. **Permission Issues**

   ```bash
   # Check output directory permissions
   ls -ld /tmp/kcs-chunks

   # Fix permissions
   sudo chown -R $USER:$USER /tmp/kcs-chunks
   chmod -R 755 /tmp/kcs-chunks
   ```

4. **Corrupted Kernel Source**

   ```bash
   # Verify kernel source integrity
   find ~/src/linux -name "*.c" | head -10 | xargs file

   # Re-clone if necessary
   git clone https://github.com/torvalds/linux.git ~/src/linux-clean
   ```

### Chunk Processing Database Failures

**Problem**: `process_chunks.py` fails to process chunks into database.

**Common Causes and Solutions**:

1. **Chunk Checksum Verification Failures**

   ```bash
   # Check chunk integrity
   sha256sum /tmp/kcs-chunks/kernel_001.json

   # Compare with manifest
   jq '.chunks[] | select(.id=="kernel_001") | .checksum_sha256' \
     /tmp/kcs-chunks/manifest.json

   # Solution: Regenerate corrupted chunks
   rm /tmp/kcs-chunks/kernel_001.json
   ./tools/index_kernel.sh --incremental \
     --manifest /tmp/kcs-chunks/manifest.json \
     ~/src/linux
   ```

2. **Database Transaction Timeouts**

   ```bash
   # Symptoms in logs
   grep -i "timeout\|deadlock" /var/log/kcs/process_chunks.log

   # Solution: Reduce parallelism and batch size
   ./tools/process_chunks.py \
     --manifest /tmp/kcs-chunks/manifest.json \
     --parallel 2 \
     --batch-size 5
   ```

3. **JSON Parsing Errors**

   ```bash
   # Validate chunk JSON structure
   jq . /tmp/kcs-chunks/kernel_001.json > /dev/null

   # If invalid, check for truncation
   tail -5 /tmp/kcs-chunks/kernel_001.json

   # Solution: Regenerate affected chunks
   ./tools/process_chunks.py \
     --manifest /tmp/kcs-chunks/manifest.json \
     --chunk-ids kernel_001 \
     --force-regenerate
   ```

4. **Chunk Size Violations**

   ```bash
   # Check for oversized chunks
   find /tmp/kcs-chunks -name "*.json" -size +50M

   # Solution: Use constitutional limit enforcement
   ./tools/index_kernel.sh \
     --chunk-size 50MB \
     --strict-size-limit \
     ~/src/linux
   ```

### Resume Processing Issues

**Problem**: Resume functionality doesn't work correctly after failures.

**Diagnostic Steps**:

1. **Check Processing Status Table**

   ```sql
   -- Connect to database
   psql -d kcs

   -- Check chunk processing status
   SELECT chunk_id, status, error_message, retry_count
   FROM chunk_processing_status
   WHERE status IN ('failed', 'processing')
   ORDER BY chunk_id;
   ```

2. **Identify Stuck Chunks**

   ```sql
   -- Find chunks stuck in processing state
   SELECT chunk_id, started_at,
          EXTRACT(EPOCH FROM (NOW() - started_at)) as seconds_stuck
   FROM chunk_processing_status
   WHERE status = 'processing'
     AND started_at < NOW() - INTERVAL '30 minutes';
   ```

3. **Reset Failed Chunks**

   ```bash
   # Reset specific failed chunks
   ./tools/process_chunks.py \
     --manifest /tmp/kcs-chunks/manifest.json \
     --reset-failed \
     --chunk-ids kernel_015,kernel_023

   # Resume processing
   ./tools/process_chunks.py \
     --manifest /tmp/kcs-chunks/manifest.json \
     --resume
   ```

## Database Connection Problems

### Connection Pool Exhaustion

**Problem**: "Too many connections" or "Connection pool exhausted" errors.

**Solutions**:

1. **Check Current Connections**

   ```sql
   SELECT count(*) as active_connections
   FROM pg_stat_activity
   WHERE datname = 'kcs' AND state = 'active';
   ```

2. **Adjust Connection Pool Settings**

   ```bash
   # Set environment variables
   export DB_POOL_MIN_SIZE=2
   export DB_POOL_MAX_SIZE=8

   # Or modify configuration
   vim src/python/kcs_mcp/database.py
   # Adjust pool size parameters
   ```

3. **Kill Stuck Connections**

   ```sql
   -- Identify long-running queries
   SELECT pid, query, state, query_start
   FROM pg_stat_activity
   WHERE datname = 'kcs'
     AND query_start < NOW() - INTERVAL '10 minutes';

   -- Kill stuck connections (careful!)
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'kcs'
     AND state = 'idle in transaction'
     AND query_start < NOW() - INTERVAL '30 minutes';
   ```

### Database Schema Issues

**Problem**: Schema migration failures or missing tables.

**Solutions**:

1. **Check Applied Migrations**

   ```sql
   -- Check if migration table exists
   \dt migrations

   -- Check applied migrations
   SELECT * FROM migrations ORDER BY applied_at;
   ```

2. **Apply Missing Migrations**

   ```bash
   # Apply all migrations in order
   for migration in src/sql/migrations/*.sql; do
     echo "Applying $migration"
     psql -d kcs -f "$migration"
   done
   ```

3. **Verify Chunk Processing Tables**

   ```sql
   -- Check required tables exist
   \dt chunk_*

   -- Verify table structure
   \d chunk_processing_status
   \d indexing_manifest
   ```

## Memory and Performance Issues

### High Memory Usage During Processing

**Problem**: Process consuming excessive memory during chunk operations.

**Diagnostic Steps**:

1. **Monitor Memory Usage**

   ```bash
   # Real-time memory monitoring
   watch -n 1 'ps aux | grep -E "(process_chunks|index_kernel)" | head -10'

   # Check system memory
   free -h
   cat /proc/meminfo | grep -E "(MemTotal|MemAvailable|SwapTotal)"
   ```

2. **Profile Memory Usage**

   ```bash
   # Run with memory profiling
   python -m memory_profiler tools/process_chunks.py \
     --manifest /tmp/kcs-chunks/manifest.json \
     --parallel 1
   ```

**Solutions**:

1. **Reduce Memory Pressure**

   ```bash
   # Use smaller chunks
   ./tools/index_kernel.sh \
     --chunk-size 10MB \
     --parallel-chunks 1 \
     ~/src/linux

   # Process with minimal parallelism
   ./tools/process_chunks.py \
     --manifest /tmp/kcs-chunks/manifest.json \
     --parallel 1 \
     --batch-size 1
   ```

2. **Enable Memory Monitoring**

   ```bash
   # Run performance tests to verify limits
   pytest tests/performance/test_memory_usage.py -v
   ```

### Slow Chunk Processing Performance

**Problem**: Chunk processing taking longer than expected.

**Diagnostic Tools**:

1. **Database Query Performance**

   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_statement = 'all';
   ALTER SYSTEM SET log_min_duration_statement = 1000;
   SELECT pg_reload_conf();

   -- Check slow queries
   tail -f /var/log/postgresql/postgresql-14-main.log | grep "duration:"
   ```

2. **Connection Pool Metrics**

   ```python
   # Add to debugging code
   import asyncio
   from kcs_mcp.database import get_connection_pool

   async def check_pool():
       pool = await get_connection_pool()
       print(f"Pool size: {pool.get_size()}")
       print(f"Available: {pool.get_idle_size()}")

   asyncio.run(check_pool())
   ```

**Performance Tuning**:

1. **Optimize Database Configuration**

   ```sql
   -- Increase work_mem for complex queries
   ALTER SYSTEM SET work_mem = '256MB';

   -- Tune checkpoint settings
   ALTER SYSTEM SET checkpoint_completion_target = 0.9;
   ALTER SYSTEM SET wal_buffers = '16MB';

   -- Reload configuration
   SELECT pg_reload_conf();
   ```

2. **Parallel Processing Optimization**

   ```bash
   # Find optimal parallelism for your system
   for parallel in 1 2 4 8; do
     echo "Testing parallelism: $parallel"
     time ./tools/process_chunks.py \
       --manifest /tmp/test-chunks/manifest.json \
       --parallel $parallel \
       --dry-run
   done
   ```

## Indexing Failures

### Tree-sitter Parsing Errors

**Problem**: Rust parser fails to parse C source files.

**Solutions**:

1. **Check Tree-sitter Installation**

   ```bash
   # Verify tree-sitter-c is available
   find ~/.cargo -name "*tree_sitter*" 2>/dev/null

   # Reinstall if necessary
   cd src/rust/kcs-parser
   cargo clean && cargo build --release
   ```

2. **Test Parsing Individual Files**

   ```bash
   # Test problematic files directly
   cd src/rust/kcs-parser
   echo "fn test() {}" | cargo run -- --stdin --format json
   ```

3. **Handle Preprocessor Issues**

   ```bash
   # Use preprocessed files for complex macros
   ./tools/index_kernel.sh \
     --preprocess \
     --subsystem drivers/simple \
     ~/src/linux
   ```

### Clang Integration Problems

**Problem**: Enhanced parsing with Clang fails.

**Solutions**:

1. **Check Clang Installation**

   ```bash
   # Verify clang is available
   which clang
   clang --version

   # Check for libclang
   find /usr -name "libclang*" 2>/dev/null
   ```

2. **Fallback to Tree-sitter Only**

   ```bash
   # Disable Clang integration
   export KCS_DISABLE_CLANG=1
   ./tools/index_kernel.sh ~/src/linux
   ```

3. **Configure Clang Paths**

   ```bash
   # Set explicit paths
   export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu/libclang.so.1
   export CLANG_PATH=/usr/bin/clang-14
   ```

## MCP Server Issues

### Server Startup Failures

**Problem**: KCS MCP server fails to start.

**Diagnostic Steps**:

1. **Check Port Availability**

   ```bash
   # Check if port is in use
   netstat -tlnp | grep :8080

   # Kill existing processes
   sudo fuser -k 8080/tcp
   ```

2. **Verify Database Connectivity**

   ```bash
   # Test database connection
   psql -d kcs -c "SELECT 1;"

   # Check environment variables
   echo $DATABASE_URL
   ```

3. **Check Dependencies**

   ```bash
   # Verify Python environment
   source .venv/bin/activate
   python -c "import kcs_mcp; print('OK')"

   # Check installed packages
   pip list | grep -E "(asyncpg|fastapi|pydantic)"
   ```

**Solutions**:

1. **Start with Debug Logging**

   ```bash
   # Enable debug mode
   export PYTHONPATH=/home/dave/src/kcs/src/python
   export LOG_LEVEL=DEBUG

   python -m kcs_mcp.cli --host 0.0.0.0 --port 8080
   ```

2. **Use Docker for Consistent Environment**

   ```bash
   # Start via Docker Compose
   docker compose -f tools/docker/docker-compose.yml up kcs
   ```

### Chunk API Endpoint Failures

**Problem**: Chunk-related MCP endpoints returning errors.

**Diagnostic Steps**:

1. **Test Endpoints Manually**

   ```bash
   # Check health endpoint
   curl http://localhost:8080/health

   # Test chunk manifest endpoint
   curl -X GET "http://localhost:8080/mcp/chunks/manifest" \
     -H "accept: application/json"
   ```

2. **Check Database State**

   ```sql
   -- Verify chunk tracking tables
   SELECT COUNT(*) FROM indexing_manifest;
   SELECT COUNT(*) FROM chunk_processing_status;
   ```

## Development Environment Problems

### Virtual Environment Issues

**Problem**: Python dependencies or virtual environment corruption.

**Solutions**:

1. **Recreate Virtual Environment**

   ```bash
   # Remove existing environment
   rm -rf .venv

   # Recreate with make setup
   make setup

   # Or manually
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,performance]"
   ```

2. **Check Python Version Compatibility**

   ```bash
   # Verify Python version
   python --version  # Should be 3.10+

   # Check for conflicting packages
   pip check
   ```

### Rust Build Issues

**Problem**: Rust components fail to build.

**Solutions**:

1. **Update Rust Toolchain**

   ```bash
   # Update Rust
   rustup update stable
   rustup default stable

   # Check version
   rustc --version  # Should be 1.70+
   ```

2. **Clean and Rebuild**

   ```bash
   cd src/rust
   cargo clean
   cargo build --release --workspace
   ```

3. **Check Cargo Dependencies**

   ```bash
   # Update dependencies
   cd src/rust
   cargo update

   # Check for conflicts
   cargo tree --duplicates
   ```

## Monitoring and Logging

### Enable Comprehensive Logging

1. **Python Application Logging**

   ```bash
   # Set log level
   export LOG_LEVEL=DEBUG
   export PYTHONPATH=/home/dave/src/kcs/src/python

   # Log to file
   python -m kcs_mcp.cli 2>&1 | tee kcs-server.log
   ```

2. **Database Query Logging**

   ```sql
   -- Enable slow query logging
   ALTER SYSTEM SET log_min_duration_statement = 500;
   ALTER SYSTEM SET log_statement = 'mod';
   SELECT pg_reload_conf();
   ```

3. **Rust Component Logging**

   ```bash
   # Enable Rust debug logging
   export RUST_LOG=debug
   export RUST_BACKTRACE=1

   ./tools/index_kernel.sh ~/src/linux
   ```

### Performance Monitoring

1. **Run Performance Test Suite**

   ```bash
   # Run all performance tests
   pytest tests/performance/ -v --tb=short

   # Run specific chunk tests
   pytest tests/performance/test_chunk_generation.py::TestChunkGeneration::test_50mb_chunk_generation_performance -v
   ```

2. **Monitor System Resources**

   ```bash
   # Continuous monitoring
   htop

   # I/O monitoring
   iotop -a

   # Database connections
   watch -n 5 'psql -d kcs -c "SELECT count(*) FROM pg_stat_activity WHERE datname = '\''kcs'\'';"'
   ```

## Emergency Recovery Procedures

### Complete Reset

If all else fails, these steps will reset the system to a clean state:

1. **Clean Database**

   ```sql
   -- Drop and recreate database
   DROP DATABASE kcs;
   CREATE DATABASE kcs OWNER kcs;

   -- Apply all migrations
   \c kcs
   \i src/sql/migrations/001_initial_schema.sql
   -- ... apply remaining migrations
   ```

2. **Clean File System**

   ```bash
   # Remove all generated chunks
   rm -rf /tmp/kcs-chunks
   rm -rf /var/cache/kcs

   # Clean build artifacts
   cd src/rust && cargo clean
   rm -rf .venv
   ```

3. **Rebuild Everything**

   ```bash
   # Full rebuild
   make clean
   make setup
   make build-rust
   make test
   ```

## Getting Help

If problems persist after following this guide:

1. **Check Project Issues**: <https://github.com/anthropics/claude-code/issues>
2. **Enable Debug Logging**: Set `LOG_LEVEL=DEBUG` for all components
3. **Gather System Information**:

   ```bash
   # System info
   uname -a
   lsb_release -a
   python --version
   rustc --version
   psql --version

   # Resource usage
   free -h
   df -h
   ```

4. **Collect Relevant Logs**:
   - Application logs: `/var/log/kcs/`
   - PostgreSQL logs: `/var/log/postgresql/`
   - System logs: `journalctl -u kcs`

5. **Performance Metrics**:

   ```bash
   # Run diagnostic script
   ./tools/run_system_test.py --diagnostic
   ```
