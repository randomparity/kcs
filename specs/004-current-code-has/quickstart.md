# Quickstart: Enhanced Kernel Entry Point and Symbol Detection

## Prerequisites

- Linux kernel source tree (v5.x or v6.x)
- KCS installed and database initialized
- Optional: compile_commands.json for Clang analysis

## Quick Test Scenarios

### 1. Extract All Entry Points

```bash
# Extract all entry point types from kernel
kcs-extract-entry-points /path/to/kernel --all-types

# Verify results
psql -d kcs -c "SELECT entry_type, COUNT(*) FROM entry_point GROUP BY entry_type;"
```

Expected output:

- syscall: ~400 entries
- ioctl: ~1000+ entries
- file_ops: ~500+ entries
- sysfs: ~2000+ entries
- procfs: ~300+ entries
- Other types with varying counts

### 2. Detect Export Patterns

```bash
# Find all GPL-exported symbols in scheduler
find kernel/sched -name "*.c" | \
  kcs-detect-patterns --pattern-type export_symbol_gpl

# Query results
psql -d kcs -c "
  SELECT s.name, s.metadata->>'export_type'
  FROM symbol s
  WHERE s.metadata->>'export_status' = 'exported'
  LIMIT 10;
"
```

### 3. Enhance Symbols with Clang

```bash
# Generate compile_commands.json if needed
cd /path/to/kernel
make clean
bear -- make -j$(nproc) defconfig all

# Enhance symbols with semantic information
kcs-enhance-symbols --compile-commands compile_commands.json \
  --files fs/read_write.c

# Check enhanced metadata
psql -d kcs -c "
  SELECT name, metadata->>'return_type', metadata->'parameters'
  FROM symbol
  WHERE file_path LIKE '%read_write.c%'
    AND metadata ? 'return_type'
  LIMIT 5;
"
```

### 4. Test Ioctl Detection

```bash
# Extract ioctl handlers from a driver
kcs-extract-entry-points drivers/gpu/drm --type ioctl

# Verify ioctl command detection
psql -d kcs -c "
  SELECT name, metadata->>'ioctl_cmd'
  FROM entry_point
  WHERE entry_type = 'Ioctl'
    AND metadata ? 'ioctl_cmd'
  LIMIT 10;
"
```

### 5. Module Parameter Discovery

```bash
# Detect module parameters
kcs-detect-patterns --pattern-type module_param \
  drivers/net/ethernet/intel/e1000

# Query parameters with descriptions
psql -d kcs -c "
  SELECT
    kp.raw_text,
    kp.metadata->>'param_desc' as description
  FROM kernel_pattern kp
  WHERE kp.pattern_type = 'ModuleParam'
    AND kp.metadata ? 'param_desc'
  LIMIT 5;
"
```

## Integration Test Scenarios

### Scenario 1: Complete Subsystem Analysis

```bash
# Full analysis of a subsystem (e.g., ext4)
./tools/index_kernel.sh --subsystem fs/ext4 /path/to/kernel

# Verify comprehensive detection
psql -d kcs -c "
  SELECT
    'Entry Points' as type, COUNT(*)
  FROM entry_point
  WHERE file_path LIKE '%ext4%'
  UNION ALL
  SELECT
    'Exported Symbols', COUNT(*)
  FROM symbol
  WHERE file_path LIKE '%ext4%'
    AND metadata->>'export_status' = 'exported'
  UNION ALL
  SELECT
    'Module Params', COUNT(*)
  FROM kernel_pattern
  WHERE pattern_type = 'ModuleParam'
    AND file_path LIKE '%ext4%';
"
```

### Scenario 2: MCP Tool Integration

```python
# Test enhanced entrypoint_flow endpoint
curl -X POST http://localhost:8080/mcp/tools/entrypoint_flow \
  -H "Content-Type: application/json" \
  -d '{
    "entry": "IOCTL_TIOCGWINSZ",
    "config": "x86_64:defconfig"
  }'

# Should now return ioctl flow with proper detection
```

### Scenario 3: Performance Validation

```bash
# Benchmark pattern detection
time kcs-detect-patterns --all-types kernel/

# Should complete in <30 seconds for kernel/ directory
# Full kernel should be <20 minutes (constitutional requirement)
```

## Verification Checklist

### Pattern Detection

- [ ] EXPORT_SYMBOL detected (check kernel/printk/printk.c)
- [ ] EXPORT_SYMBOL_GPL detected (check kernel/sched/core.c)
- [ ] module_param detected (check drivers/)
- [ ] MODULE_PARM_DESC associated with params

### Entry Points

- [ ] Ioctl handlers found in drivers/
- [ ] file_operations callbacks identified
- [ ] sysfs DEVICE_ATTR handlers detected
- [ ] procfs proc_create entries found
- [ ] debugfs file operations detected

### Symbol Enhancement

- [ ] Clang types added when compile_commands.json available
- [ ] Function signatures include return types
- [ ] Parameters have name and type
- [ ] Attributes like __init detected

### Database Integration

- [ ] Metadata JSONB columns populated
- [ ] Pattern table has entries
- [ ] Relationships correctly established

## Troubleshooting

### No Clang Enhancement

```bash
# Check clang availability
clang --version

# Verify compile_commands.json
jq '.[0]' compile_commands.json

# Run with debug logging
RUST_LOG=debug kcs-enhance-symbols --compile-commands ...
```

### Missing Patterns

```bash
# Verify regex patterns
kcs-detect-patterns --debug --pattern-type export_symbol test.c

# Check file encoding
file -bi kernel/source.c  # Should be text/x-c

# Test with simple file
echo "EXPORT_SYMBOL(test_func);" > test.c
kcs-detect-patterns test.c
```

### Performance Issues

```bash
# Use parallel processing
kcs-extract-entry-points --parallel --threads 8 /path/to/kernel

# Process subsystems separately
for dir in fs mm kernel drivers; do
  kcs-extract-entry-points /path/to/kernel/$dir &
done
wait
```

## Success Criteria

1. **Coverage**: >90% of kernel entry points detected
2. **Accuracy**: <1% false positive rate for patterns
3. **Performance**: Full kernel index <20 minutes
4. **Enhancement**: >50% symbols enriched when Clang available
5. **Integration**: All MCP endpoints use new data automatically
