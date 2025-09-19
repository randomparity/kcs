# Research: Enhanced Kernel Entry Point and Symbol Detection

## Current Implementation Status

### Entry Points - Partially Implemented

**Decision**: Enhance existing partial implementations rather than rewrite
**Rationale**: Basic structure exists in `kcs-extractor/src/entrypoints.rs`
**Current Status**:

- ✅ file_operations extraction (basic regex pattern)
- ✅ sysfs attributes extraction (DEVICE_ATTR patterns)
- ✅ module init/exit functions
- ❌ ioctl handlers (stub exists in ioctls.rs)
- ❌ procfs entry points
- ❌ debugfs interfaces
- ❌ netlink handlers
- ❌ notification chains
- ❌ interrupt handlers

### Pattern Recognition - Not Implemented

**Decision**: Implement in new `kernel_patterns.rs` module
**Rationale**: File exists but only contains TODO comment
**Required Patterns**:

- EXPORT_SYMBOL/EXPORT_SYMBOL_GPL/EXPORT_SYMBOL_NS
- module_param/module_param_array
- MODULE_PARM_DESC
- __setup (boot parameters)
- early_param
- core_param

### Clang Integration - Stub Only

**Decision**: Implement using clang-sys crate
**Rationale**: ClangBridge structure exists but not implemented
**Current Status**:

- Dependency already in Cargo.toml
- Basic structure in clang_bridge.rs
- TODO comments indicate planned functionality
- No actual clang index initialization
- No compile_commands.json parsing

## Technical Decisions

### 1. Pattern Detection Strategy

**Decision**: Hybrid approach - Tree-sitter for structure, regex for simple patterns
**Rationale**:

- Tree-sitter excels at structural parsing (functions, structs)
- Regex efficient for macro patterns (EXPORT_SYMBOL, module_param)
- Clang provides semantic information when available
**Alternatives Considered**:
- Pure tree-sitter: Too complex for macro patterns
- Pure regex: Misses structural context
- Pure Clang: Requires compile_commands.json, not always available

### 2. Entry Point Detection Order

**Decision**: Progressive enhancement - basic patterns first, Clang enrichment optional
**Rationale**:

- Works without compile_commands.json
- Graceful degradation when Clang unavailable
- Faster initial results
**Implementation Order**:

1. Regex patterns for macros (fast)
2. Tree-sitter for structures (accurate)
3. Clang for types/signatures (when available)

### 3. Database Schema

**Decision**: Extend existing entrypoint table, add metadata JSONB column
**Rationale**:

- No breaking schema changes
- Backward compatible
- Flexible metadata storage
**Metadata Fields**:
- export_type (GPL/non-GPL/NS)
- module_name
- param_type (for module_param)
- interrupt_number (for IRQ handlers)
- ioctl_cmd (magic number)

### 4. Performance Optimization

**Decision**: Streaming parser with parallel file processing
**Rationale**:

- Already implemented in extract_entrypoints_streaming.py
- Memory efficient for large kernels
- Can leverage multiple cores
**Optimizations**:
- File-level parallelism (rayon)
- Pattern compilation caching
- Incremental results via channels

### 5. Testing Strategy

**Decision**: Use kernel test fixtures from existing tests
**Rationale**:

- tests/fixtures/kernel/ already contains sample code
- Real kernel patterns for accuracy
- Fast test execution
**Test Categories**:
- Pattern detection accuracy
- Entry point classification
- Clang enhancement validation
- Performance benchmarks

## Integration Points

### Existing Tools to Enhance

1. **extract-entry-points CLI** (`kcs-extractor/src/main.rs`)
   - Add new entry point types to output
   - Support --pattern-type flag for selective extraction

2. **Python streaming wrapper** (`tools/extract_entrypoints_streaming.py`)
   - Already handles streaming output
   - Just needs new entry types in protocol

3. **Database ingestion** (`src/python/kcs_mcp/database.py`)
   - insert_entrypoints() method exists
   - Add metadata field handling

4. **MCP endpoints** (`src/python/kcs_mcp/tools.py`)
   - entrypoint_flow already searches entry points
   - Will automatically benefit from new data

## Risk Assessment

### Technical Risks

1. **Clang initialization complexity**
   - Mitigation: Make Clang optional, work without it
   - Fallback: Tree-sitter + regex patterns

2. **Pattern false positives**
   - Mitigation: Conservative patterns, validate with tests
   - Use negative lookahead for common false matches

3. **Performance regression**
   - Mitigation: Benchmark before/after
   - Use streaming and parallelism

### Compatibility Risks

1. **Database migration**
   - Mitigation: JSONB metadata column is additive only
   - No existing data modification needed

2. **API changes**
   - Mitigation: New fields are optional
   - Existing queries continue to work

## Dependencies

### Required (already present)

- tree-sitter (0.20+)
- tree-sitter-c
- clang-sys (1.6+)
- regex (1.10+)
- rayon (1.7+)

### Optional Enhancements

- tree-sitter-rust (for Rust kernel modules)
- nom (for complex parsing if needed)

## Next Steps

1. Contract definition for new entry point types
2. Test fixtures for each pattern type
3. Incremental implementation per pattern
4. Integration tests with real kernel
5. Performance validation against targets
