# Data Model: Kernel Context Server

**Date**: 2025-09-14
**Feature**: KCS - Kernel Context Server

## Entity Relationship Diagram

```mermaid
erDiagram
    File ||--o{ Symbol : contains
    Symbol ||--o{ CallEdge : calls
    Symbol ||--o{ CallEdge : called_by
    Symbol ||--o{ DependsOnKconfig : requires
    Symbol ||--o{ ModuleSymbol : exported_by
    Symbol ||--o{ EntryPoint : implements
    EntryPoint ||--o{ EntryToImpl : dispatches_to
    Symbol ||--|| Summary : has
    File ||--o{ Ownership : maintained_by
    Symbol ||--o{ TestCoverage : covered_by

    File {
        bigint id PK
        text path
        text sha
        text config
        timestamp indexed_at
    }

    Symbol {
        bigint id PK
        text name
        text kind
        bigint file_id FK
        int start_line
        int end_line
        int start_col
        int end_col
        text config
        byte config_bitmap
        text signature
    }

    EntryPoint {
        bigint id PK
        text kind
        text key
        bigint symbol_id FK
        bigint file_id FK
        jsonb details
        text config
    }

    CallEdge {
        bigint caller_id FK
        bigint callee_id FK
        text config
        byte config_bitmap
        text call_type
        bool is_indirect
    }

    KconfigOption {
        text name PK
        text help
        text type
        text default_value
        jsonb depends_on
    }

    DependsOnKconfig {
        bigint symbol_id FK
        text kconfig_name FK
        text condition_type
    }

    ModuleSymbol {
        text module_name
        bigint symbol_id FK
        text export_type
    }

    Summary {
        bigint symbol_id PK FK
        jsonb content
        timestamp generated_at
        text model_version
    }

    Ownership {
        text path_pattern PK
        jsonb maintainers
        text subsystem
    }

    TestCoverage {
        bigint symbol_id FK
        text test_suite
        text test_name
        text test_type
    }

    DriftReport {
        bigint id PK
        text feature_id
        jsonb mismatches
        timestamp checked_at
        text severity
    }
```

## Core Entities

### File
Represents a source file in the kernel repository.

**Fields**:
- `id`: Unique identifier
- `path`: Relative path from repository root
- `sha`: Git SHA for version tracking
- `config`: Configuration this file was indexed under
- `indexed_at`: Timestamp of last indexing

**Constraints**:
- Unique on (path, config)
- Path must be relative, not absolute
- SHA must be valid git object hash

### Symbol
Represents a function, struct, variable, or other code element.

**Fields**:
- `id`: Unique identifier
- `name`: Symbol name (e.g., "sys_read", "task_struct")
- `kind`: Type of symbol (function, struct, variable, macro, typedef)
- `file_id`: Reference to containing file
- `start_line`, `end_line`: Line span in file
- `start_col`, `end_col`: Column span for precise location
- `config`: Configuration this symbol exists in
- `config_bitmap`: Bitmap of all configs where symbol exists
- `signature`: Function signature or struct definition

**Constraints**:
- Unique on (name, file_id, config)
- Line numbers must be positive
- Kind must be from defined enum

### EntryPoint
Kernel boundary where external requests enter.

**Fields**:
- `id`: Unique identifier
- `kind`: Type (syscall, ioctl, netlink, file_ops, sysfs, proc, debugfs, driver, notifier, bpf)
- `key`: Unique identifier within kind (e.g., "__NR_read", "TCGETS")
- `symbol_id`: Implementing function
- `file_id`: Location of definition
- `details`: Kind-specific metadata (JSON)
- `config`: Configuration context

**Constraints**:
- Unique on (kind, key, config)
- Details schema varies by kind

**Details Schema Examples**:
```json
// Syscall
{
  "syscall_nr": 0,
  "arch": "x86_64",
  "compat": false
}

// Ioctl
{
  "cmd": "0x5401",
  "decoded": "TCGETS",
  "arg_type": "struct termios *"
}

// File operations
{
  "struct_name": "ext4_file_operations",
  "operation": "read"
}
```

## Relationship Entities

### CallEdge
Represents a function call relationship.

**Fields**:
- `caller_id`: Calling function
- `callee_id`: Called function
- `config`: Configuration context
- `config_bitmap`: Configs where edge exists
- `call_type`: direct, indirect, macro, inline
- `is_indirect`: True for function pointers

**Constraints**:
- No self-loops unless recursive
- Config must match symbol configs

### DependsOnKconfig
Links symbols to Kconfig options.

**Fields**:
- `symbol_id`: Symbol reference
- `kconfig_name`: CONFIG_* option name
- `condition_type`: ifdef, if_enabled, depends_on, select

**Constraints**:
- Kconfig_name must start with CONFIG_
- Must reference existing KconfigOption

### ModuleSymbol
Tracks symbol exports and module membership.

**Fields**:
- `module_name`: Kernel module name
- `symbol_id`: Exported symbol
- `export_type`: EXPORT_SYMBOL, EXPORT_SYMBOL_GPL, etc.

**Constraints**:
- Module_name must be valid kernel module
- Export_type from defined enum

## Aggregate Entities

### Summary
AI-generated or rule-based symbol documentation.

**Fields**:
- `symbol_id`: Associated symbol
- `content`: JSON structured summary
- `generated_at`: Creation timestamp
- `model_version`: Generator version

**Content Schema**:
```json
{
  "purpose": "High-level description",
  "inputs": ["param1: description", "param2: description"],
  "outputs": "Return value description",
  "side_effects": ["Effect 1", "Effect 2"],
  "concurrency": {
    "can_sleep": true,
    "locking": ["lock1", "lock2"],
    "rcu": "rcu_read_lock required",
    "irq_safe": false
  },
  "error_paths": ["EINVAL: Invalid input", "ENOMEM: Out of memory"],
  "invariants": ["Precondition 1", "Postcondition 1"],
  "tests": ["test1.c", "test2.c"],
  "citations": [
    {"file": "fs/read_write.c", "line": 451}
  ]
}
```

### DriftReport
Tracks mismatches between spec and implementation.

**Fields**:
- `id`: Unique identifier
- `feature_id`: Feature being checked
- `mismatches`: JSON array of issues
- `checked_at`: Timestamp
- `severity`: blocker, warning, info

**Mismatches Schema**:
```json
[
  {
    "kind": "missing_abi_doc",
    "detail": "sysfs attribute lacks ABI documentation",
    "path": "/sys/kernel/kcs/status",
    "span": {"file": "kernel/kcs.c", "line": 123}
  },
  {
    "kind": "kconfig_mismatch",
    "detail": "Help text doesn't match implementation",
    "option": "CONFIG_KCS_ENABLE",
    "span": {"file": "init/Kconfig", "line": 456}
  }
]
```

## Indexes and Performance

### Primary Indexes
```sql
-- Symbol lookups
CREATE INDEX idx_symbol_name ON symbol(name);
CREATE INDEX idx_symbol_config ON symbol(config);
CREATE INDEX idx_symbol_bitmap ON symbol(config_bitmap);

-- Call graph traversal
CREATE INDEX idx_call_caller ON call_edge(caller_id);
CREATE INDEX idx_call_callee ON call_edge(callee_id);

-- Entry point queries
CREATE INDEX idx_entry_kind ON entrypoint(kind);
CREATE INDEX idx_entry_key ON entrypoint(kind, key);

-- File operations
CREATE INDEX idx_file_path ON file(path);
CREATE INDEX idx_file_config ON file(config);
```

### Specialized Indexes
```sql
-- Full text search on summaries
CREATE INDEX idx_summary_content ON summary USING gin(content);

-- Vector similarity search (pgvector)
CREATE INDEX idx_symbol_embedding ON symbol_embedding USING hnsw(embedding vector_cosine_ops);

-- Bitmap operations
CREATE INDEX idx_config_bitmap ON symbol USING gin(config_bitmap);
```

## Validation Rules

### Symbol Validation
- Names must be valid C identifiers (except macros)
- Line spans must not overlap within same file
- Signature must parse as valid C declaration

### Entry Point Validation
- Syscall numbers must be unique per architecture
- Ioctl commands must decode to valid _IO* macros
- File operations must reference actual function pointers

### Graph Validation
- No orphaned symbols (unreachable from entry points)
- Call edges must connect existing symbols
- Config bitmaps must be consistent across related entities

### Citation Validation
- All citations must reference valid file:line locations
- Line numbers must be within file bounds
- SHA must match indexed version

## Migration Strategy

### Initial Load
1. Create schema with all tables
2. Index one configuration at a time
3. Build call graph incrementally
4. Generate summaries in background
5. Validate against known patterns

### Incremental Updates
1. Detect changed files via git diff
2. Re-index affected symbols
3. Update call edges for changes
4. Regenerate impacted summaries
5. Run drift detection

### Schema Evolution
- Use numbered migration files
- Never modify existing migrations
- Add columns as nullable initially
- Backfill data before making non-null
- Version all JSON schemas

## Storage Estimates

### Size Projections (per configuration)
- Files: ~50K files × 200 bytes = 10 MB
- Symbols: ~500K symbols × 500 bytes = 250 MB
- Call edges: ~2M edges × 50 bytes = 100 MB
- Summaries: ~100K summaries × 2KB = 200 MB
- Indexes: ~300 MB
- **Total per config**: ~860 MB
- **6 configs (3 arch × 2 types)**: ~5.2 GB
- **With overhead and growth**: <20 GB target

### Optimization Opportunities
- Compress summaries with TOAST
- Deduplicate common signatures
- Archive old configurations
- Partition by configuration
- Use columnar storage for analytics

---

*Data model designed for efficient graph traversal, configuration awareness, and citation traceability as required by the constitution.*