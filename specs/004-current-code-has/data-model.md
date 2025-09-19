# Data Model: Enhanced Kernel Entry Point and Symbol Detection

## Overview

Extensions to existing KCS data model to support comprehensive entry point detection and symbol enrichment.

## Entity Extensions

### 1. EntryPoint (Extended)

**Existing Fields**:

- id: UUID
- name: String
- entry_type: EntryType enum
- file_path: String
- line_number: Integer
- signature: String (optional)
- description: String (optional)

**New EntryType Values**:

- Ioctl (existing but unused)
- FileOps (existing)
- Sysfs (existing)
- Procfs (new)
- Debugfs (new)
- Netlink (new)
- NotificationChain (new)
- InterruptHandler (new)
- BootParam (new)

**New Metadata Field** (JSONB):

```json
{
  "export_type": "GPL|non-GPL|NS",       // For EXPORT_SYMBOL
  "namespace": "string",                 // For EXPORT_SYMBOL_NS
  "module": "string",                    // Owning module
  "param_type": "int|string|bool|array", // For module_param
  "param_desc": "string",               // From MODULE_PARM_DESC
  "ioctl_cmd": "0x1234",               // Ioctl magic number
  "irq_number": 15,                    // For interrupt handlers
  "netlink_family": "NETLINK_ROUTE",   // Netlink protocol family
  "ops_struct": "file_operations",     // Parent structure name
  "subsystem": "fs|mm|net|driver"      // Kernel subsystem
}
```

### 2. Symbol (Extended)

**Existing Fields**:

- id: UUID
- name: String
- symbol_type: SymbolType enum
- file_id: UUID (FK)
- start_line: Integer
- end_line: Integer
- signature: String (optional)

**New Metadata Field** (JSONB):

```json
{
  "export_status": "exported|static|global",
  "export_type": "GPL|non-GPL|NS",
  "clang_type": "string",              // From Clang analysis
  "return_type": "string",             // Function return type
  "parameters": [                     // Function parameters
    {"name": "fd", "type": "int"},
    {"name": "buf", "type": "char*"}
  ],
  "attributes": ["__init", "__exit"],  // Kernel attributes
  "config_deps": ["CONFIG_FOO"],      // Config dependencies
  "module_param": true,               // Is module parameter
  "documentation": "string"           // Extracted from comments
}
```

### 3. KernelPattern (New)

**Fields**:

- id: UUID
- pattern_type: PatternType enum
- symbol_id: UUID (FK to Symbol, nullable)
- entrypoint_id: UUID (FK to EntryPoint, nullable)
- file_id: UUID (FK)
- line_number: Integer
- raw_text: String
- metadata: JSONB

**PatternType Enum**:

- ExportSymbol
- ExportSymbolGPL
- ExportSymbolNS
- ModuleParam
- ModuleParamArray
- ModuleParmDesc
- EarlyParam
- CoreParam
- SetupParam

## Relationships

### Existing (unchanged)

- Symbol → File (many-to-one)
- EntryPoint → File (many-to-one)
- CallEdge → Symbol (caller/callee)

### New Relationships

- KernelPattern → Symbol (optional, when pattern exports symbol)
- KernelPattern → EntryPoint (optional, when pattern defines entry)
- Symbol ↔ EntryPoint (many-to-many, via join table)

## State Transitions

### Entry Point Discovery Flow

```
1. File Parsed (Tree-sitter)
   ↓
2. Patterns Detected (Regex)
   ↓
3. Entry Points Created
   ↓
4. Clang Enhancement (optional)
   ↓
5. Metadata Enriched
   ↓
6. Database Persisted
```

### Symbol Enhancement Flow

```
1. Basic Symbol Found (Tree-sitter)
   ↓
2. Export Pattern Matched (Regex)
   ↓
3. Clang Type Retrieved (if available)
   ↓
4. Documentation Extracted
   ↓
5. Enhanced Symbol Stored
```

## Validation Rules

### EntryPoint

- name: Required, non-empty
- entry_type: Must be valid enum value
- file_path: Must exist in indexed kernel
- line_number: Positive integer
- metadata: Valid JSON when present

### Symbol

- name: Required, valid C identifier
- symbol_type: Valid enum value
- start_line <= end_line
- metadata.parameters: Array of objects with name/type

### KernelPattern

- pattern_type: Valid enum value
- Either symbol_id OR entrypoint_id set (not both)
- line_number: Positive integer
- raw_text: Non-empty, contains actual pattern

## Indexes

### Performance Indexes (existing)

- entrypoint(name, entry_type)
- symbol(name, symbol_type)
- symbol(file_id, start_line)

### New Indexes

- entrypoint(metadata->>'export_type') - For finding GPL exports
- symbol(metadata->>'export_status') - For exported symbols
- kernel_pattern(pattern_type) - For pattern queries
- entrypoint(metadata->>'subsystem') - For subsystem filtering

## Migration Strategy

### Backward Compatibility

- All new fields are optional or have defaults
- JSONB metadata columns are nullable
- Existing queries continue to work
- No data migration required

### Schema Migration

```sql
-- Add metadata columns (non-breaking)
ALTER TABLE entrypoint
  ADD COLUMN IF NOT EXISTS metadata JSONB;

ALTER TABLE symbol
  ADD COLUMN IF NOT EXISTS metadata JSONB;

-- Create new pattern table
CREATE TABLE IF NOT EXISTS kernel_pattern (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pattern_type VARCHAR(50) NOT NULL,
  symbol_id UUID REFERENCES symbol(id),
  entrypoint_id UUID REFERENCES entrypoint(id),
  file_id UUID NOT NULL REFERENCES file(id),
  line_number INTEGER NOT NULL,
  raw_text TEXT NOT NULL,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT pattern_reference CHECK (
    (symbol_id IS NOT NULL AND entrypoint_id IS NULL) OR
    (symbol_id IS NULL AND entrypoint_id IS NOT NULL) OR
    (symbol_id IS NULL AND entrypoint_id IS NULL)
  )
);

-- Add new indexes
CREATE INDEX idx_entrypoint_export ON entrypoint((metadata->>'export_type'));
CREATE INDEX idx_symbol_export ON symbol((metadata->>'export_status'));
CREATE INDEX idx_pattern_type ON kernel_pattern(pattern_type);
```

## Query Examples

### Find all GPL-exported symbols

```sql
SELECT s.name, s.signature, s.metadata
FROM symbol s
WHERE s.metadata->>'export_status' = 'exported'
  AND s.metadata->>'export_type' = 'GPL';
```

### Find all ioctl handlers with commands

```sql
SELECT e.name, e.metadata->>'ioctl_cmd' as command
FROM entrypoint e
WHERE e.entry_type = 'Ioctl'
  AND e.metadata ? 'ioctl_cmd';
```

### Find module parameters with descriptions

```sql
SELECT
  kp.raw_text,
  s.name,
  s.metadata->>'param_type' as type,
  kp.metadata->>'description' as description
FROM kernel_pattern kp
JOIN symbol s ON kp.symbol_id = s.id
WHERE kp.pattern_type = 'ModuleParam';
```
