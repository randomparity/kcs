-- KCS Graph Tables Migration
-- Creates entry points, call graphs, and configuration dependencies

-- Entry point types
CREATE TYPE entrypoint_kind AS ENUM (
    'syscall',
    'ioctl',
    'netlink',
    'file_ops',
    'sysfs',
    'proc',
    'debugfs',
    'driver',
    'notifier',
    'bpf',
    'tracepoint',
    'module_init',
    'module_exit'
);

-- Call edge types
CREATE TYPE call_type AS ENUM (
    'direct',
    'indirect',
    'macro',
    'inline',
    'function_pointer',
    'vtable'
);

-- Kconfig condition types
CREATE TYPE kconfig_condition AS ENUM (
    'ifdef',
    'if_enabled',
    'depends_on',
    'select',
    'imply'
);

-- Entry points - kernel boundaries where external requests enter
CREATE TABLE entrypoint (
    id BIGSERIAL PRIMARY KEY,
    kind entrypoint_kind NOT NULL,
    key TEXT NOT NULL, -- Unique identifier within kind (e.g., "__NR_read", "TCGETS")
    symbol_id BIGINT REFERENCES symbol(id) ON DELETE CASCADE,
    file_id BIGINT NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    details JSONB, -- Kind-specific metadata
    config TEXT NOT NULL,

    -- Constraints
    CONSTRAINT entrypoint_key_not_empty CHECK (length(key) > 0),
    CONSTRAINT entrypoint_unique_per_config UNIQUE (kind, key, config)
);

-- Call edges - function call relationships
CREATE TABLE call_edge (
    caller_id BIGINT NOT NULL REFERENCES symbol(id) ON DELETE CASCADE,
    callee_id BIGINT NOT NULL REFERENCES symbol(id) ON DELETE CASCADE,
    config TEXT NOT NULL,
    config_bitmap BYTEA, -- Configs where edge exists
    call_type call_type NOT NULL DEFAULT 'direct',
    is_indirect BOOLEAN NOT NULL DEFAULT FALSE,
    call_site_line INTEGER, -- Line number of call site

    -- Constraints
    CONSTRAINT call_edge_no_self_loop CHECK (caller_id != callee_id OR call_type = 'direct'), -- Allow recursive calls
    PRIMARY KEY (caller_id, callee_id, config)
);

-- Kconfig options - kernel configuration system
CREATE TABLE kconfig_option (
    name TEXT PRIMARY KEY,
    help TEXT,
    type TEXT, -- bool, tristate, string, int, hex
    default_value TEXT,
    depends_on JSONB, -- Complex dependency expressions
    prompt TEXT,
    source_file TEXT,
    source_line INTEGER,

    -- Constraints
    CONSTRAINT kconfig_name_format CHECK (name ~ '^CONFIG_[A-Z0-9_]+$')
);

-- Symbol dependencies on Kconfig options
CREATE TABLE depends_on_kconfig (
    symbol_id BIGINT NOT NULL REFERENCES symbol(id) ON DELETE CASCADE,
    kconfig_name TEXT NOT NULL REFERENCES kconfig_option(name) ON DELETE CASCADE,
    condition_type kconfig_condition NOT NULL,

    PRIMARY KEY (symbol_id, kconfig_name, condition_type)
);

-- Module symbol exports
CREATE TABLE module_symbol (
    module_name TEXT NOT NULL,
    symbol_id BIGINT NOT NULL REFERENCES symbol(id) ON DELETE CASCADE,
    export_type TEXT NOT NULL, -- EXPORT_SYMBOL, EXPORT_SYMBOL_GPL, etc.

    -- Constraints
    CONSTRAINT module_name_not_empty CHECK (length(module_name) > 0),
    CONSTRAINT export_type_valid CHECK (export_type ~ '^EXPORT_SYMBOL'),
    PRIMARY KEY (module_name, symbol_id)
);

-- Entry point to implementation mapping (many-to-many)
CREATE TABLE entry_to_impl (
    entry_id BIGINT NOT NULL REFERENCES entrypoint(id) ON DELETE CASCADE,
    symbol_id BIGINT NOT NULL REFERENCES symbol(id) ON DELETE CASCADE,
    dispatch_order INTEGER DEFAULT 0, -- For ordered dispatch chains

    PRIMARY KEY (entry_id, symbol_id)
);

-- File ownership/maintainership patterns
CREATE TABLE ownership (
    path_pattern TEXT PRIMARY KEY,
    maintainers JSONB NOT NULL, -- Array of maintainer objects
    subsystem TEXT,
    status TEXT, -- Maintained, Orphan, Obsolete, etc.

    -- Constraints
    CONSTRAINT ownership_maintainers_not_empty CHECK (jsonb_array_length(maintainers) > 0)
);

-- Basic indexes for graph traversal
CREATE INDEX idx_entrypoint_kind ON entrypoint(kind);
CREATE INDEX idx_entrypoint_key ON entrypoint(kind, key);
CREATE INDEX idx_entrypoint_symbol_id ON entrypoint(symbol_id);
CREATE INDEX idx_entrypoint_config ON entrypoint(config);

CREATE INDEX idx_call_edge_caller ON call_edge(caller_id);
CREATE INDEX idx_call_edge_callee ON call_edge(callee_id);
CREATE INDEX idx_call_edge_config ON call_edge(config);
CREATE INDEX idx_call_edge_type ON call_edge(call_type);

CREATE INDEX idx_kconfig_type ON kconfig_option(type);

CREATE INDEX idx_depends_kconfig_symbol ON depends_on_kconfig(symbol_id);
CREATE INDEX idx_depends_kconfig_name ON depends_on_kconfig(kconfig_name);

CREATE INDEX idx_module_symbol_module ON module_symbol(module_name);
CREATE INDEX idx_module_symbol_symbol ON module_symbol(symbol_id);

CREATE INDEX idx_entry_impl_entry ON entry_to_impl(entry_id);
CREATE INDEX idx_entry_impl_symbol ON entry_to_impl(symbol_id);

-- GIN indexes for JSONB fields
CREATE INDEX idx_entrypoint_details_gin ON entrypoint USING gin(details);
CREATE INDEX idx_kconfig_depends_gin ON kconfig_option USING gin(depends_on);
CREATE INDEX idx_ownership_maintainers_gin ON ownership USING gin(maintainers);

-- Comments for documentation
COMMENT ON TABLE entrypoint IS 'Kernel boundaries where external requests enter';
COMMENT ON COLUMN entrypoint.key IS 'Unique identifier within kind (syscall number, ioctl cmd, etc.)';
COMMENT ON COLUMN entrypoint.details IS 'Kind-specific metadata (JSON schema varies by kind)';

COMMENT ON TABLE call_edge IS 'Function call relationships with configuration context';
COMMENT ON COLUMN call_edge.config_bitmap IS 'Bitmap of configs where this edge exists';
COMMENT ON COLUMN call_edge.is_indirect IS 'True for function pointer calls';

COMMENT ON TABLE kconfig_option IS 'Kernel configuration options from Kconfig files';
COMMENT ON COLUMN kconfig_option.depends_on IS 'Complex dependency expressions as JSON';

COMMENT ON TABLE depends_on_kconfig IS 'Symbol dependencies on configuration options';

COMMENT ON TABLE module_symbol IS 'Symbols exported by kernel modules';

COMMENT ON TABLE ownership IS 'File ownership patterns from MAINTAINERS file';
COMMENT ON COLUMN ownership.path_pattern IS 'File path pattern (supports wildcards)';
COMMENT ON COLUMN ownership.maintainers IS 'Array of maintainer objects with name, email, role';

-- Example JSONB schemas (for documentation)
/*
entrypoint.details examples:

Syscall:
{
  "syscall_nr": 0,
  "arch": "x86_64",
  "compat": false,
  "args": ["unsigned int fd", "char __user *buf", "size_t count"]
}

Ioctl:
{
  "cmd": "0x5401",
  "decoded": "TCGETS",
  "arg_type": "struct termios *",
  "direction": "read"
}

File operations:
{
  "struct_name": "ext4_file_operations",
  "operation": "read",
  "index": 0
}

ownership.maintainers example:
[
  {
    "name": "Al Viro",
    "email": "viro@zeniv.linux.org.uk",
    "role": "maintainer"
  },
  {
    "name": "Christian Brauner",
    "email": "brauner@kernel.org",
    "role": "reviewer"
  }
]
*/
