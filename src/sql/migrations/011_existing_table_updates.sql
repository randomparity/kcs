-- Migration 011: Add kernel configuration awareness to existing tables
--
-- This migration adds kernel_version and kernel_config columns to existing tables
-- to enable multi-architecture and multi-configuration support.
--
-- Author: KCS Team
-- Date: 2025-09-17

BEGIN;

-- ============================================================================
-- Add Kernel Version and Config Columns
-- ============================================================================

-- Update symbol table with kernel version and config awareness
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS kernel_version VARCHAR(100),
ADD COLUMN IF NOT EXISTS kernel_config VARCHAR(100),
ADD COLUMN IF NOT EXISTS arch CONFIG_ARCH,
ADD COLUMN IF NOT EXISTS config_metadata JSONB DEFAULT '{}';

-- Update call_graph table with kernel version and config awareness
ALTER TABLE call_graph
ADD COLUMN IF NOT EXISTS kernel_version VARCHAR(100),
ADD COLUMN IF NOT EXISTS kernel_config VARCHAR(100),
ADD COLUMN IF NOT EXISTS arch CONFIG_ARCH,
ADD COLUMN IF NOT EXISTS config_conditions JSONB DEFAULT '[]';

-- Update entrypoint table with kernel version
ALTER TABLE entrypoint
ADD COLUMN IF NOT EXISTS kernel_version VARCHAR(100),
ADD COLUMN IF NOT EXISTS arch CONFIG_ARCH;

-- Update file table with kernel version and arch
ALTER TABLE file
ADD COLUMN IF NOT EXISTS kernel_version VARCHAR(100),
ADD COLUMN IF NOT EXISTS arch CONFIG_ARCH,
ADD COLUMN IF NOT EXISTS build_flags JSONB DEFAULT '{}';

-- Update dependency table with kernel version and config
ALTER TABLE dependency
ADD COLUMN IF NOT EXISTS kernel_version VARCHAR(100),
ADD COLUMN IF NOT EXISTS kernel_config VARCHAR(100),
ADD COLUMN IF NOT EXISTS arch CONFIG_ARCH;

-- Update kconfig_map table with kernel version
ALTER TABLE kconfig_map
ADD COLUMN IF NOT EXISTS kernel_version VARCHAR(100),
ADD COLUMN IF NOT EXISTS arch CONFIG_ARCH,
ADD COLUMN IF NOT EXISTS config_metadata JSONB DEFAULT '{}';

-- ============================================================================
-- Add Config-Specific Constraints
-- ============================================================================

-- Update unique constraints to include kernel version and config
ALTER TABLE symbol
DROP CONSTRAINT IF EXISTS symbol_unique_in_file,
ADD CONSTRAINT symbol_unique_in_file_config
UNIQUE (name, file_id, start_line, kernel_version, kernel_config);

ALTER TABLE entrypoint
DROP CONSTRAINT IF EXISTS entrypoint_unique_per_config,
ADD CONSTRAINT entrypoint_unique_per_version_config
UNIQUE (kind, key, config, kernel_version);

ALTER TABLE call_graph
DROP CONSTRAINT IF EXISTS call_graph_unique_edge,
ADD CONSTRAINT call_graph_unique_edge_config
UNIQUE (caller_id, callee_id, call_site_line, kernel_version, kernel_config);

ALTER TABLE dependency
DROP CONSTRAINT IF EXISTS dependency_unique_per_file,
ADD CONSTRAINT dependency_unique_per_file_config
UNIQUE (file_id, depends_on_id, kernel_version, kernel_config);

-- ============================================================================
-- Add Cross-Config Mapping Table
-- ============================================================================

-- Table to track symbol equivalence across configurations
CREATE TABLE IF NOT EXISTS symbol_cross_config (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol_name TEXT NOT NULL,
    base_symbol_id BIGINT REFERENCES symbol (id) ON DELETE CASCADE,
    base_kernel_version VARCHAR(100) NOT NULL,
    base_kernel_config VARCHAR(100) NOT NULL,
    base_arch CONFIG_ARCH NOT NULL,
    mapped_symbol_id BIGINT REFERENCES symbol (id) ON DELETE CASCADE,
    mapped_kernel_version VARCHAR(100) NOT NULL,
    mapped_kernel_config VARCHAR(100) NOT NULL,
    mapped_arch CONFIG_ARCH NOT NULL,
    similarity_score FLOAT DEFAULT 1.0,
    mapping_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT cross_config_mapping_type_check
    CHECK (mapping_type IN ('exact', 'similar', 'conditional', 'arch_specific', 'version_specific')),
    CONSTRAINT cross_config_unique_mapping
    UNIQUE (base_symbol_id, mapped_symbol_id)
);

-- ============================================================================
-- Add Config Compatibility Matrix Table
-- ============================================================================

-- Table to track compatibility between different kernel configurations
CREATE TABLE IF NOT EXISTS kernel_config_compatibility (
    compat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_version VARCHAR(100) NOT NULL,
    source_config VARCHAR(100) NOT NULL,
    source_arch CONFIG_ARCH NOT NULL,
    target_version VARCHAR(100) NOT NULL,
    target_config VARCHAR(100) NOT NULL,
    target_arch CONFIG_ARCH NOT NULL,
    compatibility_level VARCHAR(50) NOT NULL,
    compatibility_score FLOAT DEFAULT 0.0,
    differences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT config_compat_level_check
    CHECK (compatibility_level IN ('full', 'partial', 'incompatible', 'untested')),
    CONSTRAINT config_compat_score_check
    CHECK (compatibility_score >= 0.0 AND compatibility_score <= 1.0),
    CONSTRAINT config_compat_unique
    UNIQUE (source_version, source_config, source_arch, target_version, target_config, target_arch)
);

-- ============================================================================
-- Add Indexes for Performance
-- ============================================================================

-- Symbol table indexes
CREATE INDEX IF NOT EXISTS idx_symbol_kernel_version ON symbol (kernel_version);
CREATE INDEX IF NOT EXISTS idx_symbol_kernel_config ON symbol (kernel_config);
CREATE INDEX IF NOT EXISTS idx_symbol_arch ON symbol (arch);
CREATE INDEX IF NOT EXISTS idx_symbol_config_metadata ON symbol USING gin (config_metadata);
CREATE INDEX IF NOT EXISTS idx_symbol_version_config ON symbol (kernel_version, kernel_config);

-- Call graph table indexes
CREATE INDEX IF NOT EXISTS idx_call_graph_kernel_version ON call_graph (kernel_version);
CREATE INDEX IF NOT EXISTS idx_call_graph_kernel_config ON call_graph (kernel_config);
CREATE INDEX IF NOT EXISTS idx_call_graph_arch ON call_graph (arch);
CREATE INDEX IF NOT EXISTS idx_call_graph_config_conditions ON call_graph USING gin (config_conditions);
CREATE INDEX IF NOT EXISTS idx_call_graph_version_config ON call_graph (kernel_version, kernel_config);

-- Entry point table indexes
CREATE INDEX IF NOT EXISTS idx_entrypoint_kernel_version ON entrypoint (kernel_version);
CREATE INDEX IF NOT EXISTS idx_entrypoint_arch ON entrypoint (arch);
CREATE INDEX IF NOT EXISTS idx_entrypoint_version_config ON entrypoint (kernel_version, config);

-- File table indexes
CREATE INDEX IF NOT EXISTS idx_file_kernel_version ON file (kernel_version);
CREATE INDEX IF NOT EXISTS idx_file_arch ON file (arch);
CREATE INDEX IF NOT EXISTS idx_file_build_flags ON file USING gin (build_flags);
CREATE INDEX IF NOT EXISTS idx_file_version_config ON file (kernel_version, config);

-- Dependency table indexes
CREATE INDEX IF NOT EXISTS idx_dependency_kernel_version ON dependency (kernel_version);
CREATE INDEX IF NOT EXISTS idx_dependency_kernel_config ON dependency (kernel_config);
CREATE INDEX IF NOT EXISTS idx_dependency_arch ON dependency (arch);

-- Kconfig map table indexes
CREATE INDEX IF NOT EXISTS idx_kconfig_kernel_version ON kconfig_map (kernel_version);
CREATE INDEX IF NOT EXISTS idx_kconfig_arch ON kconfig_map (arch);
CREATE INDEX IF NOT EXISTS idx_kconfig_metadata ON kconfig_map USING gin (config_metadata);

-- Cross-config mapping indexes
CREATE INDEX IF NOT EXISTS idx_cross_config_symbol ON symbol_cross_config (symbol_name);
CREATE INDEX IF NOT EXISTS idx_cross_config_base ON symbol_cross_config (base_symbol_id);
CREATE INDEX IF NOT EXISTS idx_cross_config_mapped ON symbol_cross_config (mapped_symbol_id);
CREATE INDEX IF NOT EXISTS idx_cross_config_base_version ON symbol_cross_config (
    base_kernel_version, base_kernel_config, base_arch
);
CREATE INDEX IF NOT EXISTS idx_cross_config_mapped_version ON symbol_cross_config (
    mapped_kernel_version, mapped_kernel_config, mapped_arch
);
CREATE INDEX IF NOT EXISTS idx_cross_config_type ON symbol_cross_config (mapping_type);

-- Config compatibility indexes
CREATE INDEX IF NOT EXISTS idx_compat_source ON kernel_config_compatibility (
    source_version, source_config, source_arch
);
CREATE INDEX IF NOT EXISTS idx_compat_target ON kernel_config_compatibility (
    target_version, target_config, target_arch
);
CREATE INDEX IF NOT EXISTS idx_compat_level ON kernel_config_compatibility (compatibility_level);
CREATE INDEX IF NOT EXISTS idx_compat_score ON kernel_config_compatibility (compatibility_score);

-- ============================================================================
-- Helper Functions for Config Management
-- ============================================================================

-- Function to populate kernel version from existing data
CREATE OR REPLACE FUNCTION populate_kernel_version()
RETURNS VOID AS $$
DECLARE
    default_version VARCHAR(100) := '6.6.0';
BEGIN
    -- Update symbol table
    UPDATE symbol
    SET kernel_version = default_version,
        kernel_config = 'x86_64:defconfig'
    WHERE kernel_version IS NULL;

    -- Update other tables similarly
    UPDATE call_graph
    SET kernel_version = default_version,
        kernel_config = 'x86_64:defconfig'
    WHERE kernel_version IS NULL;

    UPDATE entrypoint
    SET kernel_version = default_version
    WHERE kernel_version IS NULL;

    UPDATE file
    SET kernel_version = default_version
    WHERE kernel_version IS NULL;

    UPDATE dependency
    SET kernel_version = default_version,
        kernel_config = 'x86_64:defconfig'
    WHERE kernel_version IS NULL;

    UPDATE kconfig_map
    SET kernel_version = default_version
    WHERE kernel_version IS NULL;
END;
$$ LANGUAGE plpgsql;

-- Function to find equivalent symbols across configs
CREATE OR REPLACE FUNCTION find_equivalent_symbols(
    p_symbol_name TEXT,
    p_kernel_version VARCHAR(100)
)
RETURNS TABLE (
    symbol_id BIGINT,
    kernel_config VARCHAR(100),
    arch CONFIG_ARCH,
    file_path TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT s.id, s.kernel_config, s.arch, f.path
    FROM symbol s
    JOIN file f ON s.file_id = f.id
    WHERE s.name = p_symbol_name
      AND s.kernel_version = p_kernel_version
    ORDER BY s.kernel_config, s.arch;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON COLUMN symbol.kernel_version IS 'Kernel version this symbol belongs to (e.g., 6.6.0)';
COMMENT ON COLUMN symbol.kernel_config IS 'Kernel configuration this symbol was parsed under';
COMMENT ON COLUMN symbol.arch IS 'Architecture this symbol is compiled for';
COMMENT ON COLUMN symbol.config_metadata IS 'Additional configuration-specific metadata';

COMMENT ON COLUMN call_graph.kernel_version IS 'Kernel version this call edge belongs to';
COMMENT ON COLUMN call_graph.kernel_config IS 'Kernel configuration this edge was parsed under';
COMMENT ON COLUMN call_graph.config_conditions IS 'Configuration conditions affecting this call';

COMMENT ON COLUMN file.build_flags IS 'Compilation flags used for this file in specific config';

COMMENT ON TABLE symbol_cross_config IS 'Maps equivalent symbols across different kernel configurations';
COMMENT ON TABLE kernel_config_compatibility IS 'Tracks compatibility between kernel configurations';

COMMENT ON FUNCTION populate_kernel_version() IS
'Populates kernel version fields with default values for existing data';
COMMENT ON FUNCTION find_equivalent_symbols(TEXT, VARCHAR) IS
'Finds all instances of a symbol across different configurations';

-- ============================================================================
-- Data Migration
-- ============================================================================

-- Populate default values for existing data
SELECT populate_kernel_version();

COMMIT;
