-- Migration 006: Kernel configuration support for multi-architecture analysis
--
-- This migration adds support for kernel configuration parsing and management,
-- enabling config-aware analysis for different architectures and build configurations.
--
-- Author: KCS Team
-- Date: 2025-09-17

BEGIN;

-- ============================================================================
-- Kernel Configuration Table
-- ============================================================================

-- Stores parsed kernel configurations for different architectures and build types
CREATE TABLE kernel_config (
    config_name VARCHAR(255) PRIMARY KEY,
    architecture VARCHAR(50) NOT NULL,
    config_type VARCHAR(50) NOT NULL,
    enabled_features JSONB NOT NULL DEFAULT '[]',
    disabled_features JSONB NOT NULL DEFAULT '[]',
    module_features JSONB NOT NULL DEFAULT '[]',
    dependencies JSONB NOT NULL DEFAULT '{}',
    kernel_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Add constraints
ALTER TABLE kernel_config
    ADD CONSTRAINT kernel_config_architecture_check
    CHECK (architecture IN ('x86', 'x86_64', 'arm', 'arm64', 'riscv', 'riscv64', 'powerpc', 'powerpc64', 's390x', 'mips', 'mips64'));

ALTER TABLE kernel_config
    ADD CONSTRAINT kernel_config_type_check
    CHECK (config_type IN ('defconfig', 'allmodconfig', 'allnoconfig', 'allyesconfig', 'custom'));

-- Create indexes for efficient querying
CREATE INDEX idx_kernel_config_architecture ON kernel_config(architecture);
CREATE INDEX idx_kernel_config_type ON kernel_config(config_type);
CREATE INDEX idx_kernel_config_version ON kernel_config(kernel_version);
CREATE INDEX idx_kernel_config_created ON kernel_config(created_at);

-- JSONB indexes for feature searches
CREATE INDEX idx_kernel_config_enabled_features ON kernel_config USING gin(enabled_features);
CREATE INDEX idx_kernel_config_disabled_features ON kernel_config USING gin(disabled_features);
CREATE INDEX idx_kernel_config_module_features ON kernel_config USING gin(module_features);
CREATE INDEX idx_kernel_config_dependencies ON kernel_config USING gin(dependencies);
CREATE INDEX idx_kernel_config_metadata ON kernel_config USING gin(metadata);

-- ============================================================================
-- Configuration Dependencies Table
-- ============================================================================

-- Tracks dependencies between configuration options
CREATE TABLE config_dependency (
    id SERIAL PRIMARY KEY,
    config_name VARCHAR(255) REFERENCES kernel_config(config_name) ON DELETE CASCADE,
    option_name VARCHAR(255) NOT NULL,
    depends_on JSONB NOT NULL DEFAULT '[]',
    selects JSONB NOT NULL DEFAULT '[]',
    implies JSONB NOT NULL DEFAULT '[]',
    conflicts_with JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_config_dependency_config ON config_dependency(config_name);
CREATE INDEX idx_config_dependency_option ON config_dependency(option_name);
CREATE INDEX idx_config_dependency_depends ON config_dependency USING gin(depends_on);

-- ============================================================================
-- Add config awareness to existing tables
-- ============================================================================

-- Add config_visibility to symbol table for config-aware symbol resolution
ALTER TABLE symbol
    ADD COLUMN IF NOT EXISTS config_visibility JSONB DEFAULT '["x86_64:defconfig"]';

-- Add config_dependent to entrypoint table
ALTER TABLE entrypoint
    ADD COLUMN IF NOT EXISTS config_dependent VARCHAR(255);

-- Add config_dependent to call_edge table for config-aware call graph
ALTER TABLE call_edge
    ADD COLUMN IF NOT EXISTS config_dependent VARCHAR(255);

-- Create indexes for config columns
CREATE INDEX IF NOT EXISTS idx_symbol_config_visibility ON symbol USING gin(config_visibility);
CREATE INDEX IF NOT EXISTS idx_entrypoint_config_dependent ON entrypoint(config_dependent);
CREATE INDEX IF NOT EXISTS idx_call_edge_config_dependent ON call_edge(config_dependent);

-- ============================================================================
-- Comments and Documentation
-- ============================================================================

COMMENT ON TABLE kernel_config IS
'Stores parsed kernel configurations for different architectures and build types';

COMMENT ON COLUMN kernel_config.config_name IS
'Unique identifier for configuration (e.g., "x86_64:defconfig")';

COMMENT ON COLUMN kernel_config.architecture IS
'Target architecture for this configuration';

COMMENT ON COLUMN kernel_config.config_type IS
'Type of configuration (defconfig, allmodconfig, etc.)';

COMMENT ON COLUMN kernel_config.enabled_features IS
'Array of CONFIG_* options that are enabled (=y)';

COMMENT ON COLUMN kernel_config.disabled_features IS
'Array of CONFIG_* options that are explicitly disabled (=n)';

COMMENT ON COLUMN kernel_config.module_features IS
'Array of CONFIG_* options built as modules (=m)';

COMMENT ON COLUMN kernel_config.dependencies IS
'Map of feature dependencies and constraints';

COMMENT ON COLUMN kernel_config.kernel_version IS
'Kernel version this configuration applies to';

COMMENT ON COLUMN kernel_config.metadata IS
'Additional architecture-specific settings and metadata';

COMMENT ON TABLE config_dependency IS
'Tracks Kconfig dependencies between configuration options';

COMMENT ON COLUMN symbol.config_visibility IS
'Array of configurations where this symbol is visible/defined';

COMMENT ON COLUMN entrypoint.config_dependent IS
'CONFIG_* option that controls this entry point';

COMMENT ON COLUMN call_edge.config_dependent IS
'CONFIG_* option that controls this function call';

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to check if a feature is enabled in a configuration
CREATE OR REPLACE FUNCTION is_feature_enabled(
    p_config_name VARCHAR,
    p_feature VARCHAR
) RETURNS BOOLEAN AS $$
DECLARE
    v_enabled BOOLEAN;
BEGIN
    SELECT (enabled_features @> to_jsonb(ARRAY[p_feature]))
    INTO v_enabled
    FROM kernel_config
    WHERE config_name = p_config_name;

    RETURN COALESCE(v_enabled, FALSE);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get all active features for a configuration
CREATE OR REPLACE FUNCTION get_active_features(
    p_config_name VARCHAR
) RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'enabled', enabled_features,
        'modules', module_features
    )
    INTO v_result
    FROM kernel_config
    WHERE config_name = p_config_name;

    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;

COMMIT;
