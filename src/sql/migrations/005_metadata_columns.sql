-- Migration 005: Add metadata columns for enhanced kernel pattern detection
--
-- This migration extends the existing schema to support:
-- 1. JSONB metadata columns for entry_point and symbol tables
-- 2. New kernel_pattern table for pattern detection results
-- 3. Performance indexes for metadata queries
--
-- Author: KCS Team
-- Date: 2025-09-17

BEGIN;

-- ============================================================================
-- 1. Add metadata columns to existing tables (non-breaking)
-- ============================================================================

-- Add metadata column to entry_point table
ALTER TABLE entry_point
ADD COLUMN IF NOT EXISTS metadata JSONB;

COMMENT ON COLUMN entry_point.metadata IS
'Extended metadata for entry points including export type, module, subsystem, etc.';

-- Add metadata column to symbol table
ALTER TABLE symbol
ADD COLUMN IF NOT EXISTS metadata JSONB;

COMMENT ON COLUMN symbol.metadata IS
'Extended metadata for symbols including export status, Clang types, parameters, etc.';

-- ============================================================================
-- 2. Create kernel_pattern table
-- ============================================================================

CREATE TABLE IF NOT EXISTS kernel_pattern (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type VARCHAR(50) NOT NULL,
    symbol_id UUID REFERENCES symbol (id) ON DELETE CASCADE,
    entry_point_id UUID REFERENCES entry_point (id) ON DELETE CASCADE,
    file_id UUID NOT NULL REFERENCES file (id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL CHECK (line_number > 0),
    raw_text TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT current_timestamp,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT current_timestamp,

    -- Ensure pattern references either a symbol, an entry point, or neither (but not both)
    CONSTRAINT pattern_reference CHECK (
        (symbol_id IS NOT NULL AND entry_point_id IS NULL)
        OR (symbol_id IS NULL AND entry_point_id IS NOT NULL)
        OR (symbol_id IS NULL AND entry_point_id IS NULL)
    )
);

-- Add comments for documentation
COMMENT ON TABLE kernel_pattern IS
'Kernel-specific patterns detected in source code (EXPORT_SYMBOL, module_param, etc.)';

COMMENT ON COLUMN kernel_pattern.pattern_type IS
'Type of pattern: ExportSymbol, ExportSymbolGPL, ModuleParam, etc.';

COMMENT ON COLUMN kernel_pattern.raw_text IS
'The actual text of the pattern as found in source';

COMMENT ON COLUMN kernel_pattern.metadata IS
'Pattern-specific metadata (namespace, parameter type, description, etc.)';

-- ============================================================================
-- 3. Create performance indexes
-- ============================================================================

-- Indexes for entry_point metadata queries
CREATE INDEX IF NOT EXISTS idx_entry_point_export_type
ON entry_point ((metadata ->> 'export_type'))
WHERE metadata IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_entry_point_subsystem
ON entry_point ((metadata ->> 'subsystem'))
WHERE metadata IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_entry_point_module
ON entry_point ((metadata ->> 'module'))
WHERE metadata IS NOT NULL;

-- Indexes for symbol metadata queries
CREATE INDEX IF NOT EXISTS idx_symbol_export_status
ON symbol ((metadata ->> 'export_status'))
WHERE metadata IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_symbol_export_type
ON symbol ((metadata ->> 'export_type'))
WHERE metadata IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_symbol_module_param
ON symbol ((metadata ->> 'module_param'))
WHERE metadata IS NOT NULL;

-- Indexes for kernel_pattern queries
CREATE INDEX IF NOT EXISTS idx_kernel_pattern_type
ON kernel_pattern (pattern_type);

CREATE INDEX IF NOT EXISTS idx_kernel_pattern_file
ON kernel_pattern (file_id);

CREATE INDEX IF NOT EXISTS idx_kernel_pattern_symbol
ON kernel_pattern (symbol_id)
WHERE symbol_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_kernel_pattern_entry_point
ON kernel_pattern (entry_point_id)
WHERE entry_point_id IS NOT NULL;

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_kernel_pattern_file_type
ON kernel_pattern (file_id, pattern_type);

-- ============================================================================
-- 4. Create update trigger for kernel_pattern
-- ============================================================================

CREATE OR REPLACE FUNCTION update_kernel_pattern_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER kernel_pattern_update_timestamp
BEFORE UPDATE ON kernel_pattern
FOR EACH ROW
EXECUTE FUNCTION update_kernel_pattern_timestamp();

-- ============================================================================
-- 5. Add pattern type enum values (for reference)
-- ============================================================================

-- Note: These are the expected pattern_type values:
-- 'ExportSymbol', 'ExportSymbolGPL', 'ExportSymbolNS',
-- 'ModuleParam', 'ModuleParamArray', 'ModuleParmDesc',
-- 'EarlyParam', 'CoreParam', 'SetupParam'

-- ============================================================================
-- 6. Verification queries
-- ============================================================================

-- Verify the migration succeeded
DO $$
BEGIN
  -- Check metadata columns exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'entry_point' AND column_name = 'metadata'
  ) THEN
    RAISE EXCEPTION 'Migration failed: entry_point.metadata column not created';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'symbol' AND column_name = 'metadata'
  ) THEN
    RAISE EXCEPTION 'Migration failed: symbol.metadata column not created';
  END IF;

  -- Check kernel_pattern table exists
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'kernel_pattern'
  ) THEN
    RAISE EXCEPTION 'Migration failed: kernel_pattern table not created';
  END IF;

  RAISE NOTICE 'Migration 005_metadata_columns completed successfully';
END;
$$;

COMMIT;
