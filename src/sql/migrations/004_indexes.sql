-- KCS Performance Indexes and Constraints Migration
-- Optimizes queries for graph traversal, search, and impact analysis

-- Performance-critical composite indexes for graph traversal
CREATE INDEX idx_call_edge_caller_config ON call_edge(caller_id, config);
CREATE INDEX idx_call_edge_callee_config ON call_edge(callee_id, config);
CREATE INDEX idx_call_edge_caller_type ON call_edge(caller_id, call_type);

-- Symbol lookup optimizations
CREATE INDEX idx_symbol_name_config ON symbol(name, config);
CREATE INDEX idx_symbol_kind_config ON symbol(kind, config);
CREATE INDEX idx_symbol_file_config ON symbol(file_id, config);

-- Entry point query optimizations
CREATE INDEX idx_entrypoint_kind_config ON entrypoint(kind, config);
CREATE INDEX idx_entrypoint_symbol_config ON entrypoint(symbol_id, config);

-- Enable trigram extension for fuzzy text search (must come before gin_trgm_ops usage)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Config bitmap operations (for multi-config queries)
-- TODO: Fix GIN indexes for bytea columns - need specific operator class
-- CREATE INDEX idx_symbol_config_bitmap_gin ON symbol USING gin(config_bitmap);
-- CREATE INDEX idx_call_edge_config_bitmap_gin ON call_edge USING gin(config_bitmap);

-- Text search optimizations (requires pg_trgm extension)
CREATE INDEX idx_symbol_name_trgm ON symbol USING gin(name gin_trgm_ops);
CREATE INDEX idx_file_path_trgm ON file USING gin(path gin_trgm_ops);
CREATE INDEX idx_kconfig_name_trgm ON kconfig_option USING gin(name gin_trgm_ops);

-- Full-text search indexes
CREATE INDEX idx_summary_content_fts ON summary USING gin(to_tsvector('english', content::text));
CREATE INDEX idx_kconfig_help_fts ON kconfig_option USING gin(to_tsvector('english', coalesce(help, '')));

-- Time-based query optimizations
CREATE INDEX idx_file_indexed_at ON file(indexed_at DESC);
CREATE INDEX idx_summary_generated_at_desc ON summary(generated_at DESC);
CREATE INDEX idx_drift_checked_at_desc ON drift_report(checked_at DESC);
CREATE INDEX idx_test_coverage_last_run ON test_coverage(last_run DESC) WHERE last_run IS NOT NULL;

-- Covering indexes for common queries (include extra columns to avoid table lookups)
CREATE INDEX idx_symbol_lookup_covering ON symbol(name, config)
    INCLUDE (id, kind, file_id, start_line, end_line, signature);

CREATE INDEX idx_entrypoint_lookup_covering ON entrypoint(kind, key, config)
    INCLUDE (id, symbol_id, file_id, details);

-- Partial indexes for filtered queries
CREATE INDEX idx_symbol_functions_only ON symbol(name, config) WHERE kind = 'function';
CREATE INDEX idx_symbol_structs_only ON symbol(name, config) WHERE kind = 'struct';
CREATE INDEX idx_entrypoint_syscalls_only ON entrypoint(key, config) WHERE kind = 'syscall';
CREATE INDEX idx_entrypoint_ioctls_only ON entrypoint(key, config) WHERE kind = 'ioctl';

-- Impact analysis query optimizations
CREATE INDEX idx_depends_kconfig_covering ON depends_on_kconfig(kconfig_name)
    INCLUDE (symbol_id, condition_type);
CREATE INDEX idx_module_symbol_covering ON module_symbol(symbol_id)
    INCLUDE (module_name, export_type);

-- Risk-based filtering
CREATE INDEX idx_risk_high_risk ON risk_assessment(symbol_id) WHERE risk_score >= 7.0;
CREATE INDEX idx_summary_high_confidence ON summary(symbol_id) WHERE confidence_score >= 0.8;

-- Test coverage analysis
CREATE INDEX idx_test_coverage_by_suite_status ON test_coverage(test_suite, status, symbol_id);
CREATE INDEX idx_test_coverage_low_coverage ON test_coverage(symbol_id, test_suite)
    WHERE coverage_percentage < 50.0;

-- Ownership pattern matching optimization
CREATE INDEX idx_ownership_pattern_btree ON ownership(path_pattern);
CREATE INDEX idx_ownership_subsystem ON ownership(subsystem) WHERE subsystem IS NOT NULL;

-- Foreign key constraint indexes (for referential integrity performance)
-- Note: PostgreSQL automatically creates indexes on primary keys and unique constraints
-- but not on foreign key references

-- Already covered by existing indexes:
-- idx_symbol_file_id covers symbol.file_id -> file.id
-- idx_entrypoint_symbol_id covers entrypoint.symbol_id -> symbol.id
-- idx_entrypoint_file_id is missing, add it:
CREATE INDEX idx_entrypoint_file_id ON entrypoint(file_id);

-- Citation references
CREATE INDEX idx_citation_file_lines ON citation(file_id, start_line, end_line);

-- Concurrency and locking optimizations
-- Add advisory lock functions for coordinated operations

CREATE OR REPLACE FUNCTION acquire_index_lock(config_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    -- Use pg_advisory_lock with hash of config name
    PERFORM pg_advisory_lock(hashtext(config_name));
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION release_index_lock(config_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    PERFORM pg_advisory_unlock(hashtext(config_name));
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Query performance helper views
CREATE VIEW symbol_with_summary AS
SELECT
    s.id,
    s.name,
    s.kind,
    s.config,
    s.start_line,
    s.end_line,
    s.signature,
    f.path as file_path,
    f.sha as file_sha,
    sum.content as summary_content,
    sum.confidence_score,
    sum.human_reviewed
FROM symbol s
JOIN file f ON s.file_id = f.id
LEFT JOIN summary sum ON s.id = sum.symbol_id;

CREATE VIEW entrypoint_with_impl AS
SELECT
    e.id,
    e.kind,
    e.key,
    e.config,
    e.details,
    s.name as symbol_name,
    s.signature as symbol_signature,
    f.path as file_path,
    f.sha as file_sha,
    s.start_line,
    s.end_line
FROM entrypoint e
LEFT JOIN symbol s ON e.symbol_id = s.id
LEFT JOIN file f ON s.file_id = f.id;

-- Performance monitoring
CREATE OR REPLACE FUNCTION analyze_table_stats()
RETURNS TABLE(
    table_name TEXT,
    row_count BIGINT,
    table_size TEXT,
    index_size TEXT,
    total_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname||'.'||tablename as table_name,
        n_tup_ins - n_tup_del as row_count,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- Index usage monitoring
CREATE OR REPLACE FUNCTION check_unused_indexes()
RETURNS TABLE(
    index_name TEXT,
    table_name TEXT,
    index_size TEXT,
    index_scans BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        indexrelname as index_name,
        relname as table_name,
        pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
        idx_scan as index_scans
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
    AND idx_scan < 100  -- Indexes used less than 100 times
    AND pg_relation_size(indexrelid) > 1024 * 1024  -- Larger than 1MB
    ORDER BY pg_relation_size(indexrelid) DESC;
END;
$$ LANGUAGE plpgsql;

-- Constraint validation helpers
ALTER TABLE symbol ADD CONSTRAINT symbol_signature_json_valid
    CHECK (signature IS NULL OR (signature ~ '^[^{]*$' OR json_valid(signature)));

ALTER TABLE entrypoint ADD CONSTRAINT entrypoint_details_schema_valid
    CHECK (details IS NULL OR jsonb_typeof(details) = 'object');

-- Additional check constraints for data quality
ALTER TABLE call_edge ADD CONSTRAINT call_edge_config_consistency
    CHECK (
        -- If config_bitmap is set, config should match one of the bits
        config_bitmap IS NULL OR length(config) > 0
    );

ALTER TABLE symbol ADD CONSTRAINT symbol_location_consistency
    CHECK (
        -- End position should be after start position
        (end_line > start_line) OR
        (end_line = start_line AND end_col >= start_col)
    );

-- Performance hint comments
COMMENT ON INDEX idx_symbol_name_config IS 'Primary lookup index for symbol queries by name and config';
COMMENT ON INDEX idx_call_edge_caller_config IS 'Optimizes who_calls queries';
COMMENT ON INDEX idx_call_edge_callee_config IS 'Optimizes list_dependencies queries';
-- COMMENT ON INDEX idx_symbol_config_bitmap_gin IS 'Enables efficient multi-config symbol filtering';
COMMENT ON INDEX idx_symbol_name_trgm IS 'Enables fuzzy symbol name search';
COMMENT ON INDEX idx_summary_content_fts IS 'Full-text search on symbol summaries';

-- Recommended ANALYZE for initial statistics
-- This should be run after initial data load
-- ANALYZE symbol, call_edge, entrypoint, file, summary;
