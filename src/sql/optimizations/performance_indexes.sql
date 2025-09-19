-- Performance optimization indexes
-- Run with: psql -d kcs -f optimizations.sql

-- Symbols table indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_symbols_file_path ON symbols(file_path);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_symbols_kind ON symbols(kind);

-- Call edges indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_call_edges_caller ON call_edges(caller_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_call_edges_callee ON call_edges(callee_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_call_edges_both ON call_edges(caller_id, callee_id);

-- Entry points indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entrypoints_name ON entrypoints(name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entrypoints_type ON entrypoints(entry_type);

-- Performance monitoring
ANALYZE symbols;
ANALYZE call_edges;
ANALYZE entrypoints;
