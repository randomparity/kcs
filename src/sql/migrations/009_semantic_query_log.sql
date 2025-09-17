-- Migration 009: Semantic query logging and analytics
--
-- This migration adds support for logging and analyzing semantic search queries,
-- enabling query optimization and understanding of usage patterns.
--
-- Author: KCS Team
-- Date: 2025-09-17

BEGIN;

-- ============================================================================
-- Semantic Query Log Table
-- ============================================================================

-- Stores semantic search queries and their results for analytics and optimization
CREATE TABLE semantic_query_log (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_embedding VECTOR(768),  -- Assumes 768-dimensional embeddings (e.g., from sentence-transformers)
    query_type VARCHAR(50) NOT NULL,
    result_count INTEGER NOT NULL DEFAULT 0,
    top_k INTEGER NOT NULL DEFAULT 10,
    distance_threshold FLOAT,
    execution_time_ms INTEGER NOT NULL,
    kernel_version VARCHAR(100),
    kernel_config VARCHAR(100),
    subsystem VARCHAR(100),
    user_id VARCHAR(255),  -- Optional user/session identifier
    session_id UUID,
    query_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    results JSONB DEFAULT '[]',  -- Top results with scores
    feedback JSONB DEFAULT '{}',  -- User feedback on result quality
    metadata JSONB DEFAULT '{}',
    CONSTRAINT semantic_query_type_check
    CHECK (query_type IN ('symbol', 'function', 'concept', 'error', 'behavior', 'documentation'))
);

-- ============================================================================
-- Query Result Cache Table
-- ============================================================================

-- Caches frequently accessed query results for performance
CREATE TABLE semantic_query_cache (
    cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(64) NOT NULL,  -- SHA256 hash of normalized query
    query_embedding VECTOR(768),
    results JSONB NOT NULL,
    kernel_version VARCHAR(100),
    kernel_config VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT now(),
    CONSTRAINT semantic_cache_unique_hash UNIQUE (query_hash, kernel_version, kernel_config)
);

-- ============================================================================
-- Query Feedback Table
-- ============================================================================

-- Stores user feedback on query result quality for model improvement
CREATE TABLE semantic_query_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL REFERENCES semantic_query_log (query_id) ON DELETE CASCADE,
    result_rank INTEGER NOT NULL,
    result_id VARCHAR(255) NOT NULL,  -- ID of the result (e.g., symbol_id)
    relevance_score INTEGER,  -- User-provided relevance (1-5)
    is_correct BOOLEAN,
    is_helpful BOOLEAN,
    feedback_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    user_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT semantic_feedback_relevance_check CHECK (relevance_score >= 1 AND relevance_score <= 5)
);

-- ============================================================================
-- Query Analytics Aggregates Table
-- ============================================================================

-- Pre-aggregated statistics for query performance monitoring
CREATE TABLE semantic_query_stats (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    query_type VARCHAR(50),
    kernel_version VARCHAR(100),
    kernel_config VARCHAR(100),
    total_queries INTEGER NOT NULL DEFAULT 0,
    avg_execution_time_ms FLOAT,
    p50_execution_time_ms INTEGER,
    p95_execution_time_ms INTEGER,
    p99_execution_time_ms INTEGER,
    avg_result_count FLOAT,
    total_feedback_count INTEGER DEFAULT 0,
    avg_relevance_score FLOAT,
    cache_hit_rate FLOAT,
    unique_users INTEGER DEFAULT 0,
    top_queries JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    CONSTRAINT semantic_stats_unique_period UNIQUE (period_start, period_end, query_type, kernel_version, kernel_config)
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Semantic query log indexes
CREATE INDEX idx_semantic_query_timestamp ON semantic_query_log (query_timestamp DESC);
CREATE INDEX idx_semantic_query_type ON semantic_query_log (query_type);
CREATE INDEX idx_semantic_query_kernel ON semantic_query_log (kernel_version, kernel_config);
CREATE INDEX idx_semantic_query_subsystem ON semantic_query_log (subsystem);
CREATE INDEX idx_semantic_query_user ON semantic_query_log (user_id);
CREATE INDEX idx_semantic_query_session ON semantic_query_log (session_id);
CREATE INDEX idx_semantic_query_execution_time ON semantic_query_log (execution_time_ms);
CREATE INDEX idx_semantic_query_embedding ON semantic_query_log USING ivfflat (
    query_embedding vector_cosine_ops
) WITH (lists = 100);
CREATE INDEX idx_semantic_query_metadata ON semantic_query_log USING gin (metadata);

-- Full-text search on query text
CREATE INDEX idx_semantic_query_text_fts ON semantic_query_log
USING gin (to_tsvector('english', query_text));

-- Query cache indexes
CREATE INDEX idx_semantic_cache_hash ON semantic_query_cache (query_hash);
CREATE INDEX idx_semantic_cache_expires ON semantic_query_cache (expires_at);
CREATE INDEX idx_semantic_cache_kernel ON semantic_query_cache (kernel_version, kernel_config);
CREATE INDEX idx_semantic_cache_accessed ON semantic_query_cache (last_accessed DESC);
CREATE INDEX idx_semantic_cache_embedding ON semantic_query_cache USING ivfflat (
    query_embedding vector_cosine_ops
) WITH (lists = 100);

-- Query feedback indexes
CREATE INDEX idx_semantic_feedback_query ON semantic_query_feedback (query_id);
CREATE INDEX idx_semantic_feedback_result ON semantic_query_feedback (result_id);
CREATE INDEX idx_semantic_feedback_relevance ON semantic_query_feedback (relevance_score);
CREATE INDEX idx_semantic_feedback_created ON semantic_query_feedback (created_at DESC);
CREATE INDEX idx_semantic_feedback_user ON semantic_query_feedback (user_id);

-- Query stats indexes
CREATE INDEX idx_semantic_stats_period ON semantic_query_stats (period_start, period_end);
CREATE INDEX idx_semantic_stats_type ON semantic_query_stats (query_type);
CREATE INDEX idx_semantic_stats_kernel ON semantic_query_stats (kernel_version, kernel_config);
CREATE INDEX idx_semantic_stats_execution ON semantic_query_stats (avg_execution_time_ms);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION clean_semantic_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM semantic_query_cache
    WHERE expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update cache hit statistics
CREATE OR REPLACE FUNCTION update_cache_hit(p_cache_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE semantic_query_cache
    SET hit_count = hit_count + 1,
        last_accessed = NOW()
    WHERE cache_id = p_cache_id;
END;
$$ LANGUAGE plpgsql;

-- Function to compute query statistics for a time period
CREATE OR REPLACE FUNCTION compute_semantic_query_stats(
    p_start TIMESTAMP WITH TIME ZONE,
    p_end TIMESTAMP WITH TIME ZONE
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO semantic_query_stats (
        period_start,
        period_end,
        query_type,
        kernel_version,
        kernel_config,
        total_queries,
        avg_execution_time_ms,
        p50_execution_time_ms,
        p95_execution_time_ms,
        p99_execution_time_ms,
        avg_result_count,
        unique_users
    )
    SELECT
        p_start,
        p_end,
        query_type,
        kernel_version,
        kernel_config,
        COUNT(*) as total_queries,
        AVG(execution_time_ms) as avg_execution_time_ms,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_time_ms) as p50_execution_time_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time_ms,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY execution_time_ms) as p99_execution_time_ms,
        AVG(result_count) as avg_result_count,
        COUNT(DISTINCT user_id) as unique_users
    FROM semantic_query_log
    WHERE query_timestamp >= p_start
      AND query_timestamp < p_end
    GROUP BY query_type, kernel_version, kernel_config
    ON CONFLICT (period_start, period_end, query_type, kernel_version, kernel_config)
    DO UPDATE SET
        total_queries = EXCLUDED.total_queries,
        avg_execution_time_ms = EXCLUDED.avg_execution_time_ms,
        p50_execution_time_ms = EXCLUDED.p50_execution_time_ms,
        p95_execution_time_ms = EXCLUDED.p95_execution_time_ms,
        p99_execution_time_ms = EXCLUDED.p99_execution_time_ms,
        avg_result_count = EXCLUDED.avg_result_count,
        unique_users = EXCLUDED.unique_users;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON TABLE semantic_query_log IS 'Logs all semantic search queries for analytics and optimization';
COMMENT ON TABLE semantic_query_cache IS 'Caches frequently accessed semantic search results';
COMMENT ON TABLE semantic_query_feedback IS 'Stores user feedback on search result quality';
COMMENT ON TABLE semantic_query_stats IS 'Pre-aggregated statistics for query performance monitoring';

COMMENT ON COLUMN semantic_query_log.query_embedding IS 'Vector embedding of the query for similarity analysis';
COMMENT ON COLUMN semantic_query_log.distance_threshold IS
'Optional threshold for filtering results by vector distance';
COMMENT ON COLUMN semantic_query_log.results IS 'JSON array of top results with scores and metadata';

COMMENT ON COLUMN semantic_query_cache.query_hash IS 'SHA256 hash of normalized query for exact match lookup';
COMMENT ON COLUMN semantic_query_cache.expires_at IS 'Timestamp when cache entry should be invalidated';

COMMENT ON FUNCTION clean_semantic_cache() IS 'Removes expired entries from the semantic query cache';
COMMENT ON FUNCTION update_cache_hit(UUID) IS 'Updates hit count and last accessed time for cache entry';
COMMENT ON FUNCTION compute_semantic_query_stats(
    TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE
) IS 'Computes aggregated statistics for a time period';

COMMIT;
