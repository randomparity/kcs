-- Migration 014: Semantic search core tables
--
-- This migration creates the core tables for semantic search functionality,
-- including content indexing, vector embeddings, and search operations.
--
-- Author: KCS Team
-- Date: 2025-09-21

BEGIN;

-- ============================================================================
-- Extensions Required for Semantic Search
-- ============================================================================

-- Enable pgvector extension for vector operations
-- Note: This requires pgvector to be installed on the database server
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Indexed Content Table
-- ============================================================================

-- Stores metadata about content that has been indexed for semantic search
CREATE TABLE indexed_content (
    id SERIAL PRIMARY KEY,
    content_type VARCHAR(50) NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    content_hash VARCHAR(64) NOT NULL, -- SHA256 hash for change detection
    title TEXT,
    content TEXT NOT NULL, -- Full text content
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    chunk_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    CONSTRAINT indexed_content_status_check
    CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'))
);

-- ============================================================================
-- Vector Embedding Table
-- ============================================================================

-- Stores vector embeddings for semantic search
CREATE TABLE vector_embedding (
    id SERIAL PRIMARY KEY,
    content_id INTEGER NOT NULL REFERENCES indexed_content(id) ON DELETE CASCADE,
    embedding VECTOR(384), -- BAAI/bge-small-en-v1.5 produces 384-dimensional vectors
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE (content_id, chunk_index)
);

-- ============================================================================
-- Search Query Table
-- ============================================================================

-- Stores search queries and their processing metadata
CREATE TABLE search_query (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(384),
    preprocessing_config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    processed_at TIMESTAMP WITH TIME ZONE,
    processing_time_ms INTEGER,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    CONSTRAINT search_query_status_check
    CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'))
);

-- ============================================================================
-- Search Result Table
-- ============================================================================

-- Stores search results with scoring and ranking information
CREATE TABLE search_result (
    id SERIAL PRIMARY KEY,
    query_id INTEGER NOT NULL REFERENCES search_query(id) ON DELETE CASCADE,
    content_id INTEGER NOT NULL REFERENCES indexed_content(id) ON DELETE CASCADE,
    embedding_id INTEGER NOT NULL REFERENCES vector_embedding(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL,
    bm25_score FLOAT,
    combined_score FLOAT NOT NULL,
    result_rank INTEGER NOT NULL,
    context_lines TEXT[],
    explanation TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE (query_id, result_rank)
);

-- ============================================================================
-- Performance Indexes
-- ============================================================================

-- Indexed content indexes
CREATE INDEX idx_indexed_content_source_path ON indexed_content (source_path);
CREATE INDEX idx_indexed_content_content_type ON indexed_content (content_type);
CREATE INDEX idx_indexed_content_status ON indexed_content (status);
CREATE INDEX idx_indexed_content_hash ON indexed_content (content_hash);
CREATE INDEX idx_indexed_content_indexed_at ON indexed_content (indexed_at DESC);
CREATE INDEX idx_indexed_content_updated_at ON indexed_content (updated_at DESC);
CREATE INDEX idx_indexed_content_metadata ON indexed_content USING gin (metadata);

-- Full-text search on content
CREATE INDEX idx_indexed_content_content_fts ON indexed_content
USING gin (to_tsvector('english', content));

-- Vector embedding indexes
CREATE INDEX idx_vector_embedding_content_id ON vector_embedding (content_id);
CREATE INDEX idx_vector_embedding_chunk_index ON vector_embedding (chunk_index);
CREATE INDEX idx_vector_embedding_lines ON vector_embedding (line_start, line_end);
CREATE INDEX idx_vector_embedding_created_at ON vector_embedding (created_at DESC);

-- Vector similarity search index (IVFFlat for performance)
CREATE INDEX idx_vector_embedding_similarity ON vector_embedding
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Search query indexes
CREATE INDEX idx_search_query_text ON search_query (query_text);
CREATE INDEX idx_search_query_status ON search_query (status);
CREATE INDEX idx_search_query_created_at ON search_query (created_at DESC);
CREATE INDEX idx_search_query_processing_time ON search_query (processing_time_ms);

-- Full-text search on query text
CREATE INDEX idx_search_query_text_fts ON search_query
USING gin (to_tsvector('english', query_text));

-- Query embedding similarity index
CREATE INDEX idx_search_query_embedding ON search_query
USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);

-- Search result indexes
CREATE INDEX idx_search_result_query_id ON search_result (query_id);
CREATE INDEX idx_search_result_content_id ON search_result (content_id);
CREATE INDEX idx_search_result_embedding_id ON search_result (embedding_id);
CREATE INDEX idx_search_result_similarity_score ON search_result (similarity_score DESC);
CREATE INDEX idx_search_result_combined_score ON search_result (combined_score DESC);
CREATE INDEX idx_search_result_rank ON search_result (result_rank);
CREATE INDEX idx_search_result_created_at ON search_result (created_at DESC);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at on indexed_content
CREATE TRIGGER indexed_content_updated_at_trigger
    BEFORE UPDATE ON indexed_content
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to compute vector similarity
CREATE OR REPLACE FUNCTION vector_cosine_similarity(a vector, b vector)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to find similar embeddings
CREATE OR REPLACE FUNCTION find_similar_embeddings(
    query_embedding vector,
    similarity_threshold float DEFAULT 0.7,
    max_results integer DEFAULT 10,
    content_types text[] DEFAULT NULL
)
RETURNS TABLE (
    embedding_id integer,
    content_id integer,
    source_path text,
    chunk_text text,
    similarity_score float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ve.id as embedding_id,
        ve.content_id,
        ic.source_path,
        ve.chunk_text,
        vector_cosine_similarity(ve.embedding, query_embedding) as similarity_score
    FROM vector_embedding ve
    JOIN indexed_content ic ON ve.content_id = ic.id
    WHERE ic.status = 'COMPLETED'
      AND (content_types IS NULL OR ic.content_type = ANY(content_types))
      AND vector_cosine_similarity(ve.embedding, query_embedding) >= similarity_threshold
    ORDER BY similarity_score DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old search results
CREATE OR REPLACE FUNCTION cleanup_old_search_results(retention_days integer DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete search results older than retention period
    DELETE FROM search_result
    WHERE created_at < (now() - interval '1 day' * retention_days);

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Delete orphaned search queries (those with no results)
    DELETE FROM search_query sq
    WHERE NOT EXISTS (
        SELECT 1 FROM search_result sr WHERE sr.query_id = sq.id
    ) AND sq.created_at < (now() - interval '1 day' * retention_days);

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to reindex content (mark as pending for reprocessing)
CREATE OR REPLACE FUNCTION reindex_content(
    content_paths text[] DEFAULT NULL,
    content_types text[] DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE indexed_content
    SET status = 'PENDING',
        chunk_count = 0,
        indexed_at = NULL,
        updated_at = now()
    WHERE (content_paths IS NULL OR source_path = ANY(content_paths))
      AND (content_types IS NULL OR content_type = ANY(content_types))
      AND status != 'PROCESSING'; -- Don't interrupt active processing

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    -- Clean up existing embeddings for reindexed content
    DELETE FROM vector_embedding ve
    USING indexed_content ic
    WHERE ve.content_id = ic.id
      AND ic.status = 'PENDING'
      AND (content_paths IS NULL OR ic.source_path = ANY(content_paths))
      AND (content_types IS NULL OR ic.content_type = ANY(content_types));

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Initial Data and Constraints
-- ============================================================================

-- Add constraint to ensure embeddings exist for completed content
-- (This will be enforced by application logic rather than database constraint
-- to allow for partial indexing and error handling)

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON TABLE indexed_content IS 'Stores metadata about content indexed for semantic search';
COMMENT ON TABLE vector_embedding IS 'Stores vector embeddings for semantic similarity search';
COMMENT ON TABLE search_query IS 'Stores search queries and their processing metadata';
COMMENT ON TABLE search_result IS 'Stores search results with scoring and ranking information';

COMMENT ON COLUMN indexed_content.content_hash IS 'SHA256 hash of content for change detection';
COMMENT ON COLUMN indexed_content.content IS 'Full text content of the indexed file';
COMMENT ON COLUMN indexed_content.chunk_count IS 'Number of embedding chunks created for this content';

COMMENT ON COLUMN vector_embedding.embedding IS '384-dimensional vector from BAAI/bge-small-en-v1.5 model';
COMMENT ON COLUMN vector_embedding.chunk_text IS 'Text content that was embedded';
COMMENT ON COLUMN vector_embedding.chunk_index IS 'Sequential index of chunk within parent content';

COMMENT ON COLUMN search_query.query_embedding IS 'Vector embedding of the search query';
COMMENT ON COLUMN search_query.preprocessing_config IS 'Configuration used for query preprocessing';

COMMENT ON COLUMN search_result.similarity_score IS 'Cosine similarity score between query and result';
COMMENT ON COLUMN search_result.bm25_score IS 'BM25 text similarity score';
COMMENT ON COLUMN search_result.combined_score IS 'Hybrid score combining semantic and text similarity';

COMMENT ON FUNCTION find_similar_embeddings(vector, float, integer, text[]) IS
'Finds embeddings similar to query vector with configurable threshold and filters';
COMMENT ON FUNCTION cleanup_old_search_results(integer) IS
'Removes search results and orphaned queries older than specified retention period';
COMMENT ON FUNCTION reindex_content(text[], text[]) IS
'Marks content for reindexing and cleans up existing embeddings';

-- Grant permissions (assuming standard KCS roles exist)
-- These should be adjusted based on actual KCS permission model
GRANT SELECT, INSERT, UPDATE, DELETE ON indexed_content TO kcs_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON vector_embedding TO kcs_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON search_query TO kcs_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON search_result TO kcs_app;
GRANT USAGE ON SEQUENCE indexed_content_id_seq TO kcs_app;
GRANT USAGE ON SEQUENCE vector_embedding_id_seq TO kcs_app;
GRANT USAGE ON SEQUENCE search_query_id_seq TO kcs_app;
GRANT USAGE ON SEQUENCE search_result_id_seq TO kcs_app;

COMMIT;
