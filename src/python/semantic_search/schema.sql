-- Semantic Search Engine Schema
-- Dedicated tables for semantic search functionality with pgvector backend

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Index status tracking for content
CREATE TYPE index_status AS ENUM (
    'pending',
    'indexing',
    'completed',
    'failed',
    'stale'
);

-- Content types that can be indexed
CREATE TYPE content_type AS ENUM (
    'source_file',
    'documentation',
    'comment_block',
    'function_definition',
    'struct_definition'
);

-- Indexed content table - tracks all content that has been processed for semantic search
CREATE TABLE indexed_content (
    id BIGSERIAL PRIMARY KEY,
    content_type content_type NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL, -- SHA-256 of content for change detection
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    status index_status NOT NULL DEFAULT 'pending',
    indexed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT indexed_content_path_not_empty CHECK (length(source_path) > 0),
    CONSTRAINT indexed_content_content_not_empty CHECK (length(content) > 0),
    CONSTRAINT indexed_content_hash_format CHECK (content_hash ~ '^[a-fA-F0-9]{64}$'),
    CONSTRAINT indexed_content_unique_content UNIQUE (source_path, content_hash)
);

-- Vector embeddings for semantic search using BAAI/bge-small-en-v1.5 (384 dimensions)
CREATE TABLE vector_embedding (
    id BIGSERIAL PRIMARY KEY,
    content_id BIGINT NOT NULL REFERENCES indexed_content(id) ON DELETE CASCADE,
    embedding VECTOR(384), -- BAAI/bge-small-en-v1.5 dimension
    model_name TEXT NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5',
    model_version TEXT NOT NULL DEFAULT '1.5',
    chunk_index INTEGER NOT NULL DEFAULT 0, -- For splitting large content
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT vector_embedding_unique_chunk UNIQUE (content_id, chunk_index)
);

-- Search query log for analytics and optimization
CREATE TABLE search_query (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(384),
    filters JSONB DEFAULT '{}',
    result_count INTEGER NOT NULL DEFAULT 0,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT search_query_text_not_empty CHECK (length(query_text) > 0),
    CONSTRAINT search_query_positive_response_time CHECK (response_time_ms >= 0)
);

-- Search results for tracking relevance and performance
CREATE TABLE search_result (
    id BIGSERIAL PRIMARY KEY,
    query_id BIGINT NOT NULL REFERENCES search_query(id) ON DELETE CASCADE,
    content_id BIGINT NOT NULL REFERENCES indexed_content(id) ON DELETE CASCADE,
    rank_position INTEGER NOT NULL,
    similarity_score FLOAT NOT NULL,
    bm25_score FLOAT,
    hybrid_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT search_result_positive_rank CHECK (rank_position > 0),
    CONSTRAINT search_result_valid_similarity CHECK (similarity_score >= 0 AND similarity_score <= 1),
    CONSTRAINT search_result_unique_result UNIQUE (query_id, content_id)
);

-- Indexes for performance
CREATE INDEX idx_indexed_content_type ON indexed_content(content_type);
CREATE INDEX idx_indexed_content_status ON indexed_content(status);
CREATE INDEX idx_indexed_content_path ON indexed_content(source_path);
CREATE INDEX idx_indexed_content_hash ON indexed_content(content_hash);
CREATE INDEX idx_indexed_content_updated ON indexed_content(updated_at);

-- pgvector HNSW index for fast similarity search
CREATE INDEX idx_vector_embedding_hnsw ON vector_embedding
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_vector_embedding_content ON vector_embedding(content_id);
CREATE INDEX idx_vector_embedding_model ON vector_embedding(model_name, model_version);

CREATE INDEX idx_search_query_created ON search_query(created_at);
CREATE INDEX idx_search_query_text_gin ON search_query USING gin(to_tsvector('english', query_text));

CREATE INDEX idx_search_result_query ON search_result(query_id);
CREATE INDEX idx_search_result_content ON search_result(content_id);
CREATE INDEX idx_search_result_rank ON search_result(rank_position);
CREATE INDEX idx_search_result_score ON search_result(hybrid_score);

-- Comments for documentation
COMMENT ON TABLE indexed_content IS 'Content that has been processed for semantic search';
COMMENT ON COLUMN indexed_content.content_hash IS 'SHA-256 hash for detecting content changes';
COMMENT ON COLUMN indexed_content.metadata IS 'Additional context like file type, author, etc.';
COMMENT ON COLUMN indexed_content.status IS 'Current indexing status for the content';

COMMENT ON TABLE vector_embedding IS 'Vector embeddings using BAAI/bge-small-en-v1.5 model';
COMMENT ON COLUMN vector_embedding.chunk_index IS 'Index for content split into multiple chunks';

COMMENT ON TABLE search_query IS 'Log of search queries for analytics and optimization';
COMMENT ON COLUMN search_query.filters IS 'JSON filters applied to the search';

COMMENT ON TABLE search_result IS 'Individual search results with ranking and scores';
COMMENT ON COLUMN search_result.hybrid_score IS 'Combined semantic + BM25 relevance score';
