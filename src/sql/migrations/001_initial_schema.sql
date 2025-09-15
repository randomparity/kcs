-- KCS Initial Schema Migration
-- Creates core entities: File, Symbol, and foundational types

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create custom types
CREATE TYPE symbol_kind AS ENUM (
    'function',
    'struct',
    'variable',
    'macro',
    'typedef',
    'enum',
    'union',
    'constant'
);

CREATE TYPE config_arch AS ENUM (
    'x86_64',
    'ppc64le',
    's390x',
    'arm64',
    'riscv64'
);

-- File table - represents source files in kernel repository
CREATE TABLE file (
    id BIGSERIAL PRIMARY KEY,
    path TEXT NOT NULL,
    sha TEXT NOT NULL,
    config TEXT NOT NULL,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT file_path_not_empty CHECK (length(path) > 0),
    CONSTRAINT file_sha_format CHECK (sha ~ '^[a-fA-F0-9]{40}$'),
    CONSTRAINT file_unique_per_config UNIQUE (path, config)
);

-- Symbol table - represents functions, structs, variables, etc.
CREATE TABLE symbol (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    kind symbol_kind NOT NULL,
    file_id BIGINT NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_col INTEGER NOT NULL DEFAULT 0,
    end_col INTEGER NOT NULL DEFAULT 0,
    config TEXT NOT NULL,
    config_bitmap BYTEA, -- For multi-config tracking
    signature TEXT, -- Function signature or struct definition

    -- Constraints
    CONSTRAINT symbol_name_not_empty CHECK (length(name) > 0),
    CONSTRAINT symbol_valid_lines CHECK (start_line <= end_line AND start_line > 0),
    CONSTRAINT symbol_valid_cols CHECK (start_col <= end_col AND start_col >= 0),
    CONSTRAINT symbol_unique_per_file_config UNIQUE (name, file_id, config)
);

-- Symbol embeddings for semantic search (pgvector)
CREATE TABLE symbol_embedding (
    symbol_id BIGINT PRIMARY KEY REFERENCES symbol(id) ON DELETE CASCADE,
    embedding VECTOR(768), -- OpenAI ada-002 dimension
    model_version TEXT NOT NULL DEFAULT 'ada-002',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Basic indexes for core tables
CREATE INDEX idx_file_path ON file(path);
CREATE INDEX idx_file_config ON file(config);
CREATE INDEX idx_file_sha ON file(sha);

CREATE INDEX idx_symbol_name ON symbol(name);
CREATE INDEX idx_symbol_kind ON symbol(kind);
CREATE INDEX idx_symbol_config ON symbol(config);
CREATE INDEX idx_symbol_file_id ON symbol(file_id);
CREATE INDEX idx_symbol_lines ON symbol(start_line, end_line);

-- pgvector index for semantic similarity
CREATE INDEX idx_symbol_embedding_hnsw ON symbol_embedding
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Comments for documentation
COMMENT ON TABLE file IS 'Source files in kernel repository';
COMMENT ON COLUMN file.path IS 'Relative path from repository root';
COMMENT ON COLUMN file.sha IS 'Git SHA for version tracking';
COMMENT ON COLUMN file.config IS 'Build configuration (e.g., x86_64:defconfig)';

COMMENT ON TABLE symbol IS 'Functions, structs, variables and other code elements';
COMMENT ON COLUMN symbol.config_bitmap IS 'Bitmap of configs where symbol exists';
COMMENT ON COLUMN symbol.signature IS 'Function signature or type definition';

COMMENT ON TABLE symbol_embedding IS 'Vector embeddings for semantic code search';

-- Grant permissions for application user
-- Note: These would be run separately in production with proper role setup
-- GRANT SELECT, INSERT, UPDATE, DELETE ON file, symbol, symbol_embedding TO kcs_app;
-- GRANT USAGE ON SEQUENCE file_id_seq, symbol_id_seq TO kcs_app;
