#!/bin/bash
# Migration 014: Semantic search core tables with dynamic user grants
# This script supports any PostgreSQL user via environment variables

set -euo pipefail

# Get the database user from environment, with fallback
DB_USER="${POSTGRES_USER:-kcs}"

echo "Running migration 014 with database user: $DB_USER"

# Run the main migration SQL
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" << 'EOF'
-- Migration 014: Semantic search core tables
-- This migration creates the core tables for semantic search functionality

BEGIN;

-- Extensions Required for Semantic Search
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Indexed Content Table
CREATE TABLE indexed_content (
    id SERIAL PRIMARY KEY,
    content_type VARCHAR(50) NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    content_hash VARCHAR(64) NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    chunk_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    CONSTRAINT indexed_content_status_check
    CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'))
);

-- ... rest of table definitions ...

COMMIT;
EOF

# Now run the grants with the dynamic user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" << EOF
-- Grant permissions to the configured database user
GRANT SELECT, INSERT, UPDATE, DELETE ON indexed_content TO "$DB_USER";
GRANT SELECT, INSERT, UPDATE, DELETE ON vector_embedding TO "$DB_USER";
GRANT SELECT, INSERT, UPDATE, DELETE ON search_query TO "$DB_USER";
GRANT SELECT, INSERT, UPDATE, DELETE ON search_result TO "$DB_USER";
GRANT USAGE ON SEQUENCE indexed_content_id_seq TO "$DB_USER";
GRANT USAGE ON SEQUENCE vector_embedding_id_seq TO "$DB_USER";
GRANT USAGE ON SEQUENCE search_query_id_seq TO "$DB_USER";
GRANT USAGE ON SEQUENCE search_result_id_seq TO "$DB_USER";
EOF

echo "Migration 014 completed successfully for user: $DB_USER"
