-- Migration 012: Add chunk processing tracking for multi-file JSON output
-- Feature: 006-multi-file-json
-- Date: 2025-01-18
-- Purpose: Track chunk processing status and manifest data for resumable operations

-- Table to track processing status of individual chunks
CREATE TABLE IF NOT EXISTS chunk_processing (
    chunk_id VARCHAR(255) PRIMARY KEY,
    manifest_version VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    symbols_processed INTEGER DEFAULT 0,
    checksum_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT status_check CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);

-- Indexes for efficient querying
CREATE INDEX idx_chunk_processing_status ON chunk_processing(status);
CREATE INDEX idx_chunk_processing_manifest ON chunk_processing(manifest_version);
CREATE INDEX idx_chunk_processing_created ON chunk_processing(created_at);

-- Table to store manifest metadata
CREATE TABLE IF NOT EXISTS indexing_manifest (
    version VARCHAR(50) PRIMARY KEY,
    created TIMESTAMP NOT NULL,
    kernel_version VARCHAR(100),
    kernel_path TEXT,
    config VARCHAR(100),
    total_chunks INTEGER NOT NULL,
    total_size_bytes BIGINT,
    manifest_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient manifest queries
CREATE INDEX idx_manifest_created ON indexing_manifest(created);
CREATE INDEX idx_manifest_kernel ON indexing_manifest(kernel_version);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_chunk_processing_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update the updated_at column
DROP TRIGGER IF EXISTS trigger_update_chunk_processing_timestamp ON chunk_processing;
CREATE TRIGGER trigger_update_chunk_processing_timestamp
BEFORE UPDATE ON chunk_processing
FOR EACH ROW
EXECUTE FUNCTION update_chunk_processing_updated_at();

-- Add comments for documentation
COMMENT ON TABLE chunk_processing IS 'Tracks the processing status of individual JSON chunks during kernel indexing';
COMMENT ON COLUMN chunk_processing.chunk_id IS 'Unique identifier for the chunk (e.g., kernel_001)';
COMMENT ON COLUMN chunk_processing.manifest_version IS 'Version of the manifest this chunk belongs to';
COMMENT ON COLUMN chunk_processing.status IS 'Current processing state: pending, processing, completed, or failed';
COMMENT ON COLUMN chunk_processing.retry_count IS 'Number of retry attempts (max 3)';
COMMENT ON COLUMN chunk_processing.symbols_processed IS 'Count of symbols successfully inserted from this chunk';
COMMENT ON COLUMN chunk_processing.checksum_verified IS 'Whether SHA256 checksum was validated';

COMMENT ON TABLE indexing_manifest IS 'Stores manifest metadata for chunked kernel indexing operations';
COMMENT ON COLUMN indexing_manifest.version IS 'Unique version identifier for the manifest';
COMMENT ON COLUMN indexing_manifest.manifest_data IS 'Complete manifest JSON including all chunk metadata';
