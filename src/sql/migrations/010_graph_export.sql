-- Migration 010: Graph export and serialization management
--
-- This migration adds support for exporting and managing kernel call graphs
-- in various formats for visualization and analysis tools.
--
-- Author: KCS Team
-- Date: 2025-09-17

BEGIN;

-- ============================================================================
-- Graph Export Jobs Table
-- ============================================================================

-- Tracks graph export jobs and their status
CREATE TABLE graph_export (
    export_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_name VARCHAR(255) NOT NULL,
    export_format VARCHAR(50) NOT NULL,
    export_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    kernel_version VARCHAR(100) NOT NULL,
    kernel_config VARCHAR(100) NOT NULL,
    subsystem VARCHAR(255),
    entry_point VARCHAR(255),
    max_depth INTEGER DEFAULT 10,
    include_metadata BOOLEAN DEFAULT TRUE,
    include_annotations BOOLEAN DEFAULT TRUE,
    chunk_size INTEGER,  -- For large graph chunking
    total_chunks INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    export_size_bytes BIGINT,
    node_count INTEGER,
    edge_count INTEGER,
    output_path TEXT,
    metadata JSONB DEFAULT '{}',
    CONSTRAINT graph_export_format_check
    CHECK (export_format IN ('json', 'graphml', 'gexf', 'dot', 'cytoscape', 'gephi')),
    CONSTRAINT graph_export_status_check
    CHECK (export_status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- ============================================================================
-- Graph Export Chunks Table
-- ============================================================================

-- Stores individual chunks of large graph exports
CREATE TABLE graph_export_chunk (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_id UUID NOT NULL REFERENCES graph_export (export_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_data BYTEA NOT NULL,
    chunk_size_bytes INTEGER NOT NULL,
    node_count INTEGER,
    edge_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT graph_export_chunk_unique UNIQUE (export_id, chunk_index)
);

-- ============================================================================
-- Graph Export Templates Table
-- ============================================================================

-- Stores reusable export configuration templates
CREATE TABLE graph_export_template (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(255) NOT NULL UNIQUE,
    template_description TEXT,
    export_format VARCHAR(50) NOT NULL,
    default_config JSONB NOT NULL,
    filter_rules JSONB DEFAULT '[]',
    style_config JSONB DEFAULT '{}',
    layout_algorithm VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    is_public BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT graph_template_format_check
    CHECK (export_format IN ('json', 'graphml', 'gexf', 'dot', 'cytoscape', 'gephi'))
);

-- ============================================================================
-- Graph Visualization Presets Table
-- ============================================================================

-- Stores visualization presets for different graph tools
CREATE TABLE graph_visualization_preset (
    preset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    preset_name VARCHAR(255) NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    preset_type VARCHAR(50) NOT NULL,
    configuration JSONB NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT graph_preset_tool_check
    CHECK (tool_name IN ('cytoscape', 'gephi', 'graphviz', 'd3', 'vis.js', 'sigma.js')),
    CONSTRAINT graph_preset_type_check
    CHECK (preset_type IN ('layout', 'style', 'filter', 'analysis', 'complete'))
);

-- ============================================================================
-- Graph Export History Table
-- ============================================================================

-- Maintains history of graph exports for auditing and reuse
CREATE TABLE graph_export_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_id UUID REFERENCES graph_export (export_id) ON DELETE SET NULL,
    template_id UUID REFERENCES graph_export_template (template_id) ON DELETE SET NULL,
    user_id VARCHAR(255),
    export_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    export_parameters JSONB NOT NULL,
    export_statistics JSONB DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    retention_days INTEGER DEFAULT 90,
    metadata JSONB DEFAULT '{}',
    CONSTRAINT graph_history_retention_check CHECK (retention_days > 0)
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Graph export indexes
CREATE INDEX idx_graph_export_status ON graph_export (export_status);
CREATE INDEX idx_graph_export_format ON graph_export (export_format);
CREATE INDEX idx_graph_export_kernel ON graph_export (kernel_version, kernel_config);
CREATE INDEX idx_graph_export_subsystem ON graph_export (subsystem);
CREATE INDEX idx_graph_export_entry_point ON graph_export (entry_point);
CREATE INDEX idx_graph_export_created ON graph_export (created_at DESC);
CREATE INDEX idx_graph_export_completed ON graph_export (completed_at DESC);
CREATE INDEX idx_graph_export_metadata ON graph_export USING gin (metadata);

-- Graph export chunk indexes
CREATE INDEX idx_graph_chunk_export ON graph_export_chunk (export_id);
CREATE INDEX idx_graph_chunk_index ON graph_export_chunk (chunk_index);
CREATE INDEX idx_graph_chunk_created ON graph_export_chunk (created_at);

-- Graph export template indexes
CREATE INDEX idx_graph_template_name ON graph_export_template (template_name);
CREATE INDEX idx_graph_template_format ON graph_export_template (export_format);
CREATE INDEX idx_graph_template_public ON graph_export_template (is_public);
CREATE INDEX idx_graph_template_created_by ON graph_export_template (created_by);
CREATE INDEX idx_graph_template_config ON graph_export_template USING gin (default_config);
CREATE INDEX idx_graph_template_filters ON graph_export_template USING gin (filter_rules);

-- Graph visualization preset indexes
CREATE INDEX idx_graph_preset_name ON graph_visualization_preset (preset_name);
CREATE INDEX idx_graph_preset_tool ON graph_visualization_preset (tool_name);
CREATE INDEX idx_graph_preset_type ON graph_visualization_preset (preset_type);
CREATE INDEX idx_graph_preset_default ON graph_visualization_preset (is_default);
CREATE INDEX idx_graph_preset_config ON graph_visualization_preset USING gin (configuration);

-- Graph export history indexes
CREATE INDEX idx_graph_history_export ON graph_export_history (export_id);
CREATE INDEX idx_graph_history_template ON graph_export_history (template_id);
CREATE INDEX idx_graph_history_user ON graph_export_history (user_id);
CREATE INDEX idx_graph_history_timestamp ON graph_export_history (export_timestamp DESC);
CREATE INDEX idx_graph_history_accessed ON graph_export_history (last_accessed DESC);
CREATE INDEX idx_graph_history_params ON graph_export_history USING gin (export_parameters);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update export job status
CREATE OR REPLACE FUNCTION update_export_status(
    p_export_id UUID,
    p_status VARCHAR(50),
    p_error_message TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    UPDATE graph_export
    SET export_status = p_status,
        started_at = CASE
            WHEN p_status = 'running' THEN NOW()
            ELSE started_at
        END,
        completed_at = CASE
            WHEN p_status IN ('completed', 'failed', 'cancelled') THEN NOW()
            ELSE completed_at
        END,
        error_message = p_error_message
    WHERE export_id = p_export_id;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate export statistics
CREATE OR REPLACE FUNCTION calculate_export_statistics(p_export_id UUID)
RETURNS JSONB AS $$
DECLARE
    v_stats JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_chunks', COUNT(*),
        'total_size_bytes', SUM(chunk_size_bytes),
        'total_nodes', SUM(node_count),
        'total_edges', SUM(edge_count),
        'avg_chunk_size', AVG(chunk_size_bytes)::INTEGER
    ) INTO v_stats
    FROM graph_export_chunk
    WHERE export_id = p_export_id;

    RETURN v_stats;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old export history
CREATE OR REPLACE FUNCTION clean_export_history()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete history entries past their retention period
    DELETE FROM graph_export_history
    WHERE export_timestamp < NOW() - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Also delete orphaned export chunks from cancelled/failed exports older than 7 days
    DELETE FROM graph_export
    WHERE export_status IN ('failed', 'cancelled')
      AND completed_at < NOW() - INTERVAL '7 days';

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get default visualization preset
CREATE OR REPLACE FUNCTION get_default_preset(
    p_tool_name VARCHAR(100),
    p_preset_type VARCHAR(50)
)
RETURNS JSONB AS $$
DECLARE
    v_config JSONB;
BEGIN
    SELECT configuration INTO v_config
    FROM graph_visualization_preset
    WHERE tool_name = p_tool_name
      AND preset_type = p_preset_type
      AND is_default = TRUE
    ORDER BY updated_at DESC
    LIMIT 1;

    RETURN COALESCE(v_config, '{}'::JSONB);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Triggers
-- ============================================================================

-- Trigger to update export statistics when chunks are added
CREATE OR REPLACE FUNCTION update_export_statistics()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE graph_export
    SET node_count = (
            SELECT SUM(node_count)
            FROM graph_export_chunk
            WHERE export_id = NEW.export_id
        ),
        edge_count = (
            SELECT SUM(edge_count)
            FROM graph_export_chunk
            WHERE export_id = NEW.export_id
        ),
        export_size_bytes = (
            SELECT SUM(chunk_size_bytes)
            FROM graph_export_chunk
            WHERE export_id = NEW.export_id
        ),
        total_chunks = (
            SELECT COUNT(*)
            FROM graph_export_chunk
            WHERE export_id = NEW.export_id
        )
    WHERE export_id = NEW.export_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_export_statistics
AFTER INSERT OR UPDATE ON graph_export_chunk
FOR EACH ROW
EXECUTE FUNCTION update_export_statistics();

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON TABLE graph_export IS 'Tracks graph export jobs and their execution status';
COMMENT ON TABLE graph_export_chunk IS 'Stores chunked data for large graph exports';
COMMENT ON TABLE graph_export_template IS 'Reusable templates for graph export configurations';
COMMENT ON TABLE graph_visualization_preset IS 'Visualization presets for different graph tools';
COMMENT ON TABLE graph_export_history IS 'Historical record of graph exports for auditing';

COMMENT ON COLUMN graph_export.chunk_size IS 'Maximum size per chunk for large graph splitting';
COMMENT ON COLUMN graph_export.output_path IS 'File system path or S3 URL for export output';
COMMENT ON COLUMN graph_export_chunk.chunk_data IS 'Compressed binary data for the chunk';
COMMENT ON COLUMN graph_export_template.filter_rules IS 'JSON rules for filtering nodes and edges';
COMMENT ON COLUMN graph_export_template.style_config IS 'Visual styling configuration for the export';
COMMENT ON COLUMN graph_visualization_preset.configuration IS
'Tool-specific configuration for visualization';

COMMENT ON FUNCTION update_export_status(UUID, VARCHAR, TEXT) IS
'Updates the status of an export job with optional error message';
COMMENT ON FUNCTION calculate_export_statistics(UUID) IS
'Calculates aggregate statistics for an export from its chunks';
COMMENT ON FUNCTION clean_export_history() IS
'Removes expired export history entries based on retention settings';
COMMENT ON FUNCTION get_default_preset(VARCHAR, VARCHAR) IS
'Retrieves the default visualization preset for a tool and type';

COMMIT;
