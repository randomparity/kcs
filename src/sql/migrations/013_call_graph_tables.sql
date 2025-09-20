-- Migration 013: Call graph extraction tables
--
-- This migration adds support for storing function call relationships,
-- enabling call graph analysis and traversal functionality.
--
-- Author: KCS Team
-- Date: 2025-01-20
-- Specification: 007-call-graph-extraction/data-model.md

BEGIN;

-- ============================================================================
-- Call Edges Table
-- ============================================================================

-- Stores function call relationships between symbols
CREATE TABLE call_edges (
    id BIGSERIAL PRIMARY KEY,
    caller_id BIGINT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    callee_id BIGINT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,

    -- Call site location information
    file_path VARCHAR(1000) NOT NULL,
    line_number INTEGER NOT NULL,
    column_number INTEGER,
    function_context VARCHAR(255),
    context_before TEXT,
    context_after TEXT,

    -- Call classification and metadata
    call_type VARCHAR(20) NOT NULL,
    confidence VARCHAR(10) NOT NULL,
    conditional BOOLEAN DEFAULT FALSE,
    config_guard VARCHAR(100),

    -- Additional metadata as JSONB for extensibility
    metadata JSONB DEFAULT '{}',

    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT call_edges_caller_callee_check CHECK (caller_id != callee_id),
    CONSTRAINT call_edges_line_number_check CHECK (line_number > 0),
    CONSTRAINT call_edges_column_number_check CHECK (column_number >= 0),
    CONSTRAINT call_edges_call_type_check CHECK (
        call_type IN ('direct', 'indirect', 'macro', 'callback', 'conditional', 'assembly', 'syscall')
    ),
    CONSTRAINT call_edges_confidence_check CHECK (
        confidence IN ('high', 'medium', 'low')
    ),
    CONSTRAINT call_edges_conditional_guard_check CHECK (
        (conditional = FALSE AND config_guard IS NULL) OR
        (conditional = TRUE AND config_guard IS NOT NULL)
    )
);

-- ============================================================================
-- Function Pointers Table
-- ============================================================================

-- Stores function pointer assignments and their usage patterns
CREATE TABLE function_pointers (
    id BIGSERIAL PRIMARY KEY,
    pointer_name VARCHAR(255) NOT NULL,
    assigned_function_id BIGINT REFERENCES symbols(id) ON DELETE CASCADE,

    -- Assignment location
    assignment_file VARCHAR(1000) NOT NULL,
    assignment_line INTEGER NOT NULL,
    assignment_column INTEGER,

    -- Context information
    struct_context VARCHAR(255),
    assignment_context TEXT,

    -- Usage sites stored as JSONB array for flexibility
    usage_sites JSONB DEFAULT '[]',

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT function_pointers_line_check CHECK (assignment_line > 0),
    CONSTRAINT function_pointers_column_check CHECK (assignment_column >= 0)
);

-- ============================================================================
-- Macro Calls Table
-- ============================================================================

-- Stores function calls that occur through macro expansion
CREATE TABLE macro_calls (
    id BIGSERIAL PRIMARY KEY,
    macro_name VARCHAR(255) NOT NULL,
    macro_definition TEXT,

    -- Expansion site location
    expansion_file VARCHAR(1000) NOT NULL,
    expansion_line INTEGER NOT NULL,
    expansion_column INTEGER,

    -- Related call edges (array of call_edge IDs)
    expanded_call_ids BIGINT[],

    -- Preprocessor context
    preprocessor_context TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT macro_calls_line_check CHECK (expansion_line > 0),
    CONSTRAINT macro_calls_column_check CHECK (expansion_column >= 0)
);

-- ============================================================================
-- Call Paths Table (for caching frequently accessed paths)
-- ============================================================================

-- Stores pre-computed call paths for performance optimization
CREATE TABLE call_paths (
    id BIGSERIAL PRIMARY KEY,
    entry_point_id BIGINT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    target_function_id BIGINT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,

    -- Path information
    path_edge_ids BIGINT[], -- Array of call_edge IDs forming the path
    path_length INTEGER NOT NULL,
    total_confidence FLOAT NOT NULL,

    -- Configuration context
    config_context VARCHAR(100),
    kernel_version VARCHAR(50),

    -- Metadata and caching info
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT call_paths_different_endpoints CHECK (entry_point_id != target_function_id),
    CONSTRAINT call_paths_length_check CHECK (path_length > 0),
    CONSTRAINT call_paths_confidence_check CHECK (total_confidence >= 0.0 AND total_confidence <= 1.0)
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Call edges indexes for graph traversal
CREATE INDEX idx_call_edges_caller ON call_edges(caller_id);
CREATE INDEX idx_call_edges_callee ON call_edges(callee_id);
CREATE INDEX idx_call_edges_caller_callee ON call_edges(caller_id, callee_id);
CREATE INDEX idx_call_edges_type ON call_edges(call_type);
CREATE INDEX idx_call_edges_confidence ON call_edges(confidence);
CREATE INDEX idx_call_edges_conditional ON call_edges(conditional);
CREATE INDEX idx_call_edges_config ON call_edges(config_guard);

-- Location-based indexes for citation queries
CREATE INDEX idx_call_edges_file ON call_edges(file_path);
CREATE INDEX idx_call_edges_location ON call_edges(file_path, line_number);

-- Performance index for call site queries
CREATE INDEX idx_call_edges_function_context ON call_edges(function_context);

-- Metadata search
CREATE INDEX idx_call_edges_metadata ON call_edges USING gin(metadata);

-- Function pointers indexes
CREATE INDEX idx_function_pointers_name ON function_pointers(pointer_name);
CREATE INDEX idx_function_pointers_assigned ON function_pointers(assigned_function_id);
CREATE INDEX idx_function_pointers_file ON function_pointers(assignment_file);
CREATE INDEX idx_function_pointers_struct ON function_pointers(struct_context);
CREATE INDEX idx_function_pointers_usage ON function_pointers USING gin(usage_sites);

-- Macro calls indexes
CREATE INDEX idx_macro_calls_name ON macro_calls(macro_name);
CREATE INDEX idx_macro_calls_file ON macro_calls(expansion_file);
CREATE INDEX idx_macro_calls_location ON macro_calls(expansion_file, expansion_line);
CREATE INDEX idx_macro_calls_expanded_ids ON macro_calls USING gin(expanded_call_ids);

-- Call paths indexes for caching optimization
CREATE INDEX idx_call_paths_entry ON call_paths(entry_point_id);
CREATE INDEX idx_call_paths_target ON call_paths(target_function_id);
CREATE INDEX idx_call_paths_entry_target ON call_paths(entry_point_id, target_function_id);
CREATE INDEX idx_call_paths_config ON call_paths(config_context);
CREATE INDEX idx_call_paths_kernel_version ON call_paths(kernel_version);
CREATE INDEX idx_call_paths_length ON call_paths(path_length);
CREATE INDEX idx_call_paths_accessed ON call_paths(last_accessed DESC);

-- Composite index for path queries with config context
CREATE INDEX idx_call_paths_lookup ON call_paths(entry_point_id, target_function_id, config_context);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update call path access statistics
CREATE OR REPLACE FUNCTION update_call_path_access(p_path_id BIGINT)
RETURNS VOID AS $$
BEGIN
    UPDATE call_paths
    SET access_count = access_count + 1,
        last_accessed = NOW()
    WHERE id = p_path_id;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old cached paths (for maintenance)
CREATE OR REPLACE FUNCTION cleanup_old_call_paths(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM call_paths
    WHERE last_accessed < NOW() - INTERVAL '1 day' * days_old
      AND access_count < 5; -- Keep frequently accessed paths

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get call relationship statistics
CREATE OR REPLACE FUNCTION get_call_graph_stats()
RETURNS TABLE(
    total_call_edges BIGINT,
    direct_calls BIGINT,
    indirect_calls BIGINT,
    macro_calls BIGINT,
    function_pointers BIGINT,
    cached_paths BIGINT,
    avg_confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_call_edges,
        COUNT(*) FILTER (WHERE ce.call_type = 'direct') as direct_calls,
        COUNT(*) FILTER (WHERE ce.call_type = 'indirect') as indirect_calls,
        (SELECT COUNT(*) FROM macro_calls) as macro_calls,
        (SELECT COUNT(*) FROM function_pointers) as function_pointers,
        (SELECT COUNT(*) FROM call_paths) as cached_paths,
        AVG(CASE
            WHEN ce.confidence = 'high' THEN 1.0
            WHEN ce.confidence = 'medium' THEN 0.5
            WHEN ce.confidence = 'low' THEN 0.1
            ELSE 0.0
        END) as avg_confidence
    FROM call_edges ce;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON TABLE call_edges IS 'Function call relationships extracted from kernel source code';
COMMENT ON TABLE function_pointers IS 'Function pointer assignments and usage patterns';
COMMENT ON TABLE macro_calls IS 'Function calls occurring through macro expansion';
COMMENT ON TABLE call_paths IS 'Pre-computed call paths for performance optimization';

COMMENT ON COLUMN call_edges.caller_id IS 'Symbol ID of the calling function';
COMMENT ON COLUMN call_edges.callee_id IS 'Symbol ID of the called function';
COMMENT ON COLUMN call_edges.call_type IS 'Type of call mechanism (direct, indirect, macro, etc.)';
COMMENT ON COLUMN call_edges.confidence IS 'Reliability of call detection (high/medium/low)';
COMMENT ON COLUMN call_edges.conditional IS 'Whether call is inside conditional compilation';
COMMENT ON COLUMN call_edges.config_guard IS 'Configuration dependency if conditional';

COMMENT ON COLUMN function_pointers.pointer_name IS 'Name of the function pointer variable';
COMMENT ON COLUMN function_pointers.struct_context IS 'Struct name if pointer is a member';
COMMENT ON COLUMN function_pointers.usage_sites IS 'JSON array of call sites using this pointer';

COMMENT ON COLUMN macro_calls.macro_name IS 'Name of the macro being expanded';
COMMENT ON COLUMN macro_calls.expanded_call_ids IS 'Array of call_edge IDs from macro expansion';

COMMENT ON COLUMN call_paths.path_edge_ids IS 'Array of call_edge IDs forming the complete path';
COMMENT ON COLUMN call_paths.total_confidence IS 'Combined confidence score for entire path';

COMMENT ON FUNCTION update_call_path_access(BIGINT) IS 'Updates access statistics for call path caching';
COMMENT ON FUNCTION cleanup_old_call_paths(INTEGER) IS 'Removes old, infrequently accessed call paths';
COMMENT ON FUNCTION get_call_graph_stats() IS 'Returns comprehensive statistics about call graph data';

COMMIT;
