-- Migration 007: Specification management for drift detection
--
-- This migration adds support for storing and managing kernel specifications,
-- enabling validation of implementation against documented behavior.
--
-- Author: KCS Team
-- Date: 2025-09-17

BEGIN;

-- ============================================================================
-- Specification Documents Table
-- ============================================================================

-- Stores formal or informal specifications of kernel behavior
CREATE TABLE specification (
    spec_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    parsed_requirements JSONB NOT NULL DEFAULT '[]',
    kernel_versions JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT specification_unique_name_version UNIQUE(name, version)
);

-- Add type constraint
ALTER TABLE specification
    ADD CONSTRAINT specification_type_check
    CHECK (type IN ('api', 'behavior', 'interface', 'constraint', 'performance', 'security'));

-- Create indexes
CREATE INDEX idx_specification_name ON specification(name);
CREATE INDEX idx_specification_version ON specification(version);
CREATE INDEX idx_specification_type ON specification(type);
CREATE INDEX idx_specification_created ON specification(created_at);
CREATE INDEX idx_specification_updated ON specification(updated_at);
CREATE INDEX idx_specification_kernel_versions ON specification USING gin(kernel_versions);
CREATE INDEX idx_specification_requirements ON specification USING gin(parsed_requirements);
CREATE INDEX idx_specification_metadata ON specification USING gin(metadata);

-- Full-text search index on content
CREATE INDEX idx_specification_content_fts ON specification
    USING gin(to_tsvector('english', content));

-- ============================================================================
-- Specification Requirements Table
-- ============================================================================

-- Individual requirements extracted from specifications
CREATE TABLE specification_requirement (
    requirement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spec_id UUID REFERENCES specification(spec_id) ON DELETE CASCADE,
    requirement_key VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100),
    priority VARCHAR(20),
    testable BOOLEAN DEFAULT TRUE,
    test_criteria JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Add priority constraint
ALTER TABLE specification_requirement
    ADD CONSTRAINT requirement_priority_check
    CHECK (priority IN ('must', 'should', 'could', 'wont', 'critical', 'high', 'medium', 'low'));

-- Create indexes
CREATE INDEX idx_requirement_spec_id ON specification_requirement(spec_id);
CREATE INDEX idx_requirement_key ON specification_requirement(requirement_key);
CREATE INDEX idx_requirement_category ON specification_requirement(category);
CREATE INDEX idx_requirement_priority ON specification_requirement(priority);
CREATE INDEX idx_requirement_testable ON specification_requirement(testable);
CREATE INDEX idx_requirement_test_criteria ON specification_requirement USING gin(test_criteria);

-- ============================================================================
-- Specification Tags Table
-- ============================================================================

-- Tags for categorizing and searching specifications
CREATE TABLE specification_tag (
    tag_id SERIAL PRIMARY KEY,
    spec_id UUID REFERENCES specification(spec_id) ON DELETE CASCADE,
    tag_name VARCHAR(100) NOT NULL,
    tag_value VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_spec_tag_spec_id ON specification_tag(spec_id);
CREATE INDEX idx_spec_tag_name ON specification_tag(tag_name);
CREATE INDEX idx_spec_tag_value ON specification_tag(tag_value);
CREATE INDEX idx_spec_tag_name_value ON specification_tag(tag_name, tag_value);

-- ============================================================================
-- Comments and Documentation
-- ============================================================================

COMMENT ON TABLE specification IS
'Stores formal or informal specifications of expected kernel behavior';

COMMENT ON COLUMN specification.spec_id IS
'Unique identifier for the specification';

COMMENT ON COLUMN specification.name IS
'Name of the specification document';

COMMENT ON COLUMN specification.version IS
'Version of the specification (semantic versioning)';

COMMENT ON COLUMN specification.type IS
'Type of specification (api, behavior, interface, constraint, performance, security)';

COMMENT ON COLUMN specification.content IS
'Raw specification content (markdown, yaml, json, or plain text)';

COMMENT ON COLUMN specification.parsed_requirements IS
'Array of requirements extracted from the specification';

COMMENT ON COLUMN specification.kernel_versions IS
'Array of kernel versions this specification applies to';

COMMENT ON COLUMN specification.metadata IS
'Additional metadata including source, author, references';

COMMENT ON TABLE specification_requirement IS
'Individual testable requirements extracted from specifications';

COMMENT ON COLUMN specification_requirement.requirement_key IS
'Unique key for the requirement (e.g., FR-001, NFR-023)';

COMMENT ON COLUMN specification_requirement.description IS
'Full description of what the requirement specifies';

COMMENT ON COLUMN specification_requirement.category IS
'Category of requirement (functional, non-functional, performance, etc.)';

COMMENT ON COLUMN specification_requirement.priority IS
'Priority level using MoSCoW or severity scale';

COMMENT ON COLUMN specification_requirement.testable IS
'Whether this requirement can be automatically tested';

COMMENT ON COLUMN specification_requirement.test_criteria IS
'Array of specific test criteria for validation';

COMMENT ON TABLE specification_tag IS
'Tags for categorizing and searching specifications';

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to search specifications by content
CREATE OR REPLACE FUNCTION search_specifications(
    p_query TEXT,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    spec_id UUID,
    name VARCHAR,
    version VARCHAR,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.spec_id,
        s.name,
        s.version,
        ts_rank(to_tsvector('english', s.content), plainto_tsquery('english', p_query)) AS rank
    FROM specification s
    WHERE to_tsvector('english', s.content) @@ plainto_tsquery('english', p_query)
    ORDER BY rank DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get all requirements for a specification
CREATE OR REPLACE FUNCTION get_specification_requirements(
    p_spec_id UUID
) RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_agg(
        jsonb_build_object(
            'id', requirement_id,
            'key', requirement_key,
            'description', description,
            'category', category,
            'priority', priority,
            'testable', testable,
            'test_criteria', test_criteria
        ) ORDER BY requirement_key
    )
    INTO v_result
    FROM specification_requirement
    WHERE spec_id = p_spec_id;

    RETURN COALESCE(v_result, '[]'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to find specifications by kernel version
CREATE OR REPLACE FUNCTION find_specs_for_kernel(
    p_kernel_version VARCHAR
) RETURNS TABLE(
    spec_id UUID,
    name VARCHAR,
    version VARCHAR,
    type VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.spec_id,
        s.name,
        s.version,
        s.type
    FROM specification s
    WHERE s.kernel_versions @> to_jsonb(ARRAY[p_kernel_version])
    ORDER BY s.name, s.version DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_specification_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER specification_update_timestamp
    BEFORE UPDATE ON specification
    FOR EACH ROW
    EXECUTE FUNCTION update_specification_timestamp();

COMMIT;
