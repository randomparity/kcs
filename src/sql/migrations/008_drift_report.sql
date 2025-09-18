-- Migration 008: Drift analysis reports for spec vs implementation validation
--
-- This migration adds support for storing drift detection results,
-- tracking differences between specifications and actual kernel implementation.
--
-- Author: KCS Team
-- Date: 2025-09-17

BEGIN;

-- ============================================================================
-- Drift Report Table
-- ============================================================================

-- Drop existing drift_report table and related types if they exist (from migration 003)
-- and create new comprehensive version
DROP TABLE IF EXISTS drift_report CASCADE;
DROP TYPE IF EXISTS drift_severity CASCADE;

-- Main table for drift analysis reports
CREATE TABLE drift_report (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spec_id UUID REFERENCES specification(spec_id) ON DELETE CASCADE,
    commit_sha VARCHAR(40) NOT NULL,
    config_name VARCHAR(255) REFERENCES kernel_config(config_name),
    total_requirements INTEGER NOT NULL DEFAULT 0,
    passed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    unknown INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    severity VARCHAR(20) NOT NULL DEFAULT 'info',
    status VARCHAR(20) NOT NULL DEFAULT 'completed',
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    execution_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- Add constraints
ALTER TABLE drift_report
    ADD CONSTRAINT drift_report_severity_check
    CHECK (severity IN ('critical', 'major', 'minor', 'info', 'none'));

ALTER TABLE drift_report
    ADD CONSTRAINT drift_report_status_check
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

ALTER TABLE drift_report
    ADD CONSTRAINT drift_report_counts_check
    CHECK (passed >= 0 AND failed >= 0 AND unknown >= 0 AND skipped >= 0);

ALTER TABLE drift_report
    ADD CONSTRAINT drift_report_total_check
    CHECK (total_requirements = passed + failed + unknown + skipped);

-- Create indexes
CREATE INDEX idx_drift_report_spec ON drift_report(spec_id);
CREATE INDEX idx_drift_report_commit ON drift_report(commit_sha);
CREATE INDEX idx_drift_report_config ON drift_report(config_name);
CREATE INDEX idx_drift_report_severity ON drift_report(severity);
CREATE INDEX idx_drift_report_status ON drift_report(status);
CREATE INDEX idx_drift_report_generated ON drift_report(generated_at);
CREATE INDEX idx_drift_report_metadata ON drift_report USING gin(metadata);

-- ============================================================================
-- Drift Violations Table
-- ============================================================================

-- Individual violations detected during drift analysis
CREATE TABLE drift_violation (
    violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID REFERENCES drift_report(report_id) ON DELETE CASCADE,
    requirement_id UUID REFERENCES specification_requirement(requirement_id),
    requirement_key VARCHAR(255) NOT NULL,
    expected_behavior TEXT NOT NULL,
    actual_behavior TEXT NOT NULL,
    file_path VARCHAR(1024),
    line_number INTEGER,
    commit_sha VARCHAR(40),
    severity VARCHAR(20) NOT NULL DEFAULT 'minor',
    category VARCHAR(100),
    suggested_fix TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Add constraints
ALTER TABLE drift_violation
    ADD CONSTRAINT violation_severity_check
    CHECK (severity IN ('critical', 'major', 'minor', 'info'));

-- Create indexes
CREATE INDEX idx_violation_report ON drift_violation(report_id);
CREATE INDEX idx_violation_requirement ON drift_violation(requirement_id);
CREATE INDEX idx_violation_severity ON drift_violation(severity);
CREATE INDEX idx_violation_category ON drift_violation(category);
CREATE INDEX idx_violation_file ON drift_violation(file_path);
CREATE INDEX idx_violation_metadata ON drift_violation USING gin(metadata);

-- ============================================================================
-- Drift Conformances Table
-- ============================================================================

-- Requirements that passed validation
CREATE TABLE drift_conformance (
    conformance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID REFERENCES drift_report(report_id) ON DELETE CASCADE,
    requirement_id UUID REFERENCES specification_requirement(requirement_id),
    requirement_key VARCHAR(255) NOT NULL,
    evidence TEXT,
    file_path VARCHAR(1024),
    line_number INTEGER,
    confidence_score DECIMAL(3,2),
    metadata JSONB DEFAULT '{}'
);

-- Add constraint for confidence score
ALTER TABLE drift_conformance
    ADD CONSTRAINT conformance_confidence_check
    CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0);

-- Create indexes
CREATE INDEX idx_conformance_report ON drift_conformance(report_id);
CREATE INDEX idx_conformance_requirement ON drift_conformance(requirement_id);
CREATE INDEX idx_conformance_confidence ON drift_conformance(confidence_score);
CREATE INDEX idx_conformance_file ON drift_conformance(file_path);

-- ============================================================================
-- Drift Unknown Table
-- ============================================================================

-- Requirements that couldn't be automatically verified
CREATE TABLE drift_unknown (
    unknown_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID REFERENCES drift_report(report_id) ON DELETE CASCADE,
    requirement_id UUID REFERENCES specification_requirement(requirement_id),
    requirement_key VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    manual_review_needed BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

-- Create indexes
CREATE INDEX idx_unknown_report ON drift_unknown(report_id);
CREATE INDEX idx_unknown_requirement ON drift_unknown(requirement_id);
CREATE INDEX idx_unknown_manual_review ON drift_unknown(manual_review_needed);

-- ============================================================================
-- Comments and Documentation
-- ============================================================================

COMMENT ON TABLE drift_report IS
'Stores drift analysis reports comparing specifications to implementation';

COMMENT ON COLUMN drift_report.report_id IS
'Unique identifier for the drift analysis report';

COMMENT ON COLUMN drift_report.spec_id IS
'Reference to the specification being validated';

COMMENT ON COLUMN drift_report.commit_sha IS
'Git commit SHA of the kernel code analyzed';

COMMENT ON COLUMN drift_report.config_name IS
'Kernel configuration used for the analysis';

COMMENT ON COLUMN drift_report.total_requirements IS
'Total number of requirements checked';

COMMENT ON COLUMN drift_report.severity IS
'Overall severity of violations found';

COMMENT ON COLUMN drift_report.execution_time_ms IS
'Time taken to complete the analysis in milliseconds';

COMMENT ON TABLE drift_violation IS
'Individual violations where implementation doesn''t match specification';

COMMENT ON COLUMN drift_violation.expected_behavior IS
'What the specification requires';

COMMENT ON COLUMN drift_violation.actual_behavior IS
'What was found in the implementation';

COMMENT ON COLUMN drift_violation.suggested_fix IS
'Recommended remediation for the violation';

COMMENT ON TABLE drift_conformance IS
'Requirements that passed validation';

COMMENT ON COLUMN drift_conformance.evidence IS
'Proof that the requirement is satisfied';

COMMENT ON COLUMN drift_conformance.confidence_score IS
'Confidence level of the conformance (0.0 to 1.0)';

COMMENT ON TABLE drift_unknown IS
'Requirements that couldn''t be automatically verified';

COMMENT ON COLUMN drift_unknown.reason IS
'Why the requirement couldn''t be verified';

COMMENT ON COLUMN drift_unknown.manual_review_needed IS
'Whether manual review is required';

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to calculate drift report severity based on violations
CREATE OR REPLACE FUNCTION calculate_drift_severity(
    p_report_id UUID
) RETURNS VARCHAR AS $$
DECLARE
    v_critical_count INTEGER;
    v_major_count INTEGER;
    v_severity VARCHAR(20);
BEGIN
    SELECT
        COUNT(*) FILTER (WHERE severity = 'critical'),
        COUNT(*) FILTER (WHERE severity = 'major')
    INTO v_critical_count, v_major_count
    FROM drift_violation
    WHERE report_id = p_report_id;

    IF v_critical_count > 0 THEN
        v_severity := 'critical';
    ELSIF v_major_count > 0 THEN
        v_severity := 'major';
    ELSIF EXISTS (SELECT 1 FROM drift_violation WHERE report_id = p_report_id AND severity = 'minor') THEN
        v_severity := 'minor';
    ELSIF EXISTS (SELECT 1 FROM drift_violation WHERE report_id = p_report_id) THEN
        v_severity := 'info';
    ELSE
        v_severity := 'none';
    END IF;

    RETURN v_severity;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get drift report summary
CREATE OR REPLACE FUNCTION get_drift_summary(
    p_report_id UUID
) RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'report_id', report_id,
        'spec_id', spec_id,
        'commit_sha', commit_sha,
        'config_name', config_name,
        'total_requirements', total_requirements,
        'passed', passed,
        'failed', failed,
        'unknown', unknown,
        'skipped', skipped,
        'pass_rate', CASE
            WHEN total_requirements > 0
            THEN ROUND((passed::DECIMAL / total_requirements) * 100, 2)
            ELSE 0
        END,
        'severity', severity,
        'generated_at', generated_at,
        'execution_time_ms', execution_time_ms,
        'violations_by_severity', (
            SELECT jsonb_object_agg(severity, count)
            FROM (
                SELECT severity, COUNT(*) as count
                FROM drift_violation
                WHERE report_id = p_report_id
                GROUP BY severity
            ) t
        )
    )
    INTO v_result
    FROM drift_report
    WHERE report_id = p_report_id;

    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to find reports by specification and commit
CREATE OR REPLACE FUNCTION find_drift_reports(
    p_spec_id UUID DEFAULT NULL,
    p_commit_sha VARCHAR DEFAULT NULL,
    p_config_name VARCHAR DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    report_id UUID,
    spec_id UUID,
    commit_sha VARCHAR,
    config_name VARCHAR,
    severity VARCHAR,
    pass_rate DECIMAL,
    generated_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dr.report_id,
        dr.spec_id,
        dr.commit_sha,
        dr.config_name,
        dr.severity,
        CASE
            WHEN dr.total_requirements > 0
            THEN ROUND((dr.passed::DECIMAL / dr.total_requirements) * 100, 2)
            ELSE 0
        END as pass_rate,
        dr.generated_at
    FROM drift_report dr
    WHERE (p_spec_id IS NULL OR dr.spec_id = p_spec_id)
      AND (p_commit_sha IS NULL OR dr.commit_sha = p_commit_sha)
      AND (p_config_name IS NULL OR dr.config_name = p_config_name)
    ORDER BY dr.generated_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Trigger to update drift report severity after violations are added
CREATE OR REPLACE FUNCTION update_drift_report_severity()
RETURNS TRIGGER AS $$
DECLARE
    v_severity VARCHAR(20);
BEGIN
    -- Calculate new severity
    v_severity := calculate_drift_severity(NEW.report_id);

    -- Update the drift report
    UPDATE drift_report
    SET severity = v_severity
    WHERE report_id = NEW.report_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER drift_violation_update_severity
    AFTER INSERT OR UPDATE OR DELETE ON drift_violation
    FOR EACH ROW
    EXECUTE FUNCTION update_drift_report_severity();

COMMIT;
