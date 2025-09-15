-- KCS Aggregate Tables Migration
-- Creates summaries, drift reports, and test coverage tracking

-- Drift report severity levels
CREATE TYPE drift_severity AS ENUM (
    'blocker',
    'warning',
    'info'
);

-- Test suite types
CREATE TYPE test_suite_type AS ENUM (
    'kunit',
    'kselftest',
    'ltp',
    'kernel_ci',
    'custom'
);

-- Test coverage types
CREATE TYPE test_coverage_type AS ENUM (
    'function',
    'line',
    'branch',
    'integration',
    'regression'
);

-- Symbol summaries - AI-generated or rule-based documentation
CREATE TABLE summary (
    symbol_id BIGINT PRIMARY KEY REFERENCES symbol(id) ON DELETE CASCADE,
    content JSONB NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version TEXT NOT NULL,
    confidence_score REAL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    human_reviewed BOOLEAN DEFAULT FALSE,
    review_notes TEXT,

    -- Constraints
    CONSTRAINT summary_content_not_empty CHECK (jsonb_typeof(content) = 'object'),
    CONSTRAINT summary_model_version_not_empty CHECK (length(model_version) > 0)
);

-- Drift reports - mismatches between spec and implementation
CREATE TABLE drift_report (
    id BIGSERIAL PRIMARY KEY,
    feature_id TEXT NOT NULL,
    mismatches JSONB NOT NULL,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    severity drift_severity NOT NULL,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,

    -- Constraints
    CONSTRAINT drift_feature_id_not_empty CHECK (length(feature_id) > 0),
    CONSTRAINT drift_mismatches_array CHECK (jsonb_typeof(mismatches) = 'array'),
    CONSTRAINT drift_resolution_check CHECK (
        (resolved_at IS NULL AND resolution_notes IS NULL) OR
        (resolved_at IS NOT NULL AND length(resolution_notes) > 0)
    )
);

-- Test coverage mapping
CREATE TABLE test_coverage (
    id BIGSERIAL PRIMARY KEY,
    symbol_id BIGINT NOT NULL REFERENCES symbol(id) ON DELETE CASCADE,
    test_suite test_suite_type NOT NULL,
    test_name TEXT NOT NULL,
    test_type test_coverage_type NOT NULL,
    test_file TEXT,
    coverage_percentage REAL CHECK (coverage_percentage >= 0.0 AND coverage_percentage <= 100.0),
    last_run TIMESTAMP WITH TIME ZONE,
    status TEXT, -- passed, failed, skipped, timeout

    -- Constraints
    CONSTRAINT test_name_not_empty CHECK (length(test_name) > 0),
    CONSTRAINT test_unique_per_symbol UNIQUE (symbol_id, test_suite, test_name, test_type)
);

-- Risk assessment for symbols
CREATE TABLE risk_assessment (
    symbol_id BIGINT PRIMARY KEY REFERENCES symbol(id) ON DELETE CASCADE,
    risk_factors JSONB NOT NULL,
    risk_score REAL NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 10.0),
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    assessment_version TEXT NOT NULL,

    -- Constraints
    CONSTRAINT risk_factors_array CHECK (jsonb_typeof(risk_factors) = 'array')
);

-- Impact analysis cache
CREATE TABLE impact_analysis (
    id BIGSERIAL PRIMARY KEY,
    change_hash TEXT NOT NULL, -- Hash of the change (diff, files, symbols)
    configs TEXT[] NOT NULL,
    modules TEXT[] NOT NULL,
    tests TEXT[] NOT NULL,
    owners TEXT[] NOT NULL,
    risks TEXT[] NOT NULL,
    citations JSONB NOT NULL, -- Array of span objects
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days'),

    -- Constraints
    CONSTRAINT impact_change_hash_not_empty CHECK (length(change_hash) > 0),
    CONSTRAINT impact_citations_array CHECK (jsonb_typeof(citations) = 'array')
);

-- Citation span tracking
CREATE TABLE citation (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    context TEXT, -- Optional context description
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT citation_valid_lines CHECK (start_line <= end_line AND start_line > 0)
);

-- Indexes for aggregate tables
CREATE INDEX idx_summary_model_version ON summary(model_version);
CREATE INDEX idx_summary_generated_at ON summary(generated_at);
CREATE INDEX idx_summary_human_reviewed ON summary(human_reviewed);

CREATE INDEX idx_drift_feature_id ON drift_report(feature_id);
CREATE INDEX idx_drift_severity ON drift_report(severity);
CREATE INDEX idx_drift_checked_at ON drift_report(checked_at);
CREATE INDEX idx_drift_resolved ON drift_report(resolved_at) WHERE resolved_at IS NOT NULL;

CREATE INDEX idx_test_coverage_symbol ON test_coverage(symbol_id);
CREATE INDEX idx_test_coverage_suite ON test_coverage(test_suite);
CREATE INDEX idx_test_coverage_type ON test_coverage(test_type);
CREATE INDEX idx_test_coverage_status ON test_coverage(status);

CREATE INDEX idx_risk_score ON risk_assessment(risk_score);
CREATE INDEX idx_risk_assessed_at ON risk_assessment(assessed_at);

CREATE INDEX idx_impact_change_hash ON impact_analysis(change_hash);
CREATE INDEX idx_impact_computed_at ON impact_analysis(computed_at);
CREATE INDEX idx_impact_expires_at ON impact_analysis(expires_at);

CREATE INDEX idx_citation_file_id ON citation(file_id);
CREATE INDEX idx_citation_lines ON citation(start_line, end_line);

-- GIN indexes for JSONB fields
CREATE INDEX idx_summary_content_gin ON summary USING gin(content);
CREATE INDEX idx_drift_mismatches_gin ON drift_report USING gin(mismatches);
CREATE INDEX idx_risk_factors_gin ON risk_assessment USING gin(risk_factors);
CREATE INDEX idx_impact_citations_gin ON impact_analysis USING gin(citations);

-- Partial indexes for common queries
CREATE INDEX idx_summary_unreviewed ON summary(symbol_id) WHERE human_reviewed = FALSE;
CREATE INDEX idx_drift_unresolved ON drift_report(feature_id, severity) WHERE resolved_at IS NULL;
CREATE INDEX idx_test_failed ON test_coverage(symbol_id, test_suite) WHERE status IN ('failed', 'timeout');

-- Comments for documentation
COMMENT ON TABLE summary IS 'AI-generated or rule-based symbol documentation';
COMMENT ON COLUMN summary.content IS 'Structured summary with purpose, inputs, outputs, side effects, etc.';
COMMENT ON COLUMN summary.confidence_score IS 'AI confidence in summary accuracy (0.0-1.0)';

COMMENT ON TABLE drift_report IS 'Mismatches between specification and implementation';
COMMENT ON COLUMN drift_report.feature_id IS 'Identifier for the feature being checked';
COMMENT ON COLUMN drift_report.mismatches IS 'Array of mismatch objects with kind, detail, span';

COMMENT ON TABLE test_coverage IS 'Mapping between symbols and test coverage';
COMMENT ON COLUMN test_coverage.coverage_percentage IS 'Percentage of symbol covered by this test';

COMMENT ON TABLE risk_assessment IS 'Risk scoring for symbols based on context and usage';
COMMENT ON COLUMN risk_assessment.risk_factors IS 'Array of risk factor strings';
COMMENT ON COLUMN risk_assessment.risk_score IS 'Overall risk score (0.0-10.0)';

COMMENT ON TABLE impact_analysis IS 'Cached impact analysis results';
COMMENT ON COLUMN impact_analysis.change_hash IS 'Hash of input (diff/files/symbols) for cache key';
COMMENT ON COLUMN impact_analysis.expires_at IS 'Cache expiration time';

COMMENT ON TABLE citation IS 'File/line references for traceability';

-- Example JSONB schemas (for documentation)
/*
summary.content schema:
{
  "purpose": "High-level description of what the symbol does",
  "inputs": ["param1: description", "param2: description"],
  "outputs": "Return value description",
  "side_effects": ["Effect 1", "Effect 2"],
  "concurrency": {
    "can_sleep": true,
    "locking": ["lock1", "lock2"],
    "rcu": "rcu_read_lock required",
    "irq_safe": false
  },
  "error_paths": ["EINVAL: Invalid input", "ENOMEM: Out of memory"],
  "invariants": ["Precondition 1", "Postcondition 1"],
  "tests": ["test1.c", "test2.c"],
  "citations": [
    {"file": "fs/read_write.c", "line": 451}
  ]
}

drift_report.mismatches schema:
[
  {
    "kind": "missing_abi_doc",
    "detail": "sysfs attribute lacks ABI documentation",
    "path": "/sys/kernel/kcs/status",
    "span": {"file": "kernel/kcs.c", "line": 123}
  },
  {
    "kind": "kconfig_mismatch",
    "detail": "Help text doesn't match implementation",
    "option": "CONFIG_KCS_ENABLE",
    "span": {"file": "init/Kconfig", "line": 456}
  }
]

risk_assessment.risk_factors examples:
[
  "holds_spinlock",
  "in_irq_context",
  "sleeps_in_atomic",
  "no_rcu_protection",
  "unbounded_allocation",
  "user_input_unchecked"
]

impact_analysis.citations schema:
[
  {
    "path": "fs/read_write.c",
    "sha": "abc123def456",
    "start": 451,
    "end": 465
  }
]
*/

-- Cleanup expired impact analysis entries (run by cron job)
CREATE OR REPLACE FUNCTION cleanup_expired_impact_analysis()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM impact_analysis WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
