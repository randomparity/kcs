use crate::{DriftFinding, DriftReport, DriftSeverity, DriftType};
use anyhow::Result;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub struct ReportGenerator {
    report: DriftReport,
    output_format: OutputFormat,
}

#[derive(Debug, Clone)]
pub enum OutputFormat {
    Json,
    Html,
    Markdown,
    Text,
    Sarif, // Static Analysis Results Interchange Format
}

impl ReportGenerator {
    pub fn new(report: DriftReport) -> Self {
        Self {
            report,
            output_format: OutputFormat::Json,
        }
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn generate(&self) -> Result<String> {
        match self.output_format {
            OutputFormat::Json => self.generate_json(),
            OutputFormat::Html => self.generate_html(),
            OutputFormat::Markdown => self.generate_markdown(),
            OutputFormat::Text => self.generate_text(),
            OutputFormat::Sarif => self.generate_sarif(),
        }
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = self.generate()?;
        let mut file = File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    fn generate_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.report)?)
    }

    fn generate_html(&self) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Kernel Drift Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(include_str!("../resources/report_style.css"));
        html.push_str("</style>\n</head>\n<body>\n");

        // Header
        html.push_str("<div class=\"header\">\n");
        html.push_str("<h1>Kernel Specification Drift Report</h1>\n");
        html.push_str(&format!(
            "<div class=\"timestamp\">Generated: {}</div>\n",
            self.report.metadata.analysis_time
        ));
        html.push_str("</div>\n");

        // Executive Summary
        html.push_str("<div class=\"summary\">\n");
        html.push_str("<h2>Executive Summary</h2>\n");
        html.push_str(&format!(
            "<div class=\"compliance-score\">Compliance Score: {:.1}%</div>\n",
            self.report.compliance_score
        ));

        html.push_str("<div class=\"stats\">\n");
        html.push_str(&format!(
            "<div>Total Requirements: {}</div>\n",
            self.report.metadata.total_requirements
        ));
        html.push_str(&format!(
            "<div>Requirements Checked: {}</div>\n",
            self.report.metadata.requirements_checked
        ));
        html.push_str(&format!(
            "<div>Total Drift Findings: {}</div>\n",
            self.report.metadata.drift_count
        ));
        html.push_str(&format!(
            "<div class=\"critical\">Critical Issues: {}</div>\n",
            self.report.metadata.critical_drift_count
        ));
        html.push_str("</div>\n</div>\n");

        // Findings by severity
        html.push_str("<div class=\"findings\">\n");
        html.push_str("<h2>Drift Findings</h2>\n");

        let mut findings_by_severity: HashMap<DriftSeverity, Vec<&DriftFinding>> = HashMap::new();
        for finding in &self.report.drift_findings {
            findings_by_severity
                .entry(finding.severity.clone())
                .or_default()
                .push(finding);
        }

        for severity in &[
            DriftSeverity::Critical,
            DriftSeverity::High,
            DriftSeverity::Medium,
            DriftSeverity::Low,
            DriftSeverity::Info,
        ] {
            if let Some(findings) = findings_by_severity.get(severity) {
                html.push_str(&format!(
                    "<h3 class=\"severity-{:?}\">{:?} Severity ({} issues)</h3>\n",
                    severity,
                    severity,
                    findings.len()
                ));
                html.push_str("<div class=\"finding-list\">\n");

                for finding in findings {
                    html.push_str("<div class=\"finding\">\n");
                    html.push_str(&format!("<h4>{}</h4>\n", finding.description));
                    html.push_str(&format!(
                        "<div class=\"requirement-id\">Requirement: {}</div>\n",
                        finding.requirement_id
                    ));
                    html.push_str(&format!(
                        "<div class=\"drift-type\">Type: {:?}</div>\n",
                        finding.drift_type
                    ));
                    html.push_str(&format!(
                        "<div class=\"expected\">Expected: {}</div>\n",
                        finding.expected
                    ));
                    html.push_str(&format!(
                        "<div class=\"actual\">Actual: {}</div>\n",
                        finding.actual
                    ));

                    if !finding.file_locations.is_empty() {
                        html.push_str("<div class=\"locations\">Locations:\n<ul>\n");
                        for location in &finding.file_locations {
                            html.push_str(&format!(
                                "<li>{}:{}</li>\n",
                                location.file_path, location.line_number
                            ));
                        }
                        html.push_str("</ul></div>\n");
                    }

                    html.push_str(&format!(
                        "<div class=\"remediation\">Remediation: {}</div>\n",
                        finding.remediation_suggestion
                    ));
                    html.push_str("</div>\n");
                }

                html.push_str("</div>\n");
            }
        }
        html.push_str("</div>\n");

        // Recommendations
        if !self.report.recommendations.is_empty() {
            html.push_str("<div class=\"recommendations\">\n");
            html.push_str("<h2>Recommendations</h2>\n<ul>\n");
            for rec in &self.report.recommendations {
                html.push_str(&format!("<li>{}</li>\n", rec));
            }
            html.push_str("</ul>\n</div>\n");
        }

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    fn generate_markdown(&self) -> Result<String> {
        let mut md = String::new();

        // Header
        md.push_str("# Kernel Specification Drift Report\n\n");
        md.push_str(&format!(
            "**Generated:** {}\n\n",
            self.report.metadata.analysis_time
        ));

        // Executive Summary
        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!(
            "**Compliance Score:** {:.1}%\n\n",
            self.report.compliance_score
        ));

        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Total Requirements | {} |\n",
            self.report.metadata.total_requirements
        ));
        md.push_str(&format!(
            "| Requirements Checked | {} |\n",
            self.report.metadata.requirements_checked
        ));
        md.push_str(&format!(
            "| Total Drift Findings | {} |\n",
            self.report.metadata.drift_count
        ));
        md.push_str(&format!(
            "| Critical Issues | {} |\n",
            self.report.metadata.critical_drift_count
        ));
        md.push_str(&format!(
            "| Analysis Time | {}ms |\n\n",
            self.report.metadata.execution_time_ms
        ));

        // Specification Info
        md.push_str("## Specification Information\n\n");
        md.push_str(&format!(
            "- **Source:** {}\n",
            self.report.specification.source
        ));
        md.push_str(&format!(
            "- **Version:** {}\n",
            self.report.specification.version
        ));
        md.push_str(&format!(
            "- **Requirements:** {}\n\n",
            self.report.specification.requirements.len()
        ));

        // Implementation Info
        md.push_str("## Implementation Information\n\n");
        md.push_str(&format!(
            "- **Kernel Version:** {}\n",
            self.report.implementation.kernel_version
        ));
        md.push_str(&format!(
            "- **Commit Hash:** {}\n",
            self.report.implementation.commit_hash
        ));
        md.push_str(&format!(
            "- **Analyzed Symbols:** {}\n\n",
            self.report.implementation.analyzed_symbols
        ));

        // Findings by severity
        md.push_str("## Drift Findings\n\n");

        let mut findings_by_severity: HashMap<DriftSeverity, Vec<&DriftFinding>> = HashMap::new();
        for finding in &self.report.drift_findings {
            findings_by_severity
                .entry(finding.severity.clone())
                .or_default()
                .push(finding);
        }

        for severity in &[
            DriftSeverity::Critical,
            DriftSeverity::High,
            DriftSeverity::Medium,
            DriftSeverity::Low,
            DriftSeverity::Info,
        ] {
            if let Some(findings) = findings_by_severity.get(severity) {
                md.push_str(&format!(
                    "### {:?} Severity ({} issues)\n\n",
                    severity,
                    findings.len()
                ));

                for finding in findings {
                    md.push_str(&format!("#### {}\n\n", finding.description));
                    md.push_str(&format!("- **Requirement:** {}\n", finding.requirement_id));
                    md.push_str(&format!("- **Type:** {:?}\n", finding.drift_type));
                    md.push_str(&format!("- **Expected:** `{}`\n", finding.expected));
                    md.push_str(&format!("- **Actual:** `{}`\n", finding.actual));

                    if !finding.affected_symbols.is_empty() {
                        md.push_str("- **Affected Symbols:**\n");
                        for symbol in &finding.affected_symbols {
                            md.push_str(&format!("  - `{}`\n", symbol));
                        }
                    }

                    if !finding.file_locations.is_empty() {
                        md.push_str("- **Locations:**\n");
                        for location in &finding.file_locations {
                            md.push_str(&format!(
                                "  - `{}:{}`\n",
                                location.file_path, location.line_number
                            ));
                        }
                    }

                    md.push_str(&format!(
                        "- **Remediation:** {}\n\n",
                        finding.remediation_suggestion
                    ));
                }
            }
        }

        // Recommendations
        if !self.report.recommendations.is_empty() {
            md.push_str("## Recommendations\n\n");
            for rec in &self.report.recommendations {
                md.push_str(&format!("- {}\n", rec));
            }
            md.push('\n');
        }

        // Configuration Context
        if !self.report.implementation.config.is_empty() {
            md.push_str("## Configuration Context\n\n");
            md.push_str("| Config Option | Enabled |\n");
            md.push_str("|---------------|----------|\n");
            for (config, enabled) in &self.report.implementation.config {
                md.push_str(&format!("| {} | {} |\n", config, enabled));
            }
            md.push('\n');
        }

        Ok(md)
    }

    fn generate_text(&self) -> Result<String> {
        let mut text = String::new();

        // Header
        text.push_str("=".repeat(80).as_str());
        text.push('\n');
        text.push_str("KERNEL SPECIFICATION DRIFT REPORT\n");
        text.push_str("=".repeat(80).as_str());
        text.push_str("\n\n");

        text.push_str(&format!(
            "Generated: {}\n",
            self.report.metadata.analysis_time
        ));
        text.push_str(&format!(
            "Compliance Score: {:.1}%\n\n",
            self.report.compliance_score
        ));

        // Summary stats
        text.push_str("SUMMARY\n");
        text.push_str("-".repeat(40).as_str());
        text.push('\n');
        text.push_str(&format!(
            "Total Requirements:    {}\n",
            self.report.metadata.total_requirements
        ));
        text.push_str(&format!(
            "Requirements Checked:  {}\n",
            self.report.metadata.requirements_checked
        ));
        text.push_str(&format!(
            "Total Drift Findings:  {}\n",
            self.report.metadata.drift_count
        ));
        text.push_str(&format!(
            "Critical Issues:       {}\n",
            self.report.metadata.critical_drift_count
        ));
        text.push_str(&format!(
            "Analysis Time:         {}ms\n\n",
            self.report.metadata.execution_time_ms
        ));

        // Findings
        if !self.report.drift_findings.is_empty() {
            text.push_str("DRIFT FINDINGS\n");
            text.push_str("=".repeat(80).as_str());
            text.push_str("\n\n");

            for (index, finding) in self.report.drift_findings.iter().enumerate() {
                text.push_str(&format!("Finding #{}\n", index + 1));
                text.push_str("-".repeat(40).as_str());
                text.push('\n');
                text.push_str(&format!("Severity:     {:?}\n", finding.severity));
                text.push_str(&format!("Type:         {:?}\n", finding.drift_type));
                text.push_str(&format!("Requirement:  {}\n", finding.requirement_id));
                text.push_str(&format!("Description:  {}\n", finding.description));
                text.push_str(&format!("Expected:     {}\n", finding.expected));
                text.push_str(&format!("Actual:       {}\n", finding.actual));

                if !finding.file_locations.is_empty() {
                    text.push_str("Locations:\n");
                    for location in &finding.file_locations {
                        text.push_str(&format!(
                            "  - {}:{}\n",
                            location.file_path, location.line_number
                        ));
                    }
                }

                text.push_str(&format!(
                    "Remediation:  {}\n\n",
                    finding.remediation_suggestion
                ));
            }
        }

        // Recommendations
        if !self.report.recommendations.is_empty() {
            text.push_str("RECOMMENDATIONS\n");
            text.push_str("=".repeat(80).as_str());
            text.push_str("\n\n");
            for (index, rec) in self.report.recommendations.iter().enumerate() {
                text.push_str(&format!("{}. {}\n", index + 1, rec));
            }
            text.push('\n');
        }

        Ok(text)
    }

    fn generate_sarif(&self) -> Result<String> {
        // SARIF (Static Analysis Results Interchange Format) for CI/CD integration
        let sarif = serde_json::json!({
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "kcs-drift",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/yourusername/kcs",
                        "rules": self.generate_sarif_rules()
                    }
                },
                "results": self.generate_sarif_results(),
                "invocations": [{
                    "executionSuccessful": true,
                    "endTimeUtc": self.report.metadata.analysis_time.to_rfc3339()
                }]
            }]
        });

        Ok(serde_json::to_string_pretty(&sarif)?)
    }

    fn generate_sarif_rules(&self) -> Vec<serde_json::Value> {
        let mut rules = Vec::new();
        let mut seen_requirements = std::collections::HashSet::new();

        for finding in &self.report.drift_findings {
            if seen_requirements.insert(&finding.requirement_id) {
                rules.push(serde_json::json!({
                    "id": finding.requirement_id,
                    "name": format!("{:?}", finding.drift_type),
                    "shortDescription": {
                        "text": finding.description.clone()
                    },
                    "defaultConfiguration": {
                        "level": self.severity_to_sarif_level(&finding.severity)
                    }
                }));
            }
        }

        rules
    }

    fn generate_sarif_results(&self) -> Vec<serde_json::Value> {
        self.report
            .drift_findings
            .iter()
            .map(|finding| {
                let mut result = serde_json::json!({
                    "ruleId": finding.requirement_id,
                    "level": self.severity_to_sarif_level(&finding.severity),
                    "message": {
                        "text": format!("{}\nExpected: {}\nActual: {}",
                            finding.description, finding.expected, finding.actual)
                    }
                });

                if !finding.file_locations.is_empty() {
                    let locations: Vec<_> = finding
                        .file_locations
                        .iter()
                        .map(|loc| {
                            serde_json::json!({
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": loc.file_path
                                    },
                                    "region": {
                                        "startLine": loc.line_number,
                                        "startColumn": loc.column.unwrap_or(1)
                                    }
                                }
                            })
                        })
                        .collect();

                    result["locations"] = serde_json::json!(locations);
                }

                result["fixes"] = serde_json::json!([{
                    "description": {
                        "text": finding.remediation_suggestion.clone()
                    }
                }]);

                result
            })
            .collect()
    }

    fn severity_to_sarif_level(&self, severity: &DriftSeverity) -> &'static str {
        match severity {
            DriftSeverity::Critical | DriftSeverity::High => "error",
            DriftSeverity::Medium => "warning",
            DriftSeverity::Low | DriftSeverity::Info => "note",
        }
    }
}

pub fn generate_summary_statistics(reports: &[DriftReport]) -> SummaryStatistics {
    let total_findings: usize = reports.iter().map(|r| r.drift_findings.len()).sum();
    let total_critical: usize = reports
        .iter()
        .map(|r| r.metadata.critical_drift_count)
        .sum();

    let avg_compliance = if reports.is_empty() {
        0.0
    } else {
        reports.iter().map(|r| r.compliance_score).sum::<f64>() / reports.len() as f64
    };

    let mut finding_types: HashMap<DriftType, usize> = HashMap::new();
    for report in reports {
        for finding in &report.drift_findings {
            *finding_types.entry(finding.drift_type.clone()).or_insert(0) += 1;
        }
    }

    SummaryStatistics {
        total_reports: reports.len(),
        total_findings,
        total_critical,
        average_compliance_score: avg_compliance,
        finding_type_distribution: finding_types,
    }
}

#[derive(Debug, Clone)]
pub struct SummaryStatistics {
    pub total_reports: usize,
    pub total_findings: usize,
    pub total_critical: usize,
    pub average_compliance_score: f64,
    pub finding_type_distribution: HashMap<DriftType, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec_parser::create_example_specification;
    use chrono::Utc;

    fn create_test_report() -> DriftReport {
        DriftReport {
            specification: crate::SpecificationInfo {
                source: "test_spec.yaml".to_string(),
                version: "1.0".to_string(),
                timestamp: Utc::now(),
                requirements: create_example_specification().requirements,
            },
            implementation: crate::ImplementationInfo {
                kernel_version: "5.15.0".to_string(),
                commit_hash: "abc123".to_string(),
                config: HashMap::new(),
                analyzed_symbols: 1000,
                analysis_timestamp: Utc::now(),
            },
            drift_findings: vec![DriftFinding {
                requirement_id: "REQ_001".to_string(),
                drift_type: DriftType::MissingSymbol,
                severity: DriftSeverity::Critical,
                description: "Required symbol not found".to_string(),
                expected: "sys_test".to_string(),
                actual: "Not found".to_string(),
                affected_symbols: vec!["sys_test".to_string()],
                file_locations: vec![],
                remediation_suggestion: "Implement sys_test".to_string(),
            }],
            compliance_score: 75.0,
            recommendations: vec!["Fix critical issues".to_string()],
            metadata: crate::ReportMetadata {
                analysis_time: Utc::now(),
                execution_time_ms: 1234,
                total_requirements: 10,
                requirements_checked: 8,
                drift_count: 1,
                critical_drift_count: 1,
            },
        }
    }

    #[test]
    fn test_json_generation() {
        let report = create_test_report();
        let generator = ReportGenerator::new(report);
        let json = generator.generate().unwrap();
        assert!(json.contains("\"compliance_score\":"));
        assert!(json.contains("\"drift_findings\":"));
    }

    #[test]
    fn test_markdown_generation() {
        let report = create_test_report();
        let generator = ReportGenerator::new(report).with_format(OutputFormat::Markdown);
        let md = generator.generate().unwrap();
        assert!(md.contains("# Kernel Specification Drift Report"));
        assert!(md.contains("## Executive Summary"));
        assert!(md.contains("**Compliance Score:**"));
    }

    #[test]
    fn test_text_generation() {
        let report = create_test_report();
        let generator = ReportGenerator::new(report).with_format(OutputFormat::Text);
        let text = generator.generate().unwrap();
        assert!(text.contains("KERNEL SPECIFICATION DRIFT REPORT"));
        assert!(text.contains("SUMMARY"));
        assert!(text.contains("Compliance Score:"));
    }

    #[test]
    fn test_sarif_generation() {
        let report = create_test_report();
        let generator = ReportGenerator::new(report).with_format(OutputFormat::Sarif);
        let sarif = generator.generate().unwrap();
        let json: serde_json::Value = serde_json::from_str(&sarif).unwrap();
        assert_eq!(json["version"], "2.1.0");
        assert!(json["runs"].is_array());
    }

    #[test]
    fn test_summary_statistics() {
        let reports = vec![create_test_report(), create_test_report()];
        let stats = generate_summary_statistics(&reports);
        assert_eq!(stats.total_reports, 2);
        assert_eq!(stats.total_findings, 2);
        assert_eq!(stats.total_critical, 2);
        assert_eq!(stats.average_compliance_score, 75.0);
    }
}
