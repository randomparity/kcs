use crate::{ChangePoint, ChangeType, ImpactLevel};
use anyhow::Result;
use git2::{Diff, DiffFormat, DiffOptions, Repository};
use std::collections::HashMap;
use std::path::Path;

pub struct GitDiffAnalyzer {
    repo: Repository,
}

impl GitDiffAnalyzer {
    pub fn new<P: AsRef<Path>>(repo_path: P) -> Result<Self> {
        let repo = Repository::open(repo_path)?;
        Ok(Self { repo })
    }

    pub fn analyze_commit(&self, commit_sha: &str) -> Result<Vec<ChangePoint>> {
        let commit = self
            .repo
            .find_commit(self.repo.revparse_single(commit_sha)?.id())?;
        let parent = commit.parent(0)?;

        let commit_tree = commit.tree()?;
        let parent_tree = parent.tree()?;

        let mut diff_opts = DiffOptions::new();
        diff_opts.context_lines(3);

        let diff = self.repo.diff_tree_to_tree(
            Some(&parent_tree),
            Some(&commit_tree),
            Some(&mut diff_opts),
        )?;

        self.extract_changes_from_diff(&diff)
    }

    pub fn analyze_range(&self, from_ref: &str, to_ref: &str) -> Result<Vec<ChangePoint>> {
        let from_commit = self
            .repo
            .find_commit(self.repo.revparse_single(from_ref)?.id())?;
        let to_commit = self
            .repo
            .find_commit(self.repo.revparse_single(to_ref)?.id())?;

        let from_tree = from_commit.tree()?;
        let to_tree = to_commit.tree()?;

        let mut diff_opts = DiffOptions::new();
        diff_opts.context_lines(3);

        let diff =
            self.repo
                .diff_tree_to_tree(Some(&from_tree), Some(&to_tree), Some(&mut diff_opts))?;

        self.extract_changes_from_diff(&diff)
    }

    pub fn analyze_working_directory(&self) -> Result<Vec<ChangePoint>> {
        let head = self.repo.head()?.peel_to_tree()?;

        let mut diff_opts = DiffOptions::new();
        diff_opts.include_untracked(true);
        diff_opts.context_lines(3);

        let diff = self
            .repo
            .diff_tree_to_workdir_with_index(Some(&head), Some(&mut diff_opts))?;

        self.extract_changes_from_diff(&diff)
    }

    fn extract_changes_from_diff(&self, diff: &Diff) -> Result<Vec<ChangePoint>> {
        let mut changes = Vec::new();

        diff.print(DiffFormat::Patch, |_delta, _hunk, _line| {
            // This is a simplified extraction - in practice we'd need more sophisticated parsing
            // For now, just detect file-level changes
            true
        })?;

        diff.foreach(
            &mut |delta, _progress| {
                if let Some(new_file) = delta.new_file().path() {
                    let file_path = new_file.to_string_lossy().to_string();

                    let change_type = match delta.status() {
                        git2::Delta::Added => ChangeType::FunctionAdded,
                        git2::Delta::Deleted => ChangeType::FunctionRemoved,
                        git2::Delta::Modified => ChangeType::FunctionModified,
                        _ => ChangeType::FunctionModified,
                    };

                    changes.push(ChangePoint {
                        file_path,
                        line_number: 0, // Would need hunk analysis for exact line
                        change_type,
                        symbol_name: None,
                        description: format!(
                            "File {} {:?}",
                            delta
                                .new_file()
                                .path()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "unknown".to_string()),
                            delta.status()
                        ),
                        diff_context: String::new(),
                    });
                }
                true
            },
            None,
            None,
            None,
        )?;

        Ok(changes)
    }
}

pub struct SemanticDiffAnalyzer {
    patterns: HashMap<String, Vec<regex::Regex>>,
}

impl SemanticDiffAnalyzer {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Function-related patterns
        let function_patterns = vec![
            regex::Regex::new(
                r"^\s*(?:static\s+)?(?:inline\s+)?(?:extern\s+)?\w+\s+(\w+)\s*\([^)]*\)\s*\{",
            )
            .unwrap(),
            regex::Regex::new(r"^\s*SYSCALL_DEFINE\d+\((\w+)").unwrap(),
            regex::Regex::new(r"^\s*EXPORT_SYMBOL(?:_GPL)?\((\w+)\)").unwrap(),
        ];
        patterns.insert("function".to_string(), function_patterns);

        // Structure patterns
        let struct_patterns = vec![
            regex::Regex::new(r"^\s*struct\s+(\w+)\s*\{").unwrap(),
            regex::Regex::new(r"^\s*union\s+(\w+)\s*\{").unwrap(),
            regex::Regex::new(r"^\s*enum\s+(\w+)\s*\{").unwrap(),
            regex::Regex::new(r"^\s*typedef\s+(?:struct|union|enum)\s+\w*\s*\{[^}]+\}\s*(\w+);")
                .unwrap(),
        ];
        patterns.insert("struct".to_string(), struct_patterns);

        // Macro patterns
        let macro_patterns = vec![
            regex::Regex::new(r"^\s*#define\s+(\w+)").unwrap(),
            regex::Regex::new(r"^\s*#undef\s+(\w+)").unwrap(),
        ];
        patterns.insert("macro".to_string(), macro_patterns);

        // Config patterns
        let config_patterns = vec![
            regex::Regex::new(r"^\s*config\s+(CONFIG_\w+)").unwrap(),
            regex::Regex::new(r"^\s*#ifdef\s+(CONFIG_\w+)").unwrap(),
            regex::Regex::new(r"^\s*#ifndef\s+(CONFIG_\w+)").unwrap(),
            regex::Regex::new(r"^\s*#if.*\b(CONFIG_\w+)\b").unwrap(),
        ];
        patterns.insert("config".to_string(), config_patterns);

        Self { patterns }
    }

    pub fn analyze_change_semantic_impact(&self, change: &ChangePoint) -> Result<Vec<String>> {
        let mut impacts = Vec::new();

        // Analyze based on file type and location
        if change.file_path.ends_with(".h") {
            impacts.push("Header file change - may affect multiple compilation units".to_string());
        }

        if change.file_path.starts_with("include/") {
            impacts.push("System header change - wide impact expected".to_string());
        }

        if change.file_path.contains("uapi/") {
            impacts.push("UAPI change - may break userspace applications".to_string());
        }

        // Analyze based on change type
        match change.change_type {
            ChangeType::SignatureChanged => {
                impacts.push("Function signature change - affects all callers".to_string());
            }
            ChangeType::StructChanged => {
                impacts.push("Structure layout change - may affect ABI".to_string());
            }
            ChangeType::ConfigChanged => {
                impacts.push("Configuration change - affects build variants".to_string());
            }
            ChangeType::MacroChanged => {
                impacts.push("Macro change - affects all users of the macro".to_string());
            }
            _ => {}
        }

        // Analyze content for semantic patterns
        if let Some(symbol_name) = &change.symbol_name {
            if symbol_name.starts_with("sys_") {
                impacts.push("System call change - may affect userspace".to_string());
            }

            if symbol_name.contains("lock")
                || symbol_name.contains("mutex")
                || symbol_name.contains("sem")
            {
                impacts.push("Synchronization primitive change - check for deadlocks".to_string());
            }

            if symbol_name.contains("alloc")
                || symbol_name.contains("free")
                || symbol_name.contains("mem")
            {
                impacts.push("Memory management change - check for leaks".to_string());
            }

            if symbol_name.contains("security")
                || symbol_name.contains("auth")
                || symbol_name.contains("perm")
            {
                impacts.push("Security-related change - security review required".to_string());
            }
        }

        Ok(impacts)
    }

    pub fn classify_change_risk(&self, change: &ChangePoint) -> Result<ImpactLevel> {
        let mut risk_score = 0;

        // File-based risk
        if change.file_path.starts_with("security/") {
            risk_score += 3;
        } else if change.file_path.starts_with("mm/") || change.file_path.starts_with("kernel/") {
            risk_score += 2;
        } else if change.file_path.starts_with("drivers/") {
            risk_score += 1;
        }

        if change.file_path.ends_with(".h") {
            risk_score += 1;
        }

        if change.file_path.contains("uapi/") {
            risk_score += 2;
        }

        // Change type risk
        match change.change_type {
            ChangeType::FunctionRemoved | ChangeType::SignatureChanged => risk_score += 3,
            ChangeType::StructChanged => risk_score += 2,
            ChangeType::FunctionModified => risk_score += 1,
            ChangeType::MacroChanged => risk_score += 1,
            ChangeType::ConfigChanged => risk_score += 1,
            _ => {}
        }

        // Symbol-based risk
        if let Some(symbol_name) = &change.symbol_name {
            if symbol_name.starts_with("sys_") || symbol_name.contains("syscall") {
                risk_score += 2;
            }

            if symbol_name.contains("_init") || symbol_name.contains("_exit") {
                risk_score += 1;
            }

            if symbol_name.contains("lock") || symbol_name.contains("atomic") {
                risk_score += 2;
            }
        }

        Ok(match risk_score {
            0..=2 => ImpactLevel::Low,
            3..=5 => ImpactLevel::Medium,
            6..=8 => ImpactLevel::High,
            _ => ImpactLevel::Critical,
        })
    }

    pub fn extract_symbols_from_diff(&self, diff_content: &str) -> Result<Vec<String>> {
        let mut symbols = Vec::new();

        for line in diff_content.lines() {
            if line.starts_with('+') || line.starts_with('-') {
                let clean_line = line.trim_start_matches(&['+', '-'][..]);

                // Try each pattern type
                for patterns in self.patterns.values() {
                    for pattern in patterns {
                        if let Some(captures) = pattern.captures(clean_line) {
                            if let Some(symbol) = captures.get(1) {
                                symbols.push(symbol.as_str().to_string());
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate
        symbols.sort();
        symbols.dedup();

        Ok(symbols)
    }
}

impl Default for SemanticDiffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_analyzer_creation() {
        let analyzer = SemanticDiffAnalyzer::new();
        assert!(analyzer.patterns.contains_key("function"));
        assert!(analyzer.patterns.contains_key("struct"));
        assert!(analyzer.patterns.contains_key("macro"));
        assert!(analyzer.patterns.contains_key("config"));
    }

    #[test]
    fn test_change_risk_classification() -> Result<()> {
        let analyzer = SemanticDiffAnalyzer::new();

        // High risk change
        let high_risk_change = ChangePoint {
            file_path: "security/selinux/hooks.c".to_string(),
            line_number: 100,
            change_type: ChangeType::SignatureChanged,
            symbol_name: Some("sys_open".to_string()),
            description: "Security function signature changed".to_string(),
            diff_context: String::new(),
        };

        let risk = analyzer.classify_change_risk(&high_risk_change)?;
        assert!(matches!(risk, ImpactLevel::Critical | ImpactLevel::High));

        // Low risk change
        let low_risk_change = ChangePoint {
            file_path: "drivers/char/random.c".to_string(),
            line_number: 50,
            change_type: ChangeType::VariableAdded,
            symbol_name: None,
            description: "Added local variable".to_string(),
            diff_context: String::new(),
        };

        let risk = analyzer.classify_change_risk(&low_risk_change)?;
        assert!(matches!(risk, ImpactLevel::Low | ImpactLevel::Medium));

        Ok(())
    }

    #[test]
    fn test_symbol_extraction() -> Result<()> {
        let analyzer = SemanticDiffAnalyzer::new();

        let diff = r#"
+static int new_function(int arg) {
+    return arg * 2;
+}
+
+struct new_struct {
+    int field1;
+    char field2;
+};
+
+#define NEW_MACRO 42
"#;

        let symbols = analyzer.extract_symbols_from_diff(diff)?;

        assert!(symbols.contains(&"new_function".to_string()));
        assert!(symbols.contains(&"new_struct".to_string()));
        assert!(symbols.contains(&"NEW_MACRO".to_string()));

        Ok(())
    }
}
