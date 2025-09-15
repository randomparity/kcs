use crate::{ChangePoint, ChangeType};
use anyhow::Result;
use regex::Regex;

pub fn parse_patch(patch_content: &str) -> Result<Vec<ChangePoint>> {
    let mut changes = Vec::new();
    let mut current_file = None;
    let mut current_line = 0;

    // Regex patterns for different types of changes
    let file_header = Regex::new(r"^\+\+\+ b/(.+)$")?;
    let hunk_header = Regex::new(r"^@@ -\d+,?\d* \+(\d+),?\d* @@")?;
    let function_def =
        Regex::new(r"^[+-]\s*(?:static\s+)?(?:inline\s+)?(?:extern\s+)?(\w+\s+)+(\w+)\s*\(")?;
    let struct_def = Regex::new(r"^[+-]\s*struct\s+(\w+)\s*\{")?;
    let macro_def = Regex::new(r"^[+-]\s*#define\s+(\w+)")?;
    let config_change = Regex::new(r"^[+-]\s*(?:#\s*)?(?:config|CONFIG_)\s+(\w+)")?;

    for line in patch_content.lines() {
        // Track current file
        if let Some(captures) = file_header.captures(line) {
            current_file = Some(captures[1].to_string());
            continue;
        }

        // Track line numbers from hunk headers
        if let Some(captures) = hunk_header.captures(line) {
            current_line = captures[1].parse().unwrap_or(0);
            continue;
        }

        // Skip if we don't have a current file
        let file_path = match &current_file {
            Some(path) => path.clone(),
            None => continue,
        };

        // Analyze different types of changes
        if line.starts_with('+') || line.starts_with('-') {
            let is_addition = line.starts_with('+');
            let is_removal = line.starts_with('-');

            // Function changes
            if let Some(captures) = function_def.captures(line) {
                let function_name = captures.get(2).map(|m| m.as_str().to_string());
                let change_type = if is_addition {
                    ChangeType::FunctionAdded
                } else if is_removal {
                    ChangeType::FunctionRemoved
                } else {
                    ChangeType::FunctionModified
                };

                changes.push(ChangePoint {
                    file_path: file_path.clone(),
                    line_number: current_line,
                    change_type,
                    symbol_name: function_name,
                    description: format!(
                        "Function {} in {}",
                        if is_addition {
                            "added"
                        } else if is_removal {
                            "removed"
                        } else {
                            "modified"
                        },
                        line.trim_start_matches(&['+', '-'][..])
                    ),
                    diff_context: line.to_string(),
                });
            }
            // Struct changes
            else if let Some(captures) = struct_def.captures(line) {
                let struct_name = captures[1].to_string();

                changes.push(ChangePoint {
                    file_path: file_path.clone(),
                    line_number: current_line,
                    change_type: ChangeType::StructChanged,
                    symbol_name: Some(struct_name.clone()),
                    description: format!("Struct {} modified", struct_name),
                    diff_context: line.to_string(),
                });
            }
            // Macro changes
            else if let Some(captures) = macro_def.captures(line) {
                let macro_name = captures[1].to_string();

                changes.push(ChangePoint {
                    file_path: file_path.clone(),
                    line_number: current_line,
                    change_type: ChangeType::MacroChanged,
                    symbol_name: Some(macro_name.clone()),
                    description: format!("Macro {} modified", macro_name),
                    diff_context: line.to_string(),
                });
            }
            // Config changes
            else if let Some(captures) = config_change.captures(line) {
                let config_name = captures[1].to_string();

                changes.push(ChangePoint {
                    file_path: file_path.clone(),
                    line_number: current_line,
                    change_type: ChangeType::ConfigChanged,
                    symbol_name: Some(format!("CONFIG_{}", config_name)),
                    description: format!("Config option {} modified", config_name),
                    diff_context: line.to_string(),
                });
            }
            // Generic line changes
            else if !line.trim_start_matches(&['+', '-'][..]).trim().is_empty() {
                // Only create change points for substantial changes
                if line.len() > 10
                    && !line
                        .trim_start_matches(&['+', '-'][..])
                        .trim()
                        .starts_with("//")
                {
                    changes.push(ChangePoint {
                        file_path: file_path.clone(),
                        line_number: current_line,
                        change_type: if is_addition {
                            ChangeType::VariableAdded
                        } else if is_removal {
                            ChangeType::VariableRemoved
                        } else {
                            ChangeType::VariableModified
                        },
                        symbol_name: None,
                        description: "Line modified".to_string(),
                        diff_context: line.to_string(),
                    });
                }
            }
        }

        // Update line counter for non-removal lines
        if !line.starts_with('-')
            && !line.starts_with("@@")
            && !line.starts_with("+++")
            && !line.starts_with("---")
        {
            current_line += 1;
        }
    }

    Ok(changes)
}

pub fn detect_signature_changes(patch_content: &str) -> Result<Vec<ChangePoint>> {
    let mut signature_changes = Vec::new();
    let mut current_file = None;
    let mut in_function = false;
    let mut function_lines = Vec::new();
    let mut current_line = 0;

    let file_header = Regex::new(r"^\+\+\+ b/(.+)$")?;
    let hunk_header = Regex::new(r"^@@ -\d+,?\d* \+(\d+),?\d* @@")?;
    let function_start = Regex::new(r"^[+-].*\w+\s+\w+\s*\([^)]*\)\s*\{?")?;

    for line in patch_content.lines() {
        if let Some(captures) = file_header.captures(line) {
            current_file = Some(captures[1].to_string());
            continue;
        }

        if let Some(captures) = hunk_header.captures(line) {
            current_line = captures[1].parse().unwrap_or(0);
            continue;
        }

        let file_path = match &current_file {
            Some(path) => path.clone(),
            None => continue,
        };

        if function_start.is_match(line) {
            if !in_function {
                in_function = true;
                function_lines.clear();
            }
            function_lines.push(line.to_string());
        } else if in_function {
            if line.trim() == "}" || line.contains("};") {
                // End of function, analyze accumulated lines
                if function_lines.len() > 1 {
                    let has_additions = function_lines.iter().any(|l| l.starts_with('+'));
                    let has_removals = function_lines.iter().any(|l| l.starts_with('-'));

                    if has_additions && has_removals {
                        // Potential signature change
                        if let Some(func_name) = extract_function_name(&function_lines) {
                            signature_changes.push(ChangePoint {
                                file_path: file_path.clone(),
                                line_number: current_line,
                                change_type: ChangeType::SignatureChanged,
                                symbol_name: Some(func_name.clone()),
                                description: format!("Function signature changed: {}", func_name),
                                diff_context: function_lines.join("\n"),
                            });
                        }
                    }
                }
                in_function = false;
                function_lines.clear();
            } else if line.starts_with('+') || line.starts_with('-') || line.starts_with(' ') {
                function_lines.push(line.to_string());
            }
        }

        if !line.starts_with('-')
            && !line.starts_with("@@")
            && !line.starts_with("+++")
            && !line.starts_with("---")
        {
            current_line += 1;
        }
    }

    Ok(signature_changes)
}

fn extract_function_name(function_lines: &[String]) -> Option<String> {
    let function_pattern =
        Regex::new(r"(?:static\s+)?(?:inline\s+)?(?:extern\s+)?\w+\s+(\w+)\s*\(").ok()?;

    for line in function_lines {
        if let Some(captures) = function_pattern.captures(line) {
            return Some(captures[1].to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_patch() -> Result<()> {
        let patch = r#"
--- a/test.c
+++ b/test.c
@@ -10,3 +10,4 @@

 int existing_function(void) {
+    int new_var = 42;
     return 0;
 }
+
+static int new_function(int arg) {
+    return arg * 2;
+}
"#;

        let changes = parse_patch(patch)?;

        // Should detect the new function
        let function_changes: Vec<_> = changes
            .iter()
            .filter(|c| matches!(c.change_type, ChangeType::FunctionAdded))
            .collect();

        assert_eq!(function_changes.len(), 1);
        assert_eq!(
            function_changes[0].symbol_name,
            Some("new_function".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_detect_signature_change() -> Result<()> {
        let patch = r#"
--- a/test.c
+++ b/test.c
@@ -5,2 +5,2 @@
-int old_function(int a) {
+int old_function(int a, int b) {
     return 0;
 }
"#;

        let changes = detect_signature_changes(patch)?;

        assert_eq!(changes.len(), 1);
        assert!(matches!(
            changes[0].change_type,
            ChangeType::SignatureChanged
        ));
        assert_eq!(changes[0].symbol_name, Some("old_function".to_string()));

        Ok(())
    }

    #[test]
    fn test_config_change_detection() -> Result<()> {
        let patch = r#"
--- a/Kconfig
+++ b/Kconfig
@@ -100,1 +100,1 @@
-config DEBUG_KERNEL
+config DEBUG_KERNEL_NEW
"#;

        let changes = parse_patch(patch)?;

        let config_changes: Vec<_> = changes
            .iter()
            .filter(|c| matches!(c.change_type, ChangeType::ConfigChanged))
            .collect();

        assert_eq!(config_changes.len(), 2); // One removal, one addition

        Ok(())
    }
}
