use crate::{EntryPoint, EntryType};
use anyhow::Result;
use regex::Regex;
use std::fs;
use std::path::Path;

pub fn extract_file_operations<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut file_ops = Vec::new();

    // Pattern to match file_operations structure definitions
    let file_ops_pattern = Regex::new(r"(?s)struct\s+file_operations\s+\w+\s*=\s*\{([^}]+)\}")?;

    let field_pattern = Regex::new(r"\.(\w+)\s*=\s*(\w+)")?;

    // Search through the kernel for file_operations definitions
    search_kernel_files(
        kernel_dir.as_ref(),
        &[".c"],
        |_file_path, relative_path, content| {
            for file_ops_match in file_ops_pattern.find_iter(content) {
                let struct_content = file_ops_match.as_str();
                let line_num = content[..file_ops_match.start()].lines().count();

                // Extract individual field assignments
                for field_match in field_pattern.captures_iter(struct_content) {
                    let field_name = field_match.get(1).unwrap().as_str();
                    let function_name = field_match.get(2).unwrap().as_str();

                    // Skip NULL assignments
                    if function_name == "NULL" {
                        continue;
                    }

                    file_ops.push(EntryPoint {
                        name: function_name.to_string(),
                        entry_type: EntryType::FileOps,
                        file_path: relative_path.clone(),
                        line_number: line_num as u32,
                        signature: format!(".{} = {}", field_name, function_name),
                        description: Some(format!(
                            "File operation: {} ({})",
                            field_name, function_name
                        )),
                        metadata: None,
                    });
                }
            }
            Ok(())
        },
    )?;

    Ok(file_ops)
}

pub fn extract_sysfs_entries<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut sysfs_entries = Vec::new();

    // Patterns for sysfs attribute definitions
    let attr_patterns = [
        Regex::new(r"DEVICE_ATTR\((\w+),\s*\w+,\s*(\w+),\s*(\w+)\)")?,
        Regex::new(r"static\s+DEVICE_ATTR\((\w+),\s*\w+,\s*(\w+),\s*(\w+)\)")?,
        Regex::new(r"sysfs_create_file\([^,]+,\s*&(\w+)\.attr\)")?,
    ];

    search_kernel_files(
        kernel_dir.as_ref(),
        &[".c", ".h"],
        |_file_path, relative_path, content| {
            let lines: Vec<&str> = content.lines().collect();

            for (line_num, line) in lines.iter().enumerate() {
                for pattern in &attr_patterns {
                    if let Some(captures) = pattern.captures(line) {
                        let name = captures.get(1).map(|m| m.as_str()).unwrap_or("unknown");

                        sysfs_entries.push(EntryPoint {
                            name: format!("sysfs_{}", name),
                            entry_type: EntryType::Sysfs,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: line.trim().to_string(),
                            description: Some(format!("Sysfs attribute: {}", name)),
                            metadata: None,
                        });
                    }
                }
            }
            Ok(())
        },
    )?;

    Ok(sysfs_entries)
}

pub fn extract_procfs_entries<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut procfs_entries = Vec::new();

    // Patterns for proc_create and related functions
    let proc_create_patterns = [
        // proc_create("name", mode, parent, proc_ops)
        Regex::new(r#"proc_create\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#)?,
        // proc_create_data("name", mode, parent, proc_ops, data)
        Regex::new(
            r#"proc_create_data\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*,\s*[^)]+\s*\)"#,
        )?,
        // proc_create_single("name", mode, parent, show_func)
        Regex::new(
            r#"proc_create_single\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*(\w+)\s*\)"#,
        )?,
        // proc_create_single_data("name", mode, parent, show_func, data)
        Regex::new(
            r#"proc_create_single_data\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*(\w+)\s*,\s*[^)]+\s*\)"#,
        )?,
        // proc_create_seq("name", mode, parent, seq_ops)
        Regex::new(
            r#"proc_create_seq\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#,
        )?,
        // proc_create_seq_data("name", mode, parent, seq_ops, data)
        Regex::new(
            r#"proc_create_seq_data\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*,\s*[^)]+\s*\)"#,
        )?,
    ];

    // Pattern to match proc_ops structure definitions
    let proc_ops_pattern =
        Regex::new(r"(?s)(?:static\s+)?(?:const\s+)?struct\s+proc_ops\s+(\w+)\s*=\s*\{([^}]+)\}")?;

    // Pattern to extract fields from proc_ops
    let proc_field_pattern = Regex::new(r"\.proc_(\w+)\s*=\s*(\w+)")?;

    search_kernel_files(
        kernel_dir.as_ref(),
        &[".c", ".h"],
        |_file_path, relative_path, content| {
            let lines: Vec<&str> = content.lines().collect();

            // First, find all proc_ops structures and their handlers
            let mut proc_ops_handlers = std::collections::HashMap::new();
            for proc_ops_match in proc_ops_pattern.find_iter(content) {
                if let Some(captures) = proc_ops_pattern.captures(proc_ops_match.as_str()) {
                    let ops_name = captures.get(1).unwrap().as_str();
                    let struct_content = captures.get(2).unwrap().as_str();
                    let line_num = content[..proc_ops_match.start()].lines().count();

                    // Extract handlers from proc_ops
                    for field_match in proc_field_pattern.captures_iter(struct_content) {
                        let field_name = field_match.get(1).unwrap().as_str();
                        let function_name = field_match.get(2).unwrap().as_str();

                        // Skip NULL assignments
                        if function_name == "NULL"
                            || function_name == "seq_read"
                            || function_name == "seq_lseek"
                            || function_name == "single_release"
                        {
                            continue;
                        }

                        proc_ops_handlers.insert(
                            function_name.to_string(),
                            (ops_name.to_string(), field_name.to_string(), line_num),
                        );

                        procfs_entries.push(EntryPoint {
                            name: function_name.to_string(),
                            entry_type: EntryType::ProcFs,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: format!(".proc_{} = {}", field_name, function_name),
                            description: Some(format!(
                                "ProcFS handler: {} in {}",
                                field_name, ops_name
                            )),
                            metadata: Some({
                                let mut map = serde_json::Map::new();
                                map.insert(
                                    "proc_ops".to_string(),
                                    serde_json::Value::String(ops_name.to_string()),
                                );
                                map.insert(
                                    "handler_type".to_string(),
                                    serde_json::Value::String(field_name.to_string()),
                                );
                                map
                            }),
                        });
                    }
                }
            }

            // Now find proc_create calls
            for (line_num, line) in lines.iter().enumerate() {
                for pattern in &proc_create_patterns {
                    if let Some(captures) = pattern.captures(line) {
                        let proc_path = captures.get(1).unwrap().as_str();
                        let ops_or_handler = captures.get(2).unwrap().as_str();

                        // Create metadata
                        let mut metadata = serde_json::Map::new();
                        metadata.insert(
                            "proc_path".to_string(),
                            serde_json::Value::String(proc_path.to_string()),
                        );

                        // Check if this references a known proc_ops structure
                        if let Some((ops_name, _, _)) = proc_ops_handlers.get(ops_or_handler) {
                            metadata.insert(
                                "proc_ops".to_string(),
                                serde_json::Value::String(ops_name.clone()),
                            );
                        }

                        procfs_entries.push(EntryPoint {
                            name: format!("procfs_{}", proc_path.replace('/', "_")),
                            entry_type: EntryType::ProcFs,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: line.trim().to_string(),
                            description: Some(format!("ProcFS entry: {}", proc_path)),
                            metadata: Some(metadata),
                        });

                        break; // Only match the first pattern that matches
                    }
                }
            }
            Ok(())
        },
    )?;

    Ok(procfs_entries)
}

pub fn extract_debugfs_entries<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut debugfs_entries = Vec::new();

    // Patterns for debugfs_create functions
    let debugfs_create_patterns = [
        // debugfs_create_file("name", mode, parent, data, fops)
        Regex::new(
            r#"debugfs_create_file\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#,
        )?,
        // debugfs_create_file_unsafe("name", mode, parent, data, fops)
        Regex::new(
            r#"debugfs_create_file_unsafe\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#,
        )?,
        // debugfs_create_dir("name", parent)
        Regex::new(r#"debugfs_create_dir\s*\(\s*"([^"]+)"\s*,\s*[^)]+\s*\)"#)?,
        // debugfs_create_u8/u16/u32/u64("name", mode, parent, value)
        Regex::new(
            r#"debugfs_create_u(?:8|16|32|64)\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#,
        )?,
        // debugfs_create_x8/x16/x32/x64("name", mode, parent, value) for hex display
        Regex::new(
            r#"debugfs_create_x(?:8|16|32|64)\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#,
        )?,
        // debugfs_create_bool("name", mode, parent, value)
        Regex::new(
            r#"debugfs_create_bool\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#,
        )?,
        // debugfs_create_blob("name", mode, parent, blob)
        Regex::new(
            r#"debugfs_create_blob\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*[^)]+\s*\)"#,
        )?,
        // debugfs_create_regset32("name", mode, parent, regset)
        Regex::new(
            r#"debugfs_create_regset32\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*\)"#,
        )?,
        // debugfs_create_file_size("name", mode, parent, data, fops, size)
        Regex::new(
            r#"debugfs_create_file_size\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*&?(\w+)\s*,\s*[^)]+\s*\)"#,
        )?,
    ];

    // Pattern to match file_operations structures that might be used with debugfs
    let file_ops_pattern = Regex::new(
        r"(?s)(?:static\s+)?(?:const\s+)?struct\s+file_operations\s+(\w+_debugfs_\w+|debugfs_\w+|\w+_dbg_\w+)\s*=\s*\{([^}]+)\}",
    )?;

    search_kernel_files(
        kernel_dir.as_ref(),
        &[".c", ".h"],
        |_file_path, relative_path, content| {
            let lines: Vec<&str> = content.lines().collect();

            // Track debugfs file_operations
            let mut debugfs_fops = std::collections::HashSet::new();
            for fops_match in file_ops_pattern.find_iter(content) {
                if let Some(captures) = file_ops_pattern.captures(fops_match.as_str()) {
                    let ops_name = captures.get(1).unwrap().as_str();
                    debugfs_fops.insert(ops_name.to_string());
                }
            }

            // Find debugfs_create calls
            for (line_num, line) in lines.iter().enumerate() {
                for (pattern_idx, pattern) in debugfs_create_patterns.iter().enumerate() {
                    if let Some(captures) = pattern.captures(line) {
                        let debugfs_path = captures.get(1).unwrap().as_str();

                        // Create metadata based on the type of debugfs entry
                        let mut metadata = serde_json::Map::new();
                        metadata.insert(
                            "debugfs_path".to_string(),
                            serde_json::Value::String(debugfs_path.to_string()),
                        );

                        // Determine the debugfs type based on pattern index
                        let debugfs_type = match pattern_idx {
                            0 | 1 => "file",
                            2 => "directory",
                            3 => "u32",
                            4 => "x32",
                            5 => "bool",
                            6 => "blob",
                            7 => "regset32",
                            8 => "file_size",
                            _ => "unknown",
                        };
                        metadata.insert(
                            "debugfs_type".to_string(),
                            serde_json::Value::String(debugfs_type.to_string()),
                        );

                        // If there's a second capture group (fops or value name), include it
                        if let Some(second_param) = captures.get(2) {
                            let param_name = second_param.as_str();
                            if debugfs_fops.contains(param_name) {
                                metadata.insert(
                                    "file_operations".to_string(),
                                    serde_json::Value::String(param_name.to_string()),
                                );
                            } else if debugfs_type != "directory" && debugfs_type != "blob" {
                                metadata.insert(
                                    "value_name".to_string(),
                                    serde_json::Value::String(param_name.to_string()),
                                );
                            }
                        }

                        debugfs_entries.push(EntryPoint {
                            name: format!("debugfs_{}", debugfs_path.replace('/', "_")),
                            entry_type: EntryType::DebugFs,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: line.trim().to_string(),
                            description: Some(format!(
                                "DebugFS {}: {}",
                                debugfs_type, debugfs_path
                            )),
                            metadata: Some(metadata),
                        });

                        break; // Only match the first pattern that matches
                    }
                }
            }
            Ok(())
        },
    )?;

    Ok(debugfs_entries)
}

pub fn extract_module_entries<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut module_entries = Vec::new();

    // Patterns for module init/exit functions
    let init_pattern = Regex::new(r"module_init\((\w+)\)")?;
    let exit_pattern = Regex::new(r"module_exit\((\w+)\)")?;
    let init_func_pattern = Regex::new(r"static\s+int\s+__init\s+(\w+)\s*\(")?;
    let exit_func_pattern = Regex::new(r"static\s+void\s+__exit\s+(\w+)\s*\(")?;

    search_kernel_files(
        kernel_dir.as_ref(),
        &[".c"],
        |_file_path, relative_path, content| {
            let lines: Vec<&str> = content.lines().collect();

            for (line_num, line) in lines.iter().enumerate() {
                // Module init
                if let Some(captures) = init_pattern.captures(line) {
                    if let Some(name) = captures.get(1) {
                        module_entries.push(EntryPoint {
                            name: name.as_str().to_string(),
                            entry_type: EntryType::ModuleInit,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: line.trim().to_string(),
                            description: Some(format!("Module init function: {}", name.as_str())),
                            metadata: None,
                        });
                    }
                }

                // Module exit
                if let Some(captures) = exit_pattern.captures(line) {
                    if let Some(name) = captures.get(1) {
                        module_entries.push(EntryPoint {
                            name: name.as_str().to_string(),
                            entry_type: EntryType::ModuleExit,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: line.trim().to_string(),
                            description: Some(format!("Module exit function: {}", name.as_str())),
                            metadata: None,
                        });
                    }
                }

                // Function definitions with __init/__exit
                if let Some(captures) = init_func_pattern.captures(line) {
                    if let Some(name) = captures.get(1) {
                        module_entries.push(EntryPoint {
                            name: name.as_str().to_string(),
                            entry_type: EntryType::ModuleInit,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: line.trim().to_string(),
                            description: Some(format!("Init function: {}", name.as_str())),
                            metadata: None,
                        });
                    }
                }

                if let Some(captures) = exit_func_pattern.captures(line) {
                    if let Some(name) = captures.get(1) {
                        module_entries.push(EntryPoint {
                            name: name.as_str().to_string(),
                            entry_type: EntryType::ModuleExit,
                            file_path: relative_path.clone(),
                            line_number: (line_num + 1) as u32,
                            signature: line.trim().to_string(),
                            description: Some(format!("Exit function: {}", name.as_str())),
                            metadata: None,
                        });
                    }
                }
            }
            Ok(())
        },
    )?;

    Ok(module_entries)
}

fn search_kernel_files<P, F>(kernel_dir: P, extensions: &[&str], mut callback: F) -> Result<()>
where
    P: AsRef<Path>,
    F: FnMut(&Path, String, &str) -> Result<()>,
{
    fn visit_dir<F>(
        dir: &Path,
        kernel_root: &Path,
        extensions: &[&str],
        callback: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&Path, String, &str) -> Result<()>,
    {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    // Skip some large/irrelevant directories for performance
                    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                        if matches!(dir_name, ".git" | "Documentation" | "tools" | "scripts") {
                            continue;
                        }
                    }
                    visit_dir(&path, kernel_root, extensions, callback)?;
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let ext_with_dot = format!(".{}", ext);
                    if extensions.contains(&ext_with_dot.as_str()) {
                        let relative_path = path
                            .strip_prefix(kernel_root)
                            .unwrap_or(&path)
                            .to_string_lossy()
                            .to_string();

                        if let Ok(content) = fs::read_to_string(&path) {
                            callback(&path, relative_path, &content)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    visit_dir(
        kernel_dir.as_ref(),
        kernel_dir.as_ref(),
        extensions,
        &mut callback,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_extract_procfs_entries() -> Result<()> {
        let temp_dir = tempdir()?;
        let kernel_dir = temp_dir.path();

        // Create a sample file with proc_ops and proc_create
        let fs_dir = kernel_dir.join("fs");
        fs::create_dir_all(&fs_dir)?;

        let mut test_file = File::create(fs_dir.join("test_proc.c"))?;
        writeln!(
            test_file,
            r#"
#include <linux/proc_fs.h>

static int test_proc_open(struct inode *inode, struct file *file) {{
    return single_open(file, test_proc_show, NULL);
}}

static int test_proc_show(struct seq_file *m, void *v) {{
    seq_printf(m, "Test proc file\n");
    return 0;
}}

static const struct proc_ops test_proc_ops = {{
    .proc_open      = test_proc_open,
    .proc_read      = seq_read,
    .proc_lseek     = seq_lseek,
    .proc_release   = single_release,
}};

static int __init test_init(void) {{
    proc_create("test_entry", 0, NULL, &test_proc_ops);
    proc_create("test/nested", 0644, NULL, &test_proc_ops);
    return 0;
}}
        "#
        )?;

        let entries = extract_procfs_entries(kernel_dir)?;

        // Should find both the proc_ops handlers and the proc_create calls
        assert!(
            entries.len() >= 3,
            "Should find at least 3 procfs entries, found {}",
            entries.len()
        );

        // Check that we found the proc_ops handler
        let handler = entries.iter().find(|e| e.name == "test_proc_open");
        assert!(handler.is_some(), "Should find test_proc_open handler");

        // Check that we found the proc_create entries
        let test_entry = entries.iter().find(|e| e.name == "procfs_test_entry");
        assert!(test_entry.is_some(), "Should find procfs_test_entry");

        let nested_entry = entries.iter().find(|e| e.name == "procfs_test_nested");
        assert!(nested_entry.is_some(), "Should find procfs_test_nested");

        // Verify metadata
        if let Some(entry) = test_entry {
            assert!(entry.metadata.is_some(), "Should have metadata");
            if let Some(ref metadata) = entry.metadata {
                assert!(
                    metadata.contains_key("proc_path"),
                    "Should have proc_path in metadata"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_extract_debugfs_entries() -> Result<()> {
        let temp_dir = tempdir()?;
        let kernel_dir = temp_dir.path();

        // Create a sample file with debugfs entries
        let drivers_dir = kernel_dir.join("drivers");
        fs::create_dir_all(&drivers_dir)?;

        let mut test_file = File::create(drivers_dir.join("test_debugfs.c"))?;
        writeln!(
            test_file,
            r#"
#include <linux/debugfs.h>

static struct dentry *test_debugfs_dir;

static int test_debugfs_show(struct seq_file *m, void *v) {{
    seq_printf(m, "Test content\n");
    return 0;
}}

static int test_debugfs_open(struct inode *inode, struct file *file) {{
    return single_open(file, test_debugfs_show, NULL);
}}

static const struct file_operations test_debugfs_fops = {{
    .open = test_debugfs_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
}};

static int test_value = 42;
static bool test_enabled = true;

static int __init test_init(void) {{
    test_debugfs_dir = debugfs_create_dir("test_driver", NULL);
    debugfs_create_file("status", 0444, test_debugfs_dir, NULL, &test_debugfs_fops);
    debugfs_create_u32("test_value", 0644, test_debugfs_dir, &test_value);
    debugfs_create_bool("enabled", 0644, test_debugfs_dir, &test_enabled);
    debugfs_create_x32("test_hex", 0444, test_debugfs_dir, &test_value);
    debugfs_create_blob("test_blob", 0444, test_debugfs_dir, NULL);
    return 0;
}}
        "#
        )?;

        let entries = extract_debugfs_entries(kernel_dir)?;

        // Should find multiple debugfs entries
        assert!(
            entries.len() >= 6,
            "Should find at least 6 debugfs entries, found {}",
            entries.len()
        );

        // Check that we found the directory
        let dir_entry = entries.iter().find(|e| e.name == "debugfs_test_driver");
        assert!(
            dir_entry.is_some(),
            "Should find debugfs_test_driver directory"
        );

        // Check that we found the file with fops
        let status_entry = entries.iter().find(|e| e.name == "debugfs_status");
        assert!(status_entry.is_some(), "Should find debugfs_status file");

        // Check for simple value entries
        let value_entry = entries.iter().find(|e| e.name == "debugfs_test_value");
        assert!(value_entry.is_some(), "Should find debugfs_test_value");

        let bool_entry = entries.iter().find(|e| e.name == "debugfs_enabled");
        assert!(bool_entry.is_some(), "Should find debugfs_enabled");

        // Verify metadata
        if let Some(entry) = status_entry {
            assert!(entry.metadata.is_some(), "Should have metadata");
            if let Some(ref metadata) = entry.metadata {
                assert!(
                    metadata.contains_key("debugfs_path"),
                    "Should have debugfs_path in metadata"
                );
                assert!(
                    metadata.contains_key("debugfs_type"),
                    "Should have debugfs_type in metadata"
                );
            }
        }

        Ok(())
    }
}
