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
                    if extensions.contains(&ext) {
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
