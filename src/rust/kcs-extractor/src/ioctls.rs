use crate::{EntryPoint, EntryType};
use anyhow::Result;
use regex::Regex;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

pub fn extract_ioctls<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut ioctls = Vec::new();
    let mut magic_numbers = HashMap::new();
    let mut commands = HashMap::new();

    // Search through common directories where ioctls are defined
    let search_dirs = ["drivers", "fs", "net", "arch", "kernel", "include"];

    for search_dir in &search_dirs {
        let dir_path = kernel_dir.as_ref().join(search_dir);
        if !dir_path.exists() {
            continue;
        }

        // Recursively search for .c and .h files using walkdir
        for entry in WalkDir::new(&dir_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                let file_path = entry.path();
                if let Some(ext) = file_path.extension() {
                    if ext == "c" || ext == "h" {
                        if let Ok(relative_path) = file_path.strip_prefix(kernel_dir.as_ref()) {
                            extract_ioctls_from_file(
                                file_path,
                                relative_path.to_string_lossy().as_ref(),
                                &mut ioctls,
                                &mut magic_numbers,
                                &mut commands,
                            )?;
                        }
                    }
                }
            }
        }
    }

    Ok(ioctls)
}

fn extract_ioctls_from_file(
    file_path: &Path,
    relative_path: &str,
    ioctls: &mut Vec<EntryPoint>,
    magic_numbers: &mut HashMap<String, String>,
    commands: &mut HashMap<String, serde_json::Value>,
) -> Result<()> {
    let content = fs::read_to_string(file_path)?;
    let lines: Vec<&str> = content.lines().collect();

    // Patterns for ioctl handlers
    let handler_patterns = [
        // Function pointer assignments in file_operations
        Regex::new(r"\.unlocked_ioctl\s*=\s*(\w+)")?,
        Regex::new(r"\.compat_ioctl\s*=\s*(\w+)")?,
        // Direct ioctl function definitions
        Regex::new(r"(?:static\s+)?long\s+(\w+_ioctl)\s*\(")?,
        Regex::new(r"(?:static\s+)?int\s+(\w+_ioctl)\s*\(")?,
    ];

    // Pattern for magic number definitions
    let magic_pattern =
        Regex::new(r"#define\s+(\w+_(?:IOC_)?MAGIC)\s+(?:'(.)'|0x([0-9A-Fa-f]+)|(\d+))")?;

    // Patterns for ioctl command definitions
    let command_patterns = [
        // _IO, _IOR, _IOW, _IOWR macros
        Regex::new(r"#define\s+(\w+)\s+_IO\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)")?,
        Regex::new(r"#define\s+(\w+)\s+_IOR\s*\(\s*(\w+)\s*,\s*([^,]+),\s*([^)]+)\s*\)")?,
        Regex::new(r"#define\s+(\w+)\s+_IOW\s*\(\s*(\w+)\s*,\s*([^,]+),\s*([^)]+)\s*\)")?,
        Regex::new(r"#define\s+(\w+)\s+_IOWR\s*\(\s*(\w+)\s*,\s*([^,]+),\s*([^)]+)\s*\)")?,
    ];

    // Pattern for file_operations structures
    let fops_pattern =
        Regex::new(r"(?:static\s+)?(?:const\s+)?struct\s+file_operations\s+(\w+)\s*=")?;

    // Extract magic numbers
    for line in lines.iter() {
        if let Some(captures) = magic_pattern.captures(line) {
            let name = captures
                .get(1)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            let value = if let Some(char_val) = captures.get(2) {
                format!("'{}'", char_val.as_str())
            } else if let Some(hex_val) = captures.get(3) {
                format!("0x{}", hex_val.as_str())
            } else if let Some(dec_val) = captures.get(4) {
                dec_val.as_str().to_string()
            } else {
                continue;
            };

            magic_numbers.insert(name.clone(), value.clone());
        }
    }

    // Extract ioctl commands
    for (line_num, line) in lines.iter().enumerate() {
        // Check for _IO command
        if let Some(captures) = command_patterns[0].captures(line) {
            let cmd_name = captures.get(1).map(|m| m.as_str()).unwrap_or("");
            let magic = captures.get(2).map(|m| m.as_str()).unwrap_or("");
            let number = captures.get(3).map(|m| m.as_str()).unwrap_or("");

            let cmd_info = json!({
                "name": cmd_name,
                "macro_type": "_IO",
                "magic": magic,
                "number": number,
                "direction": "none",
                "file_path": relative_path,
                "line_number": line_num + 1
            });
            commands.insert(cmd_name.to_string(), cmd_info.clone());

            // Create an entry point for the command
            let mut metadata = serde_json::Map::new();
            metadata.insert("commands".to_string(), json!([cmd_info]));
            metadata.insert(
                "magic".to_string(),
                json!(magic_numbers.get(magic).unwrap_or(&magic.to_string())),
            );
            metadata.insert("macro_type".to_string(), json!("_IO"));

            ioctls.push(EntryPoint {
                name: cmd_name.to_string(),
                entry_type: EntryType::Ioctl,
                file_path: relative_path.to_string(),
                line_number: (line_num + 1) as u32,
                signature: line.trim().to_string(),
                description: Some(format!("IOCTL command: {}", cmd_name)),
                metadata: Some(metadata),
            });
        }

        // Check for _IOR command
        if let Some(captures) = command_patterns[1].captures(line) {
            let cmd_name = captures.get(1).map(|m| m.as_str()).unwrap_or("");
            let magic = captures.get(2).map(|m| m.as_str()).unwrap_or("");
            let number = captures.get(3).map(|m| m.as_str()).unwrap_or("");
            let data_type = captures.get(4).map(|m| m.as_str()).unwrap_or("");

            let cmd_info = json!({
                "name": cmd_name,
                "macro_type": "_IOR",
                "magic": magic,
                "number": number,
                "direction": "read",
                "type": data_type,
                "file_path": relative_path,
                "line_number": line_num + 1
            });
            commands.insert(cmd_name.to_string(), cmd_info.clone());

            // Create an entry point for the command
            let mut metadata = serde_json::Map::new();
            metadata.insert("commands".to_string(), json!([cmd_info]));
            metadata.insert(
                "magic".to_string(),
                json!(magic_numbers.get(magic).unwrap_or(&magic.to_string())),
            );
            metadata.insert("macro_type".to_string(), json!("_IOR"));

            ioctls.push(EntryPoint {
                name: cmd_name.to_string(),
                entry_type: EntryType::Ioctl,
                file_path: relative_path.to_string(),
                line_number: (line_num + 1) as u32,
                signature: line.trim().to_string(),
                description: Some(format!("IOCTL command: {} (read)", cmd_name)),
                metadata: Some(metadata),
            });
        }

        // Check for _IOW command
        if let Some(captures) = command_patterns[2].captures(line) {
            let cmd_name = captures.get(1).map(|m| m.as_str()).unwrap_or("");
            let magic = captures.get(2).map(|m| m.as_str()).unwrap_or("");
            let number = captures.get(3).map(|m| m.as_str()).unwrap_or("");
            let data_type = captures.get(4).map(|m| m.as_str()).unwrap_or("");

            let cmd_info = json!({
                "name": cmd_name,
                "macro_type": "_IOW",
                "magic": magic,
                "number": number,
                "direction": "write",
                "type": data_type,
                "file_path": relative_path,
                "line_number": line_num + 1
            });
            commands.insert(cmd_name.to_string(), cmd_info.clone());

            // Create an entry point for the command
            let mut metadata = serde_json::Map::new();
            metadata.insert("commands".to_string(), json!([cmd_info]));
            metadata.insert(
                "magic".to_string(),
                json!(magic_numbers.get(magic).unwrap_or(&magic.to_string())),
            );
            metadata.insert("macro_type".to_string(), json!("_IOW"));

            ioctls.push(EntryPoint {
                name: cmd_name.to_string(),
                entry_type: EntryType::Ioctl,
                file_path: relative_path.to_string(),
                line_number: (line_num + 1) as u32,
                signature: line.trim().to_string(),
                description: Some(format!("IOCTL command: {} (write)", cmd_name)),
                metadata: Some(metadata),
            });
        }

        // Check for _IOWR command
        if let Some(captures) = command_patterns[3].captures(line) {
            let cmd_name = captures.get(1).map(|m| m.as_str()).unwrap_or("");
            let magic = captures.get(2).map(|m| m.as_str()).unwrap_or("");
            let number = captures.get(3).map(|m| m.as_str()).unwrap_or("");
            let data_type = captures.get(4).map(|m| m.as_str()).unwrap_or("");

            let cmd_info = json!({
                "name": cmd_name,
                "macro_type": "_IOWR",
                "magic": magic,
                "number": number,
                "direction": "read_write",
                "type": data_type,
                "file_path": relative_path,
                "line_number": line_num + 1
            });
            commands.insert(cmd_name.to_string(), cmd_info.clone());

            // Create an entry point for the command
            let mut metadata = serde_json::Map::new();
            metadata.insert("commands".to_string(), json!([cmd_info]));
            metadata.insert(
                "magic".to_string(),
                json!(magic_numbers.get(magic).unwrap_or(&magic.to_string())),
            );
            metadata.insert("macro_type".to_string(), json!("_IOWR"));

            ioctls.push(EntryPoint {
                name: cmd_name.to_string(),
                entry_type: EntryType::Ioctl,
                file_path: relative_path.to_string(),
                line_number: (line_num + 1) as u32,
                signature: line.trim().to_string(),
                description: Some(format!("IOCTL command: {} (read/write)", cmd_name)),
                metadata: Some(metadata),
            });
        }
    }

    // Extract ioctl handlers and associate with commands
    let mut current_fops: Option<String> = None;
    let mut handler_commands = Vec::new();

    for (line_num, line) in lines.iter().enumerate() {
        // Check for file_operations structure
        if let Some(captures) = fops_pattern.captures(line) {
            current_fops = captures.get(1).map(|m| m.as_str().to_string());
        }

        // Check for handler patterns
        for pattern in &handler_patterns {
            if let Some(captures) = pattern.captures(line) {
                if let Some(name) = captures.get(1) {
                    let handler_name = name.as_str().to_string();

                    // Try to find which commands this handler processes
                    handler_commands.clear();

                    // Look for switch statements in the handler
                    if let Some(handler_start) =
                        find_handler_definition(&lines, &handler_name, line_num)
                    {
                        handler_commands =
                            extract_handled_commands(&lines, handler_start, commands);
                    }

                    let mut metadata = serde_json::Map::new();

                    if !handler_commands.is_empty() {
                        metadata.insert("handled_commands".to_string(), json!(handler_commands));
                    }

                    if let Some(ref fops) = current_fops {
                        metadata.insert("file_operations".to_string(), json!({"name": fops}));
                    }

                    let is_compat = handler_name.contains("compat");
                    if is_compat {
                        metadata.insert(
                            "handler".to_string(),
                            json!({"name": handler_name.clone(), "type": "compat"}),
                        );
                    } else {
                        metadata.insert(
                            "handler".to_string(),
                            json!({"name": handler_name.clone(), "type": "regular"}),
                        );
                    }

                    ioctls.push(EntryPoint {
                        name: handler_name.clone(),
                        entry_type: EntryType::Ioctl,
                        file_path: relative_path.to_string(),
                        line_number: (line_num + 1) as u32,
                        signature: line.trim().to_string(),
                        description: Some(format!("IOCTL handler: {}", handler_name)),
                        metadata: if !metadata.is_empty() {
                            Some(metadata)
                        } else {
                            None
                        },
                    });
                }
            }
        }
    }

    Ok(())
}

fn find_handler_definition(lines: &[&str], handler_name: &str, hint_line: usize) -> Option<usize> {
    // Search backwards and forwards from the hint line for the function definition
    let search_pattern = format!(
        r"(?:static\s+)?(?:long|int)\s+{}\s*\(",
        regex::escape(handler_name)
    );
    let pattern = Regex::new(&search_pattern).ok()?;

    // Search backward first (for forward declarations)
    if let Some(i) = (0..=hint_line.min(lines.len() - 1))
        .rev()
        .find(|&i| pattern.is_match(lines[i]))
    {
        return Some(i);
    }

    // Search forward if not found
    (hint_line..lines.len()).find(|&i| pattern.is_match(lines[i]))
}

fn extract_handled_commands(
    lines: &[&str],
    handler_start: usize,
    commands: &HashMap<String, serde_json::Value>,
) -> Vec<String> {
    let mut handled = Vec::new();
    let mut in_switch = false;
    let mut brace_count = 0;

    let end = lines.len().min(handler_start + 200);
    for (i, &line) in lines[handler_start..end].iter().enumerate() {
        let i = i + handler_start; // Adjust index

        // Look for switch statement
        if line.contains("switch") && line.contains("cmd") {
            in_switch = true;
        }

        if in_switch {
            // Track braces to know when we exit the switch
            brace_count += line.chars().filter(|&c| c == '{').count() as i32;
            brace_count -= line.chars().filter(|&c| c == '}').count() as i32;

            if brace_count <= 0 && i > handler_start {
                break; // Exit switch statement
            }

            // Look for case statements with known commands
            if line.trim().starts_with("case ") {
                for cmd_name in commands.keys() {
                    if line.contains(cmd_name) {
                        handled.push(cmd_name.clone());
                    }
                }
            }
        }
    }

    handled
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_extract_ioctls_from_sample() -> Result<()> {
        let temp_dir = tempdir()?;
        let kernel_dir = temp_dir.path();

        // Create a sample driver file
        let drivers_dir = kernel_dir.join("drivers");
        fs::create_dir_all(&drivers_dir)?;

        let mut driver_c = File::create(drivers_dir.join("sample.c"))?;
        writeln!(driver_c, "#define TEST_IOC_MAGIC 'T'")?;
        writeln!(driver_c, "#define TEST_IOCRESET     _IO(TEST_IOC_MAGIC, 0)")?;
        writeln!(
            driver_c,
            "#define TEST_IOCGETVAL    _IOR(TEST_IOC_MAGIC, 1, int)"
        )?;
        writeln!(driver_c)?;
        writeln!(
            driver_c,
            "static long sample_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {{"
        )?;
        writeln!(driver_c, "    switch (cmd) {{")?;
        writeln!(driver_c, "    case TEST_IOCRESET:")?;
        writeln!(driver_c, "        return 0;")?;
        writeln!(driver_c, "    case TEST_IOCGETVAL:")?;
        writeln!(driver_c, "        return 0;")?;
        writeln!(driver_c, "    }}")?;
        writeln!(driver_c, "    return -EINVAL;")?;
        writeln!(driver_c, "}}")?;
        writeln!(driver_c)?;
        writeln!(
            driver_c,
            "static const struct file_operations sample_fops = {{"
        )?;
        writeln!(driver_c, "    .unlocked_ioctl = sample_ioctl,")?;
        writeln!(driver_c, "}};")?;

        let ioctls = extract_ioctls(kernel_dir)?;
        assert!(!ioctls.is_empty());

        // Should find handler
        let found_handler = ioctls.iter().any(|ioctl| ioctl.name == "sample_ioctl");
        assert!(found_handler, "Should find sample_ioctl handler");

        // Should find commands
        let found_reset = ioctls.iter().any(|ioctl| ioctl.name == "TEST_IOCRESET");
        assert!(found_reset, "Should find TEST_IOCRESET command");

        let found_getval = ioctls.iter().any(|ioctl| ioctl.name == "TEST_IOCGETVAL");
        assert!(found_getval, "Should find TEST_IOCGETVAL command");

        // Check metadata for commands
        for ioctl in &ioctls {
            if ioctl.name == "TEST_IOCRESET" {
                assert!(
                    ioctl.metadata.is_some(),
                    "TEST_IOCRESET should have metadata"
                );
                if let Some(ref metadata) = ioctl.metadata {
                    assert!(
                        metadata.contains_key("magic"),
                        "Should have magic in metadata"
                    );
                    assert!(
                        metadata.contains_key("macro_type"),
                        "Should have macro_type in metadata"
                    );
                }
            }
        }

        Ok(())
    }
}
