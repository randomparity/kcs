use crate::{EntryPoint, EntryType};
use anyhow::Result;
use regex::Regex;
use std::fs;
use std::path::Path;

pub fn extract_ioctls<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut ioctls = Vec::new();

    // Look for ioctl definitions across the kernel

    // Search through common directories where ioctls are defined
    let search_dirs = ["drivers", "fs", "net", "arch", "kernel"];

    for search_dir in &search_dirs {
        let dir_path = kernel_dir.as_ref().join(search_dir);
        if !dir_path.exists() {
            continue;
        }

        // Recursively search for .c and .h files
        if let Ok(entries) = fs::read_dir(&dir_path) {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|ft| ft.is_file()) {
                    let file_path = entry.path();
                    if let Some(ext) = file_path.extension() {
                        if ext == "c" || ext == "h" {
                            if let Ok(relative_path) = file_path.strip_prefix(kernel_dir.as_ref()) {
                                extract_ioctls_from_file(
                                    &file_path,
                                    relative_path.to_string_lossy().as_ref(),
                                    &mut ioctls,
                                )?;
                            }
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
) -> Result<()> {
    let content = fs::read_to_string(file_path)?;
    let lines: Vec<&str> = content.lines().collect();

    let ioctl_patterns = [
        // Function pointer assignments in file_operations
        Regex::new(r"\.unlocked_ioctl\s*=\s*(\w+)")?,
        Regex::new(r"\.compat_ioctl\s*=\s*(\w+)")?,
        // Direct ioctl function definitions
        Regex::new(r"static\s+long\s+(\w+_ioctl)\s*\(")?,
    ];

    for (line_num, line) in lines.iter().enumerate() {
        for pattern in &ioctl_patterns {
            if let Some(captures) = pattern.captures(line) {
                if let Some(name) = captures.get(1) {
                    ioctls.push(EntryPoint {
                        name: name.as_str().to_string(),
                        entry_type: EntryType::Ioctl,
                        file_path: relative_path.to_string(),
                        line_number: (line_num + 1) as u32,
                        signature: line.trim().to_string(),
                        description: Some(format!("IOCTL handler: {}", name.as_str())),
                    });
                }
            }
        }
    }

    Ok(())
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
        writeln!(
            driver_c,
            "static long sample_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {{"
        )?;
        writeln!(driver_c, "    return 0;")?;
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

        let found_ioctl = ioctls.iter().any(|ioctl| ioctl.name == "sample_ioctl");
        assert!(found_ioctl);

        Ok(())
    }
}
