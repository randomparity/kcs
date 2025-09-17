use crate::{EntryPoint, EntryType};
use anyhow::Result;
use regex::Regex;
use std::fs;
use std::path::Path;

pub fn extract_syscalls<P: AsRef<Path>>(kernel_dir: P) -> Result<Vec<EntryPoint>> {
    let mut syscalls = Vec::new();

    // Look for syscall definitions in kernel/sys.c and arch-specific files
    let syscall_files = [
        "kernel/sys.c",
        "arch/x86/entry/syscalls/syscall_64.tbl",
        "include/linux/syscalls.h",
    ];

    let syscall_pattern = Regex::new(r"SYSCALL_DEFINE\d+\(([^,\)]+)")?;
    let asmlinkage_pattern = Regex::new(r"asmlinkage\s+\w+\s+sys_(\w+)\s*\(")?;

    for file_path in &syscall_files {
        let full_path = kernel_dir.as_ref().join(file_path);
        if !full_path.exists() {
            continue;
        }

        let content = fs::read_to_string(&full_path)?;
        let lines: Vec<&str> = content.lines().collect();

        for (line_num, line) in lines.iter().enumerate() {
            // Match SYSCALL_DEFINE patterns
            if let Some(captures) = syscall_pattern.captures(line) {
                if let Some(name) = captures.get(1) {
                    syscalls.push(EntryPoint {
                        name: format!("sys_{}", name.as_str()),
                        entry_type: EntryType::Syscall,
                        file_path: file_path.to_string(),
                        line_number: (line_num + 1) as u32,
                        signature: line.trim().to_string(),
                        description: Some(format!("System call: {}", name.as_str())),
                        metadata: None,
                    });
                }
            }

            // Match asmlinkage patterns
            if let Some(captures) = asmlinkage_pattern.captures(line) {
                if let Some(name) = captures.get(1) {
                    syscalls.push(EntryPoint {
                        name: format!("sys_{}", name.as_str()),
                        entry_type: EntryType::Syscall,
                        file_path: file_path.to_string(),
                        line_number: (line_num + 1) as u32,
                        signature: line.trim().to_string(),
                        description: Some(format!("System call: {}", name.as_str())),
                        metadata: None,
                    });
                }
            }
        }
    }

    Ok(syscalls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_extract_syscalls_from_sample() -> Result<()> {
        let temp_dir = tempdir()?;
        let kernel_dir = temp_dir.path();

        // Create a sample sys.c file
        let sys_c_dir = kernel_dir.join("kernel");
        fs::create_dir_all(&sys_c_dir)?;

        let mut sys_c = File::create(sys_c_dir.join("sys.c"))?;
        writeln!(sys_c, "SYSCALL_DEFINE1(getpid, void) {{")?;
        writeln!(sys_c, "    return task_tgid_vnr(current);")?;
        writeln!(sys_c, "}}")?;

        let syscalls = extract_syscalls(kernel_dir)?;
        assert_eq!(syscalls.len(), 1);
        assert_eq!(syscalls[0].name, "sys_getpid");

        Ok(())
    }
}
