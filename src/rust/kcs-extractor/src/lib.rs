use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub mod entry_points;
pub mod ioctls;
pub mod syscalls;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryPoint {
    pub name: String,
    pub entry_type: EntryType,
    pub file_path: String,
    pub line_number: u32,
    pub signature: String,
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryType {
    Syscall,
    Ioctl,
    FileOps,
    Sysfs,
    ProcFs,
    DebugFs,
    Netlink,
    ModuleInit,
    ModuleExit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    pub include_syscalls: bool,
    pub include_ioctls: bool,
    pub include_file_ops: bool,
    pub include_sysfs: bool,
    pub include_procfs: bool,
    pub include_debugfs: bool,
    pub include_netlink: bool,
    pub include_modules: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            include_syscalls: true,
            include_ioctls: true,
            include_file_ops: true,
            include_sysfs: true,
            include_procfs: true,
            include_debugfs: true,
            include_netlink: true,
            include_modules: true,
        }
    }
}

pub struct Extractor {
    config: ExtractionConfig,
}

impl Extractor {
    pub fn new(config: ExtractionConfig) -> Self {
        Self { config }
    }

    pub fn extract_from_directory<P: AsRef<Path>>(&self, kernel_dir: P) -> Result<Vec<EntryPoint>> {
        let mut entry_points = Vec::new();

        if self.config.include_syscalls {
            entry_points.extend(syscalls::extract_syscalls(kernel_dir.as_ref())?);
        }

        if self.config.include_ioctls {
            entry_points.extend(ioctls::extract_ioctls(kernel_dir.as_ref())?);
        }

        if self.config.include_file_ops {
            entry_points.extend(entry_points::extract_file_operations(kernel_dir.as_ref())?);
        }

        if self.config.include_sysfs {
            entry_points.extend(entry_points::extract_sysfs_entries(kernel_dir.as_ref())?);
        }

        if self.config.include_procfs {
            entry_points.extend(entry_points::extract_procfs_entries(kernel_dir.as_ref())?);
        }

        if self.config.include_debugfs {
            entry_points.extend(entry_points::extract_debugfs_entries(kernel_dir.as_ref())?);
        }

        if self.config.include_netlink {
            entry_points.extend(entry_points::extract_netlink_handlers(kernel_dir.as_ref())?);
        }

        if self.config.include_modules {
            entry_points.extend(entry_points::extract_module_entries(kernel_dir.as_ref())?);
        }

        Ok(entry_points)
    }

    pub fn extract_from_index(&self, index_data: &str) -> Result<Vec<EntryPoint>> {
        use regex::Regex;

        let mut entry_points = Vec::new();

        // Try to parse as array first (current parser format)
        let parsed_files: Vec<serde_json::Value> = match serde_json::from_str(index_data) {
            Ok(array) => array,
            Err(_) => {
                // Fallback: try parsing as object
                let parsed_index: HashMap<String, serde_json::Value> =
                    serde_json::from_str(index_data)?;
                // Convert object to array format if needed
                parsed_index.into_values().collect()
            }
        };

        if self.config.include_syscalls {
            // Look for syscall functions by name patterns
            let sys_pattern = Regex::new(r"^(?:__se_sys_|__do_sys_|sys_)(\w+)$")?;
            let ksys_pattern = Regex::new(r"^ksys_(\w+)$")?;

            for file_data in &parsed_files {
                if let Some(file_path) = file_data.get("path").and_then(|p| p.as_str()) {
                    if let Some(symbols) = file_data.get("symbols").and_then(|s| s.as_array()) {
                        for symbol in symbols {
                            if let Some(signature) =
                                symbol.get("signature").and_then(|s| s.as_str())
                            {
                                if let Some(name) = symbol.get("name").and_then(|n| n.as_str()) {
                                    let start_line = symbol
                                        .get("start_line")
                                        .and_then(|l| l.as_u64())
                                        .unwrap_or(0)
                                        as u32;

                                    // Check for syscall function name patterns
                                    if let Some(captures) = sys_pattern.captures(name) {
                                        if let Some(syscall_name) = captures.get(1) {
                                            entry_points.push(EntryPoint {
                                                name: format!("sys_{}", syscall_name.as_str()),
                                                entry_type: EntryType::Syscall,
                                                file_path: file_path.to_string(),
                                                line_number: start_line,
                                                signature: signature.to_string(),
                                                description: Some(format!(
                                                    "System call: {}",
                                                    syscall_name.as_str()
                                                )),
                                                metadata: None,
                                            });
                                        }
                                    }
                                    // Check for ksys_ helper functions (often real syscall implementations)
                                    else if let Some(captures) = ksys_pattern.captures(name) {
                                        if let Some(syscall_name) = captures.get(1) {
                                            entry_points.push(EntryPoint {
                                                name: format!("ksys_{}", syscall_name.as_str()),
                                                entry_type: EntryType::Syscall,
                                                file_path: file_path.to_string(),
                                                line_number: start_line,
                                                signature: signature.to_string(),
                                                description: Some(format!(
                                                    "Kernel syscall helper: {}",
                                                    syscall_name.as_str()
                                                )),
                                                metadata: None,
                                            });
                                        }
                                    }
                                    // Check for direct sys_ function names (but not if already matched above)
                                    else if name.starts_with("sys_")
                                        && !name.starts_with("__se_sys_")
                                        && !name.starts_with("__do_sys_")
                                    {
                                        entry_points.push(EntryPoint {
                                            name: name.to_string(),
                                            entry_type: EntryType::Syscall,
                                            file_path: file_path.to_string(),
                                            line_number: start_line,
                                            signature: signature.to_string(),
                                            description: Some(format!(
                                                "Direct syscall: {}",
                                                &name[4..]
                                            )),
                                            metadata: None,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // TODO: Add other entry point types (ioctls, file_ops, etc.)
        // if self.config.include_ioctls { ... }
        // if self.config.include_file_ops { ... }

        Ok(entry_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_creation() {
        let extractor = Extractor::new(ExtractionConfig::default());
        assert!(extractor.config.include_syscalls);
    }
}
