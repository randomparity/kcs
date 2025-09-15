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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryType {
    Syscall,
    Ioctl,
    FileOps,
    Sysfs,
    ProcFs,
    DebugFs,
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

        Ok(entry_points)
    }

    pub fn extract_from_index(&self, index_data: &str) -> Result<Vec<EntryPoint>> {
        let _parsed_index: HashMap<String, serde_json::Value> = serde_json::from_str(index_data)?;

        let entry_points = Vec::new();

        // Extract from parsed index data
        // This will be implemented based on the parser output format

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
