use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    pub name: String,
    pub arch: String,
    pub options: HashMap<String, ConfigValue>,
    pub dependencies: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    Bool(bool),
    String(String),
    Integer(i64),
    Module,
}

impl ConfigValue {
    pub fn is_enabled(&self) -> bool {
        match self {
            ConfigValue::Bool(b) => *b,
            ConfigValue::String(s) => !s.is_empty(),
            ConfigValue::Integer(i) => *i != 0,
            ConfigValue::Module => true,
        }
    }
}

#[derive(Debug)]
pub struct ConfigParser {
    config: KernelConfig,
}

impl ConfigParser {
    pub fn new() -> Self {
        Self {
            config: KernelConfig {
                name: "default".to_string(),
                arch: "x86_64".to_string(),
                options: HashMap::new(),
                dependencies: HashMap::new(),
            },
        }
    }

    pub fn parse_dotconfig<P: AsRef<Path>>(&mut self, config_path: P) -> Result<()> {
        let content = std::fs::read_to_string(config_path)?;

        for line in content.lines() {
            let line = line.trim();

            // Skip comments and empty lines
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            // Parse CONFIG_OPTION=value or # CONFIG_OPTION is not set
            if line.starts_with("# ") && line.ends_with(" is not set") {
                // Disabled config option
                let option_name = line
                    .strip_prefix("# ")
                    .and_then(|s| s.strip_suffix(" is not set"))
                    .unwrap_or("")
                    .to_string();

                if !option_name.is_empty() {
                    self.config.options.insert(option_name, ConfigValue::Bool(false));
                }
            } else if let Some(eq_pos) = line.find('=') {
                // Enabled config option
                let option_name = line[..eq_pos].to_string();
                let value_str = &line[eq_pos + 1..];

                let config_value = if value_str == "y" {
                    ConfigValue::Bool(true)
                } else if value_str == "n" {
                    ConfigValue::Bool(false)
                } else if value_str == "m" {
                    ConfigValue::Module
                } else if let Ok(int_val) = value_str.parse::<i64>() {
                    ConfigValue::Integer(int_val)
                } else {
                    // Remove quotes if present
                    let cleaned = value_str.trim_matches('"');
                    ConfigValue::String(cleaned.to_string())
                };

                self.config.options.insert(option_name, config_value);
            }
        }

        Ok(())
    }

    pub fn parse_kconfig<P: AsRef<Path>>(&mut self, kconfig_path: P) -> Result<()> {
        let content = std::fs::read_to_string(kconfig_path)?;

        let mut current_config: Option<String> = None;
        let mut dependencies: Vec<String> = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            if let Some(config_name_part) = line.strip_prefix("config ") {
                // Save previous config if any
                if let Some(config_name) = current_config.take() {
                    if !dependencies.is_empty() {
                        self.config.dependencies.insert(config_name, dependencies.clone());
                        dependencies.clear();
                    }
                }

                current_config = Some(config_name_part.to_string());
            } else if let Some(deps_str) = line.strip_prefix("depends on ") {
                // Parse dependencies (simplified - doesn't handle complex expressions)
                for dep in deps_str.split("&&") {
                    let dep = dep.trim().replace("!", "");
                    if dep.starts_with("CONFIG_") {
                        dependencies.push(dep);
                    }
                }
            }
        }

        // Save last config
        if let Some(config_name) = current_config {
            if !dependencies.is_empty() {
                self.config.dependencies.insert(config_name, dependencies);
            }
        }

        Ok(())
    }

    pub fn is_option_enabled(&self, option: &str) -> bool {
        self.config.options.get(option).map(|v| v.is_enabled()).unwrap_or(false)
    }

    pub fn get_enabled_options(&self) -> HashMap<String, bool> {
        self.config.options.iter().map(|(k, v)| (k.clone(), v.is_enabled())).collect()
    }

    pub fn resolve_dependencies(&self, option: &str) -> Vec<String> {
        let mut resolved = Vec::new();
        let mut to_check = vec![option.to_string()];
        let mut visited = std::collections::HashSet::new();

        while let Some(current) = to_check.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(deps) = self.config.dependencies.get(&current) {
                for dep in deps {
                    if !resolved.contains(dep) {
                        resolved.push(dep.clone());
                        to_check.push(dep.clone());
                    }
                }
            }
        }

        resolved
    }

    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    pub fn into_config(self) -> KernelConfig {
        self.config
    }
}

impl Default for ConfigParser {
    fn default() -> Self {
        Self::new()
    }
}

pub fn parse_defconfig(arch: &str, defconfig_name: &str) -> Result<KernelConfig> {
    // This would parse a defconfig file for a specific architecture
    // For now, return a minimal default config
    Ok(KernelConfig {
        name: format!("{}:{}", arch, defconfig_name),
        arch: arch.to_string(),
        options: HashMap::from([
            ("CONFIG_64BIT".to_string(), ConfigValue::Bool(true)),
            ("CONFIG_X86_64".to_string(), ConfigValue::Bool(true)),
            ("CONFIG_SMP".to_string(), ConfigValue::Bool(true)),
            ("CONFIG_VFS".to_string(), ConfigValue::Bool(true)),
        ]),
        dependencies: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_dotconfig() -> Result<()> {
        let mut config_file = NamedTempFile::new()?;
        writeln!(config_file, "CONFIG_64BIT=y")?;
        writeln!(config_file, "CONFIG_SMP=y")?;
        writeln!(config_file, "# CONFIG_DEBUG is not set")?;
        writeln!(config_file, "CONFIG_MODULES=m")?;
        writeln!(config_file, "CONFIG_MAX_CPUS=8")?;

        let mut parser = ConfigParser::new();
        parser.parse_dotconfig(config_file.path())?;

        assert!(parser.is_option_enabled("CONFIG_64BIT"));
        assert!(parser.is_option_enabled("CONFIG_SMP"));
        assert!(!parser.is_option_enabled("CONFIG_DEBUG"));
        assert!(parser.is_option_enabled("CONFIG_MODULES"));

        Ok(())
    }

    #[test]
    fn test_config_value_is_enabled() {
        assert!(ConfigValue::Bool(true).is_enabled());
        assert!(!ConfigValue::Bool(false).is_enabled());
        assert!(ConfigValue::Module.is_enabled());
        assert!(!ConfigValue::Integer(0).is_enabled());
        assert!(ConfigValue::Integer(1).is_enabled());
        assert!(!ConfigValue::String("".to_string()).is_enabled());
        assert!(ConfigValue::String("value".to_string()).is_enabled());
    }
}
