use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KernelConfig {
    pub arch: String,
    pub config_name: String,
    pub version: String,
    pub options: HashMap<String, ConfigOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigOption {
    pub name: String,
    pub value: ConfigValue,
    pub config_type: ConfigType,
    pub help_text: Option<String>,
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ConfigValue {
    Bool(bool),
    String(String),
    Number(i64),
    Module,
    NotSet,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfigType {
    Bool,
    Tristate,
    String,
    Int,
    Hex,
}

impl KernelConfig {
    pub fn parse(content: &str) -> Result<Self> {
        let mut config = KernelConfig {
            arch: String::new(),
            config_name: String::new(),
            version: String::new(),
            options: HashMap::new(),
        };

        let config_re = Regex::new(r"^CONFIG_([A-Z0-9_]+)=(.+)$")?;
        let not_set_re = Regex::new(r"^# CONFIG_([A-Z0-9_]+) is not set$")?;
        let header_re = Regex::new(r"^# Linux/([^ ]+) ([^ ]+) Kernel Configuration$")?;

        for line in content.lines() {
            let line = line.trim();

            if line.is_empty() || (line.starts_with('#') && !line.contains("CONFIG_")) {
                if let Some(caps) = header_re.captures(line) {
                    config.arch = caps.get(1).map_or("", |m| m.as_str()).to_string();
                    config.version = caps.get(2).map_or("", |m| m.as_str()).to_string();
                }
                continue;
            }

            if let Some(caps) = not_set_re.captures(line) {
                let name = caps.get(1).map_or("", |m| m.as_str()).to_string();
                config.options.insert(
                    name.clone(),
                    ConfigOption {
                        name,
                        value: ConfigValue::NotSet,
                        config_type: ConfigType::Bool,
                        help_text: None,
                        depends_on: vec![],
                    },
                );
            } else if let Some(caps) = config_re.captures(line) {
                let name = caps.get(1).map_or("", |m| m.as_str()).to_string();
                let value_str = caps.get(2).map_or("", |m| m.as_str());

                let (value, config_type) = parse_config_value(value_str)?;

                config.options.insert(
                    name.clone(),
                    ConfigOption {
                        name,
                        value,
                        config_type,
                        help_text: None,
                        depends_on: vec![],
                    },
                );
            }
        }

        if config.options.is_empty() {
            return Err(anyhow!("No configuration options found"));
        }

        Ok(config)
    }

    pub fn parse_kconfig(content: &str) -> Result<Vec<ConfigOption>> {
        let mut options = Vec::new();
        let mut current_option: Option<ConfigOption> = None;
        let mut in_help = false;
        let mut help_text = String::new();

        let config_re = Regex::new(r"^config\s+([A-Z0-9_]+)$")?;
        let bool_re = Regex::new(r#"^\s*bool\s+"(.*)"$"#)?;
        let tristate_re = Regex::new(r#"^\s*tristate\s+"(.*)"$"#)?;
        let string_re = Regex::new(r#"^\s*string\s+"(.*)"$"#)?;
        let int_re = Regex::new(r#"^\s*int\s+"(.*)"$"#)?;
        let hex_re = Regex::new(r#"^\s*hex\s+"(.*)"$"#)?;
        let default_re = Regex::new(r"^\s*default\s+(.+)$")?;
        let depends_re = Regex::new(r"^\s*depends\s+on\s+(.+)$")?;
        let help_re = Regex::new(r"^\s*help$")?;

        for line in content.lines() {
            if in_help {
                if line.starts_with('\t') || line.starts_with("  ") {
                    if !help_text.is_empty() {
                        help_text.push('\n');
                    }
                    help_text.push_str(line.trim());
                } else {
                    in_help = false;
                    if let Some(ref mut opt) = current_option {
                        opt.help_text = Some(help_text.clone());
                    }
                    help_text.clear();
                }
            }

            if let Some(caps) = config_re.captures(line) {
                if let Some(opt) = current_option.take() {
                    options.push(opt);
                }

                let name = caps.get(1).map_or("", |m| m.as_str()).to_string();
                current_option = Some(ConfigOption {
                    name,
                    value: ConfigValue::NotSet,
                    config_type: ConfigType::Bool,
                    help_text: None,
                    depends_on: vec![],
                });
            } else if bool_re.is_match(line) {
                if let Some(ref mut opt) = current_option {
                    opt.config_type = ConfigType::Bool;
                }
            } else if tristate_re.is_match(line) {
                if let Some(ref mut opt) = current_option {
                    opt.config_type = ConfigType::Tristate;
                }
            } else if string_re.is_match(line) {
                if let Some(ref mut opt) = current_option {
                    opt.config_type = ConfigType::String;
                }
            } else if int_re.is_match(line) {
                if let Some(ref mut opt) = current_option {
                    opt.config_type = ConfigType::Int;
                }
            } else if hex_re.is_match(line) {
                if let Some(ref mut opt) = current_option {
                    opt.config_type = ConfigType::Hex;
                }
            } else if let Some(caps) = default_re.captures(line) {
                if let Some(ref mut opt) = current_option {
                    let default_val = caps.get(1).map_or("", |m| m.as_str());
                    opt.value = match default_val {
                        "y" => ConfigValue::Bool(true),
                        "n" => ConfigValue::Bool(false),
                        "m" => ConfigValue::Module,
                        val if val.starts_with('"') && val.ends_with('"') => {
                            ConfigValue::String(val.trim_matches('"').to_string())
                        }
                        val => {
                            if let Ok(num) = val.parse::<i64>() {
                                ConfigValue::Number(num)
                            } else if let Some(stripped) = val.strip_prefix("0x") {
                                if let Ok(num) = i64::from_str_radix(stripped, 16) {
                                    ConfigValue::Number(num)
                                } else {
                                    ConfigValue::String(val.to_string())
                                }
                            } else {
                                ConfigValue::String(val.to_string())
                            }
                        }
                    };
                }
            } else if let Some(caps) = depends_re.captures(line) {
                if let Some(ref mut opt) = current_option {
                    let deps = caps.get(1).map_or("", |m| m.as_str());
                    opt.depends_on = deps.split("&&").map(|s| s.trim().to_string()).collect();
                }
            } else if help_re.is_match(line) {
                in_help = true;
                help_text.clear();
            }
        }

        if let Some(mut opt) = current_option {
            if in_help && !help_text.is_empty() {
                opt.help_text = Some(help_text);
            }
            options.push(opt);
        }

        Ok(options)
    }
}

pub(crate) fn parse_config_value(value_str: &str) -> Result<(ConfigValue, ConfigType)> {
    match value_str {
        "y" => Ok((ConfigValue::Bool(true), ConfigType::Tristate)),
        "n" => Ok((ConfigValue::Bool(false), ConfigType::Tristate)),
        "m" => Ok((ConfigValue::Module, ConfigType::Tristate)),
        val if val.starts_with('"') && val.ends_with('"') => Ok((
            ConfigValue::String(val.trim_matches('"').to_string()),
            ConfigType::String,
        )),
        val if val.starts_with("0x") || val.starts_with("0X") => {
            let stripped = val.trim_start_matches(|c: char| c == '0' || c == 'x' || c == 'X')
                .trim_start_matches('x');

            // If it's a valid hex string but doesn't fit in i64, keep it as a string to avoid failure
            match i64::from_str_radix(stripped, 16) {
                Ok(hex_val) => Ok((ConfigValue::Number(hex_val), ConfigType::Hex)),
                Err(_) => {
                    // Validate hex characters; if valid, preserve as string
                    if stripped.chars().all(|c| c.is_ascii_hexdigit()) {
                        Ok((ConfigValue::String(val.to_string()), ConfigType::Hex))
                    } else {
                        Err(anyhow!("Invalid hex value {}: contains non-hex characters", val))
                    }
                }
            }
        }
        val => {
            if let Ok(num) = val.parse::<i64>() {
                Ok((ConfigValue::Number(num), ConfigType::Int))
            } else {
                Ok((ConfigValue::String(val.to_string()), ConfigType::String))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config_file() {
        let config_content = r#"#
# Automatically generated file; DO NOT EDIT.
# Linux/x86 6.1.0 Kernel Configuration
#
CONFIG_64BIT=y
CONFIG_X86_64=y
CONFIG_X86=y
CONFIG_DEBUG_KERNEL=y
CONFIG_VFS=y
CONFIG_PROC_FS=y
# CONFIG_MODULES is not set
CONFIG_PRINTK_TIME=y
CONFIG_HZ=1000
CONFIG_LOCALVERSION="-test"
"#;

        let config = KernelConfig::parse(config_content).unwrap();

        assert_eq!(config.arch, "x86");
        assert_eq!(config.version, "6.1.0");
        assert_eq!(config.options.len(), 10);

        let bit64 = config.options.get("64BIT").unwrap();
        assert_eq!(bit64.value, ConfigValue::Bool(true));
        assert_eq!(bit64.config_type, ConfigType::Tristate);

        let modules = config.options.get("MODULES").unwrap();
        assert_eq!(modules.value, ConfigValue::NotSet);

        let hz = config.options.get("HZ").unwrap();
        assert_eq!(hz.value, ConfigValue::Number(1000));

        let localversion = config.options.get("LOCALVERSION").unwrap();
        assert_eq!(localversion.value, ConfigValue::String("-test".to_string()));
    }

    #[test]
    fn test_parse_kconfig() {
        let kconfig_content = r#"
config DEBUG_KERNEL
	bool "Kernel debugging"
	help
	  Enable kernel debugging features for development.

config VFS
	bool "Virtual File System support"
	default y
	help
	  Enable VFS layer for file operations.

config PROC_FS
	bool "Proc filesystem support"
	depends on VFS
	help
	  Enable /proc filesystem.

config MODULES
	tristate "Enable loadable module support"
	help
	  Enable support for loadable kernel modules.

config HZ
	int "Timer frequency"
	default 1000
	help
	  Set the kernel timer frequency.
"#;

        let options = KernelConfig::parse_kconfig(kconfig_content).unwrap();

        assert_eq!(options.len(), 5);

        let debug = options.iter().find(|o| o.name == "DEBUG_KERNEL").unwrap();
        assert_eq!(debug.config_type, ConfigType::Bool);
        assert!(debug.help_text.is_some());

        let vfs = options.iter().find(|o| o.name == "VFS").unwrap();
        assert_eq!(vfs.value, ConfigValue::Bool(true));
        assert_eq!(vfs.config_type, ConfigType::Bool);

        let proc_fs = options.iter().find(|o| o.name == "PROC_FS").unwrap();
        assert_eq!(proc_fs.depends_on, vec!["VFS"]);

        let modules = options.iter().find(|o| o.name == "MODULES").unwrap();
        assert_eq!(modules.config_type, ConfigType::Tristate);

        let hz = options.iter().find(|o| o.name == "HZ").unwrap();
        assert_eq!(hz.config_type, ConfigType::Int);
        assert_eq!(hz.value, ConfigValue::Number(1000));
    }

    #[test]
    fn test_parse_config_value() {
        assert_eq!(
            parse_config_value("y").unwrap(),
            (ConfigValue::Bool(true), ConfigType::Tristate)
        );
        assert_eq!(
            parse_config_value("n").unwrap(),
            (ConfigValue::Bool(false), ConfigType::Tristate)
        );
        assert_eq!(
            parse_config_value("m").unwrap(),
            (ConfigValue::Module, ConfigType::Tristate)
        );
        assert_eq!(
            parse_config_value("\"test\"").unwrap(),
            (ConfigValue::String("test".to_string()), ConfigType::String)
        );
        assert_eq!(
            parse_config_value("1000").unwrap(),
            (ConfigValue::Number(1000), ConfigType::Int)
        );
        assert_eq!(
            parse_config_value("0x100").unwrap(),
            (ConfigValue::Number(256), ConfigType::Hex)
        );
    }

    #[test]
    fn test_parse_large_hex_as_string() {
        // Larger than i64::MAX; should be preserved as a hex string
        let val = "0xdead000000000000";
        let parsed = parse_config_value(val).unwrap();
        assert_eq!(parsed, (ConfigValue::String(val.to_string()), ConfigType::Hex));
    }

    #[test]
    fn test_empty_config() {
        let result = KernelConfig::parse("");
        assert!(result.is_err());
    }
}
