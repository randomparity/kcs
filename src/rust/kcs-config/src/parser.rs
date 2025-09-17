use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    pub arch: String,
    pub config_name: String,
    pub version: String,
    pub options: HashMap<String, ConfigOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigOption {
    pub name: String,
    pub value: ConfigValue,
    pub config_type: ConfigType,
    pub help_text: Option<String>,
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    Bool(bool),
    String(String),
    Number(i64),
    Module,
    NotSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigType {
    Bool,
    Tristate,
    String,
    Int,
    Hex,
}

impl KernelConfig {
    pub fn parse(_content: &str) -> Result<Self> {
        todo!("Parser implementation for T013")
    }
}
