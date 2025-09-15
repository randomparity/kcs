use crate::{Requirement, RequirementCategory};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use yaml_rust::{Yaml, YamlLoader};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Specification {
    pub version: String,
    pub name: String,
    pub description: String,
    pub requirements: Vec<Requirement>,
}

pub fn parse_specification(spec_content: &str) -> Result<Specification> {
    // Try YAML first, then JSON
    if let Ok(spec) = parse_yaml_specification(spec_content) {
        Ok(spec)
    } else {
        parse_json_specification(spec_content)
    }
}

fn parse_yaml_specification(spec_content: &str) -> Result<Specification> {
    let docs = YamlLoader::load_from_str(spec_content)?;
    let doc = &docs[0];

    let version = doc["version"].as_str().unwrap_or("1.0").to_string();
    let name = doc["name"].as_str().unwrap_or("Unknown").to_string();
    let description = doc["description"].as_str().unwrap_or("").to_string();

    let mut requirements = Vec::new();

    if let Some(reqs) = doc["requirements"].as_vec() {
        for req in reqs {
            if let Some(requirement) = parse_yaml_requirement(req)? {
                requirements.push(requirement);
            }
        }
    }

    Ok(Specification {
        version,
        name,
        description,
        requirements,
    })
}

fn parse_yaml_requirement(yaml: &Yaml) -> Result<Option<Requirement>> {
    let id = match yaml["id"].as_str() {
        Some(id) => id.to_string(),
        None => return Ok(None), // Skip requirements without ID
    };

    let description = yaml["description"].as_str().unwrap_or("").to_string();
    let mandatory = yaml["mandatory"].as_bool().unwrap_or(true);
    let expected_behavior = yaml["expected_behavior"].as_str().unwrap_or("").to_string();

    let category = match yaml["category"].as_str() {
        Some("syscall") => RequirementCategory::Syscall,
        Some("api") => RequirementCategory::API,
        Some("abi") => RequirementCategory::ABI,
        Some("performance") => RequirementCategory::Performance,
        Some("security") => RequirementCategory::Security,
        Some("compatibility") => RequirementCategory::Compatibility,
        Some("feature") => RequirementCategory::Feature,
        _ => RequirementCategory::Feature,
    };

    let expected_symbols = if let Some(symbols) = yaml["expected_symbols"].as_vec() {
        symbols
            .iter()
            .filter_map(|s| s.as_str())
            .map(|s| s.to_string())
            .collect()
    } else if let Some(symbol) = yaml["expected_symbol"].as_str() {
        vec![symbol.to_string()]
    } else {
        Vec::new()
    };

    let config_dependencies = if let Some(configs) = yaml["config_dependencies"].as_vec() {
        configs
            .iter()
            .filter_map(|c| c.as_str())
            .map(|c| c.to_string())
            .collect()
    } else if let Some(config) = yaml["config_dependency"].as_str() {
        vec![config.to_string()]
    } else {
        Vec::new()
    };

    Ok(Some(Requirement {
        id,
        category,
        description,
        expected_symbols,
        expected_behavior,
        mandatory,
        config_dependencies,
    }))
}

fn parse_json_specification(spec_content: &str) -> Result<Specification> {
    let spec: serde_json::Value = serde_json::from_str(spec_content)?;

    let version = spec["version"].as_str().unwrap_or("1.0").to_string();
    let name = spec["name"].as_str().unwrap_or("Unknown").to_string();
    let description = spec["description"].as_str().unwrap_or("").to_string();

    let mut requirements = Vec::new();

    if let Some(reqs) = spec["requirements"].as_array() {
        for req in reqs {
            if let Some(requirement) = parse_json_requirement(req)? {
                requirements.push(requirement);
            }
        }
    }

    Ok(Specification {
        version,
        name,
        description,
        requirements,
    })
}

fn parse_json_requirement(json: &serde_json::Value) -> Result<Option<Requirement>> {
    let id = match json["id"].as_str() {
        Some(id) => id.to_string(),
        None => return Ok(None),
    };

    let description = json["description"].as_str().unwrap_or("").to_string();
    let mandatory = json["mandatory"].as_bool().unwrap_or(true);
    let expected_behavior = json["expected_behavior"].as_str().unwrap_or("").to_string();

    let category = match json["category"].as_str() {
        Some("syscall") => RequirementCategory::Syscall,
        Some("api") => RequirementCategory::API,
        Some("abi") => RequirementCategory::ABI,
        Some("performance") => RequirementCategory::Performance,
        Some("security") => RequirementCategory::Security,
        Some("compatibility") => RequirementCategory::Compatibility,
        Some("feature") => RequirementCategory::Feature,
        _ => RequirementCategory::Feature,
    };

    let expected_symbols = if let Some(symbols) = json["expected_symbols"].as_array() {
        symbols
            .iter()
            .filter_map(|s| s.as_str())
            .map(|s| s.to_string())
            .collect()
    } else if let Some(symbol) = json["expected_symbol"].as_str() {
        vec![symbol.to_string()]
    } else {
        Vec::new()
    };

    let config_dependencies = if let Some(configs) = json["config_dependencies"].as_array() {
        configs
            .iter()
            .filter_map(|c| c.as_str())
            .map(|c| c.to_string())
            .collect()
    } else if let Some(config) = json["config_dependency"].as_str() {
        vec![config.to_string()]
    } else {
        Vec::new()
    };

    Ok(Some(Requirement {
        id,
        category,
        description,
        expected_symbols,
        expected_behavior,
        mandatory,
        config_dependencies,
    }))
}

pub fn create_example_specification() -> Specification {
    Specification {
        version: "1.0".to_string(),
        name: "Example Kernel Specification".to_string(),
        description: "Example specification for demonstration".to_string(),
        requirements: vec![
            Requirement {
                id: "SYS_001".to_string(),
                category: RequirementCategory::Syscall,
                description: "System call open must be available".to_string(),
                expected_symbols: vec!["sys_open".to_string()],
                expected_behavior: "long sys_open(const char *filename, int flags, umode_t mode)".to_string(),
                mandatory: true,
                config_dependencies: vec![],
            },
            Requirement {
                id: "API_001".to_string(),
                category: RequirementCategory::API,
                description: "VFS read function must be available".to_string(),
                expected_symbols: vec!["vfs_read".to_string()],
                expected_behavior: "ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)".to_string(),
                mandatory: true,
                config_dependencies: vec!["CONFIG_VFS".to_string()],
            },
            Requirement {
                id: "SEC_001".to_string(),
                category: RequirementCategory::Security,
                description: "Security hooks must be available".to_string(),
                expected_symbols: vec!["security_file_open".to_string()],
                expected_behavior: "int security_file_open(struct file *file)".to_string(),
                mandatory: false,
                config_dependencies: vec!["CONFIG_SECURITY".to_string()],
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_yaml_specification() -> Result<()> {
        let yaml_spec = r#"
version: "1.0"
name: "Test Specification"
description: "Test specification for parsing"
requirements:
  - id: "REQ_001"
    category: "syscall"
    description: "Test requirement"
    expected_symbols:
      - "sys_test"
    expected_behavior: "int sys_test(void)"
    mandatory: true
    config_dependencies:
      - "CONFIG_TEST"
"#;

        let spec = parse_yaml_specification(yaml_spec)?;
        assert_eq!(spec.name, "Test Specification");
        assert_eq!(spec.requirements.len(), 1);
        assert_eq!(spec.requirements[0].id, "REQ_001");
        assert!(matches!(
            spec.requirements[0].category,
            RequirementCategory::Syscall
        ));

        Ok(())
    }

    #[test]
    fn test_parse_json_specification() -> Result<()> {
        let json_spec = r#"
{
  "version": "1.0",
  "name": "Test Specification",
  "description": "Test specification for parsing",
  "requirements": [
    {
      "id": "REQ_001",
      "category": "api",
      "description": "Test requirement",
      "expected_symbols": ["test_function"],
      "expected_behavior": "int test_function(void)",
      "mandatory": true,
      "config_dependencies": ["CONFIG_TEST"]
    }
  ]
}
"#;

        let spec = parse_json_specification(json_spec)?;
        assert_eq!(spec.name, "Test Specification");
        assert_eq!(spec.requirements.len(), 1);
        assert_eq!(spec.requirements[0].id, "REQ_001");
        assert!(matches!(
            spec.requirements[0].category,
            RequirementCategory::API
        ));

        Ok(())
    }

    #[test]
    fn test_create_example_specification() {
        let spec = create_example_specification();
        assert_eq!(spec.requirements.len(), 3);
        assert!(spec.requirements.iter().any(|r| r.id == "SYS_001"));
        assert!(spec.requirements.iter().any(|r| r.id == "API_001"));
        assert!(spec.requirements.iter().any(|r| r.id == "SEC_001"));
    }
}
