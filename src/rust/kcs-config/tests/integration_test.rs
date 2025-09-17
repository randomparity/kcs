use anyhow::Result;
use std::fs;
use std::process::Command;
use tempfile::NamedTempFile;

#[test]
fn test_cli_parse_config_file() -> Result<()> {
    let config_content = r#"#
# Automatically generated file; DO NOT EDIT.
# Linux/x86_64 6.1.0 Kernel Configuration
#
CONFIG_64BIT=y
CONFIG_X86_64=y
CONFIG_DEBUG_KERNEL=y
CONFIG_VFS=y
# CONFIG_MODULES is not set
CONFIG_PRINTK_TIME=y
CONFIG_HZ=1000
CONFIG_LOCALVERSION="-custom"
"#;

    let temp_file = NamedTempFile::new()?;
    fs::write(&temp_file, config_content)?;

    let output = Command::new("cargo")
        .args(["run", "--quiet", "--bin", "kcs-config", "--"])
        .arg(temp_file.path())
        .arg("--format")
        .arg("json")
        .env("RUST_LOG", "off")
        .output()?;

    assert!(
        output.status.success(),
        "Command failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json_output = String::from_utf8(output.stdout)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_output)?;

    assert_eq!(parsed["arch"], "x86_64");
    assert_eq!(parsed["version"], "6.1.0");
    assert_eq!(parsed["options"]["64BIT"]["value"], true);
    assert_eq!(
        parsed["options"]["MODULES"]["value"],
        serde_json::Value::Null
    );
    assert_eq!(parsed["options"]["HZ"]["value"], 1000);
    assert_eq!(parsed["options"]["LOCALVERSION"]["value"], "-custom");

    Ok(())
}

#[test]
fn test_cli_parse_kconfig() -> Result<()> {
    let kconfig_content = r#"
config DEBUG_KERNEL
	bool "Kernel debugging"
	help
	  Enable kernel debugging features.

config VFS
	bool "Virtual File System"
	default y
	help
	  Enable VFS support.

config HZ
	int "Timer frequency"
	default 1000
	help
	  Set kernel timer frequency.
"#;

    let temp_file = NamedTempFile::new()?;
    fs::write(&temp_file, kconfig_content)?;

    let output = Command::new("cargo")
        .args(["run", "--quiet", "--bin", "kcs-config", "--"])
        .env("RUST_LOG", "off")
        .arg(temp_file.path())
        .arg("--kconfig")
        .arg("--format")
        .arg("json")
        .output()?;

    assert!(
        output.status.success(),
        "Command failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json_output = String::from_utf8(output.stdout)?;
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output)?;

    assert_eq!(parsed.len(), 3);

    let debug_kernel = parsed
        .iter()
        .find(|opt| opt["name"] == "DEBUG_KERNEL")
        .expect("DEBUG_KERNEL option not found");
    assert_eq!(debug_kernel["config_type"], "Bool");
    assert!(debug_kernel["help_text"].is_string());

    let vfs = parsed
        .iter()
        .find(|opt| opt["name"] == "VFS")
        .expect("VFS option not found");
    assert_eq!(vfs["value"], true);

    let hz = parsed
        .iter()
        .find(|opt| opt["name"] == "HZ")
        .expect("HZ option not found");
    assert_eq!(hz["config_type"], "Int");
    assert_eq!(hz["value"], 1000);

    Ok(())
}

#[test]
fn test_cli_stdin_input() -> Result<()> {
    let config_content = "CONFIG_TEST=y\nCONFIG_VALUE=100";

    // Use shell to pipe content to stdin
    let output = Command::new("sh")
        .arg("-c")
        .arg(format!("echo '{}' | RUST_LOG=off cargo run --quiet --bin kcs-config -- - --format json 2>/dev/null",
                    config_content))
        .output()?;

    let json_output = String::from_utf8(output.stdout)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_output)?;

    assert_eq!(parsed["options"]["TEST"]["value"], true);
    assert_eq!(parsed["options"]["VALUE"]["value"], 100);

    Ok(())
}

#[test]
fn test_cli_filter_options() -> Result<()> {
    let config_content = r#"
CONFIG_NET_CORE=y
CONFIG_NET_ETHERNET=y
CONFIG_NET_WIRELESS=m
CONFIG_DEBUG_KERNEL=y
CONFIG_PRINTK=y
"#;

    let temp_file = NamedTempFile::new()?;
    fs::write(&temp_file, config_content)?;

    let output = Command::new("cargo")
        .args(["run", "--quiet", "--bin", "kcs-config", "--"])
        .env("RUST_LOG", "off")
        .arg(temp_file.path())
        .arg("--filter")
        .arg("NET")
        .arg("--format")
        .arg("json")
        .output()?;

    assert!(output.status.success());

    let json_output = String::from_utf8(output.stdout)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_output)?;

    // Should only have NET_ options
    let options = parsed["options"].as_object().unwrap();
    assert_eq!(options.len(), 3);
    assert!(options.contains_key("NET_CORE"));
    assert!(options.contains_key("NET_ETHERNET"));
    assert!(options.contains_key("NET_WIRELESS"));
    assert!(!options.contains_key("DEBUG_KERNEL"));
    assert!(!options.contains_key("PRINTK"));

    Ok(())
}

#[test]
fn test_cli_summary_format() -> Result<()> {
    let config_content = r#"# Linux/x86 6.1.0 Kernel Configuration
CONFIG_A=y
CONFIG_B=m
# CONFIG_C is not set
CONFIG_D=100
"#;

    let temp_file = NamedTempFile::new()?;
    fs::write(&temp_file, config_content)?;

    let output = Command::new("cargo")
        .args(["run", "--quiet", "--bin", "kcs-config", "--"])
        .env("RUST_LOG", "off")
        .arg(temp_file.path())
        .arg("--format")
        .arg("summary")
        .output()?;

    assert!(output.status.success());
    let summary = String::from_utf8(output.stdout)?;

    assert!(summary.contains("Architecture: x86"));
    assert!(summary.contains("Version: 6.1.0"));
    assert!(summary.contains("Total Options: 4"));
    assert!(summary.contains("Enabled (y): 1"));
    assert!(summary.contains("Modules (m): 1"));
    assert!(summary.contains("Not Set: 1"));

    Ok(())
}

#[test]
fn test_cli_toml_format() -> Result<()> {
    let config_content = "CONFIG_TEST=y\nCONFIG_NUM=42";

    let temp_file = NamedTempFile::new()?;
    fs::write(&temp_file, config_content)?;

    let output = Command::new("cargo")
        .args(["run", "--quiet", "--bin", "kcs-config", "--"])
        .env("RUST_LOG", "off")
        .arg(temp_file.path())
        .arg("--format")
        .arg("toml")
        .output()?;

    assert!(output.status.success());
    let toml_output = String::from_utf8(output.stdout)?;

    // Basic validation that it's valid TOML
    let parsed: toml::Value = toml::from_str(&toml_output)?;
    assert!(parsed.get("options").is_some());

    Ok(())
}
