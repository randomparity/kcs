//! Configuration management utilities
//!
//! Provides centralized configuration management for call graph extraction,
//! supporting multiple configuration sources and formats.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::call_extraction::CallExtractionConfig;
use crate::error_handling::{CallGraphError, CallGraphResult, ErrorHandlingConfig};
use kcs_graph::ConfidenceLevel;

/// Main configuration structure for the parser and call extraction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct KcsConfig {
    /// Parser configuration
    pub parser: ParserConfig,
    /// Call extraction configuration
    pub call_extraction: CallExtractionConfig,
    /// Error handling configuration
    pub error_handling: ErrorHandlingConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Performance tuning configuration
    pub performance: PerformanceConfig,
}

/// Parser-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParserConfig {
    /// Enable tree-sitter parsing
    pub tree_sitter_enabled: bool,
    /// Enable clang integration
    pub clang_enabled: bool,
    /// Target architecture
    pub target_arch: String,
    /// Kernel version
    pub kernel_version: String,
    /// Include paths for clang
    pub include_paths: Vec<PathBuf>,
    /// Preprocessor defines
    pub defines: HashMap<String, String>,
    /// Compilation database path
    pub compile_commands_path: Option<PathBuf>,
    /// Configuration name (e.g., "defconfig", "allmodconfig")
    pub config_name: String,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            tree_sitter_enabled: true,
            clang_enabled: false,
            target_arch: "x86_64".to_string(),
            kernel_version: "6.1".to_string(),
            include_paths: vec![],
            defines: HashMap::new(),
            compile_commands_path: None,
            config_name: "defconfig".to_string(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Enable structured logging
    pub structured: bool,
    /// Log to file path (None for stdout/stderr)
    pub file_path: Option<PathBuf>,
    /// Enable performance metrics logging
    pub enable_metrics: bool,
    /// Log format (json, compact, pretty)
    pub format: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            structured: true,
            file_path: None,
            enable_metrics: true,
            format: "pretty".to_string(),
        }
    }
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerformanceConfig {
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory_usage: u64,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Cache size for frequently accessed data
    pub cache_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: 0, // Auto-detect
            enable_parallel: true,
            max_memory_usage: 0, // Unlimited
            batch_size: 1000,
            cache_size: 10000,
        }
    }
}

/// Configuration source types
#[derive(Debug, Clone)]
pub enum ConfigSource {
    /// Default configuration
    Default,
    /// Configuration file
    File(PathBuf),
    /// Environment variables
    Environment,
    /// Programmatic configuration
    Programmatic(Box<KcsConfig>),
}

/// Configuration manager for loading and merging configurations
#[derive(Debug)]
pub struct ConfigManager {
    /// Current configuration
    config: KcsConfig,
    /// Configuration sources in priority order (higher index = higher priority)
    sources: Vec<ConfigSource>,
}

impl ConfigManager {
    /// Create a new configuration manager with default settings
    pub fn new() -> Self {
        Self {
            config: KcsConfig::default(),
            sources: vec![ConfigSource::Default],
        }
    }

    /// Create configuration manager with explicit initial config
    pub fn with_config(config: KcsConfig) -> Self {
        Self {
            config: config.clone(),
            sources: vec![ConfigSource::Programmatic(Box::new(config))],
        }
    }

    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> CallGraphResult<()> {
        let path = path.as_ref();
        let config = Self::load_config_file(path)
            .with_context(|| format!("Failed to load config from {}", path.display()))?;

        self.sources.push(ConfigSource::File(path.to_path_buf()));
        self.merge_config(config);
        Ok(())
    }

    /// Load configuration from environment variables
    pub fn load_from_env(&mut self) -> CallGraphResult<()> {
        let config = Self::load_env_config()?;
        self.sources.push(ConfigSource::Environment);
        self.merge_config(config);
        Ok(())
    }

    /// Set configuration programmatically
    pub fn set_config(&mut self, config: KcsConfig) {
        self.sources.push(ConfigSource::Programmatic(Box::new(config.clone())));
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &KcsConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut KcsConfig {
        &mut self.config
    }

    /// Reload configuration from all sources
    pub fn reload(&mut self) -> CallGraphResult<()> {
        let mut config = KcsConfig::default();

        for source in &self.sources {
            match source {
                ConfigSource::Default => {
                    // Already using defaults
                },
                ConfigSource::File(path) => {
                    let file_config = Self::load_config_file(path).with_context(|| {
                        format!("Failed to reload config from {}", path.display())
                    })?;
                    Self::merge_configs(&mut config, file_config);
                },
                ConfigSource::Environment => {
                    let env_config = Self::load_env_config()?;
                    Self::merge_configs(&mut config, env_config);
                },
                ConfigSource::Programmatic(prog_config) => {
                    config = (**prog_config).clone();
                },
            }
        }

        self.config = config;
        self.validate_config()?;
        Ok(())
    }

    /// Validate current configuration
    pub fn validate_config(&self) -> CallGraphResult<()> {
        // Validate parser config
        if self.config.parser.target_arch.is_empty() {
            return Err(CallGraphError::ConfigError {
                message: "target_arch cannot be empty".to_string(),
            });
        }

        if self.config.parser.kernel_version.is_empty() {
            return Err(CallGraphError::ConfigError {
                message: "kernel_version cannot be empty".to_string(),
            });
        }

        // Validate performance config
        if self.config.performance.batch_size == 0 {
            return Err(CallGraphError::ConfigError {
                message: "batch_size must be greater than 0".to_string(),
            });
        }

        // Validate logging config
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.config.logging.level.as_str()) {
            return Err(CallGraphError::ConfigError {
                message: format!(
                    "Invalid log level '{}', must be one of: {:?}",
                    self.config.logging.level, valid_levels
                ),
            });
        }

        let valid_formats = ["json", "compact", "pretty"];
        if !valid_formats.contains(&self.config.logging.format.as_str()) {
            return Err(CallGraphError::ConfigError {
                message: format!(
                    "Invalid log format '{}', must be one of: {:?}",
                    self.config.logging.format, valid_formats
                ),
            });
        }

        Ok(())
    }

    /// Save current configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> CallGraphResult<()> {
        let path = path.as_ref();
        Self::save_config_file(&self.config, path)
            .with_context(|| format!("Failed to save config to {}", path.display()))?;
        Ok(())
    }

    /// Load configuration from file based on extension
    fn load_config_file(path: &Path) -> CallGraphResult<KcsConfig> {
        let content = fs::read_to_string(path).map_err(|e| CallGraphError::FileError {
            path: path.to_string_lossy().to_string(),
            source: e,
        })?;

        let config = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => {
                serde_json::from_str(&content).map_err(|e| CallGraphError::ConfigError {
                    message: format!("Failed to parse JSON config: {}", e),
                })?
            },
            Some("toml") => toml::from_str(&content).map_err(|e| CallGraphError::ConfigError {
                message: format!("Failed to parse TOML config: {}", e),
            })?,
            Some("yaml") | Some("yml") => {
                serde_yaml::from_str(&content).map_err(|e| CallGraphError::ConfigError {
                    message: format!("Failed to parse YAML config: {}", e),
                })?
            },
            _ => {
                return Err(CallGraphError::ConfigError {
                    message: format!("Unsupported config file format for {}", path.display()),
                });
            },
        };

        Ok(config)
    }

    /// Save configuration to file based on extension
    fn save_config_file(config: &KcsConfig, path: &Path) -> CallGraphResult<()> {
        let content = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => {
                serde_json::to_string_pretty(config).map_err(|e| CallGraphError::ConfigError {
                    message: format!("Failed to serialize JSON config: {}", e),
                })?
            },
            Some("toml") => {
                toml::to_string_pretty(config).map_err(|e| CallGraphError::ConfigError {
                    message: format!("Failed to serialize TOML config: {}", e),
                })?
            },
            Some("yaml") | Some("yml") => {
                serde_yaml::to_string(config).map_err(|e| CallGraphError::ConfigError {
                    message: format!("Failed to serialize YAML config: {}", e),
                })?
            },
            _ => {
                return Err(CallGraphError::ConfigError {
                    message: format!("Unsupported config file format for {}", path.display()),
                });
            },
        };

        fs::write(path, content).map_err(|e| CallGraphError::FileError {
            path: path.to_string_lossy().to_string(),
            source: e,
        })?;

        Ok(())
    }

    /// Load configuration from environment variables
    fn load_env_config() -> CallGraphResult<KcsConfig> {
        let mut config = KcsConfig::default();

        // Parser configuration from environment
        if let Ok(value) = env::var("KCS_TREE_SITTER_ENABLED") {
            config.parser.tree_sitter_enabled =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid boolean value for KCS_TREE_SITTER_ENABLED".to_string(),
                })?;
        }

        if let Ok(value) = env::var("KCS_CLANG_ENABLED") {
            config.parser.clang_enabled =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid boolean value for KCS_CLANG_ENABLED".to_string(),
                })?;
        }

        if let Ok(value) = env::var("KCS_TARGET_ARCH") {
            config.parser.target_arch = value;
        }

        if let Ok(value) = env::var("KCS_KERNEL_VERSION") {
            config.parser.kernel_version = value;
        }

        if let Ok(value) = env::var("KCS_CONFIG_NAME") {
            config.parser.config_name = value;
        }

        // Performance configuration from environment
        if let Ok(value) = env::var("KCS_WORKER_THREADS") {
            config.performance.worker_threads =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid integer value for KCS_WORKER_THREADS".to_string(),
                })?;
        }

        if let Ok(value) = env::var("KCS_ENABLE_PARALLEL") {
            config.performance.enable_parallel =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid boolean value for KCS_ENABLE_PARALLEL".to_string(),
                })?;
        }

        if let Ok(value) = env::var("KCS_MAX_MEMORY_USAGE") {
            config.performance.max_memory_usage =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid integer value for KCS_MAX_MEMORY_USAGE".to_string(),
                })?;
        }

        if let Ok(value) = env::var("KCS_BATCH_SIZE") {
            config.performance.batch_size =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid integer value for KCS_BATCH_SIZE".to_string(),
                })?;
        }

        // Logging configuration from environment
        if let Ok(value) = env::var("KCS_LOG_LEVEL") {
            config.logging.level = value;
        }

        if let Ok(value) = env::var("KCS_LOG_STRUCTURED") {
            config.logging.structured = value.parse().map_err(|_| CallGraphError::ConfigError {
                message: "Invalid boolean value for KCS_LOG_STRUCTURED".to_string(),
            })?;
        }

        if let Ok(value) = env::var("KCS_LOG_FILE") {
            config.logging.file_path = Some(PathBuf::from(value));
        }

        if let Ok(value) = env::var("KCS_LOG_METRICS") {
            config.logging.enable_metrics =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid boolean value for KCS_LOG_METRICS".to_string(),
                })?;
        }

        if let Ok(value) = env::var("KCS_LOG_FORMAT") {
            config.logging.format = value;
        }

        // Call extraction configuration from environment
        if let Ok(value) = env::var("KCS_MAX_FILE_SIZE") {
            config.call_extraction.max_file_size =
                value.parse().map_err(|_| CallGraphError::ConfigError {
                    message: "Invalid integer value for KCS_MAX_FILE_SIZE".to_string(),
                })?;
        }

        if let Ok(value) = env::var("KCS_MIN_CONFIDENCE") {
            config.call_extraction.min_confidence = match value.to_lowercase().as_str() {
                "low" => ConfidenceLevel::Low,
                "medium" => ConfidenceLevel::Medium,
                "high" => ConfidenceLevel::High,
                _ => {
                    return Err(CallGraphError::ConfigError {
                        message: "Invalid confidence level, must be 'low', 'medium', or 'high'"
                            .to_string(),
                    })
                },
            };
        }

        Ok(config)
    }

    /// Merge two configurations with the second taking priority
    fn merge_configs(base: &mut KcsConfig, override_config: KcsConfig) {
        // This is a simple merge - in practice, you might want more sophisticated merging logic
        *base = override_config;
    }

    /// Merge a configuration into the current one
    fn merge_config(&mut self, config: KcsConfig) {
        Self::merge_configs(&mut self.config, config);
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration builder for fluent configuration creation
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    config: KcsConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: KcsConfig::default(),
        }
    }

    /// Set parser configuration
    pub fn parser(mut self, parser: ParserConfig) -> Self {
        self.config.parser = parser;
        self
    }

    /// Set call extraction configuration
    pub fn call_extraction(mut self, call_extraction: CallExtractionConfig) -> Self {
        self.config.call_extraction = call_extraction;
        self
    }

    /// Set error handling configuration
    pub fn error_handling(mut self, error_handling: ErrorHandlingConfig) -> Self {
        self.config.error_handling = error_handling;
        self
    }

    /// Set logging configuration
    pub fn logging(mut self, logging: LoggingConfig) -> Self {
        self.config.logging = logging;
        self
    }

    /// Set performance configuration
    pub fn performance(mut self, performance: PerformanceConfig) -> Self {
        self.config.performance = performance;
        self
    }

    /// Enable parallel processing
    pub fn enable_parallel(mut self, enable: bool) -> Self {
        self.config.performance.enable_parallel = enable;
        self.config.call_extraction.enable_parallel = enable;
        self
    }

    /// Set worker thread count
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.config.performance.worker_threads = threads;
        self
    }

    /// Set target architecture
    pub fn target_arch<S: Into<String>>(mut self, arch: S) -> Self {
        self.config.parser.target_arch = arch.into();
        self
    }

    /// Set kernel version
    pub fn kernel_version<S: Into<String>>(mut self, version: S) -> Self {
        self.config.parser.kernel_version = version.into();
        self
    }

    /// Set configuration name
    pub fn config_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.parser.config_name = name.into();
        self
    }

    /// Set log level
    pub fn log_level<S: Into<String>>(mut self, level: S) -> Self {
        self.config.logging.level = level.into();
        self
    }

    /// Set maximum file size
    pub fn max_file_size(mut self, size: u64) -> Self {
        self.config.call_extraction.max_file_size = size as usize;
        self
    }

    /// Set minimum confidence level
    pub fn min_confidence(mut self, confidence: ConfidenceLevel) -> Self {
        self.config.call_extraction.min_confidence = confidence;
        self
    }

    /// Build the configuration
    pub fn build(self) -> KcsConfig {
        self.config
    }

    /// Build the configuration and validate it
    pub fn build_and_validate(self) -> CallGraphResult<KcsConfig> {
        let config = self.config;
        let manager = ConfigManager::with_config(config.clone());
        manager.validate_config()?;
        Ok(config)
    }
}

/// Utility functions for common configuration scenarios
pub mod presets {
    use super::*;

    /// Configuration preset for development
    pub fn development() -> KcsConfig {
        ConfigBuilder::new()
            .log_level("debug")
            .enable_parallel(false) // For easier debugging
            .worker_threads(1)
            .build()
    }

    /// Configuration preset for production
    pub fn production() -> KcsConfig {
        ConfigBuilder::new()
            .log_level("info")
            .enable_parallel(true)
            .worker_threads(0) // Auto-detect
            .build()
    }

    /// Configuration preset for high-performance scenarios
    pub fn high_performance() -> KcsConfig {
        ConfigBuilder::new()
            .log_level("warn")
            .enable_parallel(true)
            .worker_threads(0) // Auto-detect
            .max_file_size(100 * 1024 * 1024) // 100MB
            .min_confidence(ConfidenceLevel::Medium)
            .build()
    }

    /// Configuration preset for testing
    pub fn testing() -> KcsConfig {
        ConfigBuilder::new()
            .log_level("trace")
            .enable_parallel(false)
            .worker_threads(1)
            .max_file_size(1024 * 1024) // 1MB
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_config_default() {
        let config = KcsConfig::default();
        assert!(config.parser.tree_sitter_enabled);
        assert!(!config.parser.clang_enabled);
        assert_eq!(config.parser.target_arch, "x86_64");
        assert_eq!(config.logging.level, "info");
        assert!(config.performance.enable_parallel);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .target_arch("arm64")
            .kernel_version("6.5")
            .log_level("debug")
            .enable_parallel(false)
            .build();

        assert_eq!(config.parser.target_arch, "arm64");
        assert_eq!(config.parser.kernel_version, "6.5");
        assert_eq!(config.logging.level, "debug");
        assert!(!config.performance.enable_parallel);
    }

    #[test]
    fn test_config_manager_new() {
        let manager = ConfigManager::new();
        assert_eq!(manager.config().parser.target_arch, "x86_64");
        assert_eq!(manager.sources.len(), 1);
    }

    #[test]
    fn test_config_validation() {
        let mut config = KcsConfig::default();
        config.parser.target_arch = String::new();

        let manager = ConfigManager::with_config(config);
        assert!(manager.validate_config().is_err());
    }

    #[test]
    fn test_config_file_json() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");

        let config = KcsConfig::default();
        let manager = ConfigManager::with_config(config.clone());

        manager.save_to_file(&config_path).unwrap();
        assert!(config_path.exists());

        let mut new_manager = ConfigManager::new();
        new_manager.load_from_file(&config_path).unwrap();
        assert_eq!(new_manager.config(), &config);
    }

    #[test]
    fn test_presets() {
        let dev_config = presets::development();
        assert_eq!(dev_config.logging.level, "debug");
        assert!(!dev_config.performance.enable_parallel);

        let prod_config = presets::production();
        assert_eq!(prod_config.logging.level, "info");
        assert!(prod_config.performance.enable_parallel);
    }

    #[test]
    fn test_env_config() {
        env::set_var("KCS_TARGET_ARCH", "arm64");
        env::set_var("KCS_LOG_LEVEL", "trace");
        env::set_var("KCS_ENABLE_PARALLEL", "false");

        let config = ConfigManager::load_env_config().unwrap();
        assert_eq!(config.parser.target_arch, "arm64");
        assert_eq!(config.logging.level, "trace");
        assert!(!config.performance.enable_parallel);

        // Clean up
        env::remove_var("KCS_TARGET_ARCH");
        env::remove_var("KCS_LOG_LEVEL");
        env::remove_var("KCS_ENABLE_PARALLEL");
    }
}
