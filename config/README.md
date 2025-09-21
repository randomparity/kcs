# KCS Configuration Management

This directory contains configuration management for the Kernel Context Server (KCS), providing
unified configuration for call graph extraction settings across both Rust and Python components.

## Overview

The configuration system supports:

- **Multiple file formats**: JSON, YAML, TOML
- **Environment variable overrides**: Prefix-based environment variable support
- **Configuration validation**: Built-in validation with clear error messages
- **Preset configurations**: Pre-defined configurations for common scenarios
- **CLI integration**: Command-line tools for configuration management
- **Type safety**: Strongly typed configuration with serde support

## Configuration Structure

The main configuration is organized into the following sections:

### Parser Configuration

- Tree-sitter and Clang integration settings
- Target architecture and kernel version
- Include paths and preprocessor defines
- Compilation database settings

### Call Extraction Configuration

- File size limits and parallel processing settings
- Confidence level thresholds
- Feature toggles for different call types (macros, callbacks, etc.)
- Sub-component configurations for specialized extractors

### Error Handling Configuration

- Retry policies and recovery strategies
- Detailed logging and metrics collection
- Fail-fast vs. continue-on-error behavior

### Logging Configuration

- Log levels and output formats
- Structured logging settings
- File output configuration

### Performance Configuration

- Worker thread configuration
- Memory usage limits
- Batch processing settings
- Cache configuration

### Database Configuration

- Connection settings and pool configuration
- Timeout and retry settings

## File Formats

### JSON Configuration

```json
{
  "parser": {
    "tree_sitter_enabled": true,
    "clang_enabled": false,
    "target_arch": "x86_64",
    "kernel_version": "6.1"
  },
  "call_extraction": {
    "max_file_size": 10485760,
    "enable_parallel": true,
    "min_confidence": "low"
  }
}
```

### YAML Configuration

```yaml
parser:
  tree_sitter_enabled: true
  clang_enabled: false
  target_arch: x86_64
  kernel_version: "6.1"

call_extraction:
  max_file_size: 10485760
  enable_parallel: true
  min_confidence: low
```

### TOML Configuration

```toml
[parser]
tree_sitter_enabled = true
clang_enabled = false
target_arch = "x86_64"
kernel_version = "6.1"

[call_extraction]
max_file_size = 10485760
enable_parallel = true
min_confidence = "low"
```

## Environment Variables

Configuration can be overridden using environment variables with the `KCS_` prefix:

| Environment Variable | Configuration Path | Type | Example |
|---------------------|-------------------|------|---------|
| `KCS_TARGET_ARCH` | parser.target_arch | string | `x86_64` |
| `KCS_KERNEL_VERSION` | parser.kernel_version | string | `6.1` |
| `KCS_CLANG_ENABLED` | parser.clang_enabled | bool | `true` |
| `KCS_ENABLE_PARALLEL` | call_extraction.enable_parallel | bool | `true` |
| `KCS_MAX_FILE_SIZE` | call_extraction.max_file_size | int | `10485760` |
| `KCS_MIN_CONFIDENCE` | call_extraction.min_confidence | string | `medium` |
| `KCS_WORKER_THREADS` | performance.worker_threads | int | `8` |
| `KCS_LOG_LEVEL` | logging.level | string | `debug` |
| `KCS_DATABASE_URL` | database.url | string | `postgresql://...` |

## Configuration Presets

Pre-defined configurations for common scenarios:

### Development

- Debug logging enabled
- Single-threaded for easier debugging
- Smaller batch sizes
- Enhanced error reporting

### Production

- Info-level logging
- Full parallel processing
- Optimized batch sizes
- Clang integration enabled

### High Performance

- Warning-level logging only
- Maximum parallelization
- Large file size limits
- Medium confidence threshold
- Optimized for speed over safety

### Testing

- Trace-level logging
- Single-threaded for reproducibility
- Small file size limits
- Detailed error reporting

## CLI Usage

The KCS MCP server includes configuration management commands:

### Initialize Configuration

```bash
# Create a new configuration file with default preset
kcs-mcp config init config.yaml

# Create with specific preset
kcs-mcp config init config.yaml --preset production

# Overwrite existing file
kcs-mcp config init config.yaml --preset development --force
```

### Validate Configuration

```bash
# Validate configuration file
kcs-mcp config validate config.yaml

# Verbose validation with summary
kcs-mcp config validate config.yaml --verbose
```

### Show Configuration

```bash
# Show current configuration (defaults + environment)
kcs-mcp config show

# Show configuration from file
kcs-mcp config show --config config.yaml

# Show as JSON
kcs-mcp config show --config config.yaml --format json
```

### Run Server with Configuration

```bash
# Use configuration file
kcs-mcp serve --config config.yaml

# Use environment variables only
KCS_LOG_LEVEL=debug KCS_ENABLE_PARALLEL=false kcs-mcp serve
```

## Programming Interface

### Rust Usage

```rust
use kcs_parser::config::{ConfigManager, ConfigBuilder, presets};

// Load from file
let mut manager = ConfigManager::new();
manager.load_from_file("config.yaml")?;
manager.load_from_env()?; // Apply environment overrides
let config = manager.config();

// Build programmatically
let config = ConfigBuilder::new()
    .target_arch("arm64")
    .kernel_version("6.5")
    .enable_parallel(true)
    .log_level("debug")
    .build_and_validate()?;

// Use presets
let config = presets::production();
```

### Python Usage

```python
from kcs_mcp.config import ConfigManager, ConfigBuilder, Presets

# Load from file
manager = ConfigManager()
manager.load_from_file("config.yaml")
manager.load_from_env()  # Apply environment overrides
config = manager.config

# Build programmatically
config = ConfigBuilder() \
    .target_arch("arm64") \
    .kernel_version("6.5") \
    .enable_parallel(True) \
    .log_level("debug") \
    .build_and_validate()

# Use presets
config = Presets.production()
```

## Configuration Examples

See the `examples/` directory for complete configuration examples:

- `development.json` - Development configuration with debugging enabled
- `production.yaml` - Production configuration optimized for performance
- `high-performance.toml` - High-performance configuration for large-scale analysis

## Validation Rules

The configuration system enforces the following validation rules:

1. **Required fields**: `target_arch`, `kernel_version` cannot be empty
2. **Numeric constraints**: `max_file_size`, `batch_size` must be > 0
3. **Enum values**: `log_level` must be one of `trace`, `debug`, `info`, `warn`, `error`
4. **Format validation**: `log_format` must be one of `json`, `compact`, `pretty`
5. **Confidence levels**: `min_confidence` must be one of `low`, `medium`, `high`

## Best Practices

1. **Use configuration files** for persistent settings rather than environment variables
2. **Apply environment variables** for deployment-specific overrides
3. **Validate configurations** before deployment using `kcs-mcp config validate`
4. **Use presets** as starting points and customize as needed
5. **Version control** your configuration files
6. **Document** any custom configuration changes for your deployment

## Troubleshooting

### Common Issues

1. **Configuration file not found**: Ensure the file path is correct and the file exists
2. **Parse errors**: Check YAML/JSON/TOML syntax using a validator
3. **Validation failures**: Review the error message and fix the invalid values
4. **Environment override not working**: Check the environment variable name and prefix
5. **Performance issues**: Adjust `worker_threads`, `batch_size`, and `max_file_size` settings

### Debug Configuration Loading

```bash
# Show final configuration to debug what values are being used
kcs-mcp config show --config your-config.yaml

# Validate configuration to see any errors
kcs-mcp config validate your-config.yaml --verbose
```
