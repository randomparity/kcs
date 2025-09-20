"""Configuration management for KCS MCP tools.

Provides centralized configuration management with support for multiple
sources (files, environment variables, programmatic settings) and formats
(JSON, YAML, TOML).
"""

import json
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class ParserConfig:
    """Configuration for the Rust parser component."""

    tree_sitter_enabled: bool = True
    clang_enabled: bool = False
    target_arch: str = "x86_64"
    kernel_version: str = "6.1"
    include_paths: list[str] = field(default_factory=list)
    defines: dict[str, str] = field(default_factory=dict)
    compile_commands_path: str | None = None
    config_name: str = "defconfig"


@dataclass
class CallExtractionConfig:
    """Configuration for call graph extraction."""

    max_file_size: int = 10 * 1024 * 1024  # 10MB
    enable_parallel: bool = True
    min_confidence: str = "low"  # low, medium, high
    include_indirect: bool = True
    include_macros: bool = True
    include_callbacks: bool = True
    include_conditional: bool = True

    # Sub-component configurations
    direct_calls: dict[str, Any] = field(
        default_factory=lambda: {
            "include_conditional": True,
            "include_control_flow": True,
            "validate_identifiers": False,
            "max_depth": 50,
        }
    )

    pointer_calls: dict[str, Any] = field(
        default_factory=lambda: {
            "include_callback_arrays": True,
            "include_member_pointers": True,
            "validate_identifiers": False,
            "max_depth": 30,
        }
    )

    macro_calls: dict[str, Any] = field(
        default_factory=lambda: {
            "include_function_macros": True,
            "include_complex_expansions": True,
            "validate_identifiers": False,
            "max_expansion_depth": 10,
        }
    )

    callbacks: dict[str, Any] = field(
        default_factory=lambda: {
            "include_member_calls": True,
            "include_pointer_assignments": True,
            "include_array_callbacks": True,
            "include_callback_registration": True,
            "max_depth": 20,
        }
    )

    conditional: dict[str, Any] = field(
        default_factory=lambda: {
            "track_config_dependencies": True,
            "include_ifdef_blocks": True,
            "include_switch_cases": True,
            "max_nesting_depth": 15,
        }
    )


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling and recovery."""

    enable_recovery: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 0.1
    enable_detailed_logging: bool = True
    collect_metrics: bool = True
    fail_fast: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "info"
    structured: bool = True
    file_path: str | None = None
    enable_metrics: bool = True
    format: str = "pretty"  # json, compact, pretty


@dataclass
class PerformanceConfig:
    """Configuration for performance tuning."""

    worker_threads: int = 0  # 0 = auto-detect
    enable_parallel: bool = True
    max_memory_usage: int = 0  # 0 = unlimited
    batch_size: int = 1000
    cache_size: int = 10000


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""

    url: str | None = None
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    timeout: float = 30.0


@dataclass
class KcsConfig:
    """Main configuration container for KCS MCP tools."""

    parser: ParserConfig = field(default_factory=ParserConfig)
    call_extraction: CallExtractionConfig = field(default_factory=CallExtractionConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


class ConfigManager:
    """Manages loading, merging, and validation of configuration."""

    def __init__(self, config: KcsConfig | None = None):
        """Initialize configuration manager.

        Args:
            config: Initial configuration. Uses default if None.
        """
        self._config = config or KcsConfig()
        self._sources: list[str] = []

    @property
    def config(self) -> KcsConfig:
        """Get current configuration."""
        return self._config

    def load_from_file(self, path: str | Path) -> None:
        """Load configuration from file.

        Args:
            path: Path to configuration file (JSON, YAML, or TOML)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info("Loading configuration from file", path=str(path))

        try:
            with open(path, encoding="utf-8") as f:
                if path.suffix.lower() == ".json":
                    data = json.load(f)
                elif path.suffix.lower() in (".yaml", ".yml"):
                    data = yaml.safe_load(f)
                elif path.suffix.lower() == ".toml":
                    content = f.read()
                    data = tomllib.loads(content)
                else:
                    raise ValueError(
                        f"Unsupported configuration file format: {path.suffix}"
                    )

            self._merge_config_data(data)
            self._sources.append(f"file:{path}")

        except Exception as e:
            logger.error(
                "Failed to load configuration file", path=str(path), error=str(e)
            )
            raise ValueError(f"Failed to load configuration from {path}: {e}") from e

    def load_from_env(self, prefix: str = "KCS_") -> None:
        """Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix
        """
        logger.info("Loading configuration from environment", prefix=prefix)

        env_config: dict[str, Any] = {}

        # Parser configuration
        if value := os.getenv(f"{prefix}TREE_SITTER_ENABLED"):
            env_config.setdefault("parser", {})["tree_sitter_enabled"] = (
                self._parse_bool(value)
            )
        if value := os.getenv(f"{prefix}CLANG_ENABLED"):
            env_config.setdefault("parser", {})["clang_enabled"] = self._parse_bool(
                value
            )
        if value := os.getenv(f"{prefix}TARGET_ARCH"):
            env_config.setdefault("parser", {})["target_arch"] = value
        if value := os.getenv(f"{prefix}KERNEL_VERSION"):
            env_config.setdefault("parser", {})["kernel_version"] = value
        if value := os.getenv(f"{prefix}CONFIG_NAME"):
            env_config.setdefault("parser", {})["config_name"] = value

        # Call extraction configuration
        if value := os.getenv(f"{prefix}MAX_FILE_SIZE"):
            env_config.setdefault("call_extraction", {})["max_file_size"] = int(value)
        if value := os.getenv(f"{prefix}ENABLE_PARALLEL"):
            env_config.setdefault("call_extraction", {})["enable_parallel"] = (
                self._parse_bool(value)
            )
        if value := os.getenv(f"{prefix}MIN_CONFIDENCE"):
            env_config.setdefault("call_extraction", {})["min_confidence"] = (
                value.lower()
            )

        # Performance configuration
        if value := os.getenv(f"{prefix}WORKER_THREADS"):
            env_config.setdefault("performance", {})["worker_threads"] = int(value)
        if value := os.getenv(f"{prefix}BATCH_SIZE"):
            env_config.setdefault("performance", {})["batch_size"] = int(value)

        # Logging configuration
        if value := os.getenv(f"{prefix}LOG_LEVEL"):
            env_config.setdefault("logging", {})["level"] = value.lower()
        if value := os.getenv(f"{prefix}LOG_STRUCTURED"):
            env_config.setdefault("logging", {})["structured"] = self._parse_bool(value)
        if value := os.getenv(f"{prefix}LOG_FILE"):
            env_config.setdefault("logging", {})["file_path"] = value

        # Database configuration
        if value := os.getenv(f"{prefix}DATABASE_URL"):
            env_config.setdefault("database", {})["url"] = value
        if value := os.getenv(f"{prefix}DATABASE_POOL_SIZE"):
            env_config.setdefault("database", {})["pool_size"] = int(value)

        if env_config:
            self._merge_config_data(env_config)
            self._sources.append("environment")

    def load_from_dict(self, data: dict[str, Any], source_name: str = "dict") -> None:
        """Load configuration from dictionary.

        Args:
            data: Configuration data
            source_name: Name for tracking source
        """
        logger.info("Loading configuration from dictionary", source=source_name)
        self._merge_config_data(data)
        self._sources.append(source_name)

    def save_to_file(self, path: str | Path) -> None:
        """Save current configuration to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        logger.info("Saving configuration to file", path=str(path))

        data = asdict(self._config)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.suffix.lower() == ".toml":
                import tomli_w

                with open(path, "wb") as f:
                    tomli_w.dump(data, f)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    if path.suffix.lower() == ".json":
                        json.dump(data, f, indent=2)
                    elif path.suffix.lower() in (".yaml", ".yml"):
                        yaml.dump(data, f, default_flow_style=False, indent=2)
                    else:
                        raise ValueError(f"Unsupported output format: {path.suffix}")

        except Exception as e:
            logger.error("Failed to save configuration", path=str(path), error=str(e))
            raise ValueError(f"Failed to save configuration to {path}: {e}") from e

    def validate(self) -> None:
        """Validate current configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        config = self._config

        # Validate parser config
        if not config.parser.target_arch:
            raise ValueError("target_arch cannot be empty")
        if not config.parser.kernel_version:
            raise ValueError("kernel_version cannot be empty")

        # Validate call extraction config
        if config.call_extraction.max_file_size <= 0:
            raise ValueError("max_file_size must be greater than 0")

        valid_confidence_levels = {"low", "medium", "high"}
        if config.call_extraction.min_confidence not in valid_confidence_levels:
            raise ValueError(
                f"min_confidence must be one of: {valid_confidence_levels}"
            )

        # Validate performance config
        if config.performance.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        # Validate logging config
        valid_log_levels = {"trace", "debug", "info", "warn", "error"}
        if config.logging.level not in valid_log_levels:
            raise ValueError(f"log level must be one of: {valid_log_levels}")

        valid_log_formats = {"json", "compact", "pretty"}
        if config.logging.format not in valid_log_formats:
            raise ValueError(f"log format must be one of: {valid_log_formats}")

    def get_rust_config(self) -> dict[str, Any]:
        """Get configuration formatted for Rust bridge.

        Returns:
            Configuration dictionary for Rust bridge
        """
        return {
            "tree_sitter_enabled": self._config.parser.tree_sitter_enabled,
            "clang_enabled": self._config.parser.clang_enabled,
            "target_arch": self._config.parser.target_arch,
            "kernel_version": self._config.parser.kernel_version,
        }

    def _merge_config_data(self, data: dict[str, Any]) -> None:
        """Merge configuration data into current config."""
        for section_name, section_data in data.items():
            if hasattr(self._config, section_name):
                section = getattr(self._config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        logger.warning(
                            "Unknown configuration key", section=section_name, key=key
                        )
            else:
                logger.warning("Unknown configuration section", section=section_name)

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse boolean value from string."""
        return value.lower() in ("true", "1", "yes", "on")


class ConfigBuilder:
    """Fluent builder for creating configurations."""

    def __init__(self) -> None:
        """Initialize builder with default configuration."""
        self._config = KcsConfig()

    def target_arch(self, arch: str) -> "ConfigBuilder":
        """Set target architecture."""
        self._config.parser.target_arch = arch
        return self

    def kernel_version(self, version: str) -> "ConfigBuilder":
        """Set kernel version."""
        self._config.parser.kernel_version = version
        return self

    def config_name(self, name: str) -> "ConfigBuilder":
        """Set configuration name."""
        self._config.parser.config_name = name
        return self

    def enable_clang(self, enable: bool = True) -> "ConfigBuilder":
        """Enable or disable clang integration."""
        self._config.parser.clang_enabled = enable
        return self

    def enable_parallel(self, enable: bool = True) -> "ConfigBuilder":
        """Enable or disable parallel processing."""
        self._config.call_extraction.enable_parallel = enable
        self._config.performance.enable_parallel = enable
        return self

    def worker_threads(self, threads: int) -> "ConfigBuilder":
        """Set number of worker threads."""
        self._config.performance.worker_threads = threads
        return self

    def max_file_size(self, size: int) -> "ConfigBuilder":
        """Set maximum file size for processing."""
        self._config.call_extraction.max_file_size = size
        return self

    def min_confidence(self, level: str) -> "ConfigBuilder":
        """Set minimum confidence level."""
        self._config.call_extraction.min_confidence = level
        return self

    def log_level(self, level: str) -> "ConfigBuilder":
        """Set log level."""
        self._config.logging.level = level
        return self

    def database_url(self, url: str) -> "ConfigBuilder":
        """Set database URL."""
        self._config.database.url = url
        return self

    def build(self) -> KcsConfig:
        """Build the configuration."""
        return self._config

    def build_and_validate(self) -> KcsConfig:
        """Build and validate the configuration."""
        manager = ConfigManager(self._config)
        manager.validate()
        return self._config


# Configuration presets for common scenarios
class Presets:
    """Pre-defined configuration presets."""

    @staticmethod
    def development() -> KcsConfig:
        """Configuration preset for development."""
        return (
            ConfigBuilder()
            .log_level("debug")
            .enable_parallel(False)
            .worker_threads(1)
            .build()
        )

    @staticmethod
    def production() -> KcsConfig:
        """Configuration preset for production."""
        return (
            ConfigBuilder()
            .log_level("info")
            .enable_parallel(True)
            .worker_threads(0)
            .build()
        )

    @staticmethod
    def high_performance() -> KcsConfig:
        """Configuration preset for high-performance scenarios."""
        return (
            ConfigBuilder()
            .log_level("warn")
            .enable_parallel(True)
            .worker_threads(0)
            .max_file_size(100 * 1024 * 1024)
            .min_confidence("medium")
            .build()
        )

    @staticmethod
    def testing() -> KcsConfig:
        """Configuration preset for testing."""
        return (
            ConfigBuilder()
            .log_level("trace")
            .enable_parallel(False)
            .worker_threads(1)
            .max_file_size(1024 * 1024)
            .build()
        )


def load_config_from_file(path: str | Path) -> KcsConfig:
    """Load configuration from file (convenience function).

    Args:
        path: Path to configuration file

    Returns:
        Loaded configuration
    """
    manager = ConfigManager()
    manager.load_from_file(path)
    manager.validate()
    return manager.config


def load_config_from_env(prefix: str = "KCS_") -> KcsConfig:
    """Load configuration from environment (convenience function).

    Args:
        prefix: Environment variable prefix

    Returns:
        Configuration with environment overrides
    """
    manager = ConfigManager()
    manager.load_from_env(prefix)
    manager.validate()
    return manager.config


def create_default_config() -> KcsConfig:
    """Create default configuration (convenience function).

    Returns:
        Default configuration
    """
    return KcsConfig()
