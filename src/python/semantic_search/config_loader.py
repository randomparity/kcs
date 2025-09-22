"""
Configuration loader for semantic search engine.

Provides utilities to load configuration from multiple sources including
YAML files, JSON files, and environment variables with proper precedence.
"""

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .config import SemanticSearchConfig
from .logging_integration import get_semantic_search_logger

logger = get_semantic_search_logger(__name__)


class ConfigurationError(Exception):
    """Exception raised when configuration loading fails."""

    def __init__(self, message: str, source: str | None = None) -> None:
        """
        Initialize configuration error.

        Args:
            message: Error message
            source: Configuration source that failed (optional)
        """
        super().__init__(message)
        self.source = source


class ConfigLoader:
    """
    Loads semantic search configuration from multiple sources.

    Supports loading from YAML files, JSON files, and environment variables
    with proper precedence handling and validation.
    """

    def __init__(self, config_search_paths: list[str] | None = None) -> None:
        """
        Initialize configuration loader.

        Args:
            config_search_paths: List of paths to search for config files
        """
        self.config_search_paths = (
            config_search_paths or self._get_default_search_paths()
        )

    def _get_default_search_paths(self) -> list[str]:
        """Get default configuration file search paths."""
        paths = [
            # Current directory
            ".",
            # Config directory in project root
            "config",
            # User config directory
            str(Path.home() / ".config" / "kcs"),
            # System config directory
            "/etc/kcs",
        ]

        # Add project root if we can determine it
        try:
            # Assume we're in src/python/semantic_search/
            project_root = Path(__file__).parent.parent.parent.parent
            if project_root.exists():
                paths.insert(1, str(project_root / "config"))
        except Exception:
            pass

        return paths

    def find_config_file(self, filename: str = "semantic_search.yaml") -> Path | None:
        """
        Find configuration file in search paths.

        Args:
            filename: Configuration filename to search for

        Returns:
            Path to configuration file if found, None otherwise
        """
        # Also try common variations
        filenames = [filename]
        if filename.endswith(".yaml"):
            filenames.append(filename.replace(".yaml", ".yml"))
        elif filename.endswith(".yml"):
            filenames.append(filename.replace(".yml", ".yaml"))

        # Add JSON variant
        base_name = filename.split(".")[0]
        filenames.append(f"{base_name}.json")

        for search_path in self.config_search_paths:
            search_dir = Path(search_path)
            if not search_dir.exists():
                continue

            for fname in filenames:
                config_file = search_dir / fname
                if config_file.exists() and config_file.is_file():
                    logger.info(f"Found configuration file: {config_file}")
                    return config_file

        logger.info(f"No configuration file found for {filename}")
        return None

    def load_yaml_config(self, file_path: Path) -> dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            if not isinstance(config_dict, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a dictionary, got {type(config_dict)}",
                    str(file_path),
                )

            logger.info(f"Loaded YAML configuration from {file_path}")
            return config_dict

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration: {e}", str(file_path)
            ) from e

        except OSError as e:
            raise ConfigurationError(
                f"Failed to read configuration file: {e}", str(file_path)
            ) from e

    def load_json_config(self, file_path: Path) -> dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                config_dict = json.load(f)

            if not isinstance(config_dict, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a dictionary, got {type(config_dict)}",
                    str(file_path),
                )

            logger.info(f"Loaded JSON configuration from {file_path}")
            return config_dict

        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Failed to parse JSON configuration: {e}", str(file_path)
            ) from e

        except OSError as e:
            raise ConfigurationError(
                f"Failed to read configuration file: {e}", str(file_path)
            ) from e

    def load_from_file(self, file_path: Path | str) -> dict[str, Any]:
        """
        Load configuration from file (auto-detect format).

        Args:
            file_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If file cannot be loaded
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        if file_path.suffix.lower() in [".yaml", ".yml"]:
            return self.load_yaml_config(file_path)
        elif file_path.suffix.lower() == ".json":
            return self.load_json_config(file_path)
        else:
            # Try to auto-detect
            try:
                return self.load_yaml_config(file_path)
            except ConfigurationError:
                try:
                    return self.load_json_config(file_path)
                except ConfigurationError:
                    raise ConfigurationError(
                        f"Could not parse configuration file as YAML or JSON: {file_path}"
                    ) from None

    def merge_config_dicts(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merge configuration dictionaries with override precedence.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        merged = base.copy()

        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = self.merge_config_dicts(merged[key], value)
            else:
                # Override value
                merged[key] = value

        return merged

    def load_config(
        self, config_file: str | Path | None = None, include_env: bool = True
    ) -> SemanticSearchConfig:
        """
        Load semantic search configuration from all sources.

        Args:
            config_file: Specific configuration file path (optional)
            include_env: Whether to include environment variables

        Returns:
            Loaded and validated configuration

        Raises:
            ConfigurationError: If configuration cannot be loaded or is invalid
        """
        config_dict: dict[str, Any] = {}

        # 1. Load from configuration file
        if config_file:
            # Use specific file
            file_path = Path(config_file)
            if not file_path.exists():
                raise ConfigurationError(
                    f"Specified configuration file not found: {config_file}"
                )
            file_config = self.load_from_file(file_path)
        else:
            # Search for configuration file
            found_file = self.find_config_file()
            if found_file:
                file_config = self.load_from_file(found_file)
            else:
                file_config = {}

        config_dict = file_config

        # 2. Override with environment variables if requested
        if include_env:
            try:
                env_config = SemanticSearchConfig.from_env()
                env_dict = env_config.to_dict()

                # Merge environment config with file config
                config_dict = self.merge_config_dicts(config_dict, env_dict)

                logger.info("Merged environment variables with file configuration")

            except Exception as e:
                logger.warning(f"Failed to load environment configuration: {e}")

        # 3. Create and validate final configuration
        try:
            if config_dict:
                config = SemanticSearchConfig.from_dict(config_dict)
            else:
                # Fall back to environment-only configuration
                config = SemanticSearchConfig.from_env()

            # Validate configuration
            errors = config.validate()
            if errors:
                raise ConfigurationError(
                    f"Configuration validation failed: {'; '.join(errors)}"
                )

            logger.info(
                "Successfully loaded and validated semantic search configuration"
            )
            return config

        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration: {e}") from e

    def save_config(
        self,
        config: SemanticSearchConfig,
        output_path: Path | str,
        format_type: str = "yaml",
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            output_path: Output file path
            format_type: Output format ("yaml" or "json")

        Raises:
            ConfigurationError: If saving fails
        """
        output_path = Path(output_path)

        try:
            config_dict = config.to_dict()

            if format_type.lower() == "yaml":
                with open(output_path, "w", encoding="utf-8") as f:
                    yaml.dump(
                        config_dict,
                        f,
                        default_flow_style=False,
                        indent=2,
                        sort_keys=False,
                    )
            elif format_type.lower() == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, sort_keys=True)
            else:
                raise ConfigurationError(f"Unsupported format: {format_type}")

            logger.info(f"Saved configuration to {output_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}") from e


# Convenience functions


def load_semantic_search_config(
    config_file: str | Path | None = None,
) -> SemanticSearchConfig:
    """
    Load semantic search configuration with default settings.

    Args:
        config_file: Optional path to configuration file

    Returns:
        Loaded configuration

    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    loader = ConfigLoader()
    return loader.load_config(config_file)


def find_config_file(filename: str = "semantic_search.yaml") -> Path | None:
    """
    Find configuration file in standard locations.

    Args:
        filename: Configuration filename to search for

    Returns:
        Path to configuration file if found
    """
    loader = ConfigLoader()
    return loader.find_config_file(filename)


def validate_model_path(model_path: str) -> bool:
    """
    Validate that a model path is accessible.

    Args:
        model_path: Path to model directory or Hugging Face model name

    Returns:
        True if path is valid and accessible
    """
    try:
        # Check if it's a local path
        if os.path.exists(model_path):
            path = Path(model_path)
            # Check if it looks like a model directory
            if path.is_dir():
                # Look for common model files
                model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
                if any((path / f).exists() for f in model_files):
                    return True

        # Assume it's a Hugging Face model name if not a local path
        # We can't easily validate remote models without downloading,
        # so we'll do basic format validation
        if "/" in model_path and len(model_path.split("/")) == 2:
            # Looks like org/model format
            return True

        return False

    except Exception:
        return False


# CLI integration
def main() -> None:
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Configuration Loader")
    parser.add_argument(
        "command",
        choices=["load", "validate", "show", "create-example"],
        help="Configuration command to execute",
    )
    parser.add_argument("--config-file", help="Path to configuration file")
    parser.add_argument("--output", help="Output file path for create-example command")
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format for create-example command",
    )

    args = parser.parse_args()

    try:
        if args.command == "load":
            config = load_semantic_search_config(args.config_file)
            print("Configuration loaded successfully")
            print(f"Model: {config.embedding.model_name}")
            print(
                f"Database: {config.database.host}:{config.database.port}/{config.database.database}"
            )

        elif args.command == "validate":
            config = load_semantic_search_config(args.config_file)
            errors = config.validate()
            if errors:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("Configuration is valid")

        elif args.command == "show":
            config = load_semantic_search_config(args.config_file)
            config_dict = config.to_dict()
            print(yaml.dump(config_dict, default_flow_style=False, indent=2))

        elif args.command == "create-example":
            output_path = args.output or f"semantic_search_example.{args.format}"
            config = SemanticSearchConfig()
            loader = ConfigLoader()
            loader.save_config(config, output_path, args.format)
            print(f"Created example configuration: {output_path}")

    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        if e.source:
            print(f"Source: {e.source}")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
