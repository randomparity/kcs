"""
Command-line interface for KCS MCP server.

Provides the main entry point for the kcs-mcp command.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

import structlog
import uvicorn

from .config import (
    ConfigManager,
    Presets,
    create_default_config,
    load_config_from_file,
)

logger = structlog.get_logger(__name__)


def config_init_command(args: argparse.Namespace) -> None:
    """Initialize configuration file."""
    config_path = Path(args.output)

    if config_path.exists() and not args.force:
        print(f"Error: Configuration file already exists: {config_path}")
        print("Use --force to overwrite")
        sys.exit(1)

    # Select preset
    if args.preset == "development":
        config = Presets.development()
    elif args.preset == "production":
        config = Presets.production()
    elif args.preset == "high-performance":
        config = Presets.high_performance()
    elif args.preset == "testing":
        config = Presets.testing()
    else:
        config = create_default_config()

    # Save configuration
    manager = ConfigManager(config)
    try:
        manager.save_to_file(config_path)
        print(f"✓ Configuration file created: {config_path}")
        print(f"  Preset: {args.preset}")
        print(f"  Format: {config_path.suffix}")
    except Exception as e:
        print(f"Error: Failed to create configuration file: {e}")
        sys.exit(1)


def config_validate_command(args: argparse.Namespace) -> None:
    """Validate configuration file."""
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        config = load_config_from_file(config_path)
        manager = ConfigManager(config)
        manager.validate()
        print(f"✓ Configuration is valid: {config_path}")

        if args.verbose:
            print("\nConfiguration summary:")
            print(
                f"  Parser: {'Clang' if config.parser.clang_enabled else 'Tree-sitter only'}"
            )
            print(f"  Target: {config.parser.target_arch}")
            print(f"  Parallel: {config.performance.enable_parallel}")
            print(f"  Workers: {config.performance.worker_threads or 'auto'}")
            print(f"  Log level: {config.logging.level}")

    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)


def config_show_command(args: argparse.Namespace) -> None:
    """Show configuration."""
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        try:
            config = load_config_from_file(config_path)
            print(f"Configuration from: {config_path}")
        except Exception as e:
            print(f"Error: Failed to load configuration: {e}")
            sys.exit(1)
    else:
        # Show configuration from environment and defaults
        manager = ConfigManager()
        manager.load_from_env()
        config = manager.config
        print("Configuration (defaults + environment):")

    # Format output
    if args.format == "json":
        import json
        from dataclasses import asdict

        print(json.dumps(asdict(config), indent=2))
    else:
        # Pretty print
        print("\nParser Configuration:")
        print(f"  Tree-sitter: {config.parser.tree_sitter_enabled}")
        print(f"  Clang: {config.parser.clang_enabled}")
        print(f"  Architecture: {config.parser.target_arch}")
        print(f"  Kernel version: {config.parser.kernel_version}")
        print(f"  Config name: {config.parser.config_name}")

        print("\nCall Extraction:")
        print(f"  Max file size: {config.call_extraction.max_file_size:,} bytes")
        print(f"  Parallel: {config.call_extraction.enable_parallel}")
        print(f"  Min confidence: {config.call_extraction.min_confidence}")
        print(f"  Include macros: {config.call_extraction.include_macros}")
        print(f"  Include callbacks: {config.call_extraction.include_callbacks}")

        print("\nPerformance:")
        print(f"  Worker threads: {config.performance.worker_threads or 'auto'}")
        print(f"  Parallel: {config.performance.enable_parallel}")
        print(f"  Batch size: {config.performance.batch_size}")
        print(f"  Cache size: {config.performance.cache_size}")

        print("\nLogging:")
        print(f"  Level: {config.logging.level}")
        print(f"  Format: {config.logging.format}")
        print(f"  Structured: {config.logging.structured}")
        print(f"  File: {config.logging.file_path or 'stdout/stderr'}")


def add_config_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Add configuration management subcommands."""
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management commands",
        description="Manage KCS configuration files and settings",
    )

    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        help="Configuration commands",
        required=True,
    )

    # config init
    init_parser = config_subparsers.add_parser(
        "init",
        help="Initialize configuration file",
        description="Create a new configuration file with specified preset",
    )
    init_parser.add_argument(
        "output",
        help="Output configuration file path",
    )
    init_parser.add_argument(
        "--preset",
        choices=["default", "development", "production", "high-performance", "testing"],
        default="default",
        help="Configuration preset to use",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration file",
    )
    init_parser.set_defaults(func=config_init_command)

    # config validate
    validate_parser = config_subparsers.add_parser(
        "validate",
        help="Validate configuration file",
        description="Check configuration file for errors and consistency",
    )
    validate_parser.add_argument(
        "config",
        help="Configuration file to validate",
    )
    validate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show configuration summary",
    )
    validate_parser.set_defaults(func=config_validate_command)

    # config show
    show_parser = config_subparsers.add_parser(
        "show",
        help="Show current configuration",
        description="Display current configuration from file or environment",
    )
    show_parser.add_argument(
        "--config",
        "-c",
        help="Configuration file to show (default: environment + defaults)",
    )
    show_parser.add_argument(
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format",
    )
    show_parser.set_defaults(func=config_show_command)


def serve_command(args: argparse.Namespace) -> NoReturn:
    """Run the KCS MCP server."""
    # Configure logging
    log_level = args.log_level.upper()

    # Validate required environment variables
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set, server may fail to start")

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warning("JWT_SECRET not set, using insecure default")

    # Log startup information
    logger.info(
        "Starting KCS MCP server",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=log_level,
        reload=args.reload,
    )

    # Start the server
    try:
        uvicorn.run(
            "kcs_mcp.app:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True,
            server_header=False,
            date_header=False,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error("Server failed to start", error=str(e), exc_info=True)
        sys.exit(1)


def main() -> NoReturn:
    """Main entry point for the KCS MCP server CLI."""
    parser = argparse.ArgumentParser(
        description="Kernel Context Server MCP API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=False,  # Default to serve command if no subcommand given
    )

    # Add configuration management subcommands
    add_config_subcommands(subparsers)

    # Add serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the KCS MCP server",
        description="Run the KCS MCP server with specified configuration",
    )

    serve_parser.add_argument(
        "--host",
        default=os.getenv("KCS_HOST", "127.0.0.1"),
        help="Host to bind to",
    )

    serve_parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("KCS_PORT", "8080")),
        help="Port to bind to",
    )

    serve_parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("KCS_WORKERS", "1")),
        help="Number of worker processes",
    )

    serve_parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("KCS_RELOAD", "false").lower() == "true",
        help="Enable auto-reload for development",
    )

    serve_parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info").lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level",
    )

    serve_parser.add_argument(
        "--config",
        "-c",
        help="Configuration file to use",
    )

    serve_parser.set_defaults(func=serve_command)

    args = parser.parse_args()

    # If no command is given, default to serve
    if not hasattr(args, "func"):
        # Re-parse with serve as default
        sys.argv.insert(1, "serve")
        args = parser.parse_args()

    # Handle config subcommands
    if args.command == "config":
        args.func(args)
        sys.exit(0)

    # Handle serve command (or default behavior)
    args.func(args)
    # Note: args.func(args) should call sys.exit(), but mypy can't infer this
    sys.exit(0)  # pragma: no cover


if __name__ == "__main__":
    main()
