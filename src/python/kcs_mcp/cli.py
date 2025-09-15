"""
Command-line interface for KCS MCP server.

Provides the main entry point for the kcs-mcp command.
"""

import argparse
import os
import sys
from typing import NoReturn

import structlog
import uvicorn

logger = structlog.get_logger(__name__)


def main() -> NoReturn:
    """Main entry point for the KCS MCP server CLI."""
    parser = argparse.ArgumentParser(
        description="Kernel Context Server MCP API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        default=os.getenv("KCS_HOST", "127.0.0.1"),
        help="Host to bind to",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("KCS_PORT", "8080")),
        help="Port to bind to",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("KCS_WORKERS", "1")),
        help="Number of worker processes",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("KCS_RELOAD", "false").lower() == "true",
        help="Enable auto-reload for development",
    )

    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info").lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level",
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
