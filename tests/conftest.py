"""
Global pytest configuration and fixtures.

This file defines pytest markers, fixtures, and configuration
that apply to all tests in the KCS project.
"""

import os

import pytest
import requests


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring external services",
    )
    config.addinivalue_line(
        "markers", "requires_mcp_server: marks tests that require a running MCP server"
    )
    config.addinivalue_line(
        "markers",
        "requires_database: marks tests that require a running PostgreSQL database",
    )


def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible."""
    try:
        response = requests.get("http://localhost:8080", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def is_database_available() -> bool:
    """Check if PostgreSQL database is accessible."""
    try:
        import asyncio

        import asyncpg

        async def check_db():
            try:
                conn = await asyncpg.connect(get_database_url())
                await conn.close()
                return True
            except Exception:
                return False

        return asyncio.run(check_db())
    except Exception:
        return False


# Skip integration tests in CI environments unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)

# Skip tests requiring MCP server when it's not running
skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server not running"
)

# Skip tests requiring database when it's not available
skip_without_database = pytest.mark.skipif(
    not is_database_available(), reason="PostgreSQL database not available"
)


def get_mcp_jwt_token() -> str:
    """
    Get JWT token for MCP authentication from environment.

    In development mode, this returns the development token.
    For production, proper JWT tokens should be implemented.

    Returns:
        JWT token string for authentication
    """
    environment = os.getenv("ENVIRONMENT", "production")
    if environment == "development":
        return "dev-token"
    else:
        # In production, proper JWT verification would be implemented
        return os.getenv(
            "JWT_SECRET",
            "dev_jwt_secret_change_in_production_use_64_char_random_string",
        )


def get_mcp_auth_headers() -> dict[str, str]:
    """
    Get common headers for MCP API requests including authentication.

    Returns:
        Dictionary with Content-Type and Authorization headers
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_mcp_jwt_token()}",
    }


def get_database_url() -> str:
    """
    Get PostgreSQL database URL from environment.

    This reads the DATABASE_URL from environment variables, using the same
    fallback as the main application. Tests should use this function
    instead of hard-coding database URLs.

    Returns:
        PostgreSQL database connection URL
    """
    url = os.getenv("DATABASE_URL")
    if not url:  # Handle empty string or None
        return (
            "postgresql://kcs:kcs_dev_password_change_in_production@localhost:5432/kcs"
        )
    return url


def get_test_database_url() -> str:
    """
    Get PostgreSQL database URL for testing purposes.

    Similar to get_database_url() but uses a test database name to avoid
    conflicts with the main database during testing.

    Returns:
        PostgreSQL database connection URL for testing
    """
    # Use the main database URL but replace only the database name with _test suffix
    main_url = get_database_url()
    if main_url.endswith("/kcs"):
        # Replace only the last occurrence of "/kcs" to avoid touching the username
        return main_url[:-4] + "/kcs_test"
    else:
        # For URLs that don't end with /kcs, carefully append _test to just the database name
        # This is a fallback - the standard case should be main_url ending with /kcs
        if "/" in main_url:
            parts = main_url.rsplit("/", 1)
            return f"{parts[0]}/{parts[1]}_test"
        return main_url + "_test"
