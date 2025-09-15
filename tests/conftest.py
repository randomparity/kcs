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
                conn = await asyncpg.connect(
                    "postgresql://kcs:kcs_dev_password@localhost:5432/kcs"
                )
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
