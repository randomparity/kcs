"""
Configuration for integration tests.

Skips integration tests when running in CI or TESTING mode since they require
a real database connection.
"""

import os

import pytest

# Check if we're in CI or TESTING mode (no database available)
IN_CI = (
    os.getenv("CI", "").lower() == "true" or os.getenv("TESTING", "").lower() == "true"
)

# Skip all integration tests in CI
pytestmark = pytest.mark.skipif(
    IN_CI, reason="Integration tests require database connection, not available in CI"
)


def pytest_collection_modifyitems(config, items):
    """Mark all integration tests to skip in CI environment."""
    if IN_CI:
        skip_marker = pytest.mark.skip(
            reason="Integration tests skipped in CI environment"
        )
        for item in items:
            # Only mark integration tests (those in the integration directory)
            if "/integration/" in str(item.fspath):
                item.add_marker(skip_marker)
