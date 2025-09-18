"""
Database operations for KCS MCP server.

This package contains database query modules for different domain areas.
"""

# Import Database and get_database from the database.py module using importlib
import importlib.util
from pathlib import Path

# Load the database.py module directly
_database_file = Path(__file__).parent.parent / "database.py"
_spec = importlib.util.spec_from_file_location("database_module", _database_file)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_database_file}")
_database_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_database_module)

# Import classes from the loaded module
Database = _database_module.Database
get_database = _database_module.get_database

# Clean up
del _database_file, _spec, _database_module

from .chunk_queries import ChunkQueries  # noqa: E402

__all__ = ["ChunkQueries", "Database", "get_database"]
