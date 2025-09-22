"""CLI commands for semantic search."""

from .index_commands import add_index_command
from .search_commands import add_search_command
from .status_commands import add_status_command

__all__ = [
    "add_index_command",
    "add_search_command",
    "add_status_command",
]
