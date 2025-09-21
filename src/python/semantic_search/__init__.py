"""
Semantic Search Engine for KCS

This module provides semantic search capabilities for knowledge bases using
embeddings and vector similarity search with pgvector backend.

Integrates with KCS logging infrastructure for consistent structured logging.
"""

__version__ = "0.1.0"

# Import logging integration to ensure it's initialized when the package is imported
try:
    from .logging_integration import initialize_logging

    initialize_logging()
except ImportError:
    # Logging integration not available, continue without it
    pass
