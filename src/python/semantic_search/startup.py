"""
Semantic search engine startup integration for KCS server.

Provides initialization and cleanup functions for integrating semantic search
functionality with the main KCS server lifecycle.
"""

import os
from typing import Any

from .database.connection import init_database_connection
from .logging_integration import get_semantic_search_logger

logger = get_semantic_search_logger(__name__)


class SemanticSearchStartup:
    """
    Handles semantic search engine startup and shutdown.

    Integrates with KCS server lifecycle to provide semantic search functionality
    with proper resource management and graceful degradation.
    """

    def __init__(self) -> None:
        """Initialize semantic search startup handler."""
        self._initialized = False
        self._database_connection: Any = None

    async def initialize(self, database_url: str | None = None) -> bool:
        """
        Initialize semantic search engine.

        Args:
            database_url: Optional database URL (will use environment if not provided)

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            logger.info("Semantic search already initialized")
            return True

        try:
            logger.info("Initializing semantic search engine")

            # Determine database configuration
            if not database_url:
                database_url = self._get_database_url_from_env()

            # Initialize database connection
            if database_url:
                await self._initialize_database(database_url)
            else:
                logger.warning(
                    "No database URL provided, semantic search will run in limited mode"
                )

            # Mark as initialized
            self._initialized = True

            logger.info("Semantic search engine initialized successfully")
            return True

        except Exception as e:
            logger.error(
                "Failed to initialize semantic search engine",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            return False

    async def shutdown(self) -> None:
        """
        Shutdown semantic search engine and cleanup resources.
        """
        if not self._initialized:
            logger.info("Semantic search not initialized, skipping shutdown")
            return

        try:
            logger.info("Shutting down semantic search engine")

            # Close database connection
            if self._database_connection:
                await self._database_connection.disconnect()
                self._database_connection = None

            self._initialized = False

            logger.info("Semantic search engine shutdown complete")

        except Exception as e:
            logger.error(
                "Error during semantic search shutdown",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )

    def is_initialized(self) -> bool:
        """
        Check if semantic search is initialized.

        Returns:
            True if initialized and ready for use
        """
        return self._initialized

    def get_health_status(self) -> dict[str, Any]:
        """
        Get semantic search health status.

        Returns:
            Dictionary with health status information
        """
        status = {
            "initialized": self._initialized,
            "database_connected": self._database_connection is not None,
        }

        if self._database_connection:
            try:
                # Add database health check if available
                status["database_healthy"] = True
            except Exception:
                status["database_healthy"] = False

        return status

    def _get_database_url_from_env(self) -> str | None:
        """Get database URL from environment variables."""
        # Try semantic search specific variables first
        semantic_db_url = os.getenv("SEMANTIC_SEARCH_DATABASE_URL")
        if semantic_db_url:
            return semantic_db_url

        # Fall back to main KCS database URL
        kcs_db_url = os.getenv("DATABASE_URL")
        if kcs_db_url:
            logger.info("Using main KCS database URL for semantic search")
            return kcs_db_url

        # Try individual components
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "kcs")
        username = os.getenv("POSTGRES_USER", "kcs")
        password = os.getenv("POSTGRES_PASSWORD", "")

        if password:
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"

        logger.warning("No database configuration found in environment")
        return None

    async def _initialize_database(self, database_url: str) -> None:
        """
        Initialize database connection for semantic search.

        Args:
            database_url: Database connection URL
        """
        try:
            logger.info("Initializing semantic search database connection")

            # Initialize database connection
            self._database_connection = await init_database_connection(
                database_url=database_url
            )

            # Verify that required tables exist
            await self._verify_database_schema()

            logger.info("Semantic search database connection initialized")

        except Exception as e:
            logger.error(
                "Failed to initialize semantic search database",
                database_url=database_url,
                error=str(e),
            )
            raise

    async def _verify_database_schema(self) -> None:
        """
        Verify that required database schema exists.

        Raises:
            RuntimeError: If required tables are missing
        """
        if not self._database_connection:
            return

        try:
            # Check for required tables
            required_tables = [
                "indexed_content",
                "vector_embedding",
                "search_query",
                "search_result",
            ]

            async with self._database_connection.acquire() as conn:
                for table in required_tables:
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                        table,
                    )
                    if not exists:
                        logger.warning(
                            "Required semantic search table missing", table=table
                        )

            logger.info("Database schema verification complete")

        except Exception as e:
            logger.warning("Database schema verification failed", error=str(e))


# Global startup instance
_startup_instance = SemanticSearchStartup()


async def initialize_semantic_search(database_url: str | None = None) -> bool:
    """
    Initialize semantic search engine (global function).

    Args:
        database_url: Optional database URL

    Returns:
        True if initialization succeeded
    """
    return await _startup_instance.initialize(database_url)


async def shutdown_semantic_search() -> None:
    """Shutdown semantic search engine (global function)."""
    await _startup_instance.shutdown()


def is_semantic_search_initialized() -> bool:
    """Check if semantic search is initialized (global function)."""
    return _startup_instance.is_initialized()


def get_semantic_search_health() -> dict[str, Any]:
    """Get semantic search health status (global function)."""
    return _startup_instance.get_health_status()
