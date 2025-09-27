"""
Database connection pool management for semantic search engine.

Provides PostgreSQL connection pooling with pgvector support for
semantic search operations. Follows KCS database patterns.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Required environment variables for database connection
REQUIRED_ENV_VARS = {
    "POSTGRES_USER": "Database username is required",
    "POSTGRES_PASSWORD": "Database password is required",
}

# Optional environment variables with defaults
OPTIONAL_ENV_VARS = {
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "kcs",
}


def verify_environment(load_env_file: bool = True) -> None:
    """
    Verify required environment variables are set.

    Args:
        load_env_file: Whether to load .env file before verification

    Raises:
        OSError: If required variables are missing
    """
    if load_env_file:
        # Load environment variables from .env file
        load_dotenv()

    missing_vars = []
    for var_name, error_msg in REQUIRED_ENV_VARS.items():
        value = os.getenv(var_name)
        if not value:
            missing_vars.append(f"  - {var_name}: {error_msg}")

    if missing_vars:
        logger.error("Missing required environment variables")
        raise OSError(
            "Missing required environment variables:\n"
            + "\n".join(missing_vars)
            + "\n\nPlease ensure these are set in your .env file or environment."
        )

    # Log optional variables for debugging
    logger.debug("Database environment configuration:")
    logger.debug(f"  POSTGRES_USER: {os.getenv('POSTGRES_USER')}")
    logger.debug(
        f"  POSTGRES_PASSWORD: {'***' if os.getenv('POSTGRES_PASSWORD') else 'NOT SET'}"
    )
    for var_name, default in OPTIONAL_ENV_VARS.items():
        value = os.getenv(var_name, default)
        logger.debug(f"  {var_name}: {value}")


# Global connection instance
_connection_instance: Optional["DatabaseConnection"] = None


class DatabaseConfig(BaseModel):
    """Database configuration for semantic search."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="kcs", description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password")
    min_pool_size: int = Field(default=2, ge=1, description="Minimum pool size")
    max_pool_size: int = Field(default=10, ge=1, description="Maximum pool size")
    command_timeout: int = Field(
        default=30, ge=1, description="Command timeout in seconds"
    )

    @classmethod
    def from_env(cls, verify: bool = True) -> "DatabaseConfig":
        """
        Create config from environment variables.

        Args:
            verify: Whether to verify environment variables first

        Returns:
            DatabaseConfig instance

        Raises:
            OSError: If required variables are missing
            ValueError: If environment variables are invalid
        """
        if verify:
            # Verify and load environment
            verify_environment(load_env_file=True)

        try:
            return cls(
                host=os.getenv("POSTGRES_HOST", OPTIONAL_ENV_VARS["POSTGRES_HOST"]),
                port=int(
                    os.getenv("POSTGRES_PORT", OPTIONAL_ENV_VARS["POSTGRES_PORT"])
                ),
                database=os.getenv("POSTGRES_DB", OPTIONAL_ENV_VARS["POSTGRES_DB"]),
                username=os.getenv("POSTGRES_USER", ""),  # Required, verified above
                password=os.getenv("POSTGRES_PASSWORD", ""),  # Required, verified above
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid environment variable format: {e}")
            raise ValueError(
                f"Failed to create database config from environment: {e}"
            ) from e

    @classmethod
    def from_url(cls, url: str) -> "DatabaseConfig":
        """Create config from database URL."""
        # Parse URL format: postgresql://user:pass@host:port/database
        if not url.startswith("postgresql://"):
            raise ValueError("Database URL must start with 'postgresql://'")

        # This is a simple implementation - for production use a proper URL parser
        import urllib.parse

        parsed = urllib.parse.urlparse(url)

        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "kcs",
            username=parsed.username or "kcs",
            password=parsed.password or "",
        )

    def to_url(self) -> str:
        """Convert config to database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class DatabaseConnection:
    """
    PostgreSQL database connection manager for semantic search.

    Provides connection pooling, transaction management, and schema validation
    for semantic search operations with pgvector support.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """
        Initialize database connection manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self.pool: asyncpg.Pool | None = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish database connection pool."""
        async with self._lock:
            if self._connected:
                logger.info("Database already connected")
                return

            try:
                logger.info(
                    f"Connecting to database at {self.config.host}:{self.config.port}"
                )

                self.pool = await asyncpg.create_pool(
                    self.config.to_url(),
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=self.config.command_timeout,
                )

                # Test connection and verify schema
                async with self.pool.acquire() as conn:
                    # Verify basic connectivity
                    await conn.fetchval("SELECT 1")

                    # Check if pgvector extension is available
                    has_vector = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                    if not has_vector:
                        logger.warning(
                            "pgvector extension not found - vector operations will fail"
                        )

                    # Check if semantic search tables exist
                    tables_exist = await self._verify_schema(conn)
                    if not tables_exist:
                        logger.warning(
                            "Semantic search tables not found - run migrations first"
                        )

                self._connected = True
                logger.info(
                    f"Database connection pool created (min={self.config.min_pool_size}, "
                    f"max={self.config.max_pool_size})"
                )

            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                if self.pool:
                    await self.pool.close()
                    self.pool = None
                raise ConnectionError(f"Failed to connect to database: {e}") from e

    async def disconnect(self) -> None:
        """Close database connection pool."""
        async with self._lock:
            if not self._connected:
                return

            if self.pool:
                await self.pool.close()
                self.pool = None
                logger.info("Database connection pool closed")

            self._connected = False

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Acquire database connection from pool.

        Returns:
            Database connection context manager

        Raises:
            ConnectionError: If not connected to database
        """
        if not self.pool or not self._connected:
            raise ConnectionError("Database connection not established")

        async with self.pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Acquire database connection with transaction.

        Returns:
            Database connection with transaction context

        Raises:
            ConnectionError: If not connected to database
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """
        Execute a query that doesn't return results.

        Args:
            query: SQL query to execute
            *args: Query parameters

        Returns:
            Command status
        """
        async with self.acquire() as conn:
            result = await conn.execute(query, *args)
            return str(result)

    async def fetch_one(self, query: str, *args: Any) -> asyncpg.Record | None:
        """
        Fetch a single row from query.

        Args:
            query: SQL query to execute
            *args: Query parameters

        Returns:
            Single row or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch_all(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """
        Fetch all rows from query.

        Args:
            query: SQL query to execute
            *args: Query parameters

        Returns:
            List of rows
        """
        async with self.acquire() as conn:
            result = await conn.fetch(query, *args)
            return list(result)

    async def fetch_val(self, query: str, *args: Any) -> Any:
        """
        Fetch a single value from query.

        Args:
            query: SQL query to execute
            *args: Query parameters

        Returns:
            Single value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def _verify_schema(self, conn: asyncpg.Connection) -> bool:
        """
        Verify semantic search schema exists.

        Args:
            conn: Database connection

        Returns:
            True if schema is valid
        """
        required_tables = [
            "indexed_content",
            "vector_embedding",
            "search_query",
            "search_result",
        ]

        for table in required_tables:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_name = $1)",
                table,
            )
            if not exists:
                logger.warning(f"Required table '{table}' not found")
                return False

        return True

    async def get_pool_stats(self) -> dict[str, Any]:
        """
        Get connection pool statistics.

        Returns:
            Pool statistics dictionary
        """
        if not self.pool:
            return {"connected": False}

        return {
            "connected": self._connected,
            "size": self.pool.get_size(),
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "idle_size": self.pool.get_idle_size(),
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Perform database health check.

        Returns:
            Health status dictionary
        """
        if not self._connected or not self.pool:
            return {"healthy": False, "error": "Not connected"}

        try:
            async with self.acquire() as conn:
                # Test basic connectivity
                await conn.fetchval("SELECT 1")

                # Check response time
                import time

                start = time.time()
                await conn.fetchval("SELECT pg_database_size(current_database())")
                response_time = (time.time() - start) * 1000

                # Get pool stats
                pool_stats = await self.get_pool_stats()

                return {
                    "healthy": True,
                    "response_time_ms": round(response_time, 2),
                    "pool": pool_stats,
                    "database": self.config.database,
                    "host": self.config.host,
                }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}


async def init_database_connection(
    config: DatabaseConfig | None = None, database_url: str | None = None
) -> DatabaseConnection:
    """
    Initialize global database connection.

    Args:
        config: Database configuration (optional)
        database_url: Database URL string (optional)

    Returns:
        Database connection instance

    Raises:
        ValueError: If neither config nor URL provided
        ConnectionError: If connection fails
    """
    global _connection_instance

    if _connection_instance and _connection_instance._connected:
        logger.info("Database connection already initialized")
        return _connection_instance

    # Determine configuration
    if config:
        db_config = config
    elif database_url:
        db_config = DatabaseConfig.from_url(database_url)
    else:
        # Try environment variables
        try:
            db_config = DatabaseConfig.from_env()
        except Exception as e:
            raise ValueError(
                "Must provide either config, database_url, or set environment variables"
            ) from e

    # Create and connect
    _connection_instance = DatabaseConnection(db_config)
    await _connection_instance.connect()

    return _connection_instance


def get_database_connection() -> DatabaseConnection:
    """
    Get the global database connection instance.

    Returns:
        Database connection instance

    Raises:
        RuntimeError: If connection not initialized
    """
    if not _connection_instance:
        raise RuntimeError(
            "Database connection not initialized. Call init_database_connection() first."
        )

    return _connection_instance


async def close_database_connection() -> None:
    """Close the global database connection."""
    global _connection_instance

    if _connection_instance:
        await _connection_instance.disconnect()
        _connection_instance = None
        logger.info("Global database connection closed")
