"""
Database migration utilities for semantic search.

Provides Python interface for applying and managing semantic search
database migrations with proper error handling and logging.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

import asyncpg

from .logging_integration import get_semantic_search_logger

logger = get_semantic_search_logger(__name__)


class MigrationError(Exception):
    """Exception raised when migration operations fail."""

    def __init__(self, message: str, migration_name: str | None = None) -> None:
        """
        Initialize migration error.

        Args:
            message: Error message
            migration_name: Name of migration that failed (optional)
        """
        super().__init__(message)
        self.migration_name = migration_name


class SemanticSearchMigrations:
    """
    Handles semantic search database migrations.

    Provides methods to check, apply, and verify semantic search database
    schema with proper error handling and rollback capabilities.
    """

    def __init__(self, database_url: str) -> None:
        """
        Initialize migration manager.

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.migrations_dir = (
            Path(__file__).parent.parent.parent.parent / "sql" / "migrations"
        )

    async def check_prerequisites(self) -> dict[str, bool]:
        """
        Check if all prerequisites for semantic search are met.

        Returns:
            Dictionary with prerequisite check results
        """
        results = {
            "database_connection": False,
            "pgvector_available": False,
            "uuid_extension": False,
        }

        try:
            conn = await asyncpg.connect(self.database_url)

            # Test basic connection
            await conn.fetchval("SELECT 1")
            results["database_connection"] = True

            # Check pgvector extension
            pgvector_available = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
            )
            results["pgvector_available"] = bool(pgvector_available)

            # Check uuid extension
            uuid_available = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'uuid-ossp')"
            )
            results["uuid_extension"] = bool(uuid_available)

            await conn.close()

        except Exception as e:
            logger.error(f"Failed to check prerequisites: {e}")

        return results

    async def check_migration_status(self) -> dict[str, Any]:
        """
        Check the current migration status.

        Returns:
            Dictionary with migration status information
        """
        status: dict[str, Any] = {
            "core_tables_exist": False,
            "query_log_tables_exist": False,
            "indexes_exist": False,
            "functions_exist": False,
            "migration_needed": True,
        }

        try:
            conn = await asyncpg.connect(self.database_url)

            # Check core semantic search tables
            core_tables = [
                "indexed_content",
                "vector_embedding",
                "search_query",
                "search_result",
            ]

            all_core_exist = True
            for table in core_tables:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                    table,
                )
                if not exists:
                    all_core_exist = False
                    break

            status["core_tables_exist"] = all_core_exist

            # Check query log tables (from migration 009)
            query_tables = [
                "semantic_query_log",
                "semantic_query_cache",
                "semantic_query_feedback",
                "semantic_query_stats",
            ]

            all_query_exist = True
            for table in query_tables:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                    table,
                )
                if not exists:
                    all_query_exist = False
                    break

            status["query_log_tables_exist"] = all_query_exist

            # Check key indexes
            key_indexes = [
                "idx_vector_embedding_similarity",
                "idx_indexed_content_content_fts",
            ]

            all_indexes_exist = True
            for index in key_indexes:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = $1)",
                    index,
                )
                if not exists:
                    all_indexes_exist = False
                    break

            status["indexes_exist"] = all_indexes_exist

            # Check key functions
            key_functions = [
                "find_similar_embeddings",
                "cleanup_old_search_results",
            ]

            all_functions_exist = True
            for function in key_functions:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = $1)", function
                )
                if not exists:
                    all_functions_exist = False
                    break

            status["functions_exist"] = all_functions_exist

            # Determine if migration is needed
            status["migration_needed"] = not (
                status["core_tables_exist"]
                and status["indexes_exist"]
                and status["functions_exist"]
            )

            await conn.close()

        except Exception as e:
            logger.error(f"Failed to check migration status: {e}")
            status["error"] = str(e)

        return status

    async def apply_core_migration(self, force: bool = False) -> bool:
        """
        Apply the core semantic search migration.

        Args:
            force: Whether to force re-apply if already exists

        Returns:
            True if migration was successful

        Raises:
            MigrationError: If migration fails
        """
        migration_file = self.migrations_dir / "014_semantic_search_core.sql"

        if not migration_file.exists():
            raise MigrationError(
                f"Migration file not found: {migration_file}",
                "014_semantic_search_core",
            )

        # Check if migration is needed
        status = await self.check_migration_status()

        if not status["migration_needed"] and not force:
            logger.info("Semantic search migration not needed")
            return True

        if not force and status["core_tables_exist"]:
            logger.warning("Core tables already exist. Use force=True to re-apply.")
            return False

        logger.info("Applying semantic search core migration")

        try:
            # Read migration SQL
            migration_sql = migration_file.read_text()

            # Apply migration
            conn = await asyncpg.connect(self.database_url)

            try:
                # Execute migration in a transaction
                async with conn.transaction():
                    await conn.execute(migration_sql)

                logger.info("Core migration applied successfully")

            finally:
                await conn.close()

            # Verify migration was successful
            verification_status = await self.check_migration_status()

            if verification_status["migration_needed"]:
                raise MigrationError(
                    "Migration verification failed", "014_semantic_search_core"
                )

            logger.info("Migration verification successful")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise MigrationError(
                f"Failed to apply core migration: {e}", "014_semantic_search_core"
            ) from e

    async def rollback_core_migration(self) -> bool:
        """
        Rollback the core semantic search migration.

        WARNING: This will delete all semantic search data!

        Returns:
            True if rollback was successful
        """
        logger.warning("Rolling back semantic search core migration")
        logger.warning("This will DELETE ALL semantic search data!")

        try:
            conn = await asyncpg.connect(self.database_url)

            try:
                async with conn.transaction():
                    # Drop tables in reverse dependency order
                    await conn.execute("DROP TABLE IF EXISTS search_result CASCADE")
                    await conn.execute("DROP TABLE IF EXISTS vector_embedding CASCADE")
                    await conn.execute("DROP TABLE IF EXISTS search_query CASCADE")
                    await conn.execute("DROP TABLE IF EXISTS indexed_content CASCADE")

                    # Drop functions
                    await conn.execute(
                        "DROP FUNCTION IF EXISTS find_similar_embeddings CASCADE"
                    )
                    await conn.execute(
                        "DROP FUNCTION IF EXISTS cleanup_old_search_results CASCADE"
                    )
                    await conn.execute(
                        "DROP FUNCTION IF EXISTS reindex_content CASCADE"
                    )
                    await conn.execute(
                        "DROP FUNCTION IF EXISTS vector_cosine_similarity CASCADE"
                    )
                    await conn.execute(
                        "DROP FUNCTION IF EXISTS update_updated_at_column CASCADE"
                    )

                logger.info("Core migration rollback completed")

            finally:
                await conn.close()

            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise MigrationError(
                f"Failed to rollback core migration: {e}", "014_semantic_search_core"
            ) from e

    async def get_migration_info(self) -> dict[str, Any]:
        """
        Get comprehensive migration information.

        Returns:
            Dictionary with migration and database information
        """
        info = {
            "migration_files": [],
            "prerequisites": {},
            "status": {},
            "database_url": self.database_url.split("@")[-1]
            if "@" in self.database_url
            else "unknown",
        }

        try:
            # Find migration files
            if self.migrations_dir.exists():
                migration_files = list(self.migrations_dir.glob("*semantic*.sql"))
                info["migration_files"] = [f.name for f in migration_files]

            # Get prerequisites
            info["prerequisites"] = await self.check_prerequisites()

            # Get status
            info["status"] = await self.check_migration_status()

        except Exception as e:
            logger.error(f"Failed to get migration info: {e}")
            info["error"] = str(e)

        return info


# Convenience functions for common operations


async def check_semantic_search_schema(database_url: str) -> dict[str, Any]:
    """
    Check semantic search schema status.

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        Schema status information
    """
    migrations = SemanticSearchMigrations(database_url)
    return await migrations.get_migration_info()


async def apply_semantic_search_migrations(
    database_url: str, force: bool = False
) -> bool:
    """
    Apply semantic search migrations.

    Args:
        database_url: PostgreSQL connection URL
        force: Whether to force re-apply existing migrations

    Returns:
        True if successful

    Raises:
        MigrationError: If migration fails
    """
    migrations = SemanticSearchMigrations(database_url)

    # Check prerequisites first
    prereqs = await migrations.check_prerequisites()

    if not prereqs["database_connection"]:
        raise MigrationError("Cannot connect to database")

    if not prereqs["pgvector_available"]:
        raise MigrationError("pgvector extension is not available")

    # Apply core migration
    return await migrations.apply_core_migration(force=force)


async def verify_semantic_search_setup(database_url: str) -> bool:
    """
    Verify semantic search database setup is complete.

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        True if setup is complete and functional
    """
    migrations = SemanticSearchMigrations(database_url)

    try:
        status = await migrations.check_migration_status()
        return not status["migration_needed"]

    except Exception as e:
        logger.error(f"Failed to verify setup: {e}")
        return False


# CLI integration for standalone usage
async def main() -> None:
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Database Migrations")
    parser.add_argument(
        "command",
        choices=["check", "apply", "rollback", "info"],
        help="Migration command to execute",
    )
    parser.add_argument("--database-url", help="PostgreSQL connection URL")
    parser.add_argument(
        "--force", action="store_true", help="Force re-apply migrations"
    )

    args = parser.parse_args()

    # Get database URL
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: Database URL not provided")
        return

    migrations = SemanticSearchMigrations(database_url)

    try:
        if args.command == "check":
            status = await migrations.check_migration_status()
            print("Migration Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")

        elif args.command == "apply":
            success = await migrations.apply_core_migration(force=args.force)
            if success:
                print("Migration applied successfully")
            else:
                print("Migration not applied")

        elif args.command == "rollback":
            success = await migrations.rollback_core_migration()
            if success:
                print("Migration rolled back successfully")
            else:
                print("Rollback failed")

        elif args.command == "info":
            info = await migrations.get_migration_info()
            print("Migration Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")

    except MigrationError as e:
        print(f"Migration error: {e}")
        if e.migration_name:
            print(f"Migration: {e.migration_name}")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
