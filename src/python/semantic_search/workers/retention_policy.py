"""
Data retention policy enforcement for semantic search.

Implements automatic cleanup of old data according to FR-012 requirements:
- Vector embeddings: 1 year retention
- Search queries: Raw for 7 days, anonymized analytics for 90 days
- Search results: Linked to query retention
- Indexed content: Indefinite (purged only on file deletion)
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

from ..config_loader import load_semantic_search_config
from ..database.connection import get_database_connection
from ..logging_integration import get_semantic_search_logger

logger = get_semantic_search_logger(__name__)


class RetentionPolicy:
    """
    Enforces data retention policies for semantic search data.

    Implements automatic cleanup according to FR-012 specifications with
    configurable retention periods and safe deletion practices.
    """

    def __init__(
        self,
        config_file: str | None = None,
        vector_retention_days: int = 365,  # 1 year
        query_retention_days: int = 7,  # Raw queries
        analytics_retention_days: int = 90,  # Anonymized analytics
        dry_run: bool = False,
    ) -> None:
        """
        Initialize retention policy enforcer.

        Args:
            config_file: Path to configuration file
            vector_retention_days: Days to retain vector embeddings (default: 365)
            query_retention_days: Days to retain raw search queries (default: 7)
            analytics_retention_days: Days to retain anonymized analytics (default: 90)
            dry_run: If True, only log what would be deleted without actual deletion
        """
        # Load configuration
        try:
            self.config = load_semantic_search_config(config_file)
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise

        # Retention periods
        self.vector_retention_days = vector_retention_days
        self.query_retention_days = query_retention_days
        self.analytics_retention_days = analytics_retention_days
        self.dry_run = dry_run

        # Stats tracking
        self._stats: dict[str, Any] = {
            "last_run": None,
            "queries_cleaned": 0,
            "queries_anonymized": 0,
            "results_cleaned": 0,
            "vectors_cleaned": 0,
            "orphaned_cleaned": 0,
            "errors": 0,
        }

    async def run_cleanup(self) -> dict[str, Any]:
        """
        Run complete data retention cleanup.

        Executes all retention policies in the correct order to maintain
        referential integrity and avoid cascading deletes.

        Returns:
            Dictionary with cleanup statistics
        """
        logger.info(
            "Starting data retention cleanup",
            dry_run=self.dry_run,
            vector_retention_days=self.vector_retention_days,
            query_retention_days=self.query_retention_days,
            analytics_retention_days=self.analytics_retention_days,
        )

        start_time = datetime.now()
        cleanup_stats = {
            "started_at": start_time,
            "dry_run": self.dry_run,
            "queries_cleaned": 0,
            "queries_anonymized": 0,
            "results_cleaned": 0,
            "vectors_cleaned": 0,
            "orphaned_cleaned": 0,
            "errors": 0,
        }

        try:
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                # Step 1: Anonymize old queries (before deletion)
                anonymized = await self._anonymize_old_queries(conn)
                cleanup_stats["queries_anonymized"] = anonymized

                # Step 2: Clean up old search results (before queries)
                results_cleaned = await self._cleanup_search_results(conn)
                cleanup_stats["results_cleaned"] = results_cleaned

                # Step 3: Clean up old search queries
                queries_cleaned = await self._cleanup_search_queries(conn)
                cleanup_stats["queries_cleaned"] = queries_cleaned

                # Step 4: Clean up old vector embeddings
                vectors_cleaned = await self._cleanup_vector_embeddings(conn)
                cleanup_stats["vectors_cleaned"] = vectors_cleaned

                # Step 5: Clean up orphaned data
                orphaned_cleaned = await self._cleanup_orphaned_data(conn)
                cleanup_stats["orphaned_cleaned"] = orphaned_cleaned

            # Update internal stats
            self._stats["last_run"] = start_time
            self._stats["queries_cleaned"] = (
                self._stats.get("queries_cleaned", 0) + cleanup_stats["queries_cleaned"]
            )
            self._stats["queries_anonymized"] = (
                self._stats.get("queries_anonymized", 0)
                + cleanup_stats["queries_anonymized"]
            )
            self._stats["results_cleaned"] = (
                self._stats.get("results_cleaned", 0) + cleanup_stats["results_cleaned"]
            )
            self._stats["vectors_cleaned"] = (
                self._stats.get("vectors_cleaned", 0) + cleanup_stats["vectors_cleaned"]
            )
            self._stats["orphaned_cleaned"] = (
                self._stats.get("orphaned_cleaned", 0)
                + cleanup_stats["orphaned_cleaned"]
            )

            completion_time = datetime.now()
            duration = (completion_time - start_time).total_seconds()

            cleanup_stats.update(
                {
                    "completed_at": completion_time,
                    "duration_seconds": duration,
                    "success": True,
                }
            )

            logger.info(
                "Data retention cleanup completed",
                duration_seconds=duration,
                **{
                    k: v
                    for k, v in cleanup_stats.items()
                    if k not in ["started_at", "completed_at"]
                },
            )

        except Exception as e:
            self._stats["errors"] = self._stats.get("errors", 0) + 1
            cleanup_stats["errors"] = 1
            cleanup_stats["success"] = False
            cleanup_stats["error"] = str(e)

            logger.error(
                "Data retention cleanup failed",
                error=str(e),
                exc_info=True,
            )

        return cleanup_stats

    async def _anonymize_old_queries(self, conn: Any) -> int:
        """
        Anonymize old search queries while preserving analytics data.

        Removes personally identifiable information from queries older than
        the raw retention period but keeps them for analytics purposes.

        Args:
            conn: Database connection

        Returns:
            Number of queries anonymized
        """
        cutoff_date = datetime.now() - timedelta(days=self.query_retention_days)

        logger.debug(
            "Anonymizing old search queries",
            cutoff_date=cutoff_date,
            dry_run=self.dry_run,
        )

        try:
            # Find queries that need anonymization
            queries_to_anonymize = await conn.fetch(
                """
                SELECT id, query_text, created_at
                FROM search_query
                WHERE created_at < $1
                  AND query_text IS NOT NULL
                  AND query_text != '[ANONYMIZED]'
                ORDER BY created_at
                """,
                cutoff_date,
            )

            if not queries_to_anonymize:
                logger.debug("No queries found for anonymization")
                return 0

            anonymized_count = 0

            for query in queries_to_anonymize:
                query_id = query["id"]
                original_text = query["query_text"]
                created_at = query["created_at"]

                if self.dry_run:
                    logger.info(
                        "Would anonymize query",
                        query_id=query_id,
                        created_at=created_at,
                        original_length=len(original_text),
                    )
                    anonymized_count += 1
                else:
                    # Create anonymized analytics record
                    analytics_data = {
                        "original_length": len(original_text),
                        "word_count": len(original_text.split()),
                        "has_code_keywords": any(
                            keyword in original_text.lower()
                            for keyword in [
                                "function",
                                "struct",
                                "class",
                                "def",
                                "return",
                                "if",
                                "for",
                            ]
                        ),
                        "has_file_patterns": any(
                            pattern in original_text
                            for pattern in [".c", ".h", ".py", "/", "\\"]
                        ),
                        "anonymized_at": datetime.now().isoformat(),
                    }

                    # Update query with anonymized data
                    await conn.execute(
                        """
                        UPDATE search_query
                        SET query_text = '[ANONYMIZED]',
                            preprocessing_config = COALESCE(preprocessing_config, '{}')::jsonb || $2::jsonb
                        WHERE id = $1
                        """,
                        query_id,
                        json.dumps(analytics_data),
                    )

                    anonymized_count += 1

                    logger.debug(
                        "Anonymized query",
                        query_id=query_id,
                        created_at=created_at,
                        analytics_preserved=bool(analytics_data),
                    )

            logger.info(
                "Query anonymization completed",
                anonymized_count=anonymized_count,
                dry_run=self.dry_run,
            )

            return anonymized_count

        except Exception as e:
            logger.error(
                "Failed to anonymize old queries",
                error=str(e),
                exc_info=True,
            )
            raise

    async def _cleanup_search_results(self, conn: Any) -> int:
        """
        Clean up old search results based on query retention.

        Removes search results for queries that are beyond the analytics
        retention period.

        Args:
            conn: Database connection

        Returns:
            Number of search results cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=self.analytics_retention_days)

        logger.debug(
            "Cleaning up old search results",
            cutoff_date=cutoff_date,
            dry_run=self.dry_run,
        )

        try:
            if self.dry_run:
                # Count what would be deleted
                count_result = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM search_result sr
                    JOIN search_query sq ON sr.query_id = sq.id
                    WHERE sq.created_at < $1
                    """,
                    cutoff_date,
                )
                count = int(count_result) if count_result is not None else 0

                logger.info(
                    "Would clean up search results",
                    count=count,
                    cutoff_date=cutoff_date,
                )

                return count

            else:
                # Delete old search results
                result = await conn.execute(
                    """
                    DELETE FROM search_result
                    WHERE query_id IN (
                        SELECT id FROM search_query
                        WHERE created_at < $1
                    )
                    """,
                    cutoff_date,
                )

                # Extract number of deleted rows from result
                deleted_count = (
                    int(str(result).split()[-1])
                    if result and "DELETE" in str(result)
                    else 0
                )

                logger.info(
                    "Cleaned up search results",
                    deleted_count=deleted_count,
                    cutoff_date=cutoff_date,
                )

                return deleted_count

        except Exception as e:
            logger.error(
                "Failed to cleanup search results",
                error=str(e),
                exc_info=True,
            )
            raise

    async def _cleanup_search_queries(self, conn: Any) -> int:
        """
        Clean up old search queries beyond analytics retention.

        Removes anonymized queries that are older than the analytics
        retention period.

        Args:
            conn: Database connection

        Returns:
            Number of search queries cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=self.analytics_retention_days)

        logger.debug(
            "Cleaning up old search queries",
            cutoff_date=cutoff_date,
            dry_run=self.dry_run,
        )

        try:
            if self.dry_run:
                # Count what would be deleted
                count_result = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM search_query
                    WHERE created_at < $1
                    """,
                    cutoff_date,
                )
                count = int(count_result) if count_result is not None else 0

                logger.info(
                    "Would clean up search queries",
                    count=count,
                    cutoff_date=cutoff_date,
                )

                return count

            else:
                # Delete old queries
                result = await conn.execute(
                    """
                    DELETE FROM search_query
                    WHERE created_at < $1
                    """,
                    cutoff_date,
                )

                # Extract number of deleted rows from result
                deleted_count = (
                    int(str(result).split()[-1])
                    if result and "DELETE" in str(result)
                    else 0
                )

                logger.info(
                    "Cleaned up search queries",
                    deleted_count=deleted_count,
                    cutoff_date=cutoff_date,
                )

                return deleted_count

        except Exception as e:
            logger.error(
                "Failed to cleanup search queries",
                error=str(e),
                exc_info=True,
            )
            raise

    async def _cleanup_vector_embeddings(self, conn: Any) -> int:
        """
        Clean up old vector embeddings beyond retention period.

        Removes vector embeddings that are older than the vector retention
        period, but preserves embeddings for content that still exists.

        Args:
            conn: Database connection

        Returns:
            Number of vector embeddings cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=self.vector_retention_days)

        logger.debug(
            "Cleaning up old vector embeddings",
            cutoff_date=cutoff_date,
            dry_run=self.dry_run,
        )

        try:
            if self.dry_run:
                # Count what would be deleted
                count_result = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM vector_embedding ve
                    JOIN indexed_content ic ON ve.content_id = ic.id
                    WHERE ve.created_at < $1
                      AND ic.status != 'COMPLETED'  -- Don't delete active embeddings
                    """,
                    cutoff_date,
                )
                count = int(count_result) if count_result is not None else 0

                logger.info(
                    "Would clean up vector embeddings",
                    count=count,
                    cutoff_date=cutoff_date,
                )

                return count

            else:
                # Delete old vector embeddings for non-active content
                result = await conn.execute(
                    """
                    DELETE FROM vector_embedding
                    WHERE created_at < $1
                      AND content_id IN (
                          SELECT id FROM indexed_content
                          WHERE status IN ('FAILED', 'PENDING')
                             OR updated_at < $1
                      )
                    """,
                    cutoff_date,
                )

                # Extract number of deleted rows from result
                deleted_count = (
                    int(str(result).split()[-1])
                    if result and "DELETE" in str(result)
                    else 0
                )

                logger.info(
                    "Cleaned up vector embeddings",
                    deleted_count=deleted_count,
                    cutoff_date=cutoff_date,
                )

                return deleted_count

        except Exception as e:
            logger.error(
                "Failed to cleanup vector embeddings",
                error=str(e),
                exc_info=True,
            )
            raise

    async def _cleanup_orphaned_data(self, conn: Any) -> int:
        """
        Clean up orphaned data with broken references.

        Removes data that has lost its parent references due to cascading
        deletes or data corruption.

        Args:
            conn: Database connection

        Returns:
            Number of orphaned records cleaned up
        """
        logger.debug(
            "Cleaning up orphaned data",
            dry_run=self.dry_run,
        )

        total_cleaned = 0

        try:
            # Clean up search results without valid queries
            if self.dry_run:
                orphaned_results = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM search_result sr
                    WHERE NOT EXISTS (
                        SELECT 1 FROM search_query sq WHERE sq.id = sr.query_id
                    )
                    """
                )
                logger.info(
                    "Would clean up orphaned search results",
                    count=orphaned_results,
                )
                total_cleaned += orphaned_results

            else:
                result = await conn.execute(
                    """
                    DELETE FROM search_result
                    WHERE NOT EXISTS (
                        SELECT 1 FROM search_query sq WHERE sq.id = query_id
                    )
                    """
                )
                orphaned_results = (
                    int(result.split()[-1]) if result and "DELETE" in result else 0
                )
                total_cleaned += orphaned_results

                if orphaned_results > 0:
                    logger.info(
                        "Cleaned up orphaned search results",
                        count=orphaned_results,
                    )

            # Clean up vector embeddings without valid content
            if self.dry_run:
                orphaned_embeddings = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM vector_embedding ve
                    WHERE NOT EXISTS (
                        SELECT 1 FROM indexed_content ic WHERE ic.id = ve.content_id
                    )
                    """
                )
                logger.info(
                    "Would clean up orphaned vector embeddings",
                    count=orphaned_embeddings,
                )
                total_cleaned += orphaned_embeddings

            else:
                result = await conn.execute(
                    """
                    DELETE FROM vector_embedding
                    WHERE NOT EXISTS (
                        SELECT 1 FROM indexed_content ic WHERE ic.id = content_id
                    )
                    """
                )
                orphaned_embeddings = (
                    int(result.split()[-1]) if result and "DELETE" in result else 0
                )
                total_cleaned += orphaned_embeddings

                if orphaned_embeddings > 0:
                    logger.info(
                        "Cleaned up orphaned vector embeddings",
                        count=orphaned_embeddings,
                    )

            logger.info(
                "Orphaned data cleanup completed",
                total_cleaned=total_cleaned,
                dry_run=self.dry_run,
            )

            return total_cleaned

        except Exception as e:
            logger.error(
                "Failed to cleanup orphaned data",
                error=str(e),
                exc_info=True,
            )
            raise

    def get_stats(self) -> dict[str, Any]:
        """
        Get retention policy statistics.

        Returns:
            Dictionary with cleanup statistics and configuration
        """
        stats = self._stats.copy()
        stats.update(
            {
                "vector_retention_days": self.vector_retention_days,
                "query_retention_days": self.query_retention_days,
                "analytics_retention_days": self.analytics_retention_days,
                "dry_run": self.dry_run,
            }
        )

        return stats


class RetentionScheduler:
    """
    Scheduler for automatic retention policy enforcement.

    Runs retention cleanup on a configurable schedule with proper
    error handling and monitoring.
    """

    def __init__(
        self,
        retention_policy: RetentionPolicy,
        interval_hours: int = 24,
    ) -> None:
        """
        Initialize retention scheduler.

        Args:
            retention_policy: RetentionPolicy instance to run
            interval_hours: Hours between cleanup runs
        """
        self.retention_policy = retention_policy
        self.interval_hours = interval_hours
        self.is_running = False
        self.shutdown_requested = False

        self._stats: dict[str, Any] = {
            "started_at": None,
            "last_run": None,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
        }

    async def start(self) -> None:
        """
        Start the retention policy scheduler.

        Runs cleanup tasks on the configured interval until shutdown.
        """
        if self.is_running:
            logger.warning("Retention scheduler already running")
            return

        logger.info(
            "Starting retention policy scheduler",
            interval_hours=self.interval_hours,
        )

        self.is_running = True
        self.shutdown_requested = False
        self._stats["started_at"] = datetime.now()

        try:
            while self.is_running and not self.shutdown_requested:
                try:
                    # Run retention cleanup
                    cleanup_result = await self.retention_policy.run_cleanup()

                    self._stats["last_run"] = datetime.now()
                    self._stats["total_runs"] = self._stats.get("total_runs", 0) + 1

                    if cleanup_result.get("success", False):
                        self._stats["successful_runs"] = (
                            self._stats.get("successful_runs", 0) + 1
                        )
                        logger.info(
                            "Scheduled retention cleanup completed successfully",
                            **{
                                k: v
                                for k, v in cleanup_result.items()
                                if k != "success"
                            },
                        )
                    else:
                        self._stats["failed_runs"] = (
                            self._stats.get("failed_runs", 0) + 1
                        )
                        logger.error(
                            "Scheduled retention cleanup failed",
                            error=cleanup_result.get("error", "Unknown error"),
                        )

                except Exception as e:
                    self._stats["failed_runs"] = self._stats.get("failed_runs", 0) + 1
                    logger.error(
                        "Error in retention scheduler",
                        error=str(e),
                        exc_info=True,
                    )

                # Wait for next interval
                await asyncio.sleep(self.interval_hours * 3600)

        except Exception as e:
            logger.error(
                "Fatal error in retention scheduler",
                error=str(e),
                exc_info=True,
            )
        finally:
            self.is_running = False

        logger.info("Retention policy scheduler stopped")

    async def stop(self) -> None:
        """Stop the retention policy scheduler."""
        logger.info("Stopping retention policy scheduler")
        self.shutdown_requested = True

        # Wait for current run to complete
        timeout = 300  # 5 minutes
        start_time = datetime.now()

        while (
            self.is_running and (datetime.now() - start_time).total_seconds() < timeout
        ):
            await asyncio.sleep(1)

        if self.is_running:
            logger.warning("Retention scheduler did not stop gracefully within timeout")

    def get_stats(self) -> dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler performance information
        """
        stats = self._stats.copy()
        stats.update(
            {
                "is_running": self.is_running,
                "shutdown_requested": self.shutdown_requested,
                "interval_hours": self.interval_hours,
            }
        )

        if stats["started_at"] and isinstance(stats["started_at"], datetime):
            stats["uptime_seconds"] = (
                datetime.now() - stats["started_at"]
            ).total_seconds()

        return stats


# CLI and standalone execution support
async def run_retention_cleanup(
    config_file: str | None = None,
    dry_run: bool = False,
    vector_retention_days: int = 365,
    query_retention_days: int = 7,
    analytics_retention_days: int = 90,
) -> None:
    """
    Run data retention cleanup as standalone operation.

    Args:
        config_file: Path to configuration file
        dry_run: If True, only log what would be deleted
        vector_retention_days: Days to retain vector embeddings
        query_retention_days: Days to retain raw search queries
        analytics_retention_days: Days to retain anonymized analytics
    """
    policy = RetentionPolicy(
        config_file=config_file,
        dry_run=dry_run,
        vector_retention_days=vector_retention_days,
        query_retention_days=query_retention_days,
        analytics_retention_days=analytics_retention_days,
    )

    try:
        result = await policy.run_cleanup()

        if result.get("success", False):
            logger.info("Data retention cleanup completed successfully")
        else:
            logger.error("Data retention cleanup failed", error=result.get("error"))

    except Exception as e:
        logger.error("Retention cleanup failed", error=str(e), exc_info=True)


async def run_retention_scheduler(
    config_file: str | None = None,
    interval_hours: int = 24,
    dry_run: bool = False,
) -> None:
    """
    Run retention policy scheduler as standalone process.

    Args:
        config_file: Path to configuration file
        interval_hours: Hours between cleanup runs
        dry_run: If True, only log what would be deleted
    """
    policy = RetentionPolicy(config_file=config_file, dry_run=dry_run)
    scheduler = RetentionScheduler(policy, interval_hours=interval_hours)

    try:
        await scheduler.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await scheduler.stop()


def main() -> None:
    """Main function for CLI execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Data Retention")
    parser.add_argument(
        "command",
        choices=["cleanup", "schedule"],
        help="Command to execute",
    )
    parser.add_argument(
        "--config-file",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only log what would be deleted without actual deletion",
    )
    parser.add_argument(
        "--vector-retention-days",
        type=int,
        default=365,
        help="Days to retain vector embeddings (default: 365)",
    )
    parser.add_argument(
        "--query-retention-days",
        type=int,
        default=7,
        help="Days to retain raw search queries (default: 7)",
    )
    parser.add_argument(
        "--analytics-retention-days",
        type=int,
        default=90,
        help="Days to retain anonymized analytics (default: 90)",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=24,
        help="Hours between scheduled cleanup runs (default: 24)",
    )

    args = parser.parse_args()

    if args.command == "cleanup":
        asyncio.run(
            run_retention_cleanup(
                config_file=args.config_file,
                dry_run=args.dry_run,
                vector_retention_days=args.vector_retention_days,
                query_retention_days=args.query_retention_days,
                analytics_retention_days=args.analytics_retention_days,
            )
        )
    elif args.command == "schedule":
        asyncio.run(
            run_retention_scheduler(
                config_file=args.config_file,
                interval_hours=args.interval_hours,
                dry_run=args.dry_run,
            )
        )


if __name__ == "__main__":
    main()
