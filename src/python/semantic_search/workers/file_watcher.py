"""
File watcher for incremental index updates.

Monitors file system changes and queues content for reindexing when files
are modified, added, or deleted. Provides efficient incremental updates
for the semantic search index.
"""

import asyncio
import hashlib
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from ..config_loader import load_semantic_search_config
from ..database.connection import get_database_connection
from ..logging_integration import get_semantic_search_logger

logger = get_semantic_search_logger(__name__)


class SemanticSearchFileHandler(FileSystemEventHandler):
    """
    File system event handler for semantic search indexing.

    Handles file events and queues content for reindexing when changes occur.
    Supports filtering by file types and patterns.
    """

    def __init__(
        self,
        on_file_change: Callable[[str, str], None],
        supported_extensions: set[str] | None = None,
        ignore_patterns: set[str] | None = None,
    ) -> None:
        """
        Initialize file handler.

        Args:
            on_file_change: Callback function for file changes (path, event_type)
            supported_extensions: File extensions to monitor (e.g., {'.c', '.h', '.py'})
            ignore_patterns: Path patterns to ignore (e.g., {'__pycache__', '.git'})
        """
        super().__init__()
        self.on_file_change = on_file_change
        self.supported_extensions = supported_extensions or {
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cc",
            ".hh",
            ".py",
            ".rs",
            ".go",
            ".js",
            ".ts",
            ".java",
            ".md",
            ".txt",
            ".rst",
            ".asciidoc",
        }
        self.ignore_patterns = ignore_patterns or {
            "__pycache__",
            ".git",
            ".svn",
            ".hg",
            "node_modules",
            "target",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }

    def _should_process_path(self, path: str) -> bool:
        """
        Check if a file path should be processed.

        Args:
            path: File path to check

        Returns:
            True if path should be processed
        """
        path_obj = Path(path)

        # Check if any part of the path matches ignore patterns
        for part in path_obj.parts:
            if any(pattern in part for pattern in self.ignore_patterns):
                return False

        # Check file extension
        if self.supported_extensions:
            return path_obj.suffix.lower() in self.supported_extensions

        return True

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory and self._should_process_path(str(event.src_path)):
            logger.debug(
                "File modified",
                path=event.src_path,
                event_type="modified",
            )
            self.on_file_change(str(event.src_path), "modified")

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory and self._should_process_path(str(event.src_path)):
            logger.debug(
                "File created",
                path=event.src_path,
                event_type="created",
            )
            self.on_file_change(str(event.src_path), "created")

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory and self._should_process_path(str(event.src_path)):
            logger.debug(
                "File deleted",
                path=event.src_path,
                event_type="deleted",
            )
            self.on_file_change(str(event.src_path), "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events."""
        # Handle move as delete + create
        if hasattr(event, "src_path") and hasattr(event, "dest_path"):
            if self._should_process_path(str(event.src_path)):
                self.on_file_change(str(event.src_path), "deleted")
            if self._should_process_path(str(event.dest_path)):
                self.on_file_change(str(event.dest_path), "created")


class IncrementalIndexer:
    """
    Manages incremental index updates based on file system changes.

    Watches configured directories for file changes and automatically
    queues content for reindexing when modifications occur.
    """

    def __init__(
        self,
        watch_paths: list[str],
        config_file: str | None = None,
        batch_size: int = 50,
        flush_interval: int = 30,
    ) -> None:
        """
        Initialize incremental indexer.

        Args:
            watch_paths: Directories to watch for file changes
            config_file: Path to configuration file
            batch_size: Maximum number of changes to batch before processing
            flush_interval: Seconds between processing batches
        """
        self.watch_paths = [Path(p).resolve() for p in watch_paths]
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Load configuration
        try:
            self.config = load_semantic_search_config(config_file)
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise

        # Initialize components
        self.observer: Any | None = None
        self.is_running = False
        self.shutdown_requested = False

        # Change tracking
        self._pending_changes: dict[str, str] = {}  # path -> event_type
        self._last_flush = datetime.now()
        self._stats: dict[str, Any] = {
            "changes_detected": 0,
            "files_queued": 0,
            "files_removed": 0,
            "errors": 0,
            "started_at": None,
        }

    async def start(self) -> None:
        """
        Start file watching and incremental indexing.

        Sets up file system observers and begins monitoring for changes.
        """
        if self.is_running:
            logger.warning("Incremental indexer already running")
            return

        try:
            logger.info("Starting incremental indexer", watch_paths=self.watch_paths)

            # Validate watch paths
            valid_paths = []
            for path in self.watch_paths:
                if path.exists() and path.is_dir():
                    valid_paths.append(path)
                    logger.info("Watching directory", path=str(path))
                else:
                    logger.warning(
                        "Watch path does not exist or is not a directory",
                        path=str(path),
                    )

            if not valid_paths:
                raise ValueError("No valid watch paths provided")

            self.watch_paths = valid_paths

            # Create file system observer
            self.observer = Observer()

            # Create event handler
            handler = SemanticSearchFileHandler(
                on_file_change=self._handle_file_change,
                supported_extensions=self._get_supported_extensions(),
                ignore_patterns=self._get_ignore_patterns(),
            )

            # Add watchers for each path
            for path in self.watch_paths:
                self.observer.schedule(handler, str(path), recursive=True)

            # Start observer
            self.observer.start()

            self.is_running = True
            self.shutdown_requested = False
            self._stats["started_at"] = datetime.now()

            logger.info("Incremental indexer started successfully")

            # Start background processing loop
            await self._run_processing_loop()

        except Exception as e:
            logger.error(
                "Failed to start incremental indexer",
                error=str(e),
                exc_info=True,
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """
        Stop file watching and incremental indexing.

        Gracefully shuts down the file system observer and processes any
        remaining pending changes.
        """
        logger.info("Stopping incremental indexer")

        self.shutdown_requested = True

        # Process any remaining changes
        if self._pending_changes:
            await self._process_pending_changes()

        # Stop file system observer
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=10)
            self.observer = None

        self.is_running = False
        logger.info("Incremental indexer stopped")

    def _handle_file_change(self, file_path: str, event_type: str) -> None:
        """
        Handle file change event.

        Args:
            file_path: Path to changed file
            event_type: Type of change (created, modified, deleted)
        """
        try:
            # Normalize path
            normalized_path = str(Path(file_path).resolve())

            # Track the change
            self._pending_changes[normalized_path] = event_type
            self._stats["changes_detected"] = self._stats.get("changes_detected", 0) + 1

            logger.debug(
                "File change detected",
                path=normalized_path,
                event_type=event_type,
                pending_count=len(self._pending_changes),
            )

            # Process immediately if batch is full
            if len(self._pending_changes) >= self.batch_size:
                # Schedule immediate processing in the background
                task = asyncio.create_task(self._process_pending_changes())
                # Don't await the task to avoid blocking file events
                task.add_done_callback(
                    lambda t: t.exception() if t.exception() else None
                )

        except Exception as e:
            logger.error(
                "Error handling file change",
                file_path=file_path,
                event_type=event_type,
                error=str(e),
            )
            self._stats["errors"] = self._stats.get("errors", 0) + 1

    async def _run_processing_loop(self) -> None:
        """
        Main processing loop for batched file changes.

        Runs until shutdown is requested, processing pending changes
        at regular intervals.
        """
        while self.is_running and not self.shutdown_requested:
            try:
                # Check if we should flush pending changes
                time_since_flush = (datetime.now() - self._last_flush).total_seconds()

                if time_since_flush >= self.flush_interval and self._pending_changes:
                    await self._process_pending_changes()

                # Wait before next check
                await asyncio.sleep(min(self.flush_interval / 4, 5))

            except Exception as e:
                logger.error(
                    "Error in processing loop",
                    error=str(e),
                    exc_info=True,
                )
                self._stats["errors"] = self._stats.get("errors", 0) + 1
                await asyncio.sleep(5)

    async def _process_pending_changes(self) -> None:
        """
        Process all pending file changes.

        Updates the database with changes and queues files for reindexing
        or removes deleted files from the index.
        """
        if not self._pending_changes:
            return

        # Get current batch and clear pending
        changes = self._pending_changes.copy()
        self._pending_changes.clear()
        self._last_flush = datetime.now()

        logger.info(
            "Processing file changes",
            change_count=len(changes),
        )

        try:
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                for file_path, event_type in changes.items():
                    try:
                        if event_type == "deleted":
                            await self._handle_file_deletion(conn, file_path)
                        else:  # created or modified
                            await self._handle_file_update(conn, file_path)

                    except Exception as e:
                        logger.error(
                            "Failed to process file change",
                            file_path=file_path,
                            event_type=event_type,
                            error=str(e),
                        )
                        self._stats["errors"] = self._stats.get("errors", 0) + 1

        except Exception as e:
            logger.error(
                "Failed to process pending changes",
                error=str(e),
                exc_info=True,
            )
            self._stats["errors"] = self._stats.get("errors", 0) + 1

    async def _handle_file_deletion(self, conn: Any, file_path: str) -> None:
        """
        Handle file deletion by removing from index.

        Args:
            conn: Database connection
            file_path: Path to deleted file
        """
        # Remove from indexed_content (cascades to embeddings)
        result = await conn.execute(
            "DELETE FROM indexed_content WHERE source_path = $1",
            file_path,
        )

        if result:
            self._stats["files_removed"] = self._stats.get("files_removed", 0) + 1
            logger.debug(
                "Removed deleted file from index",
                file_path=file_path,
            )

    async def _handle_file_update(self, conn: Any, file_path: str) -> None:
        """
        Handle file creation or modification by queuing for reindexing.

        Args:
            conn: Database connection
            file_path: Path to updated file
        """
        try:
            # Check if file still exists
            if not os.path.exists(file_path):
                await self._handle_file_deletion(conn, file_path)
                return

            # Read file content
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding or skip binary files
                try:
                    with open(file_path, encoding="latin-1") as f:
                        content = f.read()
                except Exception:
                    logger.debug(
                        "Skipping binary or unreadable file",
                        file_path=file_path,
                    )
                    return

            # Generate content hash
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Check if content has actually changed
            existing_hash = await conn.fetchval(
                "SELECT content_hash FROM indexed_content WHERE source_path = $1",
                file_path,
            )

            if existing_hash == content_hash:
                logger.debug(
                    "File content unchanged, skipping reindex",
                    file_path=file_path,
                )
                return

            # Determine content type based on file extension
            content_type = self._determine_content_type(file_path)

            # Extract title (filename without extension)
            title = Path(file_path).stem

            # Upsert content record
            await conn.execute(
                """
                INSERT INTO indexed_content
                (content_type, source_path, content_hash, title, content, status, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, 'PENDING', now(), now())
                ON CONFLICT (source_path)
                DO UPDATE SET
                    content_hash = $3,
                    title = $4,
                    content = $5,
                    status = 'PENDING',
                    chunk_count = 0,
                    indexed_at = NULL,
                    updated_at = now()
                """,
                content_type,
                file_path,
                content_hash,
                title,
                content,
            )

            # Clear existing embeddings for reprocessing
            await conn.execute(
                """
                DELETE FROM vector_embedding ve
                USING indexed_content ic
                WHERE ve.content_id = ic.id AND ic.source_path = $1
                """,
                file_path,
            )

            self._stats["files_queued"] = self._stats.get("files_queued", 0) + 1
            logger.debug(
                "Queued file for reindexing",
                file_path=file_path,
                content_type=content_type,
                content_hash=content_hash[:8],
            )

        except Exception as e:
            logger.error(
                "Failed to handle file update",
                file_path=file_path,
                error=str(e),
            )
            raise

    def _determine_content_type(self, file_path: str) -> str:
        """
        Determine content type based on file extension.

        Args:
            file_path: Path to file

        Returns:
            Content type string
        """
        extension = Path(file_path).suffix.lower()

        type_mapping = {
            ".c": "source_file",
            ".h": "source_file",
            ".cpp": "source_file",
            ".hpp": "source_file",
            ".cc": "source_file",
            ".hh": "source_file",
            ".py": "source_file",
            ".rs": "source_file",
            ".go": "source_file",
            ".js": "source_file",
            ".ts": "source_file",
            ".java": "source_file",
            ".md": "documentation",
            ".txt": "documentation",
            ".rst": "documentation",
            ".asciidoc": "documentation",
        }

        return type_mapping.get(extension, "unknown")

    def _get_supported_extensions(self) -> set[str]:
        """Get supported file extensions from configuration."""
        # Could be made configurable in the future
        return {
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cc",
            ".hh",
            ".py",
            ".rs",
            ".go",
            ".js",
            ".ts",
            ".java",
            ".md",
            ".txt",
            ".rst",
            ".asciidoc",
        }

    def _get_ignore_patterns(self) -> set[str]:
        """Get ignore patterns from configuration."""
        # Could be made configurable in the future
        return {
            "__pycache__",
            ".git",
            ".svn",
            ".hg",
            "node_modules",
            "target",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".venv",
            "venv",
            ".env",
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get incremental indexer statistics.

        Returns:
            Dictionary with performance and status information
        """
        stats = self._stats.copy()
        stats.update(
            {
                "is_running": self.is_running,
                "shutdown_requested": self.shutdown_requested,
                "watch_paths": [str(p) for p in self.watch_paths],
                "pending_changes": len(self._pending_changes),
                "batch_size": self.batch_size,
                "flush_interval": self.flush_interval,
            }
        )

        if stats["started_at"] and isinstance(stats["started_at"], datetime):
            stats["uptime_seconds"] = (
                datetime.now() - stats["started_at"]
            ).total_seconds()

        return stats


# CLI and standalone execution support
async def run_file_watcher(
    watch_paths: list[str],
    config_file: str | None = None,
    batch_size: int = 50,
    flush_interval: int = 30,
) -> None:
    """
    Run incremental indexer as standalone process.

    Args:
        watch_paths: Directories to watch for changes
        config_file: Path to configuration file
        batch_size: Changes to batch before processing
        flush_interval: Seconds between processing batches
    """
    indexer = IncrementalIndexer(
        watch_paths=watch_paths,
        config_file=config_file,
        batch_size=batch_size,
        flush_interval=flush_interval,
    )

    try:
        await indexer.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("File watcher failed", error=str(e), exc_info=True)
    finally:
        await indexer.stop()


def main() -> None:
    """Main function for CLI execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search File Watcher")
    parser.add_argument(
        "watch_paths",
        nargs="+",
        help="Directories to watch for file changes",
    )
    parser.add_argument(
        "--config-file",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum changes to batch before processing",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=30,
        help="Seconds between processing batches",
    )

    args = parser.parse_args()

    # Run file watcher
    asyncio.run(
        run_file_watcher(
            watch_paths=args.watch_paths,
            config_file=args.config_file,
            batch_size=args.batch_size,
            flush_interval=args.flush_interval,
        )
    )


if __name__ == "__main__":
    main()
