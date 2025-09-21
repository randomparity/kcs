"""
Content indexing background worker for semantic search.

Processes content files in background queues, generating embeddings and
storing vector representations for semantic search functionality.
"""

import asyncio
import signal
import time
from datetime import datetime
from typing import Any

from ..config_loader import load_semantic_search_config
from ..database.connection import get_database_connection
from ..database.vector_store import VectorStore
from ..logging_integration import get_semantic_search_logger
from ..services.embedding_service import EmbeddingService

logger = get_semantic_search_logger(__name__)


class IndexingWorker:
    """
    Background worker for content indexing and embedding generation.

    Monitors queued content and processes it asynchronously to generate
    vector embeddings for semantic search. Supports graceful shutdown
    and error recovery with exponential backoff.
    """

    def __init__(
        self,
        worker_id: str = "worker-1",
        poll_interval: int = 30,
        batch_size: int = 10,
        max_retries: int = 3,
        config_file: str | None = None,
    ) -> None:
        """
        Initialize indexing worker.

        Args:
            worker_id: Unique identifier for this worker instance
            poll_interval: Seconds between polling for new content
            batch_size: Maximum number of items to process per batch
            max_retries: Maximum retry attempts for failed items
            config_file: Path to configuration file (optional)
        """
        self.worker_id = worker_id
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Load configuration
        try:
            self.config = load_semantic_search_config(config_file)
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise

        # Initialize components
        self.vector_store: VectorStore | None = None
        self.embedding_service: EmbeddingService | None = None

        # Worker state
        self.is_running = False
        self.shutdown_requested = False
        self._stats: dict[str, Any] = {
            "processed_count": 0,
            "error_count": 0,
            "retry_count": 0,
            "started_at": None,
            "last_activity": None,
        }

    async def initialize(self) -> None:
        """
        Initialize worker components and connections.

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Initializing indexing worker", worker_id=self.worker_id)

            # Initialize vector store
            self.vector_store = VectorStore()

            # Initialize embedding service (no parameters needed)
            self.embedding_service = EmbeddingService()

            logger.info("Indexing worker initialized successfully")

        except Exception as e:
            logger.error(
                "Failed to initialize indexing worker",
                worker_id=self.worker_id,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Worker initialization failed: {e}") from e

    async def start(self) -> None:
        """
        Start the background indexing worker.

        Runs continuously until shutdown is requested or an unrecoverable
        error occurs. Processes batches of pending content items.
        """
        if self.is_running:
            logger.warning("Worker already running", worker_id=self.worker_id)
            return

        try:
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            self.is_running = True
            self.shutdown_requested = False
            self._stats["started_at"] = datetime.now()

            logger.info("Starting indexing worker", worker_id=self.worker_id)

            # Main processing loop
            while self.is_running and not self.shutdown_requested:
                try:
                    await self._process_batch()
                    self._stats["last_activity"] = datetime.now()

                    # Wait before next poll
                    await asyncio.sleep(self.poll_interval)

                except Exception as e:
                    logger.error(
                        "Error in worker main loop",
                        worker_id=self.worker_id,
                        error=str(e),
                        exc_info=True,
                    )
                    self._stats["error_count"] = self._stats.get("error_count", 0) + 1

                    # Back off on repeated errors
                    await asyncio.sleep(min(self.poll_interval * 2, 300))

            logger.info("Indexing worker stopped", worker_id=self.worker_id)

        except Exception as e:
            logger.error(
                "Fatal error in indexing worker",
                worker_id=self.worker_id,
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            self.is_running = False
            await self._cleanup()

    async def stop(self) -> None:
        """
        Request graceful shutdown of the worker.

        Sets shutdown flag and waits for current batch to complete.
        """
        logger.info("Shutdown requested for indexing worker", worker_id=self.worker_id)
        self.shutdown_requested = True

        # Wait for worker to stop (with timeout)
        timeout = 60  # seconds
        start_time = time.time()

        while self.is_running and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)

        if self.is_running:
            logger.warning(
                "Worker did not stop gracefully within timeout",
                worker_id=self.worker_id,
                timeout=timeout,
            )
        else:
            logger.info("Worker stopped gracefully", worker_id=self.worker_id)

    async def _process_batch(self) -> None:
        """
        Process a batch of pending content items.

        Retrieves pending items from database, processes each one to generate
        embeddings, and updates status. Handles errors with retry logic.
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return

        try:
            # Get pending content items
            pending_items = await self._get_pending_items()

            if not pending_items:
                logger.debug("No pending items to process", worker_id=self.worker_id)
                return

            logger.info(
                "Processing batch",
                worker_id=self.worker_id,
                item_count=len(pending_items),
            )

            # Process each item
            for item in pending_items:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping batch processing")
                    break

                try:
                    await self._process_content_item(item)
                    self._stats["processed_count"] = (
                        self._stats.get("processed_count", 0) + 1
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process content item",
                        worker_id=self.worker_id,
                        item_id=item.get("id"),
                        source_path=item.get("source_path"),
                        error=str(e),
                    )

                    await self._handle_processing_error(item, e)
                    self._stats["error_count"] = self._stats.get("error_count", 0) + 1

        except Exception as e:
            logger.error(
                "Error processing batch",
                worker_id=self.worker_id,
                error=str(e),
                exc_info=True,
            )

    async def _get_pending_items(self) -> list[dict[str, Any]]:
        """
        Get pending content items from database.

        Returns:
            List of content items that need processing
        """
        if not self.vector_store:
            return []

        try:
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                # Get items with status 'PENDING', oldest first
                rows = await conn.fetch(
                    """
                    SELECT id, content_type, source_path, content_hash,
                           title, content, metadata, created_at
                    FROM indexed_content
                    WHERE status = 'PENDING'
                    ORDER BY created_at ASC
                    LIMIT $1
                    """,
                    self.batch_size,
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error("Failed to fetch pending items", error=str(e))
            return []

    async def _process_content_item(self, item: dict[str, Any]) -> None:
        """
        Process a single content item to generate embeddings.

        Args:
            item: Content item dictionary from database
        """
        if not self.vector_store or not self.embedding_service:
            raise RuntimeError("Services not initialized")

        content_id = item["id"]
        source_path = item["source_path"]
        content = item["content"]

        logger.debug(
            "Processing content item",
            worker_id=self.worker_id,
            content_id=content_id,
            source_path=source_path,
        )

        try:
            # Mark as processing
            await self._update_content_status(content_id, "PROCESSING")

            # Create content chunks
            chunks = self._create_chunks(
                content,
                self.config.search.chunk_size,
                self.config.search.chunk_overlap,
            )

            if not chunks:
                logger.warning(
                    "No chunks created for content",
                    content_id=content_id,
                    source_path=source_path,
                )
                await self._update_content_status(content_id, "FAILED")
                return

            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.embed_batch(chunk_texts)

            # Store embeddings in database
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                async with conn.transaction():
                    # Clear any existing embeddings for this content
                    await conn.execute(
                        "DELETE FROM vector_embedding WHERE content_id = $1",
                        content_id,
                    )

                    # Insert new embeddings
                    for i, (chunk, embedding) in enumerate(
                        zip(chunks, embeddings, strict=False)
                    ):
                        await conn.execute(
                            """
                            INSERT INTO vector_embedding
                            (content_id, embedding, chunk_index, chunk_text,
                             line_start, line_end, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """,
                            content_id,
                            embedding,  # Embedding is already a list
                            i,
                            chunk["text"],
                            chunk.get("line_start"),
                            chunk.get("line_end"),
                            chunk.get("metadata", {}),
                        )

            # Update content status
            await self._update_content_status(
                content_id,
                "COMPLETED",
                chunk_count=len(chunks),
                indexed_at=datetime.now(),
            )

            logger.info(
                "Content processing completed",
                worker_id=self.worker_id,
                content_id=content_id,
                source_path=source_path,
                chunk_count=len(chunks),
            )

        except Exception:
            # Mark as failed
            await self._update_content_status(content_id, "FAILED")
            raise

    def _create_chunks(
        self, content: str, chunk_size: int, overlap: int
    ) -> list[dict[str, Any]]:
        """
        Create text chunks from content.

        Args:
            content: Text content to chunk
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not content.strip():
            return []

        chunks = []
        lines = content.splitlines()
        current_chunk = ""
        current_line_start = 0
        current_line_end = 0

        for line_num, line in enumerate(lines):
            # Check if adding this line would exceed chunk size
            if current_chunk and len(current_chunk) + len(line) + 1 > chunk_size:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "line_start": current_line_start,
                            "line_end": current_line_end,
                            "metadata": {
                                "chunk_size": len(current_chunk),
                                "line_count": current_line_end - current_line_start + 1,
                            },
                        }
                    )

                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Take last `overlap` characters for overlap
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + "\n" if overlap_text else ""
                else:
                    current_chunk = ""

                current_line_start = line_num

            # Add line to current chunk
            if current_chunk:
                current_chunk += "\n"
            current_chunk += line
            current_line_end = line_num

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "line_start": current_line_start,
                    "line_end": current_line_end,
                    "metadata": {
                        "chunk_size": len(current_chunk),
                        "line_count": current_line_end - current_line_start + 1,
                    },
                }
            )

        return chunks

    async def _update_content_status(
        self,
        content_id: int,
        status: str,
        chunk_count: int | None = None,
        indexed_at: datetime | None = None,
    ) -> None:
        """
        Update content status in database.

        Args:
            content_id: Content item ID
            status: New status (PENDING, PROCESSING, COMPLETED, FAILED)
            chunk_count: Number of chunks created (optional)
            indexed_at: Timestamp when indexing completed (optional)
        """
        if not self.vector_store:
            return

        try:
            db_conn = get_database_connection()
            async with db_conn.acquire() as conn:
                if chunk_count is not None and indexed_at is not None:
                    await conn.execute(
                        """
                        UPDATE indexed_content
                        SET status = $1, chunk_count = $2, indexed_at = $3, updated_at = now()
                        WHERE id = $4
                        """,
                        status,
                        chunk_count,
                        indexed_at,
                        content_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE indexed_content
                        SET status = $1, updated_at = now()
                        WHERE id = $2
                        """,
                        status,
                        content_id,
                    )

        except Exception as e:
            logger.error(
                "Failed to update content status",
                content_id=content_id,
                status=status,
                error=str(e),
            )

    async def _handle_processing_error(
        self, item: dict[str, Any], error: Exception
    ) -> None:
        """
        Handle processing error with retry logic.

        Args:
            item: Content item that failed processing
            error: Exception that occurred
        """
        content_id = item["id"]

        # Increment retry count in metadata
        metadata = item.get("metadata", {})
        retry_count = metadata.get("retry_count", 0) + 1
        metadata["retry_count"] = retry_count
        metadata["last_error"] = str(error)
        metadata["last_error_at"] = datetime.now().isoformat()

        if retry_count <= self.max_retries:
            # Reset to PENDING for retry
            await self._update_content_status(content_id, "PENDING")

            # Update metadata with retry info
            if self.vector_store:
                db_conn = get_database_connection()
                async with db_conn.acquire() as conn:
                    await conn.execute(
                        "UPDATE indexed_content SET metadata = $1 WHERE id = $2",
                        metadata,
                        content_id,
                    )

            logger.info(
                "Content item queued for retry",
                content_id=content_id,
                retry_count=retry_count,
                max_retries=self.max_retries,
            )
            self._stats["retry_count"] = self._stats.get("retry_count", 0) + 1
        else:
            # Mark as permanently failed
            await self._update_content_status(content_id, "FAILED")

            logger.error(
                "Content item failed permanently after max retries",
                content_id=content_id,
                retry_count=retry_count,
                max_retries=self.max_retries,
            )

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(sig: int, frame: Any) -> None:
            logger.info(
                "Received shutdown signal",
                worker_id=self.worker_id,
                signal=signal.Signals(sig).name,
            )
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _cleanup(self) -> None:
        """Cleanup resources and connections."""
        logger.info("Cleaning up worker resources", worker_id=self.worker_id)

        try:
            # Embedding service doesn't need explicit cleanup
            pass

            # Note: vector_store connection cleanup is handled by the connection pool

        except Exception as e:
            logger.error(
                "Error during worker cleanup",
                worker_id=self.worker_id,
                error=str(e),
            )

    def get_stats(self) -> dict[str, Any]:
        """
        Get worker statistics.

        Returns:
            Dictionary with worker performance and status information
        """
        stats = self._stats.copy()
        stats.update(
            {
                "worker_id": self.worker_id,
                "is_running": self.is_running,
                "shutdown_requested": self.shutdown_requested,
                "poll_interval": self.poll_interval,
                "batch_size": self.batch_size,
                "max_retries": self.max_retries,
            }
        )

        if stats["started_at"] and isinstance(stats["started_at"], datetime):
            stats["uptime_seconds"] = (
                datetime.now() - stats["started_at"]
            ).total_seconds()

        return stats


# CLI and standalone execution support
async def run_worker(
    worker_id: str = "worker-1",
    config_file: str | None = None,
    poll_interval: int = 30,
    batch_size: int = 10,
) -> None:
    """
    Run indexing worker as standalone process.

    Args:
        worker_id: Unique worker identifier
        config_file: Path to configuration file
        poll_interval: Seconds between polling
        batch_size: Items to process per batch
    """
    worker = IndexingWorker(
        worker_id=worker_id,
        poll_interval=poll_interval,
        batch_size=batch_size,
        config_file=config_file,
    )

    try:
        await worker.initialize()
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Worker failed", error=str(e), exc_info=True)
    finally:
        await worker.stop()


def main() -> None:
    """Main function for CLI execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Indexing Worker")
    parser.add_argument(
        "--worker-id",
        default="worker-1",
        help="Unique worker identifier",
    )
    parser.add_argument(
        "--config-file",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between polling for new content",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum items to process per batch",
    )

    args = parser.parse_args()

    # Run worker
    asyncio.run(
        run_worker(
            worker_id=args.worker_id,
            config_file=args.config_file,
            poll_interval=args.poll_interval,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
