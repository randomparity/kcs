"""
Async chunk loading module for KCS chunk processing.

Provides functionality to load and validate chunk manifests and individual
chunk files with checksum verification and structured error handling.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from pydantic import ValidationError

from .bridge import RUST_BRIDGE_AVAILABLE
from .database import Database
from .models.chunk_models import ChunkManifest, ChunkMetadata

if RUST_BRIDGE_AVAILABLE:
    try:
        import kcs_python_bridge
    except ImportError:
        RUST_BRIDGE_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ChunkLoadError(Exception):
    """Base exception for chunk loading errors."""

    def __init__(self, message: str, chunk_id: str | None = None):
        super().__init__(message)
        self.chunk_id = chunk_id


class ChecksumMismatchError(ChunkLoadError):
    """Raised when chunk checksum verification fails."""

    def __init__(self, chunk_id: str, expected: str, actual: str):
        super().__init__(
            f"Checksum mismatch for chunk {chunk_id}: expected {expected}, got {actual}",
            chunk_id,
        )
        self.expected = expected
        self.actual = actual


class ChunkLoader:
    """
    Async loader for chunk manifests and chunk files with validation.

    Provides methods to load, validate, and verify chunks with proper
    error handling and structured logging.
    """

    def __init__(
        self,
        verify_checksums: bool = True,
        buffer_size: int = 1024 * 1024,
        database: Database | None = None,
    ):
        """
        Initialize chunk loader.

        Args:
            verify_checksums: Whether to verify SHA256 checksums on load
            buffer_size: Buffer size for streaming file operations (1MB default)
            database: Optional database instance for caching and logging
        """
        self.verify_checksums = verify_checksums
        self.buffer_size = buffer_size
        self.database = database
        self._checksum_calculator = None

        if RUST_BRIDGE_AVAILABLE and verify_checksums:
            try:
                # Initialize Rust checksum calculator with default config
                self._checksum_calculator = kcs_python_bridge.PyChecksumCalculator()
                logger.debug("Initialized Rust checksum calculator")
            except Exception as e:
                logger.warning(
                    "Failed to initialize Rust checksum calculator", error=str(e)
                )

    async def load_manifest(self, manifest_path: Path | str) -> ChunkManifest:
        """
        Load and validate a chunk manifest from file.

        Args:
            manifest_path: Path to manifest.json file

        Returns:
            Validated ChunkManifest object

        Raises:
            ChunkLoadError: If manifest cannot be loaded or is invalid
            FileNotFoundError: If manifest file doesn't exist
        """
        manifest_path = Path(manifest_path)

        logger.info("Loading chunk manifest", path=str(manifest_path))

        try:
            async with aiofiles.open(manifest_path, encoding="utf-8") as f:
                content = await f.read()

            manifest_data = json.loads(content)
            manifest = ChunkManifest.model_validate(manifest_data)

            logger.info(
                "Manifest loaded successfully",
                path=str(manifest_path),
                version=manifest.version,
                total_chunks=manifest.total_chunks,
            )

            # Cache manifest in database if available
            if self.database:
                await self._cache_manifest_in_database(manifest, manifest_path)

            return manifest

        except FileNotFoundError:
            logger.error("Manifest file not found", path=str(manifest_path))
            raise
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON in manifest", path=str(manifest_path), error=str(e)
            )
            raise ChunkLoadError(
                f"Invalid JSON in manifest {manifest_path}: {e}"
            ) from e
        except ValidationError as e:
            logger.error(
                "Manifest validation failed", path=str(manifest_path), error=str(e)
            )
            raise ChunkLoadError(f"Manifest validation failed: {e}") from e
        except Exception as e:
            logger.error(
                "Unexpected error loading manifest",
                path=str(manifest_path),
                error=str(e),
            )
            raise ChunkLoadError(f"Failed to load manifest {manifest_path}: {e}") from e

    async def load_chunk(
        self,
        chunk_metadata: ChunkMetadata,
        base_path: Path | str | None = None,
        verify_checksum: bool | None = None,
    ) -> dict[str, Any]:
        """
        Load and validate an individual chunk file.

        Args:
            chunk_metadata: Metadata for the chunk to load
            base_path: Base directory path (defaults to chunk file's parent dir)
            verify_checksum: Override checksum verification (defaults to loader config)

        Returns:
            Dictionary containing chunk data (symbols, entry_points, etc.)

        Raises:
            ChunkLoadError: If chunk cannot be loaded
            ChecksumMismatchError: If checksum verification fails
            FileNotFoundError: If chunk file doesn't exist
        """
        if verify_checksum is None:
            verify_checksum = self.verify_checksums

        # Resolve chunk file path
        if base_path:
            chunk_path = Path(base_path) / chunk_metadata.file
        else:
            chunk_path = Path(chunk_metadata.file)

        logger.debug(
            "Loading chunk file",
            chunk_id=chunk_metadata.id,
            path=str(chunk_path),
            verify_checksum=verify_checksum,
        )

        try:
            # Read chunk file
            async with aiofiles.open(chunk_path, encoding="utf-8") as f:
                content = await f.read()

            # Verify checksum if requested
            if verify_checksum:
                await self._verify_chunk_checksum(
                    content, chunk_metadata.checksum_sha256, chunk_metadata.id
                )

            # Parse JSON
            chunk_data = json.loads(content)

            # Basic validation that this is a chunk file
            if not isinstance(chunk_data, dict):
                raise ChunkLoadError(f"Chunk {chunk_metadata.id} is not a JSON object")

            # The chunk file contains just the data (e.g., {"files": [...]})
            # Metadata comes from the manifest, not from the chunk file itself
            # Return chunk_data directly since processor now handles it properly

            logger.info(
                "Chunk loaded successfully",
                chunk_id=chunk_metadata.id,
                path=str(chunk_path),
                size_bytes=len(content),
                checksum_verified=verify_checksum,
            )

            # Log chunk load to database if available
            if self.database:
                await self._log_chunk_load(
                    chunk_metadata, len(content), verify_checksum
                )

            return chunk_data

        except FileNotFoundError:
            logger.error(
                "Chunk file not found", chunk_id=chunk_metadata.id, path=str(chunk_path)
            )
            raise
        except ChecksumMismatchError:
            # Re-raise checksum errors as-is
            raise
        except json.JSONDecodeError as e:
            logger.error(
                "Invalid JSON in chunk file",
                chunk_id=chunk_metadata.id,
                path=str(chunk_path),
                error=str(e),
            )
            raise ChunkLoadError(
                f"Invalid JSON in chunk {chunk_metadata.id}: {e}", chunk_metadata.id
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected error loading chunk",
                chunk_id=chunk_metadata.id,
                path=str(chunk_path),
                error=str(e),
            )
            raise ChunkLoadError(
                f"Failed to load chunk {chunk_metadata.id}: {e}", chunk_metadata.id
            ) from e

    async def load_chunks_batch(
        self,
        chunks: list[ChunkMetadata],
        base_path: Path | str | None = None,
        max_concurrency: int = 4,
        verify_checksums: bool | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Load multiple chunks concurrently with controlled parallelism.

        Args:
            chunks: List of chunk metadata to load
            base_path: Base directory path for chunk files
            max_concurrency: Maximum number of concurrent loads
            verify_checksums: Override checksum verification setting

        Returns:
            Dictionary mapping chunk_id to chunk data

        Note:
            Failed chunks are logged but not included in results.
            Check logs for load failures.
        """
        logger.info(
            "Starting batch chunk loading",
            total_chunks=len(chunks),
            max_concurrency=max_concurrency,
        )

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def load_single_chunk(
            chunk_metadata: ChunkMetadata,
        ) -> tuple[str, dict[str, Any] | None]:
            async with semaphore:
                try:
                    chunk_data = await self.load_chunk(
                        chunk_metadata, base_path, verify_checksums
                    )
                    return chunk_metadata.id, chunk_data
                except Exception as e:
                    logger.error(
                        "Failed to load chunk in batch",
                        chunk_id=chunk_metadata.id,
                        error=str(e),
                    )
                    return chunk_metadata.id, None

        # Execute all loads concurrently
        tasks = [load_single_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter successful loads
        loaded_chunks = {
            chunk_id: chunk_data
            for chunk_id, chunk_data in results
            if chunk_data is not None
        }

        logger.info(
            "Batch chunk loading completed",
            total_chunks=len(chunks),
            successful_loads=len(loaded_chunks),
            failed_loads=len(chunks) - len(loaded_chunks),
        )

        return loaded_chunks

    async def verify_manifest_integrity(
        self, manifest: ChunkManifest, base_path: Path | str | None = None
    ) -> dict[str, Any]:
        """
        Verify the integrity of all chunks referenced in a manifest.

        Args:
            manifest: The manifest to verify
            base_path: Base directory path for chunk files

        Returns:
            Dictionary with verification results:
            {
                "valid": bool,
                "total_chunks": int,
                "verified_chunks": int,
                "failed_chunks": list[str],
                "errors": dict[str, str]
            }
        """
        logger.info(
            "Starting manifest integrity verification",
            manifest_version=manifest.version,
            total_chunks=manifest.total_chunks,
        )

        errors: dict[str, str] = {}
        verified_count = 0

        for chunk in manifest.chunks:
            try:
                # Only verify checksum, don't load full content
                if base_path:
                    chunk_path = Path(base_path) / chunk.file
                else:
                    chunk_path = Path(chunk.file)

                # Check file exists and size matches
                if not chunk_path.exists():
                    errors[chunk.id] = f"File not found: {chunk_path}"
                    continue

                file_size = chunk_path.stat().st_size
                if file_size != chunk.size_bytes:
                    errors[chunk.id] = (
                        f"Size mismatch: expected {chunk.size_bytes}, got {file_size}"
                    )
                    continue

                # Verify checksum if enabled
                if self.verify_checksums:
                    async with aiofiles.open(chunk_path, encoding="utf-8") as f:
                        content = await f.read()

                    await self._verify_chunk_checksum(
                        content, chunk.checksum_sha256, chunk.id
                    )

                verified_count += 1

            except Exception as e:
                errors[chunk.id] = str(e)

        failed_chunks = list(errors.keys())
        is_valid = len(failed_chunks) == 0

        result = {
            "valid": is_valid,
            "total_chunks": manifest.total_chunks,
            "verified_chunks": verified_count,
            "failed_chunks": failed_chunks,
            "errors": errors,
        }

        logger.info(
            "Manifest integrity verification completed",
            valid=is_valid,
            verified_chunks=verified_count,
            failed_chunks=len(failed_chunks),
        )

        return result

    async def _verify_chunk_checksum(
        self, content: str, expected_checksum: str, chunk_id: str
    ) -> None:
        """
        Verify chunk content against expected SHA256 checksum.

        Args:
            content: Chunk file content
            expected_checksum: Expected SHA256 hex string
            chunk_id: Chunk identifier for error reporting

        Raises:
            ChecksumMismatchError: If checksum doesn't match
        """
        if not self._checksum_calculator:
            # Fallback to Python hashlib if Rust bridge unavailable
            import hashlib

            actual_checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            try:
                # Use Rust bridge for fast checksum calculation
                actual_checksum = self._checksum_calculator.calculate_sha256(
                    content.encode("utf-8")
                )
            except Exception as e:
                logger.warning(
                    "Rust checksum calculation failed, falling back to Python",
                    chunk_id=chunk_id,
                    error=str(e),
                )
                import hashlib

                actual_checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if actual_checksum.lower() != expected_checksum.lower():
            logger.error(
                "Checksum verification failed",
                chunk_id=chunk_id,
                expected=expected_checksum,
                actual=actual_checksum,
            )
            raise ChecksumMismatchError(chunk_id, expected_checksum, actual_checksum)

        logger.debug("Checksum verified", chunk_id=chunk_id, checksum=actual_checksum)

    async def _cache_manifest_in_database(
        self, manifest: ChunkManifest, manifest_path: Path
    ) -> None:
        """
        Cache manifest data in the database for faster access.

        Args:
            manifest: The loaded manifest
            manifest_path: Path to the manifest file
        """
        try:
            if not self.database:
                return
            async with self.database.acquire() as conn:
                # Check if manifest already exists
                existing = await conn.fetchrow(
                    "SELECT version FROM indexing_manifest WHERE version = $1",
                    manifest.version,
                )

                if existing:
                    logger.debug(
                        "Manifest already cached in database",
                        version=manifest.version,
                    )
                    return

                # Insert manifest record
                await conn.execute(
                    """
                    INSERT INTO indexing_manifest (
                        version, created, kernel_version, kernel_path, config,
                        total_chunks, total_size_bytes, manifest_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (version) DO NOTHING
                    """,
                    manifest.version,
                    manifest.created,
                    manifest.kernel_version,
                    manifest.kernel_path,
                    manifest.config,
                    manifest.total_chunks,
                    manifest.total_size_bytes,
                    manifest.model_dump(),  # Store full manifest as JSONB
                )

                logger.info(
                    "Manifest cached in database",
                    version=manifest.version,
                    path=str(manifest_path),
                )

        except Exception as e:
            logger.warning(
                "Failed to cache manifest in database",
                version=manifest.version,
                error=str(e),
            )
            # Don't re-raise - manifest caching is optional

    async def _log_chunk_load(
        self, chunk_metadata: ChunkMetadata, content_size: int, checksum_verified: bool
    ) -> None:
        """
        Log chunk load operation to database for auditing.

        Args:
            chunk_metadata: Metadata of the loaded chunk
            content_size: Size of loaded content in bytes
            checksum_verified: Whether checksum was verified
        """
        try:
            if not self.database:
                return
            async with self.database.acquire() as conn:
                # Log to a chunk_loads audit table (if it exists)
                await conn.execute(
                    """
                    INSERT INTO chunk_loads (
                        chunk_id, loaded_at, content_size_bytes, checksum_verified
                    ) VALUES ($1, CURRENT_TIMESTAMP, $2, $3)
                    ON CONFLICT DO NOTHING
                    """,
                    chunk_metadata.id,
                    content_size,
                    checksum_verified,
                )

                logger.debug(
                    "Chunk load logged to database",
                    chunk_id=chunk_metadata.id,
                    content_size=content_size,
                )

        except Exception as e:
            logger.debug(
                "Failed to log chunk load to database",
                chunk_id=chunk_metadata.id,
                error=str(e),
            )
            # Don't re-raise - load logging is optional


# Convenience functions for common operations


async def load_manifest_from_path(manifest_path: Path | str) -> ChunkManifest:
    """Load a chunk manifest from the given path."""
    loader = ChunkLoader()
    return await loader.load_manifest(manifest_path)


async def load_chunk_by_id(
    manifest: ChunkManifest,
    chunk_id: str,
    base_path: Path | str | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    """
    Load a specific chunk by ID from a manifest.

    Args:
        manifest: The manifest containing chunk metadata
        chunk_id: ID of the chunk to load
        base_path: Base directory path for chunk files
        verify_checksum: Whether to verify checksum

    Returns:
        Chunk data dictionary

    Raises:
        ChunkLoadError: If chunk ID not found or load fails
    """
    # Find chunk metadata
    chunk_metadata = None
    for chunk in manifest.chunks:
        if chunk.id == chunk_id:
            chunk_metadata = chunk
            break

    if chunk_metadata is None:
        raise ChunkLoadError(f"Chunk ID {chunk_id} not found in manifest")

    loader = ChunkLoader(verify_checksums=verify_checksum)
    return await loader.load_chunk(chunk_metadata, base_path)


async def verify_all_chunks(
    manifest: ChunkManifest, base_path: Path | str | None = None
) -> bool:
    """
    Verify all chunks in a manifest are valid and accessible.

    Args:
        manifest: The manifest to verify
        base_path: Base directory path for chunk files

    Returns:
        True if all chunks are valid, False otherwise
    """
    loader = ChunkLoader()
    result = await loader.verify_manifest_integrity(manifest, base_path)
    return bool(result["valid"])
