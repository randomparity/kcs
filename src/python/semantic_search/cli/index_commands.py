"""
CLI index command implementation for semantic search engine.

Provides command-line interface for content indexing, index management,
and batch operations for the semantic search system.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import structlog

from ..database.connection import DatabaseConfig, init_database_connection
from ..database.index_manager import IndexManager, VectorIndexConfig
from ..database.vector_store import VectorStore
from ..services.embedding_service import EmbeddingService

logger = structlog.get_logger(__name__)


async def index_file(
    file_path: str,
    content_type: str | None = None,
    database_url: str | None = None,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Index a single file for semantic search.

    Args:
        file_path: Path to file to index
        content_type: Override content type detection
        database_url: Database connection URL
        force_reindex: Force reindexing if already indexed

    Returns:
        Indexing result dictionary
    """
    try:
        # Initialize database connection
        if database_url:
            db_config = DatabaseConfig.from_url(database_url)
        else:
            db_config = DatabaseConfig.from_env()

        await init_database_connection(db_config)

        # Initialize services
        vector_store = VectorStore()
        embedding_service = EmbeddingService()

        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if not path.is_file():
            return {"error": f"Path is not a file: {file_path}"}

        # Read file content
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return {"error": f"Could not read file as UTF-8: {file_path}"}

        # Detect content type if not provided
        if not content_type:
            if path.suffix in [".c", ".h"]:
                content_type = "c_source"
            elif path.suffix in [".py"]:
                content_type = "python_source"
            elif path.suffix in [".md"]:
                content_type = "markdown"
            elif path.suffix in [".txt"]:
                content_type = "text"
            else:
                content_type = "unknown"

        # Check if already indexed (unless force_reindex)
        # Note: VectorStore doesn't have get_content_by_path method yet
        # This would need to be implemented or we skip this check for now
        if not force_reindex:
            # TODO: Implement get_content_by_path in VectorStore
            pass

        # Create indexed content using VectorStore interface
        # IndexedContent model has different constructor signature
        metadata = {"file_size": path.stat().st_size, "file_ext": path.suffix}

        # Store content and generate embedding
        content_id = await vector_store.store_content(
            content_type=content_type,
            source_path=str(path.absolute()),
            content=content,
            title=path.name,
            metadata=metadata,
        )

        # Generate embedding
        embedding = embedding_service.embed_query(content)

        # Store embedding
        await vector_store.store_embedding(
            content_id=content_id,
            embedding=embedding,
            model_name="BAAI/bge-small-en-v1.5",
            model_version="1.0",
        )

        return {
            "status": "indexed",
            "content_id": content_id,
            "path": str(path.absolute()),
            "content_type": content_type,
            "size": len(content),
        }

    except Exception as e:
        logger.error(f"Failed to index file {file_path}", error=str(e), exc_info=True)
        return {"error": str(e)}


async def index_directory(
    directory_path: str,
    recursive: bool = True,
    file_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_files: int | None = None,
    database_url: str | None = None,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Index all files in a directory.

    Args:
        directory_path: Path to directory to index
        recursive: Include subdirectories
        file_patterns: File patterns to include (glob style)
        exclude_patterns: File patterns to exclude (glob style)
        max_files: Maximum number of files to index
        database_url: Database connection URL
        force_reindex: Force reindexing of existing files

    Returns:
        Batch indexing results
    """
    try:
        directory = Path(directory_path)
        if not directory.exists():
            return {"error": f"Directory not found: {directory_path}"}

        if not directory.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}

        # Default file patterns if none provided
        if not file_patterns:
            file_patterns = ["*.c", "*.h", "*.py", "*.md", "*.txt"]

        # Default exclude patterns
        if not exclude_patterns:
            exclude_patterns = ["*.o", "*.so", "*.pyc", "__pycache__", ".git", ".svn"]

        # Find files to index
        files_to_index = []

        if recursive:
            pattern_files: list[Path] = []
            for pattern in file_patterns:
                pattern_files.extend(list(directory.rglob(pattern)))
        else:
            pattern_files = []
            for pattern in file_patterns:
                pattern_files.extend(list(directory.glob(pattern)))

        # Filter out excluded patterns
        for file_path in pattern_files:
            skip_file = False
            for exclude_pattern in exclude_patterns:
                if exclude_pattern in str(file_path):
                    skip_file = True
                    break

            if not skip_file and file_path.is_file():
                files_to_index.append(file_path)

        # Limit number of files if specified
        if max_files and len(files_to_index) > max_files:
            files_to_index = files_to_index[:max_files]

        logger.info(f"Found {len(files_to_index)} files to index in {directory_path}")

        # Index files in batches
        results: dict[str, Any] = {
            "total_files": len(files_to_index),
            "indexed": 0,
            "already_indexed": 0,
            "errors": 0,
            "file_results": [],
        }

        for i, file_path in enumerate(files_to_index):
            logger.info(f"Indexing file {i + 1}/{len(files_to_index)}: {file_path}")

            result = await index_file(
                str(file_path),
                database_url=database_url,
                force_reindex=force_reindex,
            )

            results["file_results"].append(
                {
                    "file": str(file_path),
                    "result": result,
                }
            )

            if result.get("status") == "indexed":
                results["indexed"] += 1
            elif result.get("status") == "already_indexed":
                results["already_indexed"] += 1
            elif "error" in result:
                results["errors"] += 1

        return results

    except Exception as e:
        logger.error(
            f"Failed to index directory {directory_path}", error=str(e), exc_info=True
        )
        return {"error": str(e)}


async def manage_indexes(
    operation: str,
    index_name: str | None = None,
    database_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Manage database indexes.

    Args:
        operation: Operation to perform (optimize, rebuild, stats, vacuum)
        index_name: Specific index name (for rebuild operation)
        database_url: Database connection URL
        **kwargs: Additional operation parameters

    Returns:
        Operation results
    """
    try:
        # Initialize database connection
        if database_url:
            db_config = DatabaseConfig.from_url(database_url)
        else:
            db_config = DatabaseConfig.from_env()

        await init_database_connection(db_config)

        # Initialize index manager
        index_manager = IndexManager()

        if operation == "optimize":
            # Optimize vector indexes
            vector_config = VectorIndexConfig(
                m=kwargs.get("m", 16),
                ef_construction=kwargs.get("ef_construction", 64),
                ef_search=kwargs.get("ef_search", 40),
            )
            return await index_manager.optimize_vector_index(vector_config)

        elif operation == "rebuild":
            if not index_name:
                return {"error": "Index name required for rebuild operation"}
            return await index_manager.rebuild_index(index_name)

        elif operation == "stats":
            return {
                "index_statistics": [
                    stat.dict() for stat in await index_manager.get_index_statistics()
                ],
                "performance_metrics": await index_manager.get_performance_metrics(),
                "usage_analysis": await index_manager.check_index_usage(),
            }

        elif operation == "vacuum":
            return await index_manager.vacuum_analyze_tables()

        elif operation == "optimize_all":
            return await index_manager.optimize_all_indexes()

        else:
            return {"error": f"Unknown operation: {operation}"}

    except Exception as e:
        logger.error("Index management operation failed", error=str(e), exc_info=True)
        return {"error": str(e)}


def format_index_results(results: dict[str, Any], output_format: str = "pretty") -> str:
    """
    Format indexing results for display.

    Args:
        results: Results to format
        output_format: Output format ('pretty', 'json', 'summary')

    Returns:
        Formatted results string
    """
    if output_format == "json":
        return json.dumps(results, indent=2, default=str)

    elif output_format == "summary":
        if "total_files" in results:
            # Directory indexing summary
            return (
                f"Indexing Summary:\n"
                f"  Total files: {results['total_files']}\n"
                f"  Indexed: {results['indexed']}\n"
                f"  Already indexed: {results['already_indexed']}\n"
                f"  Errors: {results['errors']}"
            )
        elif "status" in results:
            # Single file indexing summary
            return f"File {results.get('path', 'unknown')}: {results['status']}"
        else:
            # Generic summary
            return f"Operation completed: {results.get('success', 'unknown status')}"

    else:  # pretty format
        output_lines = []

        if "error" in results:
            output_lines.append(f"❌ Error: {results['error']}")

        elif "total_files" in results:
            # Directory indexing results
            output_lines.extend(
                [
                    "📁 Directory Indexing Results",
                    "=" * 30,
                    f"Total files processed: {results['total_files']}",
                    f"Successfully indexed: {results['indexed']}",
                    f"Already indexed: {results['already_indexed']}",
                    f"Errors encountered: {results['errors']}",
                    "",
                ]
            )

            if results.get("file_results"):
                output_lines.append("📝 File Details:")
                for file_result in results["file_results"][:10]:  # Show first 10
                    file_path = file_result["file"]
                    result = file_result["result"]
                    status = result.get("status", "error")

                    if status == "indexed":
                        output_lines.append(f"  ✅ {file_path}")
                    elif status == "already_indexed":
                        output_lines.append(f"  ➡️  {file_path} (already indexed)")
                    else:
                        error_msg = result.get("error", "unknown error")
                        output_lines.append(f"  ❌ {file_path}: {error_msg}")

                if len(results["file_results"]) > 10:
                    output_lines.append(
                        f"  ... and {len(results['file_results']) - 10} more files"
                    )

        elif "status" in results:
            # Single file indexing result
            status = results["status"]
            path = results.get("path", "unknown")

            if status == "indexed":
                output_lines.extend(
                    [
                        f"✅ Successfully indexed: {path}",
                        f"   Content ID: {results.get('content_id')}",
                        f"   Content Type: {results.get('content_type')}",
                        f"   Size: {results.get('size')} characters",
                    ]
                )
            elif status == "already_indexed":
                output_lines.extend(
                    [
                        f"➡️  Already indexed: {path}",
                        f"   Content ID: {results.get('content_id')}",
                    ]
                )

        elif "index_statistics" in results:
            # Index management results
            output_lines.extend(
                [
                    "📊 Index Statistics",
                    "=" * 20,
                ]
            )

            for stat in results["index_statistics"]:
                output_lines.append(
                    f"  {stat['index_name']}: {stat['index_type']} "
                    f"({stat['size_bytes']:,} bytes, {stat['usage_count']} uses)"
                )

            if "performance_metrics" in results:
                perf = results["performance_metrics"]
                output_lines.extend(
                    [
                        "",
                        "⚡ Performance Metrics:",
                        f"  Cache hit ratio: {perf.get('cache_hit_ratio', 0):.1f}%",
                        f"  Active connections: {perf.get('active_connections', 0)}",
                        f"  Index count: {perf.get('index_count', 0)}",
                    ]
                )

        else:
            # Generic results
            output_lines.append("Operation completed successfully")
            for key, value in results.items():
                if key != "error":
                    output_lines.append(f"  {key}: {value}")

        return "\n".join(output_lines)


async def index_command(args: argparse.Namespace) -> None:
    """Execute index command with given arguments."""
    try:
        if args.index_command == "file":
            # Index single file
            result = await index_file(
                file_path=args.path,
                content_type=args.content_type,
                database_url=args.database_url,
                force_reindex=args.force,
            )

        elif args.index_command == "directory":
            # Index directory
            file_patterns = None
            if args.patterns:
                file_patterns = [p.strip() for p in args.patterns.split(",")]

            exclude_patterns = None
            if args.exclude:
                exclude_patterns = [p.strip() for p in args.exclude.split(",")]

            result = await index_directory(
                directory_path=args.path,
                recursive=not args.no_recursive,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                max_files=args.max_files,
                database_url=args.database_url,
                force_reindex=args.force,
            )

        elif args.index_command == "manage":
            # Manage indexes
            result = await manage_indexes(
                operation=args.operation,
                index_name=getattr(args, "index_name", None),
                database_url=args.database_url,
                m=getattr(args, "m", 16),
                ef_construction=getattr(args, "ef_construction", 64),
                ef_search=getattr(args, "ef_search", 40),
            )

        else:
            result = {"error": f"Unknown index command: {args.index_command}"}

        # Format and display results
        output = format_index_results(result, args.format)
        print(output)

    except Exception as e:
        if args.verbose:
            logger.error("Index command failed", error=str(e), exc_info=True)
            print(f"Error: {e}", file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def add_index_command(subparsers: argparse._SubParsersAction) -> None:
    """Add index command to argument parser."""
    index_parser = subparsers.add_parser(
        "index",
        help="Content indexing and index management",
        description="Index content for semantic search and manage database indexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    index_subparsers = index_parser.add_subparsers(
        dest="index_command",
        help="Index commands",
        required=True,
    )

    # File indexing command
    file_parser = index_subparsers.add_parser(
        "file",
        help="Index a single file",
        description="Index a single file for semantic search",
    )

    file_parser.add_argument("path", help="Path to file to index")
    file_parser.add_argument(
        "--content-type",
        help="Override content type detection",
    )
    file_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing if already indexed",
    )

    # Directory indexing command
    dir_parser = index_subparsers.add_parser(
        "directory",
        help="Index all files in a directory",
        description="Index all matching files in a directory",
    )

    dir_parser.add_argument("path", help="Path to directory to index")
    dir_parser.add_argument(
        "--patterns",
        default="*.c,*.h,*.py,*.md,*.txt",
        help="File patterns to include (comma-separated)",
    )
    dir_parser.add_argument(
        "--exclude",
        default="*.o,*.so,*.pyc,__pycache__,.git,.svn",
        help="File patterns to exclude (comma-separated)",
    )
    dir_parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to index",
    )
    dir_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't include subdirectories",
    )
    dir_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing of existing files",
    )

    # Index management command
    manage_parser = index_subparsers.add_parser(
        "manage",
        help="Manage database indexes",
        description="Optimize, rebuild, or analyze database indexes",
    )

    manage_parser.add_argument(
        "operation",
        choices=["optimize", "rebuild", "stats", "vacuum", "optimize_all"],
        help="Management operation to perform",
    )
    manage_parser.add_argument(
        "--index-name",
        help="Specific index name (for rebuild operation)",
    )
    manage_parser.add_argument(
        "--m",
        type=int,
        default=16,
        help="HNSW parameter: max connections per node",
    )
    manage_parser.add_argument(
        "--ef-construction",
        type=int,
        default=64,
        help="HNSW parameter: construction time",
    )
    manage_parser.add_argument(
        "--ef-search",
        type=int,
        default=40,
        help="HNSW parameter: search time",
    )

    # Common arguments for all subcommands
    for parser in [file_parser, dir_parser, manage_parser]:
        parser.add_argument(
            "--format",
            "-f",
            choices=["pretty", "json", "summary"],
            default="pretty",
            help="Output format",
        )
        parser.add_argument(
            "--database-url",
            help="Database connection URL (overrides environment)",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose error output",
        )

    # Set command function
    def sync_index_command(args: argparse.Namespace) -> NoReturn:
        """Synchronous wrapper for async index command."""
        try:
            asyncio.run(index_command(args))
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nIndexing cancelled", file=sys.stderr)
            sys.exit(130)

    index_parser.set_defaults(func=sync_index_command)


# For testing purposes
if __name__ == "__main__":
    # Simple test interface
    parser = argparse.ArgumentParser(description="Test index command")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_index_command(subparsers)

    args = parser.parse_args()
    args.func(args)
