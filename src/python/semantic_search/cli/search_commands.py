"""
CLI search command implementation for semantic search engine.

Provides command-line interface for performing semantic searches with
various filtering and output options.
"""

import argparse
import asyncio
import json
import sys
from typing import Any, NoReturn

import structlog

from ..database.connection import DatabaseConfig, init_database_connection
from ..models.search_query import SearchQuery
from ..services.ranking_service import RankingService
from ..services.vector_search_service import SearchFilters, VectorSearchService

logger = structlog.get_logger(__name__)


async def execute_search(
    query_text: str,
    max_results: int = 20,
    similarity_threshold: float = 0.0,
    content_types: list[str] | None = None,
    file_paths: list[str] | None = None,
    path_patterns: list[str] | None = None,
    include_context: bool = True,
    config_context: list[str] | None = None,
    user_id: str | None = None,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """
    Execute semantic search query.

    Args:
        query_text: Search query text
        max_results: Maximum results to return
        similarity_threshold: Minimum similarity score
        content_types: Filter by content types
        file_paths: Filter by specific file paths
        path_patterns: Filter by path patterns
        include_context: Include context lines
        config_context: Kernel configuration filters
        user_id: User identifier
        database_url: Database connection URL

    Returns:
        List of search results as dictionaries
    """
    try:
        # Initialize database connection
        if database_url:
            db_config = DatabaseConfig.from_url(database_url)
        else:
            db_config = DatabaseConfig.from_env()

        await init_database_connection(db_config)

        # Create search query
        search_query = SearchQuery(
            query_text=query_text,
            user_id=user_id,
            config_context=config_context or [],
        )

        # Initialize services
        vector_service = VectorSearchService(db_config.to_url())
        await vector_service.initialize()

        ranking_service = RankingService()

        try:
            # Define search filters
            filters = SearchFilters(
                content_types=content_types,
                file_paths=file_paths,
                path_patterns=path_patterns,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                include_context=include_context,
            )

            # Perform vector search
            search_results = await vector_service.similarity_search(
                search_query.embedding, filters
            )

            # Apply hybrid ranking - convert SearchResult to tuple format
            search_tuples = [
                (result.content_id, result.similarity_score, "")
                for result in search_results
            ]
            ranked_results = ranking_service.rank_results(
                search_query.processed_query, search_tuples, len(search_results)
            )

            # Convert to dictionaries for output
            results_dict = []
            for result in ranked_results:
                result_dict = {
                    "content_id": result.content_id,
                    "similarity_score": result.similarity_score,
                    "bm25_score": result.bm25_score,
                    "combined_score": result.combined_score,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                }

                if include_context:
                    result_dict["context_lines"] = result.context_lines

                results_dict.append(result_dict)

            return results_dict

        finally:
            await vector_service.close()

    except Exception as e:
        logger.error("Search execution failed", error=str(e), exc_info=True)
        raise


def format_search_results(
    results: list[dict[str, Any]], output_format: str = "pretty"
) -> str:
    """
    Format search results for display.

    Args:
        results: Search results to format
        output_format: Output format ('pretty', 'json', 'compact')

    Returns:
        Formatted results string
    """
    if not results:
        return "No results found."

    if output_format == "json":
        return json.dumps(results, indent=2)

    elif output_format == "compact":
        output_lines = []
        for i, result in enumerate(results, 1):
            output_lines.append(
                f"{i}. {result['explanation']} (score: {result['combined_score']:.3f})"
            )
        return "\n".join(output_lines)

    else:  # pretty format
        output_lines = []
        output_lines.append(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            output_lines.append(f"=== Result {i} ===")
            output_lines.append(f"Content ID: {result['content_id']}")
            output_lines.append(f"Similarity: {result['similarity_score']:.3f}")
            output_lines.append(f"BM25: {result['bm25_score']:.3f}")
            output_lines.append(f"Combined Score: {result['combined_score']:.3f}")
            output_lines.append(f"Confidence: {result['confidence']:.3f}")
            output_lines.append(f"Explanation: {result['explanation']}")

            if result.get("context_lines"):
                output_lines.append("Context:")
                for line in result["context_lines"]:
                    output_lines.append(f"  {line}")

            output_lines.append("")  # Empty line between results

        return "\n".join(output_lines)


async def search_command(args: argparse.Namespace) -> None:
    """Execute search command with given arguments."""
    try:
        # Parse content types
        content_types = None
        if args.content_types:
            content_types = [ct.strip() for ct in args.content_types.split(",")]

        # Parse file paths
        file_paths = None
        if args.file_paths:
            file_paths = [fp.strip() for fp in args.file_paths.split(",")]

        # Parse path patterns
        path_patterns = None
        if args.path_patterns:
            path_patterns = [pp.strip() for pp in args.path_patterns.split(",")]

        # Parse config context
        config_context = None
        if args.config_context:
            config_context = [cc.strip() for cc in args.config_context.split(",")]

        # Execute search
        results = await execute_search(
            query_text=args.query,
            max_results=args.max_results,
            similarity_threshold=args.threshold,
            content_types=content_types,
            file_paths=file_paths,
            path_patterns=path_patterns,
            include_context=not args.no_context,
            config_context=config_context,
            user_id=args.user_id,
            database_url=args.database_url,
        )

        # Format and display results
        output = format_search_results(results, args.format)
        print(output)

    except Exception as e:
        if args.verbose:
            logger.error("Search command failed", error=str(e), exc_info=True)
            print(f"Error: {e}", file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def add_search_command(subparsers: argparse._SubParsersAction) -> None:
    """Add search command to argument parser."""
    search_parser = subparsers.add_parser(
        "search",
        help="Perform semantic search",
        description="Search through indexed content using semantic similarity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    search_parser.add_argument(
        "query",
        help="Search query text",
    )

    # Search filtering options
    search_parser.add_argument(
        "--max-results",
        "-n",
        type=int,
        default=20,
        help="Maximum number of results to return",
    )

    search_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.0,
        help="Minimum similarity threshold (0.0-1.0)",
    )

    search_parser.add_argument(
        "--content-types",
        help="Filter by content types (comma-separated)",
    )

    search_parser.add_argument(
        "--file-paths",
        help="Filter by specific file paths (comma-separated)",
    )

    search_parser.add_argument(
        "--path-patterns",
        help="Filter by path patterns (comma-separated, SQL LIKE syntax)",
    )

    search_parser.add_argument(
        "--config-context",
        help="Kernel configuration filters (comma-separated, e.g., CONFIG_KALLSYMS,!CONFIG_DEBUG)",
    )

    # Output options
    search_parser.add_argument(
        "--format",
        "-f",
        choices=["pretty", "json", "compact"],
        default="pretty",
        help="Output format",
    )

    search_parser.add_argument(
        "--no-context",
        action="store_true",
        help="Don't include context lines in results",
    )

    # Authentication and database
    search_parser.add_argument(
        "--user-id",
        help="User identifier for the search",
    )

    search_parser.add_argument(
        "--database-url",
        help="Database connection URL (overrides environment)",
    )

    # Debugging
    search_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose error output",
    )

    # Set command function
    def sync_search_command(args: argparse.Namespace) -> NoReturn:
        """Synchronous wrapper for async search command."""
        try:
            asyncio.run(search_command(args))
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nSearch cancelled", file=sys.stderr)
            sys.exit(130)

    search_parser.set_defaults(func=sync_search_command)


# For testing purposes
if __name__ == "__main__":
    # Simple test interface
    parser = argparse.ArgumentParser(description="Test search command")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_search_command(subparsers)

    args = parser.parse_args()
    args.func(args)
