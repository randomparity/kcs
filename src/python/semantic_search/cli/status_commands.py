"""
CLI status command implementation for semantic search engine.

Provides command-line interface for checking system status, health,
and operational metrics of the semantic search system.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any, NoReturn

import structlog

from ..database.connection import DatabaseConfig, init_database_connection
from ..database.index_manager import IndexManager
from ..database.vector_store import VectorStore
from ..services.vector_search_service import VectorSearchService

logger = structlog.get_logger(__name__)


async def get_system_status(
    database_url: str | None = None,
    include_detailed: bool = False,
) -> dict[str, Any]:
    """
    Get comprehensive system status.

    Args:
        database_url: Database connection URL
        include_detailed: Include detailed diagnostics

    Returns:
        System status dictionary
    """
    try:
        # Initialize database connection
        if database_url:
            db_config = DatabaseConfig.from_url(database_url)
        else:
            db_config = DatabaseConfig.from_env()

        connection = await init_database_connection(db_config)

        status: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "database": {},
            "indexes": {},
            "content": {},
            "services": {},
            "overall_health": "unknown",
        }

        # Database health check
        try:
            db_health = await connection.health_check()
            status["database"] = {
                "connected": db_health.get("healthy", False),
                "response_time_ms": db_health.get("response_time_ms", 0),
                "pool_stats": db_health.get("pool", {}),
                "host": db_health.get("host", "unknown"),
                "database": db_health.get("database", "unknown"),
            }
        except Exception as e:
            status["database"] = {
                "connected": False,
                "error": str(e),
            }

        # Index manager status
        try:
            index_manager = IndexManager()

            # Get index statistics
            index_stats = await index_manager.get_index_statistics()
            performance_metrics = await index_manager.get_performance_metrics()

            status["indexes"] = {
                "total_indexes": len(index_stats),
                "cache_hit_ratio": performance_metrics.get("cache_hit_ratio", 0),
                "active_connections": performance_metrics.get("active_connections", 0),
                "total_index_size": performance_metrics.get("total_index_size", 0),
            }

            if include_detailed:
                status["indexes"]["detailed_stats"] = [
                    stat.dict() for stat in index_stats
                ]
                status["indexes"]["performance_metrics"] = performance_metrics

        except Exception as e:
            status["indexes"] = {
                "error": str(e),
            }

        # Content statistics
        try:
            VectorStore()

            # Get content counts by status - this method doesn't exist yet
            # TODO: Implement get_content_statistics in VectorStore
            content_stats = {"total_content": 0, "indexed_content": 0}

            status["content"] = {
                "total_content": content_stats.get("total_content", 0),
                "indexed_content": content_stats.get("indexed_content", 0),
                "pending_content": content_stats.get("pending_content", 0),
                "failed_content": content_stats.get("failed_content", 0),
                "content_types": content_stats.get("content_types", []),
                "total_embeddings": content_stats.get("total_embeddings", 0),
            }

            if include_detailed:
                status["content"]["detailed_stats"] = content_stats

        except Exception as e:
            status["content"] = {
                "error": str(e),
            }

        # Service health checks
        try:
            # Vector search service
            vector_service = VectorSearchService(db_config.to_url())
            await vector_service.initialize()

            try:
                embedding_stats = await vector_service.get_embedding_stats()
                status["services"]["vector_search"] = {
                    "available": True,
                    "health_status": embedding_stats.get("health_status", "unknown"),
                    "total_embeddings": embedding_stats.get("total_embeddings", 0),
                    "unique_content": embedding_stats.get("unique_content", 0),
                }
            finally:
                await vector_service.close()

        except Exception as e:
            status["services"]["vector_search"] = {
                "available": False,
                "error": str(e),
            }

        # Determine overall health
        overall_health = "healthy"

        if not status["database"].get("connected", False):
            overall_health = "critical"
        elif status["indexes"].get("error") or status["content"].get("error"):
            overall_health = "degraded"
        elif not status["services"]["vector_search"].get("available", False):
            overall_health = "degraded"
        elif status["services"]["vector_search"].get("health_status") == "degraded":
            overall_health = "degraded"

        status["overall_health"] = overall_health

        return status

    except Exception as e:
        logger.error("Failed to get system status", error=str(e), exc_info=True)
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "critical",
            "error": str(e),
        }


async def get_search_performance_metrics(
    database_url: str | None = None,
    sample_queries: list[str] | None = None,
) -> dict[str, Any]:
    """
    Get search performance metrics.

    Args:
        database_url: Database connection URL
        sample_queries: Sample queries to test performance

    Returns:
        Performance metrics dictionary
    """
    try:
        # Initialize database connection
        if database_url:
            db_config = DatabaseConfig.from_url(database_url)
        else:
            db_config = DatabaseConfig.from_env()

        await init_database_connection(db_config)

        # Default sample queries if none provided
        if not sample_queries:
            sample_queries = [
                "memory allocation",
                "buffer overflow",
                "kernel configuration",
                "system call",
                "error handling",
            ]

        # Initialize services
        vector_service = VectorSearchService(db_config.to_url())
        await vector_service.initialize()

        try:
            from ..models.search_query import SearchQuery
            from ..services.vector_search_service import SearchFilters

            metrics: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(sample_queries),
                "query_results": [],
                "average_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "success_rate": 0,
            }

            response_times = []
            successful_queries = 0

            for query_text in sample_queries:
                try:
                    start_time = datetime.now()

                    # Create search query
                    search_query = SearchQuery(query_text=query_text)

                    # Perform search
                    filters = SearchFilters(
                        content_types=None,
                        file_paths=None,
                        path_patterns=None,
                        max_results=5,
                        similarity_threshold=0.0,
                        include_context=False,
                    )
                    results = await vector_service.similarity_search(
                        search_query.embedding, filters
                    )

                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds() * 1000

                    response_times.append(response_time)
                    successful_queries += 1

                    metrics["query_results"].append(
                        {
                            "query": query_text,
                            "response_time_ms": round(response_time, 2),
                            "results_count": len(results),
                            "success": True,
                        }
                    )

                except Exception as e:
                    metrics["query_results"].append(
                        {
                            "query": query_text,
                            "error": str(e),
                            "success": False,
                        }
                    )

            # Calculate statistics
            if response_times:
                metrics["average_response_time_ms"] = round(
                    sum(response_times) / len(response_times), 2
                )

                # Calculate p95
                sorted_times = sorted(response_times)
                p95_index = int(0.95 * len(sorted_times))
                metrics["p95_response_time_ms"] = round(
                    sorted_times[min(p95_index, len(sorted_times) - 1)], 2
                )

            metrics["success_rate"] = round(
                (successful_queries / len(sample_queries)) * 100, 1
            )

            return metrics

        finally:
            await vector_service.close()

    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e), exc_info=True)
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


async def get_index_health(database_url: str | None = None) -> dict[str, Any]:
    """
    Get detailed index health information.

    Args:
        database_url: Database connection URL

    Returns:
        Index health dictionary
    """
    try:
        # Initialize database connection
        if database_url:
            db_config = DatabaseConfig.from_url(database_url)
        else:
            db_config = DatabaseConfig.from_env()

        await init_database_connection(db_config)

        index_manager = IndexManager()

        # Get comprehensive index analysis
        usage_analysis = await index_manager.check_index_usage()
        performance_metrics = await index_manager.get_performance_metrics()

        health = {
            "timestamp": datetime.now().isoformat(),
            "index_usage": usage_analysis,
            "performance": performance_metrics,
            "health_score": 0,
            "recommendations": [],
        }

        # Calculate health score (0-100)
        score = 100

        # Deduct points for unused indexes
        unused_count = len(usage_analysis.get("unused_indexes", []))
        score -= min(unused_count * 5, 25)  # Max 25 points deduction

        # Deduct points for low cache hit ratio
        cache_hit_ratio = performance_metrics.get("cache_hit_ratio", 0)
        if cache_hit_ratio < 90:
            score -= (90 - cache_hit_ratio) * 2  # 2 points per % below 90%

        # Deduct points for tables with low index usage
        scan_ratios = usage_analysis.get("scan_ratios", [])
        for ratio in scan_ratios:
            if ratio.get("index_usage_percent", 100) < 50:
                score -= 10

        health["health_score"] = max(0, round(score))
        health["recommendations"] = usage_analysis.get("recommendations", [])

        return health

    except Exception as e:
        logger.error("Failed to get index health", error=str(e), exc_info=True)
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


def format_status_results(
    results: dict[str, Any], output_format: str = "pretty"
) -> str:
    """
    Format status results for display.

    Args:
        results: Results to format
        output_format: Output format ('pretty', 'json', 'summary')

    Returns:
        Formatted results string
    """
    if output_format == "json":
        return json.dumps(results, indent=2, default=str)

    elif output_format == "summary":
        if "overall_health" in results:
            health = results["overall_health"]
            emoji = (
                "✅" if health == "healthy" else "⚠️" if health == "degraded" else "❌"
            )
            return f"{emoji} System Health: {health.upper()}"
        elif "health_score" in results:
            score = results["health_score"]
            emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            return f"{emoji} Index Health Score: {score}/100"
        elif "success_rate" in results:
            rate = results["success_rate"]
            emoji = "✅" if rate >= 90 else "⚠️" if rate >= 70 else "❌"
            return f"{emoji} Search Success Rate: {rate}%"
        else:
            return "Status check completed"

    else:  # pretty format
        output_lines = []

        if "error" in results:
            output_lines.append(f"❌ Error: {results['error']}")
            return "\n".join(output_lines)

        if "overall_health" in results:
            # System status format
            health = results["overall_health"]
            health_emoji = {
                "healthy": "✅",
                "degraded": "⚠️",
                "critical": "❌",
                "unknown": "❓",
            }.get(health, "❓")

            output_lines.extend(
                [
                    "🔍 Semantic Search System Status",
                    "=" * 35,
                    f"{health_emoji} Overall Health: {health.upper()}",
                    f"📅 Timestamp: {results.get('timestamp', 'unknown')}",
                    "",
                ]
            )

            # Database status
            db = results.get("database", {})
            if db:
                db_emoji = "✅" if db.get("connected", False) else "❌"
                output_lines.extend(
                    [
                        "🗄️  Database:",
                        f"  {db_emoji} Connected: {db.get('connected', False)}",
                        f"  ⏱️  Response time: {db.get('response_time_ms', 0):.1f}ms",
                        f"  🏠 Host: {db.get('host', 'unknown')}",
                        "",
                    ]
                )

            # Index status
            indexes = results.get("indexes", {})
            if indexes and "error" not in indexes:
                output_lines.extend(
                    [
                        "📊 Indexes:",
                        f"  📈 Total indexes: {indexes.get('total_indexes', 0)}",
                        f"  💾 Cache hit ratio: {indexes.get('cache_hit_ratio', 0):.1f}%",
                        f"  🔗 Active connections: {indexes.get('active_connections', 0)}",
                        "",
                    ]
                )

            # Content status
            content = results.get("content", {})
            if content and "error" not in content:
                output_lines.extend(
                    [
                        "📝 Content:",
                        f"  📄 Total content: {content.get('total_content', 0):,}",
                        f"  ✅ Indexed: {content.get('indexed_content', 0):,}",
                        f"  ⏳ Pending: {content.get('pending_content', 0):,}",
                        f"  ❌ Failed: {content.get('failed_content', 0):,}",
                        f"  🧬 Total embeddings: {content.get('total_embeddings', 0):,}",
                        "",
                    ]
                )

            # Services status
            services = results.get("services", {})
            if services:
                output_lines.append("🔧 Services:")

                vector_search = services.get("vector_search", {})
                if vector_search:
                    vs_emoji = "✅" if vector_search.get("available", False) else "❌"
                    output_lines.extend(
                        [
                            f"  {vs_emoji} Vector Search: {'Available' if vector_search.get('available', False) else 'Unavailable'}",
                            f"    Health: {vector_search.get('health_status', 'unknown')}",
                        ]
                    )

        elif "health_score" in results:
            # Index health format
            score = results["health_score"]
            score_emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"

            output_lines.extend(
                [
                    "📊 Index Health Report",
                    "=" * 22,
                    f"{score_emoji} Health Score: {score}/100",
                    f"📅 Timestamp: {results.get('timestamp', 'unknown')}",
                    "",
                ]
            )

            recommendations = results.get("recommendations", [])
            if recommendations:
                output_lines.append("💡 Recommendations:")
                for rec in recommendations[:5]:  # Show top 5
                    output_lines.append(f"  • {rec}")
                if len(recommendations) > 5:
                    output_lines.append(
                        f"  ... and {len(recommendations) - 5} more recommendations"
                    )

        elif "success_rate" in results:
            # Performance metrics format
            rate = results["success_rate"]
            rate_emoji = "✅" if rate >= 90 else "⚠️" if rate >= 70 else "❌"

            output_lines.extend(
                [
                    "⚡ Search Performance Metrics",
                    "=" * 28,
                    f"{rate_emoji} Success Rate: {rate}%",
                    f"📊 Total Queries: {results.get('total_queries', 0)}",
                    f"⏱️  Average Response: {results.get('average_response_time_ms', 0):.1f}ms",
                    f"📈 P95 Response: {results.get('p95_response_time_ms', 0):.1f}ms",
                    f"📅 Timestamp: {results.get('timestamp', 'unknown')}",
                    "",
                ]
            )

            query_results = results.get("query_results", [])
            if query_results:
                output_lines.append("📋 Query Details:")
                for result in query_results[:5]:  # Show first 5
                    if result.get("success", False):
                        output_lines.append(
                            f"  ✅ '{result['query']}': {result['response_time_ms']:.1f}ms "
                            f"({result['results_count']} results)"
                        )
                    else:
                        output_lines.append(
                            f"  ❌ '{result['query']}': {result.get('error', 'failed')}"
                        )

        return "\n".join(output_lines)


async def status_command(args: argparse.Namespace) -> None:
    """Execute status command with given arguments."""
    try:
        if args.status_command == "system":
            # Get system status
            result = await get_system_status(
                database_url=args.database_url,
                include_detailed=args.detailed,
            )

        elif args.status_command == "performance":
            # Get performance metrics
            sample_queries = None
            if args.queries:
                sample_queries = [q.strip() for q in args.queries.split(",")]

            result = await get_search_performance_metrics(
                database_url=args.database_url,
                sample_queries=sample_queries,
            )

        elif args.status_command == "indexes":
            # Get index health
            result = await get_index_health(database_url=args.database_url)

        else:
            result = {"error": f"Unknown status command: {args.status_command}"}

        # Format and display results
        output = format_status_results(result, args.format)
        print(output)

        # Exit with error code if health is critical
        if result.get("overall_health") == "critical":
            sys.exit(1)
        elif result.get("overall_health") == "degraded":
            sys.exit(2)

    except Exception as e:
        if args.verbose:
            logger.error("Status command failed", error=str(e), exc_info=True)
            print(f"Error: {e}", file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def add_status_command(subparsers: argparse._SubParsersAction) -> None:
    """Add status command to argument parser."""
    status_parser = subparsers.add_parser(
        "status",
        help="Check system status and health",
        description="Get status, health, and performance information for the semantic search system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    status_subparsers = status_parser.add_subparsers(
        dest="status_command",
        help="Status commands",
        required=True,
    )

    # System status command
    system_parser = status_subparsers.add_parser(
        "system",
        help="Get overall system status",
        description="Check the health and status of all system components",
    )

    system_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed diagnostic information",
    )

    # Performance metrics command
    perf_parser = status_subparsers.add_parser(
        "performance",
        help="Get search performance metrics",
        description="Test search performance with sample queries",
    )

    perf_parser.add_argument(
        "--queries",
        help="Sample queries to test (comma-separated)",
    )

    # Index health command
    index_parser = status_subparsers.add_parser(
        "indexes",
        help="Get index health information",
        description="Check database index health and optimization status",
    )

    # Common arguments for all subcommands
    for parser in [system_parser, perf_parser, index_parser]:
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
    def sync_status_command(args: argparse.Namespace) -> NoReturn:
        """Synchronous wrapper for async status command."""
        try:
            asyncio.run(status_command(args))
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nStatus check cancelled", file=sys.stderr)
            sys.exit(130)

    status_parser.set_defaults(func=sync_status_command)


# For testing purposes
if __name__ == "__main__":
    # Simple test interface
    parser = argparse.ArgumentParser(description="Test status command")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_status_command(subparsers)

    args = parser.parse_args()
    args.func(args)
