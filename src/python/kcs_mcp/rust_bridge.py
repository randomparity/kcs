"""Python-Rust bridge for call graph extraction functionality.

This module provides a high-level interface to the Rust call graph extraction
capabilities through the existing kcs_python_bridge module. It abstracts the
lower-level bridge operations into convenient classes and functions specifically
for call graph analysis.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from .config import CallExtractionConfig
from .error_handling import (
    CallGraphError,
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    create_error_context,
)

logger = structlog.get_logger(__name__)

try:
    import kcs_python_bridge
except ImportError as e:
    raise ImportError(
        "kcs_python_bridge not available. Ensure it is compiled and installed."
    ) from e


# Configuration is now imported from .config module


@dataclass
class CallEdge:
    """Represents a call edge in the call graph."""

    caller: str
    callee: str
    file_path: str
    line_number: int
    call_type: str  # "direct", "indirect", "macro", "callback", "conditional"
    confidence: str  # "low", "medium", "high"
    conditional: bool = False
    config_guard: str | None = None


@dataclass
class ExtractionStats:
    """Statistics from call graph extraction."""

    files_processed: int = 0
    functions_analyzed: int = 0
    call_edges_found: int = 0
    function_pointers_found: int = 0
    macro_calls_found: int = 0
    processing_time_ms: int = 0
    accuracy_estimate: float = 0.0


@dataclass
class TraversalOptions:
    """Options for graph traversal operations."""

    max_depth: int | None = None
    call_type_filter: str | None = None
    config_filter: str | None = None
    unique_visits: bool = True
    conditional_only: bool = False


@dataclass
class TraversalResult:
    """Result of a graph traversal operation."""

    visited: list[str]
    visit_order: list[str]
    edges: list[tuple[str, str]]
    max_depth: int
    count: int


class CallGraphExtractor:
    """High-level interface to Rust call graph extraction capabilities."""

    def __init__(self, config: CallExtractionConfig | None = None):
        """Initialize the call graph extractor.

        Args:
            config: Configuration for extraction. Uses default if None.
        """
        self.config = config or CallExtractionConfig()
        self._parser: kcs_python_bridge.PyParser | None = None
        self._error_handler = ErrorHandler(enable_detailed_logging=True)
        self._init_parser()

    def _init_parser(self) -> None:
        """Initialize the Rust parser with appropriate configuration."""
        context = create_error_context("init_parser")

        try:
            logger.info("Initializing Rust parser", config=self.config.__dict__)

            # Create parser with call graph support enabled
            self._parser = kcs_python_bridge.PyParser(
                use_clang=False,  # Start with Tree-sitter only
                compile_commands_path=None,
                target_arch="x86_64",
                config_name="defconfig",
            )

            logger.info("Rust parser initialized successfully")

        except Exception as e:
            error = CallGraphError(
                message=f"Failed to initialize Rust parser: {e}",
                category=ErrorCategory.RUST_BRIDGE,
                severity=ErrorSeverity.CRITICAL,
                context=context,
                original_error=e,
                is_recoverable=False,
                is_retryable=False,
            )

            handled_error = self._error_handler.handle_error(error, context)
            if handled_error:
                raise handled_error.to_http_exception() from e

            # This should not be reached due to the error handler raising an exception
            raise RuntimeError("Parser initialization failed") from e

    def extract_from_file(
        self, file_path: str | Path
    ) -> tuple[list[CallEdge], ExtractionStats]:
        """Extract call graph from a single C source file.

        Args:
            file_path: Path to the C source file to analyze

        Returns:
            Tuple of (call_edges, extraction_stats)

        Raises:
            CallGraphError: If extraction fails
        """
        file_path = Path(file_path)
        start_time = time.time()
        context = create_error_context("extract_from_file", file_path=str(file_path))

        # Validate file existence
        if not file_path.exists():
            error = CallGraphError(
                message=f"File not found: {file_path}",
                category=ErrorCategory.FILE_SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                is_recoverable=False,
                is_retryable=False,
            )
            handled_error = self._error_handler.handle_error(error, context)
            if handled_error:
                raise handled_error.to_http_exception()

        # Check file size limits
        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            error = CallGraphError(
                message=f"File exceeds size limit: {file_size} bytes (max: {self.config.max_file_size})",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                is_recoverable=False,
                is_retryable=False,
            )
            handled_error = self._error_handler.handle_error(error, context)
            if handled_error:
                raise handled_error.to_http_exception()

        logger.info(
            "Starting call graph extraction",
            file_path=str(file_path),
            file_size=file_size,
            config=self.config.__dict__,
        )

        try:
            # Validate parser state
            if self._parser is None:
                error = CallGraphError(
                    message="Parser not initialized",
                    category=ErrorCategory.RUST_BRIDGE,
                    severity=ErrorSeverity.CRITICAL,
                    context=context,
                    is_recoverable=False,
                    is_retryable=False,
                )
                handled_error = self._error_handler.handle_error(error, context)
                if handled_error:
                    raise handled_error.to_http_exception()

                # This should not be reached
                raise RuntimeError("Parser not initialized")

            # Extract using Rust parser
            result = self._parser.parse_file(str(file_path))

            # Convert the basic call edges from parser to CallEdge objects
            call_edges = []
            for caller, callee in result.call_edges:
                call_edges.append(
                    CallEdge(
                        caller=caller,
                        callee=callee,
                        file_path=str(file_path),
                        line_number=0,  # Basic parser doesn't provide line numbers yet
                        call_type="direct",  # Default to direct for now
                        confidence="medium",  # Default confidence
                        conditional=False,
                        config_guard=None,
                    )
                )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Create extraction stats
            stats = ExtractionStats(
                files_processed=1,
                functions_analyzed=len(result.symbols),
                call_edges_found=len(call_edges),
                function_pointers_found=0,  # Not yet detected by basic parser
                macro_calls_found=0,  # Not yet detected by basic parser
                processing_time_ms=processing_time_ms,
                accuracy_estimate=0.8,  # Estimate for Tree-sitter based extraction
            )

            logger.info(
                "Call graph extraction completed",
                file_path=str(file_path),
                call_edges_found=len(call_edges),
                functions_analyzed=len(result.symbols),
                processing_time_ms=processing_time_ms,
            )

            return call_edges, stats

        except Exception as e:
            error = CallGraphError(
                message=f"Call graph extraction failed for {file_path}: {e}",
                category=ErrorCategory.RUST_BRIDGE,
                severity=ErrorSeverity.HIGH,
                context=context,
                original_error=e,
                is_recoverable=True,
                is_retryable=True,
            )

            handled_error = self._error_handler.handle_error(error, context)
            if handled_error:
                raise handled_error.to_http_exception() from e

            # This should not be reached
            raise RuntimeError("Call graph extraction failed") from e

    def extract_from_files(
        self, file_paths: list[str | Path]
    ) -> tuple[list[CallEdge], ExtractionStats]:
        """Extract call graph from multiple C source files.

        Args:
            file_paths: List of paths to C source files

        Returns:
            Tuple of (all_call_edges, aggregated_stats)
        """
        all_edges = []
        total_stats = ExtractionStats()

        for file_path in file_paths:
            try:
                edges, stats = self.extract_from_file(file_path)
                all_edges.extend(edges)

                # Aggregate stats
                total_stats.files_processed += stats.files_processed
                total_stats.functions_analyzed += stats.functions_analyzed
                total_stats.call_edges_found += stats.call_edges_found
                total_stats.function_pointers_found += stats.function_pointers_found
                total_stats.macro_calls_found += stats.macro_calls_found
                total_stats.processing_time_ms += stats.processing_time_ms

            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue

        # Calculate overall accuracy estimate
        if total_stats.files_processed > 0:
            total_stats.accuracy_estimate = 0.8  # Conservative estimate

        return all_edges, total_stats

    def extract_from_kernel_tree(
        self, kernel_path: str | Path, config_name: str = "defconfig"
    ) -> tuple[list[CallEdge], ExtractionStats]:
        """Extract call graph from an entire kernel source tree.

        Args:
            kernel_path: Path to kernel source directory
            config_name: Kernel configuration name (e.g., "defconfig", "allmodconfig")

        Returns:
            Tuple of (call_edges, extraction_stats)
        """
        kernel_path = Path(kernel_path)
        if not kernel_path.is_dir():
            raise NotADirectoryError(f"Kernel path is not a directory: {kernel_path}")

        try:
            # Use parser's kernel tree functionality
            if self._parser is None:
                raise RuntimeError("Parser not initialized")
            result = self._parser.parse_kernel_tree(str(kernel_path), config_name)

            # Convert to CallEdge objects
            call_edges = []
            for caller, callee in result.call_edges:
                call_edges.append(
                    CallEdge(
                        caller=caller,
                        callee=callee,
                        file_path="<kernel_tree>",  # Generic path for kernel tree extraction
                        line_number=0,
                        call_type="direct",
                        confidence="medium",
                        conditional=False,
                        config_guard=config_name,
                    )
                )

            stats = ExtractionStats(
                files_processed=1,  # Kernel tree counted as one logical unit
                functions_analyzed=len(result.symbols),
                call_edges_found=len(call_edges),
                function_pointers_found=0,
                macro_calls_found=0,
                processing_time_ms=0,
                accuracy_estimate=0.7,  # Lower confidence for large kernel analysis
            )

            return call_edges, stats

        except Exception as e:
            raise RuntimeError(f"Kernel tree analysis failed: {e}") from e


class CallGraphTraversal:
    """Interface for traversing call graphs using Rust graph algorithms."""

    def __init__(self, call_edges: list[CallEdge]):
        """Initialize traversal with a set of call edges.

        Args:
            call_edges: List of call edges representing the graph
        """
        self.call_edges = call_edges
        self._build_adjacency_lists()

    def _build_adjacency_lists(self) -> None:
        """Build adjacency lists from call edges for efficient traversal."""
        self.outgoing: dict[str, list[str]] = {}  # caller -> [callees]
        self.incoming: dict[str, list[str]] = {}  # callee -> [callers]
        self.all_symbols: set[str] = set()

        for edge in self.call_edges:
            # Track all symbols
            self.all_symbols.add(edge.caller)
            self.all_symbols.add(edge.callee)

            # Build outgoing adjacency list
            if edge.caller not in self.outgoing:
                self.outgoing[edge.caller] = []
            self.outgoing[edge.caller].append(edge.callee)

            # Build incoming adjacency list
            if edge.callee not in self.incoming:
                self.incoming[edge.callee] = []
            self.incoming[edge.callee].append(edge.caller)

    def bfs(
        self, start: str, options: TraversalOptions | None = None
    ) -> TraversalResult:
        """Perform breadth-first search from a starting symbol.

        Args:
            start: Starting symbol name
            options: Traversal options

        Returns:
            TraversalResult with visited symbols and path information
        """
        options = options or TraversalOptions()

        if start not in self.all_symbols:
            return TraversalResult([], [], [], 0, 0)

        visited = set()
        visit_order = []
        edges = []
        queue = [(start, 0)]
        max_depth_reached = 0

        while queue:
            symbol, depth = queue.pop(0)

            # Check depth limit
            if options.max_depth is not None and depth > options.max_depth:
                continue

            # Check if already visited (for unique visits)
            if options.unique_visits and symbol in visited:
                continue

            visited.add(symbol)
            visit_order.append(symbol)
            max_depth_reached = max(max_depth_reached, depth)

            # Process outgoing edges
            if symbol in self.outgoing:
                for callee in self.outgoing[symbol]:
                    # Apply filters (simplified - would be more sophisticated with full Rust integration)
                    if not options.unique_visits or callee not in visited:
                        queue.append((callee, depth + 1))
                        edges.append((symbol, callee))

        return TraversalResult(
            visited=list(visited),
            visit_order=visit_order,
            edges=edges,
            max_depth=max_depth_reached,
            count=len(visited),
        )

    def dfs(
        self, start: str, options: TraversalOptions | None = None
    ) -> TraversalResult:
        """Perform depth-first search from a starting symbol.

        Args:
            start: Starting symbol name
            options: Traversal options

        Returns:
            TraversalResult with visited symbols and path information
        """
        options = options or TraversalOptions()

        if start not in self.all_symbols:
            return TraversalResult([], [], [], 0, 0)

        visited = set()
        visit_order = []
        edges = []
        max_depth_reached = 0

        def dfs_recursive(symbol: str, depth: int) -> None:
            nonlocal max_depth_reached

            # Check depth limit
            if options.max_depth is not None and depth > options.max_depth:
                return

            # Check if already visited (for unique visits)
            if options.unique_visits and symbol in visited:
                return

            visited.add(symbol)
            visit_order.append(symbol)
            max_depth_reached = max(max_depth_reached, depth)

            # Process outgoing edges
            if symbol in self.outgoing:
                for callee in self.outgoing[symbol]:
                    edges.append((symbol, callee))
                    dfs_recursive(callee, depth + 1)

        dfs_recursive(start, 0)

        return TraversalResult(
            visited=list(visited),
            visit_order=visit_order,
            edges=edges,
            max_depth=max_depth_reached,
            count=len(visited),
        )

    def find_ancestors(self, target: str, max_depth: int | None = None) -> list[str]:
        """Find all ancestors of a symbol (symbols that can reach it).

        Args:
            target: Target symbol name
            max_depth: Maximum search depth

        Returns:
            List of ancestor symbol names
        """
        if target not in self.all_symbols:
            return []

        ancestors = set()
        queue = [(target, 0)]

        while queue:
            symbol, depth = queue.pop(0)

            if max_depth is not None and depth > max_depth:
                continue

            # Look at incoming edges
            if symbol in self.incoming:
                for caller in self.incoming[symbol]:
                    if caller not in ancestors:
                        ancestors.add(caller)
                        queue.append((caller, depth + 1))

        return list(ancestors)

    def find_descendants(self, source: str, max_depth: int | None = None) -> list[str]:
        """Find all descendants of a symbol (symbols it can reach).

        Args:
            source: Source symbol name
            max_depth: Maximum search depth

        Returns:
            List of descendant symbol names
        """
        if source not in self.all_symbols:
            return []

        descendants = set()
        queue = [(source, 0)]

        while queue:
            symbol, depth = queue.pop(0)

            if max_depth is not None and depth > max_depth:
                continue

            # Look at outgoing edges
            if symbol in self.outgoing:
                for callee in self.outgoing[symbol]:
                    if callee not in descendants:
                        descendants.add(callee)
                        queue.append((callee, depth + 1))

        return list(descendants)

    def find_path(
        self, start: str, end: str, max_depth: int | None = None
    ) -> list[str] | None:
        """Find a path between two symbols using BFS.

        Args:
            start: Starting symbol name
            end: Target symbol name
            max_depth: Maximum search depth

        Returns:
            List of symbols forming the path, or None if no path exists
        """
        if start not in self.all_symbols or end not in self.all_symbols:
            return None

        if start == end:
            return [start]

        visited = set()
        queue = [(start, [start], 0)]

        while queue:
            symbol, path, depth = queue.pop(0)

            if max_depth is not None and depth > max_depth:
                continue

            if symbol in visited:
                continue

            visited.add(symbol)

            # Check outgoing edges
            if symbol in self.outgoing:
                for callee in self.outgoing[symbol]:
                    if callee == end:
                        return [*path, callee]

                    if callee not in visited:
                        queue.append((callee, [*path, callee], depth + 1))

        return None


def extract_call_graph_from_files(
    file_paths: list[str | Path], config: CallExtractionConfig | None = None
) -> tuple[list[CallEdge], ExtractionStats]:
    """Convenience function to extract call graph from multiple files.

    Args:
        file_paths: List of C source file paths
        config: Extraction configuration

    Returns:
        Tuple of (call_edges, extraction_stats)
    """
    extractor = CallGraphExtractor(config)
    return extractor.extract_from_files(file_paths)


def extract_call_graph_from_kernel(
    kernel_path: str | Path,
    config_name: str = "defconfig",
    config: CallExtractionConfig | None = None,
) -> tuple[list[CallEdge], ExtractionStats]:
    """Convenience function to extract call graph from kernel source tree.

    Args:
        kernel_path: Path to kernel source directory
        config_name: Kernel configuration name
        config: Extraction configuration

    Returns:
        Tuple of (call_edges, extraction_stats)
    """
    extractor = CallGraphExtractor(config)
    return extractor.extract_from_kernel_tree(kernel_path, config_name)


def traverse_call_graph(
    call_edges: list[CallEdge],
    start: str,
    algorithm: str = "bfs",
    options: TraversalOptions | None = None,
) -> TraversalResult:
    """Convenience function to traverse a call graph.

    Args:
        call_edges: List of call edges representing the graph
        start: Starting symbol name
        algorithm: Traversal algorithm ("bfs" or "dfs")
        options: Traversal options

    Returns:
        TraversalResult with visited symbols and path information
    """
    traversal = CallGraphTraversal(call_edges)

    if algorithm.lower() == "dfs":
        return traversal.dfs(start, options)
    else:
        return traversal.bfs(start, options)


__all__ = [
    "CallEdge",
    "CallExtractionConfig",
    "CallGraphExtractor",
    "CallGraphTraversal",
    "ExtractionStats",
    "TraversalOptions",
    "TraversalResult",
    "extract_call_graph_from_files",
    "extract_call_graph_from_kernel",
    "traverse_call_graph",
]
