"""
Pydantic models for KCS MCP API.

Defines request/response schemas matching the OpenAPI specification.
"""

import typing
from typing import Any

from pydantic import BaseModel, Field, model_validator


# Core data types
class Span(BaseModel):
    """File location span with exact line/column references."""

    path: str = Field(..., description="File path relative to repository root")
    sha: str = Field(
        ..., description="Git SHA of file version", min_length=40, max_length=40
    )
    start: int = Field(..., description="Starting line number", gt=0)
    end: int = Field(..., description="Ending line number", gt=0)

    class Config:
        json_schema_extra: typing.ClassVar[dict[str, typing.Any]] = {
            "example": {
                "path": "fs/read_write.c",
                "sha": "a1b2c3d4e5f6789012345678901234567890abcd",
                "start": 451,
                "end": 465,
            }
        }


class Citation(BaseModel):
    """Citation with optional context."""

    span: Span
    context: str | None = Field(None, description="Optional context around citation")


# Tool request/response models
class SearchCodeRequest(BaseModel):
    """Request for code search."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int | None = Field(
        10, alias="topK", description="Maximum results to return", ge=1, le=100
    )


class SearchHit(BaseModel):
    """Search result hit."""

    span: Span
    snippet: str = Field(..., description="Code snippet with match highlighted")
    score: float | None = Field(None, description="Relevance score")


class SearchCodeResponse(BaseModel):
    """Response for code search."""

    hits: list[SearchHit]


class GetSymbolRequest(BaseModel):
    """Request for symbol information."""

    symbol: str = Field(..., description="Symbol name", min_length=1)


class SymbolInfo(BaseModel):
    """Symbol information response."""

    name: str
    kind: str = Field(
        ...,
        description="Symbol kind",
        pattern="^(function|struct|variable|macro|typedef)$",
    )
    decl: Span = Field(..., description="Declaration location")
    summary: dict[str, Any] | None = Field(
        None, description="Optional AI-generated summary"
    )


class WhoCallsRequest(BaseModel):
    """Request for caller analysis."""

    symbol: str = Field(..., description="Symbol name", min_length=1)
    depth: int | None = Field(1, description="Call graph depth", ge=1, le=10)


class CallerInfo(BaseModel):
    """Information about a function caller."""

    symbol: str
    span: Span
    call_type: str | None = Field(None, pattern="^(direct|indirect|macro|inline)$")


class WhoCallsResponse(BaseModel):
    """Response for caller analysis."""

    callers: list[CallerInfo]


class ListDependenciesRequest(BaseModel):
    """Request for dependency analysis."""

    symbol: str = Field(..., description="Symbol name", min_length=1)
    depth: int | None = Field(1, description="Dependency depth", ge=1, le=10)


class ListDependenciesResponse(BaseModel):
    """Response for dependency analysis."""

    callees: list[CallerInfo]  # Same structure as callers


class EntrypointFlowRequest(BaseModel):
    """Request for entry point flow tracing."""

    entry: str = Field(..., description="Entry point identifier", min_length=1)


class FlowStep(BaseModel):
    """Step in an execution flow."""

    edge: str = Field(..., description="Edge type")
    from_symbol: str = Field(..., alias="from", description="Source symbol")
    to_symbol: str = Field(..., alias="to", description="Target symbol")
    span: Span


class EntrypointFlowResponse(BaseModel):
    """Response for entry point flow tracing."""

    steps: list[FlowStep]


class ImpactOfRequest(BaseModel):
    """Request for impact analysis."""

    diff: str | None = Field(None, description="Git diff content")
    files: list[str] | None = Field(None, description="Files to analyze")
    symbols: list[str] | None = Field(None, description="Symbols to analyze")
    config: str | None = Field(None, description="Configuration context")

    @model_validator(mode="after")
    def validate_at_least_one_input(self) -> "ImpactOfRequest":
        """Ensure at least one input parameter is provided."""
        if not any([self.diff, self.files, self.symbols]):
            raise ValueError(
                "At least one of 'diff', 'files', or 'symbols' must be provided"
            )
        return self


class ImpactResult(BaseModel):
    """Impact analysis result."""

    configs: list[str] = Field(..., description="Affected configurations")
    modules: list[str] = Field(..., description="Affected modules")
    tests: list[str] = Field(..., description="Relevant tests")
    owners: list[str] = Field(..., description="Maintainer emails")
    risks: list[str] = Field(..., description="Risk factors")
    cites: list[Span] = Field(..., description="Supporting citations")


class SearchDocsRequest(BaseModel):
    """Request for documentation search."""

    query: str = Field(..., description="Search query", min_length=1)
    corpus: list[str] | None = Field(None, description="Document collections to search")


class DocHit(BaseModel):
    """Documentation search hit."""

    source: str = Field(..., description="Document source")
    anchor: str = Field(..., description="Section/anchor")
    span: Span | None = Field(None, description="File location if applicable")


class SearchDocsResponse(BaseModel):
    """Response for documentation search."""

    hits: list[DocHit]


class DiffSpecVsCodeRequest(BaseModel):
    """Request for spec vs code drift detection."""

    feature_id: str = Field(..., description="Feature identifier", min_length=1)


class DriftMismatch(BaseModel):
    """Specification vs code mismatch."""

    kind: str = Field(
        ...,
        description="Mismatch type",
        pattern="^(missing_abi_doc|kconfig_mismatch|test_missing|contract_violation)$",
    )
    detail: str = Field(..., description="Detailed description")
    span: Span | None = Field(None, description="Related code location")


class DiffSpecVsCodeResponse(BaseModel):
    """Response for drift detection."""

    mismatches: list[DriftMismatch]


class OwnersForRequest(BaseModel):
    """Request for maintainer information."""

    paths: list[str] | None = Field(None, description="File paths")
    symbols: list[str] | None = Field(None, description="Symbol names")


class MaintainerInfo(BaseModel):
    """Maintainer information."""

    section: str = Field(..., description="MAINTAINERS section")
    emails: list[str] = Field(..., description="Maintainer email addresses")
    paths: list[str] = Field(..., description="Maintained paths")


class OwnersForResponse(BaseModel):
    """Response for maintainer information."""

    maintainers: list[MaintainerInfo]


class ParseKernelConfigRequest(BaseModel):
    """Request for kernel configuration parsing."""

    config_path: str = Field(
        ..., description="Path to kernel config file", min_length=1
    )
    arch: str | None = Field(
        None,
        description="Target architecture",
        pattern="^(x86_64|arm64|arm|riscv|powerpc|s390|mips)$",
    )
    config_name: str | None = Field(None, description="Configuration name")
    incremental: bool | None = Field(False, description="Incremental parsing mode")
    base_config_id: str | None = Field(
        None, description="Base config UUID for incremental"
    )
    filters: dict[str, Any] | None = Field(None, description="Filtering options")
    resolve_dependencies: bool | None = Field(False, description="Resolve dependencies")
    max_depth: int | None = Field(3, description="Max dependency depth", ge=1, le=10)


class ConfigOption(BaseModel):
    """Kernel configuration option."""

    value: str | bool | int | None = Field(..., description="Option value")
    type: str = Field(
        ...,
        description="Option type",
        pattern="^(bool|tristate|string|int|hex)$",
    )


class ConfigDependency(BaseModel):
    """Configuration dependency relationship."""

    option: str = Field(..., description="Dependent option name")
    depends_on: list[str] = Field(..., description="List of dependencies")
    chain: list[str] | None = Field(None, description="Dependency chain")


class ParseKernelConfigResponse(BaseModel):
    """Response for kernel configuration parsing."""

    config_id: str = Field(..., description="Unique configuration identifier (UUID)")
    arch: str = Field(..., description="Target architecture")
    config_name: str = Field(..., description="Configuration name")
    options: dict[str, ConfigOption] = Field(..., description="Configuration options")
    dependencies: list[ConfigDependency] = Field(..., description="Option dependencies")
    parsed_at: str = Field(..., description="Parse timestamp (ISO format)")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    changes: dict[str, Any] | None = Field(
        None, description="Changes for incremental mode"
    )
    diff: dict[str, Any] | None = Field(None, description="Diff for incremental mode")


class SpecificationBehavior(BaseModel):
    """Expected behavior specification."""

    description: str = Field(..., description="Behavior description")
    preconditions: list[str] | None = Field(None, description="Required preconditions")
    postconditions: list[str] | None = Field(
        None, description="Expected postconditions"
    )
    error_conditions: list[str] | None = Field(
        None, description="Expected error conditions"
    )


class SpecificationParameter(BaseModel):
    """Function parameter specification."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str | None = Field(None, description="Parameter description")


class ImplementationHints(BaseModel):
    """Hints for finding implementation."""

    file_pattern: str | None = Field(None, description="File pattern to search")
    subsystem: str | None = Field(None, description="Kernel subsystem")
    related_symbols: list[str] | None = Field(None, description="Related symbols")


class Specification(BaseModel):
    """Specification definition."""

    name: str = Field(..., description="Specification name")
    version: str = Field(
        ..., description="Version (semver)", pattern=r"^\d+\.\d+\.\d+$"
    )
    entry_point: str = Field(..., description="Entry point symbol name")
    expected_behavior: SpecificationBehavior | None = Field(
        None, description="Expected behavior"
    )
    parameters: list[SpecificationParameter] | None = Field(
        None, description="Parameters"
    )
    implementation_hints: ImplementationHints | None = Field(
        None, description="Implementation hints"
    )
    previous_version: str | None = Field(
        None, description="Previous version for comparison"
    )


class ValidateSpecRequest(BaseModel):
    """Request for specification validation."""

    specification: Specification = Field(..., description="Specification to validate")
    specifications: list[Specification] | None = Field(
        None, description="For batch validation"
    )
    kernel_version: str | None = Field(None, description="Target kernel version")
    config: str | None = Field(None, description="Kernel configuration")
    drift_threshold: float | None = Field(
        0.7, description="Compliance threshold (0.0-1.0)", ge=0.0, le=1.0
    )
    include_suggestions: bool | None = Field(
        False, description="Include improvement suggestions"
    )
    compare_with_previous: bool | None = Field(
        False, description="Compare with previous version"
    )


class SpecDeviation(BaseModel):
    """Specification deviation."""

    type: str = Field(
        ...,
        description="Deviation type",
        pattern="^(missing_implementation|behavior_mismatch|parameter_mismatch|error_handling|performance)$",
    )
    severity: str = Field(
        ...,
        description="Deviation severity",
        pattern="^(critical|major|minor|info)$",
    )
    description: str = Field(..., description="Detailed description")
    location: Span | None = Field(None, description="Related code location")


class ImplementationDetails(BaseModel):
    """Implementation details found during validation."""

    entry_point: dict[str, Any] | None = Field(
        None, description="Entry point information"
    )
    call_graph: list[dict[str, Any]] | None = Field(
        None, description="Call graph analysis"
    )
    parameters_found: list[dict[str, Any]] | None = Field(
        None, description="Parameters found"
    )


class ValidationSuggestion(BaseModel):
    """Improvement suggestion."""

    type: str = Field(..., description="Suggestion type")
    description: str = Field(..., description="Suggestion description")
    priority: str | None = Field(None, description="Priority level")


class ValidationComparison(BaseModel):
    """Comparison with previous validation."""

    compliance_delta: float = Field(..., description="Change in compliance score")
    new_deviations: list[SpecDeviation] = Field(..., description="New deviations")
    resolved_deviations: list[SpecDeviation] = Field(
        ..., description="Resolved deviations"
    )


class ValidateSpecResponse(BaseModel):
    """Response for specification validation."""

    validation_id: str = Field(..., description="Unique validation identifier (UUID)")
    specification_id: str = Field(..., description="Specification identifier (UUID)")
    is_valid: bool = Field(..., description="Whether specification is valid")
    compliance_score: float = Field(
        ..., description="Compliance score (0-100)", ge=0, le=100
    )
    deviations: list[SpecDeviation] = Field(..., description="Specification deviations")
    implementation_details: ImplementationDetails = Field(
        ..., description="Implementation details"
    )
    validated_at: str = Field(..., description="Validation timestamp (ISO format)")
    suggestions: list[ValidationSuggestion] | None = Field(
        None, description="Improvement suggestions"
    )
    comparison: ValidationComparison | None = Field(
        None, description="Previous version comparison"
    )


class SemanticSearchFilters(BaseModel):
    """Filters for semantic search."""

    subsystems: list[str] | None = Field(None, description="Subsystem filters")
    file_patterns: list[str] | None = Field(None, description="File pattern filters")
    symbol_types: list[str] | None = Field(None, description="Symbol type filters")
    exclude_tests: bool | None = Field(False, description="Exclude test files")


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""

    query: str = Field(..., description="Search query", min_length=1)
    limit: int | None = Field(10, description="Maximum results", ge=1, le=1000)
    offset: int | None = Field(0, description="Result offset for pagination", ge=0)
    similarity_threshold: float | None = Field(
        0.5, description="Similarity threshold", ge=0.0, le=1.0
    )
    filters: SemanticSearchFilters | None = Field(None, description="Search filters")
    rerank: bool | None = Field(False, description="Enable reranking")
    rerank_model: str | None = Field("cross-encoder", description="Reranking model")
    search_mode: str | None = Field(
        "semantic", description="Search mode", pattern="^(semantic|hybrid|keyword)$"
    )
    keyword_weight: float | None = Field(
        0.3, description="Keyword weight in hybrid mode", ge=0.0, le=1.0
    )
    semantic_weight: float | None = Field(
        0.7, description="Semantic weight in hybrid mode", ge=0.0, le=1.0
    )
    use_cache: bool | None = Field(True, description="Use embeddings cache")
    expand_query: bool | None = Field(
        False, description="Expand query for better recall"
    )
    expansion_terms: int | None = Field(
        5, description="Number of expansion terms", ge=1, le=20
    )
    explain: bool | None = Field(False, description="Include explanations")


class SemanticSearchContext(BaseModel):
    """Context information for search result."""

    subsystem: str | None = Field(None, description="Kernel subsystem")
    function_type: str | None = Field(None, description="Function type")
    related_symbols: list[str] | None = Field(None, description="Related symbols")


class SearchResultExplanation(BaseModel):
    """Explanation for why a result was returned."""

    matching_terms: list[str] | None = Field(None, description="Matching terms")
    relevance_factors: dict[str, Any] | None = Field(
        None, description="Relevance factors"
    )


class SemanticSearchResult(BaseModel):
    """Individual semantic search result."""

    symbol: str = Field(..., description="Symbol name")
    span: Span = Field(..., description="Location in code")
    similarity_score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    snippet: str = Field(..., description="Code snippet")
    context: SemanticSearchContext = Field(..., description="Context information")
    keyword_score: float | None = Field(
        None, description="Keyword matching score", ge=0.0, le=1.0
    )
    hybrid_score: float | None = Field(None, description="Hybrid score", ge=0.0, le=1.0)
    explanation: SearchResultExplanation | None = Field(
        None, description="Result explanation"
    )


class SemanticSearchResponse(BaseModel):
    """Response for semantic search."""

    results: list[SemanticSearchResult] = Field(..., description="Search results")
    query_id: str = Field(..., description="Unique query identifier (UUID)")
    total_results: int = Field(..., description="Total number of results", ge=0)
    search_time_ms: float = Field(..., description="Search time in milliseconds", ge=0)
    reranking_applied: bool | None = Field(
        None, description="Whether reranking was applied"
    )
    rerank_time_ms: float | None = Field(
        None, description="Reranking time in milliseconds", ge=0
    )
    has_more: bool | None = Field(
        None, description="Whether more results are available"
    )
    next_offset: int | None = Field(
        None, description="Next offset for pagination", ge=0
    )
    cache_hit: bool | None = Field(None, description="Whether query hit cache")
    expanded_query: str | None = Field(None, description="Expanded query")
    expansion_terms_used: list[str] | None = Field(
        None, description="Expansion terms used"
    )


class TraversalFilters(BaseModel):
    """Filters for call graph traversal."""

    exclude_patterns: list[str] | None = Field(
        None, description="Symbol patterns to exclude"
    )
    include_subsystems: list[str] | None = Field(
        None, description="Subsystems to include"
    )
    exclude_subsystems: list[str] | None = Field(
        None, description="Subsystems to exclude"
    )
    include_only_exported: bool | None = Field(
        False, description="Include only exported symbols"
    )
    min_complexity: int | None = Field(
        None, description="Minimum complexity threshold", ge=1
    )
    exclude_static: bool | None = Field(False, description="Exclude static functions")


class TraverseCallGraphRequest(BaseModel):
    """Request for call graph traversal."""

    start_symbol: str = Field(
        ..., description="Starting symbol for traversal", min_length=1
    )
    direction: str = Field(
        "forward",
        description="Traversal direction",
        pattern="^(forward|backward|bidirectional)$",
    )
    max_depth: int | None = Field(
        5, description="Maximum traversal depth", ge=1, le=100
    )
    include_indirect: bool | None = Field(False, description="Include indirect calls")
    detect_cycles: bool | None = Field(
        True, description="Detect cycles during traversal"
    )
    find_all_paths: bool | None = Field(False, description="Find all paths to target")
    target_symbol: str | None = Field(
        None, description="Target symbol for path finding"
    )
    filters: TraversalFilters | None = Field(None, description="Traversal filters")
    include_metrics: bool | None = Field(
        False, description="Include complexity metrics"
    )
    include_visualization: bool | None = Field(
        False, description="Include visualization data"
    )
    layout: str | None = Field("hierarchical", description="Visualization layout")
    incremental: bool | None = Field(False, description="Incremental expansion")
    base_traversal_id: str | None = Field(
        None, description="Base traversal ID for incremental"
    )


class CallGraphNode(BaseModel):
    """Node in the call graph."""

    symbol: str = Field(..., description="Symbol name")
    span: Span = Field(..., description="Source location")
    depth: int = Field(..., description="Depth in traversal", ge=0)
    node_type: str = Field(..., description="Node type (function, macro, etc.)")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    is_entry_point: bool | None = Field(
        None, description="Whether this is an entry point"
    )
    metrics: dict[str, Any] | None = Field(None, description="Complexity metrics")


class CallGraphEdge(BaseModel):
    """Edge in the call graph."""

    from_symbol: str = Field(..., description="Source symbol", alias="from")
    to_symbol: str = Field(..., description="Target symbol", alias="to")
    edge_type: str = Field(
        ...,
        description="Edge type",
        pattern="^(direct|indirect|macro|inline|virtual)$",
    )
    weight: float | None = Field(None, description="Edge weight", ge=0.0)
    call_site: Span | None = Field(None, description="Call site location")


class TraversalStatistics(BaseModel):
    """Statistics from graph traversal."""

    total_nodes: int = Field(..., description="Total nodes found", ge=0)
    total_edges: int = Field(..., description="Total edges found", ge=0)
    max_depth_reached: int = Field(..., description="Maximum depth reached", ge=0)
    cycles_detected: int = Field(..., description="Number of cycles detected", ge=0)
    traversal_time_ms: float | None = Field(None, description="Traversal time", ge=0)


class VisualizationData(BaseModel):
    """Visualization data for the call graph."""

    layout: str = Field(..., description="Layout algorithm used")
    node_positions: dict[str, dict[str, float]] | None = Field(
        None, description="Node positions (symbol -> {x, y})"
    )
    suggested_colors: dict[str, str] | None = Field(
        None, description="Suggested colors (symbol -> color)"
    )
    graph_bounds: dict[str, float] | None = Field(
        None, description="Graph bounds {min_x, max_x, min_y, max_y}"
    )


class TraverseCallGraphResponse(BaseModel):
    """Response for call graph traversal."""

    nodes: list[CallGraphNode] = Field(..., description="Graph nodes")
    edges: list[CallGraphEdge] = Field(..., description="Graph edges")
    paths: list[list[str]] = Field(..., description="Paths found (for path finding)")
    cycles: list[list[str]] | None = Field(None, description="Cycles detected")
    statistics: TraversalStatistics = Field(..., description="Traversal statistics")
    traversal_id: str = Field(..., description="Unique traversal identifier (UUID)")
    visualization: VisualizationData | None = Field(
        None, description="Visualization data"
    )


# Error response
class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")

    class Config:
        json_schema_extra: typing.ClassVar[dict[str, typing.Any]] = {
            "example": {
                "error": "symbol_not_found",
                "message": "Symbol 'nonexistent_func' not found in any configuration",
            }
        }


# Resource models
class ResourceList(BaseModel):
    """List of available MCP resources."""

    resources: list[str]


# Health and metrics
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., pattern="^healthy$")
    version: str
    indexed_at: str | None = Field(None, description="Last index timestamp")


# Export Graph models
class ExportFilters(BaseModel):
    """Filters for graph export."""

    exclude_patterns: list[str] | None = Field(
        None, description="Symbol patterns to exclude"
    )
    include_subsystems: list[str] | None = Field(
        None, description="Subsystems to include"
    )
    exclude_subsystems: list[str] | None = Field(
        None, description="Subsystems to exclude"
    )
    min_edge_weight: float | None = Field(
        None, description="Minimum edge weight", ge=0.0
    )
    exclude_indirect: bool | None = Field(False, description="Exclude indirect calls")


class StylingOptions(BaseModel):
    """Styling options for graph visualization."""

    node_color: str | None = Field(None, description="Default node color")
    edge_color: str | None = Field(None, description="Default edge color")
    font_size: int | None = Field(None, description="Font size", ge=6, le=72)
    node_shape: str | None = Field(None, description="Node shape")
    edge_style: str | None = Field(None, description="Edge style")


class ExportGraphRequest(BaseModel):
    """Request for graph export."""

    root_symbol: str | None = Field(
        None, description="Root symbol for export", min_length=1
    )
    format: str = Field(
        ..., description="Export format", pattern="^(json|graphml|dot|csv)$"
    )
    depth: int | None = Field(5, description="Maximum depth", ge=1, le=100)
    include_metadata: bool | None = Field(False, description="Include node metadata")
    include_attributes: bool | None = Field(
        False, description="Include GraphML attributes"
    )
    pretty: bool | None = Field(False, description="Pretty print output")
    layout: str | None = Field("hierarchical", description="Graph layout algorithm")
    styling: StylingOptions | None = Field(None, description="Styling options")
    csv_type: str | None = Field(
        "edge_list", description="CSV export type", pattern="^(edge_list|adjacency)$"
    )
    filters: ExportFilters | None = Field(None, description="Export filters")
    chunk_size: int | None = Field(
        None, description="Chunk size for large exports", ge=1, le=10000
    )
    chunk_index: int | None = Field(
        None, description="Chunk index for pagination", ge=0
    )
    compress: bool | None = Field(False, description="Enable compression")
    compression_format: str | None = Field(
        "gzip", description="Compression format", pattern="^(gzip|zlib|bzip2)$"
    )
    async_export: bool | None = Field(
        False, description="Asynchronous export", alias="async"
    )
    callback_url: str | None = Field(None, description="Callback URL for async export")
    include_annotations: bool | None = Field(
        False, description="Include code annotations"
    )
    annotation_types: list[str] | None = Field(
        None, description="Types of annotations to include"
    )
    include_statistics: bool | None = Field(
        False, description="Include graph statistics"
    )


class GraphNode(BaseModel):
    """Node in exported graph."""

    id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Node label")
    type: str = Field(..., description="Node type")
    metadata: dict[str, Any] | None = Field(None, description="Node metadata")
    annotations: dict[str, Any] | None = Field(None, description="Code annotations")


class GraphEdge(BaseModel):
    """Edge in exported graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Edge type")
    weight: float | None = Field(None, description="Edge weight", ge=0.0)
    metadata: dict[str, Any] | None = Field(None, description="Edge metadata")


class ExportedGraph(BaseModel):
    """Exported graph structure."""

    nodes: list[GraphNode] = Field(..., description="Graph nodes")
    edges: list[GraphEdge] = Field(..., description="Graph edges")
    metadata: dict[str, Any] = Field(..., description="Graph metadata")


class ChunkInfo(BaseModel):
    """Information about chunked export."""

    total_chunks: int = Field(..., description="Total number of chunks", ge=1)
    current_chunk: int = Field(..., description="Current chunk index", ge=0)
    chunk_size: int = Field(..., description="Chunk size", ge=1)
    has_more: bool = Field(..., description="Whether more chunks are available")


class SizeInfo(BaseModel):
    """Size information for compressed exports."""

    original_size: int = Field(..., description="Original size in bytes", ge=0)
    compressed_size: int = Field(..., description="Compressed size in bytes", ge=0)
    compression_ratio: float = Field(..., description="Compression ratio", ge=0.0)


class GraphStatistics(BaseModel):
    """Graph analysis statistics."""

    total_nodes: int = Field(..., description="Total number of nodes", ge=0)
    total_edges: int = Field(..., description="Total number of edges", ge=0)
    max_depth_reached: int = Field(..., description="Maximum depth reached", ge=0)
    avg_degree: float = Field(..., description="Average node degree", ge=0.0)
    density: float = Field(..., description="Graph density", ge=0.0, le=1.0)
    connected_components: int = Field(
        ..., description="Number of connected components", ge=0
    )
    cycles_count: int = Field(..., description="Number of cycles detected", ge=0)
    longest_path: int = Field(..., description="Longest path length", ge=0)


class AsyncJobInfo(BaseModel):
    """Information about asynchronous export job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(
        ..., description="Job status", pattern="^(pending|processing|completed|failed)$"
    )
    status_url: str = Field(..., description="URL to check job status")
    estimated_time: int | None = Field(
        None, description="Estimated completion time in seconds", ge=0
    )
    progress: float | None = Field(
        None, description="Job progress percentage", ge=0.0, le=100.0
    )


class ExportGraphResponse(BaseModel):
    """Response for graph export."""

    export_id: str = Field(..., description="Unique export identifier")
    format: str = Field(..., description="Export format")
    exported_at: str = Field(..., description="Export timestamp")

    # Format-specific fields
    graph: ExportedGraph | None = Field(None, description="JSON graph data")
    graphml: str | None = Field(None, description="GraphML XML data")
    dot: str | None = Field(None, description="DOT/Graphviz data")
    csv: str | None = Field(None, description="CSV data")

    # Compression fields
    compressed: bool | None = Field(None, description="Whether data is compressed")
    compression_format: str | None = Field(None, description="Compression format used")
    graph_data: str | None = Field(None, description="Base64 encoded compressed data")
    size_info: SizeInfo | None = Field(None, description="Size information")

    # Chunking fields
    chunk_info: ChunkInfo | None = Field(None, description="Chunk information")

    # Statistics and metadata
    statistics: GraphStatistics | None = Field(None, description="Graph statistics")

    # Async export fields
    job_info: AsyncJobInfo | None = Field(None, description="Async job information")
