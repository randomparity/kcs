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
