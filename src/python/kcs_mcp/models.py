"""
Pydantic models for KCS MCP API.

Defines request/response schemas matching the OpenAPI specification.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


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
        json_schema_extra = {
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
    context: Optional[str] = Field(None, description="Optional context around citation")


# Tool request/response models
class SearchCodeRequest(BaseModel):
    """Request for code search."""

    query: str = Field(..., description="Search query", min_length=1)
    topK: Optional[int] = Field(
        10, description="Maximum results to return", ge=1, le=100
    )


class SearchHit(BaseModel):
    """Search result hit."""

    span: Span
    snippet: str = Field(..., description="Code snippet with match highlighted")
    score: Optional[float] = Field(None, description="Relevance score")


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
    summary: Optional[dict[str, Any]] = Field(
        None, description="Optional AI-generated summary"
    )


class WhoCallsRequest(BaseModel):
    """Request for caller analysis."""

    symbol: str = Field(..., description="Symbol name", min_length=1)
    depth: Optional[int] = Field(1, description="Call graph depth", ge=1, le=10)


class CallerInfo(BaseModel):
    """Information about a function caller."""

    symbol: str
    span: Span
    call_type: Optional[str] = Field(None, pattern="^(direct|indirect|macro|inline)$")


class WhoCallsResponse(BaseModel):
    """Response for caller analysis."""

    callers: list[CallerInfo]


class ListDependenciesRequest(BaseModel):
    """Request for dependency analysis."""

    symbol: str = Field(..., description="Symbol name", min_length=1)
    depth: Optional[int] = Field(1, description="Dependency depth", ge=1, le=10)


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

    diff: Optional[str] = Field(None, description="Git diff content")
    files: Optional[list[str]] = Field(None, description="Files to analyze")
    symbols: Optional[list[str]] = Field(None, description="Symbols to analyze")
    config: Optional[str] = Field(None, description="Configuration context")


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
    corpus: Optional[list[str]] = Field(
        None, description="Document collections to search"
    )


class DocHit(BaseModel):
    """Documentation search hit."""

    source: str = Field(..., description="Document source")
    anchor: str = Field(..., description="Section/anchor")
    span: Optional[Span] = Field(None, description="File location if applicable")


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
    span: Optional[Span] = Field(None, description="Related code location")


class DiffSpecVsCodeResponse(BaseModel):
    """Response for drift detection."""

    mismatches: list[DriftMismatch]


class OwnersForRequest(BaseModel):
    """Request for maintainer information."""

    paths: Optional[list[str]] = Field(None, description="File paths")
    symbols: Optional[list[str]] = Field(None, description="Symbol names")


class MaintainerInfo(BaseModel):
    """Maintainer information."""

    section: str = Field(..., description="MAINTAINERS section")
    emails: list[str] = Field(..., description="Maintainer email addresses")
    paths: list[str] = Field(..., description="Maintained paths")


class OwnersForResponse(BaseModel):
    """Response for maintainer information."""

    maintainers: list[MaintainerInfo]


# Error response
class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")

    class Config:
        json_schema_extra = {
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
    indexed_at: Optional[str] = Field(None, description="Last index timestamp")
