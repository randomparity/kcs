"""
Pydantic models for the KCS MCP API.

This package contains all data models used for request/response validation
and database operations.
"""

# Import models directly from the models.py file using importlib
import importlib.util
import sys
from pathlib import Path

# Load the models.py module directly
_models_file = Path(__file__).parent.parent / "models.py"
_spec = importlib.util.spec_from_file_location("models_module", _models_file)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_models_file}")
_models_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models_module)

# Import all the models from the loaded module
AsyncJobInfo = _models_module.AsyncJobInfo
CallerInfo = _models_module.CallerInfo
CallGraphEdge = _models_module.CallGraphEdge
CallGraphNode = _models_module.CallGraphNode
ChunkInfo = _models_module.ChunkInfo
ConfigDependency = _models_module.ConfigDependency
ConfigOption = _models_module.ConfigOption
DiffSpecVsCodeRequest = _models_module.DiffSpecVsCodeRequest
DiffSpecVsCodeResponse = _models_module.DiffSpecVsCodeResponse
EntrypointFlowRequest = _models_module.EntrypointFlowRequest
EntrypointFlowResponse = _models_module.EntrypointFlowResponse
ErrorResponse = _models_module.ErrorResponse
ExportedGraph = _models_module.ExportedGraph
ExportGraphRequest = _models_module.ExportGraphRequest
ExportGraphResponse = _models_module.ExportGraphResponse
FlowStep = _models_module.FlowStep
GetSymbolRequest = _models_module.GetSymbolRequest
GraphEdge = _models_module.GraphEdge
GraphNode = _models_module.GraphNode
GraphStatistics = _models_module.GraphStatistics
HealthResponse = _models_module.HealthResponse
ImpactOfRequest = _models_module.ImpactOfRequest
ImpactResult = _models_module.ImpactResult
ImplementationDetails = _models_module.ImplementationDetails
ListDependenciesRequest = _models_module.ListDependenciesRequest
ListDependenciesResponse = _models_module.ListDependenciesResponse
MaintainerInfo = _models_module.MaintainerInfo
OwnersForRequest = _models_module.OwnersForRequest
OwnersForResponse = _models_module.OwnersForResponse
ParseKernelConfigRequest = _models_module.ParseKernelConfigRequest
ParseKernelConfigResponse = _models_module.ParseKernelConfigResponse
SearchCodeRequest = _models_module.SearchCodeRequest
SearchCodeResponse = _models_module.SearchCodeResponse
SearchDocsRequest = _models_module.SearchDocsRequest
SearchDocsResponse = _models_module.SearchDocsResponse
SearchHit = _models_module.SearchHit
SearchResultExplanation = _models_module.SearchResultExplanation
SemanticSearchContext = _models_module.SemanticSearchContext
SemanticSearchRequest = _models_module.SemanticSearchRequest
SemanticSearchResponse = _models_module.SemanticSearchResponse
SemanticSearchResult = _models_module.SemanticSearchResult
SizeInfo = _models_module.SizeInfo
Span = _models_module.Span
SpecDeviation = _models_module.SpecDeviation
SymbolInfo = _models_module.SymbolInfo
TraversalStatistics = _models_module.TraversalStatistics
TraverseCallGraphRequest = _models_module.TraverseCallGraphRequest
TraverseCallGraphResponse = _models_module.TraverseCallGraphResponse
ValidateSpecRequest = _models_module.ValidateSpecRequest
ValidateSpecResponse = _models_module.ValidateSpecResponse
ValidationSuggestion = _models_module.ValidationSuggestion
VisualizationData = _models_module.VisualizationData
WhoCallsRequest = _models_module.WhoCallsRequest
WhoCallsResponse = _models_module.WhoCallsResponse

# Clean up
del _models_file, _spec, _models_module

# Re-export the chunk models for convenience
from .chunk_models import (  # noqa: E402
    ChunkManifest,
    ChunkMetadata,
    ChunkProcessingRecord,
    IndexingManifestRecord,
    ProcessBatchRequest,
    ProcessBatchResponse,
    ProcessChunkRequest,
    ProcessChunkResponse,
    ProcessingStatus,
)

__all__ = [
    # Main models (imported via importlib above)
    "AsyncJobInfo",
    "CallGraphEdge",
    "CallGraphNode",
    "CallerInfo",
    "ChunkInfo",
    # Chunk models
    "ChunkManifest",
    "ChunkMetadata",
    "ChunkProcessingRecord",
    "ConfigDependency",
    "ConfigOption",
    "DiffSpecVsCodeRequest",
    "DiffSpecVsCodeResponse",
    "EntrypointFlowRequest",
    "EntrypointFlowResponse",
    "ErrorResponse",
    "ExportGraphRequest",
    "ExportGraphResponse",
    "ExportedGraph",
    "FlowStep",
    "GetSymbolRequest",
    "GraphEdge",
    "GraphNode",
    "GraphStatistics",
    "HealthResponse",
    "ImpactOfRequest",
    "ImpactResult",
    "ImplementationDetails",
    "IndexingManifestRecord",
    "ListDependenciesRequest",
    "ListDependenciesResponse",
    "MaintainerInfo",
    "OwnersForRequest",
    "OwnersForResponse",
    "ParseKernelConfigRequest",
    "ParseKernelConfigResponse",
    "ProcessBatchRequest",
    "ProcessBatchResponse",
    "ProcessChunkRequest",
    "ProcessChunkResponse",
    "ProcessingStatus",
    "SearchCodeRequest",
    "SearchCodeResponse",
    "SearchDocsRequest",
    "SearchDocsResponse",
    "SearchHit",
    "SearchResultExplanation",
    "SemanticSearchContext",
    "SemanticSearchRequest",
    "SemanticSearchResponse",
    "SemanticSearchResult",
    "SizeInfo",
    "Span",
    "SpecDeviation",
    "SymbolInfo",
    "TraversalStatistics",
    "TraverseCallGraphRequest",
    "TraverseCallGraphResponse",
    "ValidateSpecRequest",
    "ValidateSpecResponse",
    "ValidationSuggestion",
    "VisualizationData",
    "WhoCallsRequest",
    "WhoCallsResponse",
]
