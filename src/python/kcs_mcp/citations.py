"""Citation formatter for KCS MCP responses.

Ensures all results include proper file:line citations per constitution.
Formats citations in consistent format for MCP protocol.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Span:
    """File span representing a location in source code."""

    path: str
    sha: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate span data."""
        if self.start <= 0:
            raise ValueError(f"Start line must be positive, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"End line {self.end} must be >= start line {self.start}")
        if not self.path:
            raise ValueError("Path cannot be empty")
        if not self.sha:
            raise ValueError("SHA cannot be empty")


@dataclass
class Citation:
    """Citation with span and optional context."""

    span: Span
    context: str | None = None


class CitationFormatter:
    """Formats citations for MCP protocol responses."""

    def __init__(self, repo_root: str | None = None):
        """Initialize formatter.

        Args:
            repo_root: Repository root path for relativizing paths
        """
        self.repo_root = Path(repo_root) if repo_root else None

    def create_span(
        self, file_path: str | Path, start_line: int, end_line: int, sha: str
    ) -> Span:
        """Create a span from file location.

        Args:
            file_path: Absolute or relative file path
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            sha: Git SHA of file version

        Returns:
            Validated Span object

        Raises:
            ValueError: If span parameters are invalid
        """
        # Convert to relative path if repo_root is set
        path_obj = Path(file_path)
        if self.repo_root and path_obj.is_absolute():
            try:
                relative_path = path_obj.relative_to(self.repo_root)
                path_str = str(relative_path)
            except ValueError:
                # Path is not under repo_root, use as-is
                path_str = str(path_obj)
        else:
            path_str = str(path_obj)

        return Span(
            path=path_str,
            sha=sha[:8],  # Truncate SHA to 8 chars for readability
            start=start_line,
            end=end_line,
        )

    def create_citation(
        self,
        file_path: str | Path,
        start_line: int,
        end_line: int,
        sha: str,
        context: str | None = None,
    ) -> Citation:
        """Create a citation from file location.

        Args:
            file_path: File path
            start_line: Starting line number
            end_line: Ending line number
            sha: Git SHA
            context: Optional context description

        Returns:
            Citation object
        """
        span = self.create_span(file_path, start_line, end_line, sha)
        return Citation(span=span, context=context)

    def format_span_text(self, span: Span) -> str:
        """Format span as human-readable text.

        Args:
            span: Span to format

        Returns:
            Formatted string like "fs/read_write.c:451-465@a1b2c3d4"
        """
        if span.start == span.end:
            return f"{span.path}:{span.start}@{span.sha}"
        else:
            return f"{span.path}:{span.start}-{span.end}@{span.sha}"

    def format_citation_text(self, citation: Citation) -> str:
        """Format citation as human-readable text.

        Args:
            citation: Citation to format

        Returns:
            Formatted string with optional context
        """
        span_text = self.format_span_text(citation.span)
        if citation.context:
            return f"{span_text} ({citation.context})"
        return span_text

    def to_dict(self, span_or_citation: Span | Citation) -> dict[str, Any]:
        """Convert span or citation to dictionary for JSON serialization.

        Args:
            span_or_citation: Span or Citation object

        Returns:
            Dictionary representation
        """
        if isinstance(span_or_citation, Span):
            return {
                "path": span_or_citation.path,
                "sha": span_or_citation.sha,
                "start": span_or_citation.start,
                "end": span_or_citation.end,
            }
        elif isinstance(span_or_citation, Citation):
            result: dict[str, Any] = {"span": self.to_dict(span_or_citation.span)}
            if span_or_citation.context:
                result["context"] = span_or_citation.context
            return result
        else:
            raise TypeError(f"Expected Span or Citation, got {type(span_or_citation)}")

    def from_dict(self, data: dict[str, Any]) -> Span | Citation:
        """Create span or citation from dictionary.

        Args:
            data: Dictionary with span or citation data

        Returns:
            Span or Citation object
        """
        if "span" in data:
            # Citation format
            span_data = data["span"]
            span = Span(
                path=span_data["path"],
                sha=span_data["sha"],
                start=span_data["start"],
                end=span_data["end"],
            )
            return Citation(span=span, context=data.get("context"))
        else:
            # Span format
            return Span(
                path=data["path"], sha=data["sha"], start=data["start"], end=data["end"]
            )

    def parse_span_text(self, text: str) -> Span:
        """Parse span from text format.

        Args:
            text: Text like "fs/read_write.c:451-465@a1b2c3d4"

        Returns:
            Span object

        Raises:
            ValueError: If text format is invalid
        """
        # Pattern: path:start[-end]@sha
        pattern = r"^(.+):(\d+)(?:-(\d+))?@([a-f0-9]+)$"
        match = re.match(pattern, text)

        if not match:
            raise ValueError(f"Invalid span format: {text}")

        path, start_str, end_str, sha = match.groups()
        start = int(start_str)
        end = int(end_str) if end_str else start

        return Span(path=path, sha=sha, start=start, end=end)

    def merge_spans(self, spans: list[Span]) -> list[Span]:
        """Merge overlapping spans from the same file.

        Args:
            spans: List of spans to merge

        Returns:
            List of merged spans
        """
        if not spans:
            return []

        # Group by (path, sha)
        by_file: dict[tuple, list[Span]] = {}
        for span in spans:
            key = (span.path, span.sha)
            if key not in by_file:
                by_file[key] = []
            by_file[key].append(span)

        merged = []
        for file_spans in by_file.values():
            # Sort by start line
            file_spans.sort(key=lambda s: s.start)

            current = file_spans[0]
            for next_span in file_spans[1:]:
                # Check if spans overlap or are adjacent
                if next_span.start <= current.end + 1:
                    # Merge spans
                    current = Span(
                        path=current.path,
                        sha=current.sha,
                        start=current.start,
                        end=max(current.end, next_span.end),
                    )
                else:
                    # No overlap, add current and start new
                    merged.append(current)
                    current = next_span

            merged.append(current)

        return merged

    def validate_citations(self, citations: list[Citation]) -> list[str]:
        """Validate a list of citations.

        Args:
            citations: Citations to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for i, citation in enumerate(citations):
            try:
                # Validate span
                if not citation.span.path:
                    errors.append(f"Citation {i}: Empty path")
                if not citation.span.sha:
                    errors.append(f"Citation {i}: Empty SHA")
                if citation.span.start <= 0:
                    errors.append(
                        f"Citation {i}: Invalid start line {citation.span.start}"
                    )
                if citation.span.end < citation.span.start:
                    errors.append(
                        f"Citation {i}: End line {citation.span.end} < start line {citation.span.start}"
                    )

                # Validate path format
                if not re.match(r"^[a-zA-Z0-9/_.-]+$", citation.span.path):
                    errors.append(
                        f"Citation {i}: Invalid path format: {citation.span.path}"
                    )

                # Validate SHA format
                if not re.match(r"^[a-f0-9]{6,40}$", citation.span.sha):
                    errors.append(
                        f"Citation {i}: Invalid SHA format: {citation.span.sha}"
                    )

            except Exception as e:
                errors.append(f"Citation {i}: {e!s}")

        return errors


def ensure_citations(
    response_data: dict[str, Any], formatter: CitationFormatter
) -> dict[str, Any]:
    """Ensure MCP response includes proper citations.

    This is a constitutional requirement - every claim must have citations.

    Args:
        response_data: MCP response data
        formatter: Citation formatter instance

    Returns:
        Response data with validated citations

    Raises:
        ValueError: If required citations are missing
    """
    # Check for citation fields in common response patterns
    citation_fields = ["cites", "citations", "spans", "references"]
    has_citations = any(field in response_data for field in citation_fields)

    if not has_citations:
        # Check if response contains symbol/search results that should have citations
        requires_citations = any(
            field in response_data
            for field in [
                "hits",
                "callers",
                "callees",
                "steps",
                "symbols",
                "mismatches",
            ]
        )

        if requires_citations:
            raise ValueError(
                "Constitutional violation: Response contains claims without citations. "
                "All results must include file:line references."
            )

    # Validate existing citations
    for field in citation_fields:
        if field in response_data:
            citations_data = response_data[field]
            if isinstance(citations_data, list):
                citations = [
                    formatter.from_dict(c) if isinstance(c, dict) else c
                    for c in citations_data
                ]
                # Filter to only Citations for validation
                citation_objects = [
                    c
                    if isinstance(c, Citation)
                    else Citation(span=c)
                    if isinstance(c, Span)
                    else c
                    for c in citations
                    if isinstance(c, Citation | Span)
                ]
                errors = formatter.validate_citations(citation_objects)
                if errors:
                    raise ValueError(
                        f"Invalid citations in {field}: {'; '.join(errors)}"
                    )

    return response_data


# Convenience functions for common use cases


def span_from_symbol(
    symbol_name: str, file_path: str, start_line: int, end_line: int, sha: str
) -> Span:
    """Create span for a symbol definition."""
    formatter = CitationFormatter()
    return formatter.create_span(file_path, start_line, end_line, sha)


def cite_function_def(
    func_name: str, file_path: str, start_line: int, end_line: int, sha: str
) -> Citation:
    """Create citation for function definition."""
    formatter = CitationFormatter()
    return formatter.create_citation(
        file_path, start_line, end_line, sha, context=f"Function {func_name} definition"
    )


def cite_call_site(
    caller: str, callee: str, file_path: str, line: int, sha: str
) -> Citation:
    """Create citation for function call site."""
    formatter = CitationFormatter()
    return formatter.create_citation(
        file_path, line, line, sha, context=f"{caller} calls {callee}"
    )
