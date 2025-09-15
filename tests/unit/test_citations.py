"""Unit tests for citation formatter.

Tests the citation formatting functionality to ensure constitutional compliance
with file:line references for all claims.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from kcs_mcp.citations import (
    Citation,
    CitationFormatter,
    Span,
    cite_call_site,
    cite_function_def,
    ensure_citations,
    span_from_symbol,
)


class TestSpan:
    """Tests for Span class."""

    def test_valid_span_creation(self):
        """Test creating valid spans."""
        span = Span(path="fs/read_write.c", sha="a1b2c3d4", start=451, end=465)

        assert span.path == "fs/read_write.c"
        assert span.sha == "a1b2c3d4"
        assert span.start == 451
        assert span.end == 465

    def test_single_line_span(self):
        """Test span with same start and end line."""
        span = Span(path="include/linux/fs.h", sha="abcdef12", start=100, end=100)

        assert span.start == span.end == 100

    def test_invalid_start_line(self):
        """Test span with invalid start line."""
        with pytest.raises(ValueError, match="Start line must be positive"):
            Span(path="test.c", sha="abc123", start=0, end=10)

        with pytest.raises(ValueError, match="Start line must be positive"):
            Span(path="test.c", sha="abc123", start=-5, end=10)

    def test_invalid_end_line(self):
        """Test span with end line before start line."""
        with pytest.raises(ValueError, match=r"End line .* must be >= start line"):
            Span(path="test.c", sha="abc123", start=10, end=5)

    def test_empty_path(self):
        """Test span with empty path."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            Span(path="", sha="abc123", start=1, end=1)

    def test_empty_sha(self):
        """Test span with empty SHA."""
        with pytest.raises(ValueError, match="SHA cannot be empty"):
            Span(path="test.c", sha="", start=1, end=1)


class TestCitation:
    """Tests for Citation class."""

    def test_citation_with_context(self):
        """Test citation with context."""
        span = Span("test.c", "abc123", 10, 15)
        citation = Citation(span=span, context="Function definition")

        assert citation.span == span
        assert citation.context == "Function definition"

    def test_citation_without_context(self):
        """Test citation without context."""
        span = Span("test.c", "abc123", 10, 15)
        citation = Citation(span=span)

        assert citation.span == span
        assert citation.context is None


class TestCitationFormatter:
    """Tests for CitationFormatter class."""

    def test_formatter_without_repo_root(self):
        """Test formatter creation without repo root."""
        formatter = CitationFormatter()
        assert formatter.repo_root is None

    def test_formatter_with_repo_root(self):
        """Test formatter creation with repo root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            formatter = CitationFormatter(repo_root=temp_dir)
            assert formatter.repo_root == Path(temp_dir)

    def test_create_span_basic(self):
        """Test basic span creation."""
        formatter = CitationFormatter()
        span = formatter.create_span(
            file_path="fs/read_write.c",
            start_line=451,
            end_line=465,
            sha="a1b2c3d4e5f6789",
        )

        assert span.path == "fs/read_write.c"
        assert span.start == 451
        assert span.end == 465
        assert span.sha == "a1b2c3d4"  # Truncated to 8 chars

    def test_create_span_with_repo_root(self):
        """Test span creation with repo root path relativization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            formatter = CitationFormatter(repo_root=temp_dir)

            # Absolute path under repo root
            abs_path = Path(temp_dir) / "fs" / "read_write.c"
            span = formatter.create_span(
                file_path=str(abs_path),
                start_line=100,
                end_line=200,
                sha="abcdef123456",
            )

            assert span.path == "fs/read_write.c"
            assert span.sha == "abcdef12"

    def test_create_span_absolute_path_outside_repo(self):
        """Test span creation with absolute path outside repo root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            formatter = CitationFormatter(repo_root=temp_dir)

            # Absolute path outside repo root
            outside_path = "/usr/include/stdio.h"
            span = formatter.create_span(
                file_path=outside_path, start_line=50, end_line=50, sha="external123"
            )

            assert span.path == outside_path

    def test_create_citation(self):
        """Test citation creation."""
        formatter = CitationFormatter()
        citation = formatter.create_citation(
            file_path="kernel/sched.c",
            start_line=1000,
            end_line=1050,
            sha="fedcba987654",
            context="Schedule function",
        )

        assert citation.span.path == "kernel/sched.c"
        assert citation.span.start == 1000
        assert citation.span.end == 1050
        assert citation.span.sha == "fedcba98"
        assert citation.context == "Schedule function"

    def test_format_span_text_single_line(self):
        """Test formatting span text for single line."""
        formatter = CitationFormatter()
        span = Span("test.c", "abc12345", 42, 42)

        formatted = formatter.format_span_text(span)
        assert formatted == "test.c:42@abc12345"

    def test_format_span_text_multiple_lines(self):
        """Test formatting span text for multiple lines."""
        formatter = CitationFormatter()
        span = Span("fs/namei.c", "def67890", 100, 150)

        formatted = formatter.format_span_text(span)
        assert formatted == "fs/namei.c:100-150@def67890"

    def test_format_citation_text_with_context(self):
        """Test formatting citation text with context."""
        formatter = CitationFormatter()
        span = Span("mm/mmap.c", "123abc", 500, 600)
        citation = Citation(span=span, context="Memory mapping")

        formatted = formatter.format_citation_text(citation)
        assert formatted == "mm/mmap.c:500-600@123abc (Memory mapping)"

    def test_format_citation_text_without_context(self):
        """Test formatting citation text without context."""
        formatter = CitationFormatter()
        span = Span("net/socket.c", "456def", 200, 200)
        citation = Citation(span=span)

        formatted = formatter.format_citation_text(citation)
        assert formatted == "net/socket.c:200@456def"

    def test_to_dict_span(self):
        """Test converting span to dictionary."""
        formatter = CitationFormatter()
        span = Span("drivers/net/e1000.c", "789ghi", 300, 400)

        result = formatter.to_dict(span)
        expected = {
            "path": "drivers/net/e1000.c",
            "sha": "789ghi",
            "start": 300,
            "end": 400,
        }

        assert result == expected

    def test_to_dict_citation(self):
        """Test converting citation to dictionary."""
        formatter = CitationFormatter()
        span = Span("arch/x86/kernel/cpu.c", "abc789", 150, 180)
        citation = Citation(span=span, context="CPU initialization")

        result = formatter.to_dict(citation)
        expected = {
            "span": {
                "path": "arch/x86/kernel/cpu.c",
                "sha": "abc789",
                "start": 150,
                "end": 180,
            },
            "context": "CPU initialization",
        }

        assert result == expected

    def test_to_dict_citation_without_context(self):
        """Test converting citation without context to dictionary."""
        formatter = CitationFormatter()
        span = Span("crypto/aes.c", "def456", 75, 75)
        citation = Citation(span=span)

        result = formatter.to_dict(citation)
        expected = {
            "span": {"path": "crypto/aes.c", "sha": "def456", "start": 75, "end": 75}
        }

        assert result == expected

    def test_to_dict_invalid_type(self):
        """Test to_dict with invalid type."""
        formatter = CitationFormatter()

        with pytest.raises(TypeError, match="Expected Span or Citation"):
            formatter.to_dict("invalid")

    def test_from_dict_span(self):
        """Test creating span from dictionary."""
        formatter = CitationFormatter()
        data = {"path": "fs/ext4/inode.c", "sha": "xyz123", "start": 50, "end": 100}

        span = formatter.from_dict(data)
        assert isinstance(span, Span)
        assert span.path == "fs/ext4/inode.c"
        assert span.sha == "xyz123"
        assert span.start == 50
        assert span.end == 100

    def test_from_dict_citation(self):
        """Test creating citation from dictionary."""
        formatter = CitationFormatter()
        data = {
            "span": {
                "path": "security/selinux/hooks.c",
                "sha": "selinux1",
                "start": 200,
                "end": 250,
            },
            "context": "SELinux hook",
        }

        citation = formatter.from_dict(data)
        assert isinstance(citation, Citation)
        assert citation.span.path == "security/selinux/hooks.c"
        assert citation.span.sha == "selinux1"
        assert citation.span.start == 200
        assert citation.span.end == 250
        assert citation.context == "SELinux hook"

    def test_parse_span_text_single_line(self):
        """Test parsing span text for single line."""
        formatter = CitationFormatter()

        span = formatter.parse_span_text("init/main.c:123@abcd1234")
        assert span.path == "init/main.c"
        assert span.start == 123
        assert span.end == 123
        assert span.sha == "abcd1234"

    def test_parse_span_text_multiple_lines(self):
        """Test parsing span text for multiple lines."""
        formatter = CitationFormatter()

        span = formatter.parse_span_text("kernel/fork.c:500-600@deadbeef")
        assert span.path == "kernel/fork.c"
        assert span.start == 500
        assert span.end == 600
        assert span.sha == "deadbeef"

    def test_parse_span_text_complex_path(self):
        """Test parsing span text with complex path."""
        formatter = CitationFormatter()

        span = formatter.parse_span_text(
            "drivers/gpu/drm/i915/intel_display.c:1000-2000@cafebabe"
        )
        assert span.path == "drivers/gpu/drm/i915/intel_display.c"
        assert span.start == 1000
        assert span.end == 2000
        assert span.sha == "cafebabe"

    def test_parse_span_text_invalid_format(self):
        """Test parsing invalid span text formats."""
        formatter = CitationFormatter()

        invalid_formats = [
            "no_colon_or_at",
            "path.c:notanumber@abc123",
            "path.c:123",  # Missing SHA
            "path.c@abc123",  # Missing line number
            ":123@abc123",  # Missing path
            "path.c:123@",  # Missing SHA
            "path.c:123-abc@def456",  # Invalid end line
        ]

        for invalid_format in invalid_formats:
            with pytest.raises(ValueError, match="Invalid span format"):
                formatter.parse_span_text(invalid_format)

    def test_merge_spans_empty(self):
        """Test merging empty span list."""
        formatter = CitationFormatter()

        result = formatter.merge_spans([])
        assert result == []

    def test_merge_spans_no_overlap(self):
        """Test merging non-overlapping spans."""
        formatter = CitationFormatter()
        spans = [
            Span("test.c", "abc123", 10, 20),
            Span("test.c", "abc123", 30, 40),
            Span("other.c", "def456", 5, 15),
        ]

        result = formatter.merge_spans(spans)
        assert len(result) == 3

        # Should be sorted and unchanged
        paths_and_ranges = [(s.path, s.start, s.end) for s in result]
        assert ("test.c", 10, 20) in paths_and_ranges
        assert ("test.c", 30, 40) in paths_and_ranges
        assert ("other.c", 5, 15) in paths_and_ranges

    def test_merge_spans_overlapping(self):
        """Test merging overlapping spans."""
        formatter = CitationFormatter()
        spans = [
            Span("test.c", "abc123", 10, 20),
            Span("test.c", "abc123", 15, 25),
            Span("test.c", "abc123", 23, 30),
        ]

        result = formatter.merge_spans(spans)
        assert len(result) == 1
        assert result[0].path == "test.c"
        assert result[0].sha == "abc123"
        assert result[0].start == 10
        assert result[0].end == 30

    def test_merge_spans_adjacent(self):
        """Test merging adjacent spans."""
        formatter = CitationFormatter()
        spans = [Span("test.c", "abc123", 10, 20), Span("test.c", "abc123", 21, 30)]

        result = formatter.merge_spans(spans)
        assert len(result) == 1
        assert result[0].start == 10
        assert result[0].end == 30

    def test_merge_spans_different_files(self):
        """Test merging spans from different files."""
        formatter = CitationFormatter()
        spans = [
            Span("file1.c", "abc123", 10, 20),
            Span("file2.c", "abc123", 10, 20),
            Span("file1.c", "abc123", 15, 25),
        ]

        result = formatter.merge_spans(spans)
        assert len(result) == 2

        # Check that file1.c spans were merged
        file1_spans = [s for s in result if s.path == "file1.c"]
        file2_spans = [s for s in result if s.path == "file2.c"]

        assert len(file1_spans) == 1
        assert len(file2_spans) == 1
        assert file1_spans[0].start == 10
        assert file1_spans[0].end == 25

    def test_merge_spans_different_shas(self):
        """Test merging spans with different SHAs."""
        formatter = CitationFormatter()
        spans = [Span("test.c", "abc123", 10, 20), Span("test.c", "def456", 15, 25)]

        result = formatter.merge_spans(spans)
        assert len(result) == 2  # Different SHAs, shouldn't merge

    def test_validate_citations_valid(self):
        """Test validating valid citations."""
        formatter = CitationFormatter()
        citations = [
            Citation(Span("test.c", "abc123", 10, 20)),
            Citation(Span("other.h", "def456", 5, 5), "Header file"),
        ]

        errors = formatter.validate_citations(citations)
        assert errors == []

    def test_validate_citations_invalid_paths(self):
        """Test validating citations with invalid paths."""
        # Test empty path
        with pytest.raises(ValueError, match="Path cannot be empty"):
            Citation(Span("", "abc123", 10, 20))

        # Valid citations with valid paths
        formatter = CitationFormatter()
        citations = [
            Citation(Span("test.c", "abc123", 10, 20)),
            Citation(Span("src/file.h", "def456", 5, 10)),
        ]
        errors = formatter.validate_citations(citations)
        assert len(errors) == 0

    def test_validate_citations_invalid_shas(self):
        """Test validating citations with invalid SHAs."""
        # Test empty SHA
        with pytest.raises(ValueError, match="SHA cannot be empty"):
            Citation(Span("test.c", "", 10, 20))

        # Valid citations for further validation
        formatter = CitationFormatter()
        citations = [
            Citation(Span("test.c", "abc123def456", 10, 20)),
            Citation(Span("test.c", "0123456789abcdef", 5, 10)),
        ]
        errors = formatter.validate_citations(citations)
        assert len(errors) == 0

    def test_validate_citations_invalid_lines(self):
        """Test validating citations with invalid line numbers."""
        # Test invalid start line (0)
        with pytest.raises(ValueError, match="Start line must be positive, got 0"):
            Citation(Span("test.c", "abc123", 0, 20))

        # Test negative start line
        with pytest.raises(ValueError, match="Start line must be positive, got -5"):
            Citation(Span("test.c", "def456", -5, 5))

        # Test end < start
        with pytest.raises(ValueError, match="End line 10 must be >= start line 20"):
            Citation(Span("test.c", "ghi789", 20, 10))

        # Valid citations
        formatter = CitationFormatter()
        citations = [
            Citation(Span("test.c", "abc123", 1, 20)),
            Citation(Span("test.c", "def456", 5, 5)),
        ]
        errors = formatter.validate_citations(citations)
        assert len(errors) == 0


class TestEnsureCitations:
    """Tests for ensure_citations function."""

    def test_response_with_valid_citations(self):
        """Test response that already has valid citations."""
        formatter = CitationFormatter()
        response_data = {
            "hits": [{"snippet": "code snippet", "score": 0.9}],
            "cites": [
                {"span": {"path": "test.c", "sha": "abc123", "start": 10, "end": 20}}
            ],
        }

        result = ensure_citations(response_data, formatter)
        assert result == response_data

    def test_response_without_citations_but_no_claims(self):
        """Test response without citations that doesn't need them."""
        formatter = CitationFormatter()
        response_data = {"status": "healthy", "message": "Service is running"}

        result = ensure_citations(response_data, formatter)
        assert result == response_data

    def test_response_with_claims_but_no_citations(self):
        """Test response with claims but missing citations."""
        formatter = CitationFormatter()
        response_data = {"hits": [{"snippet": "code snippet", "score": 0.9}]}

        with pytest.raises(ValueError, match="Constitutional violation"):
            ensure_citations(response_data, formatter)

    def test_response_with_invalid_citations(self):
        """Test response with invalid citation data."""
        formatter = CitationFormatter()
        response_data = {
            "symbols": [{"name": "test_func"}],
            "cites": [
                {
                    "span": {
                        "path": "",  # Invalid empty path
                        "sha": "abc123",
                        "start": 10,
                        "end": 20,
                    }
                }
            ],
        }

        with pytest.raises(ValueError, match="Invalid citations"):
            ensure_citations(response_data, formatter)

    def test_multiple_citation_fields(self):
        """Test response with multiple citation field types."""
        formatter = CitationFormatter()
        response_data = {
            "results": ["some data"],
            "cites": [formatter.to_dict(Span("test1.c", "abc123", 10, 20))],
            "references": [
                formatter.to_dict(Citation(Span("test2.c", "def456", 30, 40)))
            ],
        }

        result = ensure_citations(response_data, formatter)
        assert result == response_data


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_span_from_symbol(self):
        """Test span_from_symbol convenience function."""
        span = span_from_symbol(
            symbol_name="test_function",
            file_path="kernel/test.c",
            start_line=100,
            end_line=150,
            sha="abcdef123456",
        )

        assert isinstance(span, Span)
        assert span.path == "kernel/test.c"
        assert span.start == 100
        assert span.end == 150
        assert span.sha == "abcdef12"

    def test_cite_function_def(self):
        """Test cite_function_def convenience function."""
        citation = cite_function_def(
            func_name="sys_read",
            file_path="fs/read_write.c",
            start_line=500,
            end_line=600,
            sha="fedcba987654",
        )

        assert isinstance(citation, Citation)
        assert citation.span.path == "fs/read_write.c"
        assert citation.span.start == 500
        assert citation.span.end == 600
        assert citation.context == "Function sys_read definition"

    def test_cite_call_site(self):
        """Test cite_call_site convenience function."""
        citation = cite_call_site(
            caller="vfs_read",
            callee="generic_file_read",
            file_path="fs/read_write.c",
            line=450,
            sha="123456789abc",
        )

        assert isinstance(citation, Citation)
        assert citation.span.path == "fs/read_write.c"
        assert citation.span.start == 450
        assert citation.span.end == 450
        assert citation.context == "vfs_read calls generic_file_read"


class TestSerializationRoundTrip:
    """Tests for serialization and deserialization."""

    def test_span_json_roundtrip(self):
        """Test span JSON serialization round trip."""
        formatter = CitationFormatter()
        original_span = Span("drivers/block/loop.c", "deadbeef", 200, 300)

        # To dict and back
        span_dict = formatter.to_dict(original_span)
        restored_span = formatter.from_dict(span_dict)

        assert restored_span.path == original_span.path
        assert restored_span.sha == original_span.sha
        assert restored_span.start == original_span.start
        assert restored_span.end == original_span.end

    def test_citation_json_roundtrip(self):
        """Test citation JSON serialization round trip."""
        formatter = CitationFormatter()
        original_citation = Citation(
            span=Span("net/ipv4/tcp.c", "cafebabe", 1000, 1100),
            context="TCP socket handling",
        )

        # To dict and back
        citation_dict = formatter.to_dict(original_citation)
        restored_citation = formatter.from_dict(citation_dict)

        assert restored_citation.span.path == original_citation.span.path
        assert restored_citation.span.sha == original_citation.span.sha
        assert restored_citation.span.start == original_citation.span.start
        assert restored_citation.span.end == original_citation.span.end
        assert restored_citation.context == original_citation.context

    def test_span_text_roundtrip(self):
        """Test span text format round trip."""
        formatter = CitationFormatter()
        original_span = Span("arch/arm64/kernel/entry.S", "12345678", 50, 75)

        # To text and back
        span_text = formatter.format_span_text(original_span)
        restored_span = formatter.parse_span_text(span_text)

        assert restored_span.path == original_span.path
        assert restored_span.sha == original_span.sha
        assert restored_span.start == original_span.start
        assert restored_span.end == original_span.end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
