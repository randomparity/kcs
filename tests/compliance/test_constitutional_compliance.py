#!/usr/bin/env python3
"""
Constitutional compliance validation for semantic search engine.

Validates the implementation against KCS constitutional requirements:
- Read-Only Safety
- Citation-Based Truth
- MCP-First Interface
- Performance Boundaries
- Configuration Awareness
"""

import asyncio
import importlib.util
import inspect
import json
import time
from pathlib import Path
from typing import Any

import pytest
import structlog

logger = structlog.get_logger(__name__)


class ConstitutionalComplianceValidator:
    """Validates semantic search against KCS constitutional requirements."""

    def __init__(self):
        self.violations: list[str] = []
        self.warnings: list[str] = []

    def add_violation(self, requirement: str, message: str) -> None:
        """Add a constitutional violation."""
        violation = f"VIOLATION [{requirement}]: {message}"
        self.violations.append(violation)
        logger.error(
            "Constitutional violation detected",
            requirement=requirement,
            message=message,
        )

    def add_warning(self, requirement: str, message: str) -> None:
        """Add a constitutional warning."""
        warning = f"WARNING [{requirement}]: {message}"
        self.warnings.append(warning)
        logger.warning(
            "Constitutional warning", requirement=requirement, message=message
        )

    def is_compliant(self) -> bool:
        """Check if implementation is constitutionally compliant."""
        return len(self.violations) == 0


@pytest.mark.compliance
class TestConstitutionalCompliance:
    """Test suite for constitutional compliance validation."""

    def setup_method(self):
        """Setup for each test method."""
        self.validator = ConstitutionalComplianceValidator()

    def test_read_only_safety(self):
        """
        Requirement I: Read-Only Safety
        The KCS system must operate in read-only mode on kernel repositories.
        No code generation, modification, or mutation of kernel source is permitted.
        """
        logger.info("Validating Read-Only Safety compliance")

        # Check that semantic search tools don't have write operations
        mcp_tool_files = [
            Path("src/python/semantic_search/mcp/search_tool.py"),
            Path("src/python/semantic_search/mcp/index_tool.py"),
            Path("src/python/semantic_search/mcp/status_tool.py"),
        ]

        for tool_file in mcp_tool_files:
            if not tool_file.exists():
                self.validator.add_warning(
                    "Read-Only Safety", f"MCP tool file not found: {tool_file}"
                )
                continue

            source_content = tool_file.read_text()

            # Check for prohibited write operations
            prohibited_write_patterns = [
                ("write(", "file write operation"),
                ('w"', "file opened in write mode"),
                ("w'", "file opened in write mode"),
                ('wb"', "file opened in binary write mode"),
                ("wb'", "file opened in binary write mode"),
                ('wa"', "file opened in append mode"),
                ("wa'", "file opened in append mode"),
                (".write(", "method write call"),
                ("file.write", "file write method"),
                ("shutil.move", "file move operation"),
                ("shutil.copy", "file copy operation"),
                ("os.remove", "file deletion"),
                ("os.unlink", "file deletion"),
                ("pathlib.unlink", "file deletion"),
                ('mode="w"', "file opened in write mode"),
                ("mode='w'", "file opened in write mode"),
                ('mode="wb"', "file opened in binary write mode"),
                ("mode='wb'", "file opened in binary write mode"),
            ]

            for pattern, description in prohibited_write_patterns:
                if pattern in source_content:
                    # Additional check: ensure it's not in comments or strings used for other purposes
                    lines = source_content.split("\n")
                    for i, line in enumerate(lines, 1):
                        if pattern in line and not line.strip().startswith("#"):
                            # Check if it's in a docstring or comment
                            if (
                                '"""' in line
                                or "'''" in line
                                or line.strip().startswith("*")
                            ):
                                continue
                            # Check if it's a read-only open call
                            if "open(" in pattern and (
                                "encoding=" in line
                                or 'mode="r"' in line
                                or "mode='r'" in line
                            ):
                                continue  # This is a read-only operation
                            self.validator.add_violation(
                                "Read-Only Safety",
                                f"Tool {tool_file.name} contains {description} '{pattern}' at line {i}",
                            )

        # Verify that semantic search only reads from database
        search_service_path = Path(
            "src/python/semantic_search/services/vector_search_service.py"
        )
        if search_service_path.exists():
            content = search_service_path.read_text()
            # Check for SQL operations that modify kernel source data (not allowed)
            # vs. operations that modify search indexes/metadata (allowed)
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                line_upper = line.upper()
                if not line.strip().startswith("#") and '"""' not in line:
                    # Check for operations that would modify kernel source
                    if any(
                        op in line_upper
                        for op in [
                            "INSERT INTO kernel",
                            "UPDATE kernel",
                            "DELETE FROM kernel",
                        ]
                    ):
                        self.validator.add_violation(
                            "Read-Only Safety",
                            f"SQL operation modifying kernel data found at line {i}: {line.strip()}",
                        )

                    # Operations on search indexes/embeddings are allowed
                    # CREATE TABLE for search infrastructure is allowed
                    # UPDATE on search metadata is allowed

        assert self.validator.is_compliant(), (
            f"Read-Only Safety violations: {self.validator.violations}"
        )

    def test_citation_based_truth(self):
        """
        Requirement II: Citation-Based Truth
        Every claim, finding, or recommendation must include exact file/line citations.
        No unsourced assertions are permitted.
        """
        logger.info("Validating Citation-Based Truth compliance")

        # Check citation compliance through architectural relationships
        # SearchResult may reference VectorEmbedding/IndexedContent that contain citations

        # Check that VectorEmbedding has citation fields
        vector_embedding_path = Path(
            "src/python/semantic_search/models/vector_embedding.py"
        )
        if vector_embedding_path.exists():
            content = vector_embedding_path.read_text()
            citation_fields = ["file_path", "line_start", "line_end"]
            missing_fields = []
            for field in citation_fields:
                if field not in content:
                    missing_fields.append(field)

            if missing_fields:
                self.validator.add_violation(
                    "Citation-Based Truth",
                    f"VectorEmbedding model missing citation fields: {missing_fields}",
                )
        else:
            self.validator.add_violation(
                "Citation-Based Truth", "VectorEmbedding model file not found"
            )

        # Check that IndexedContent has file path
        indexed_content_path = Path(
            "src/python/semantic_search/models/indexed_content.py"
        )
        if indexed_content_path.exists():
            content = indexed_content_path.read_text()
            if "file_path" not in content:
                self.validator.add_violation(
                    "Citation-Based Truth",
                    "IndexedContent model missing file_path field",
                )
        else:
            self.validator.add_violation(
                "Citation-Based Truth", "IndexedContent model file not found"
            )

        # Check that SearchResult can retrieve citation information via relationships
        search_result_path = Path("src/python/semantic_search/models/search_result.py")
        if search_result_path.exists():
            content = search_result_path.read_text()
            # Check for content_id reference that should link to citation-containing models
            if "content_id" not in content:
                self.validator.add_violation(
                    "Citation-Based Truth",
                    "SearchResult model missing content_id field for citation retrieval",
                )

            # Check for methods to retrieve citation information
            citation_methods = ["get_content", "get_query"]
            has_citation_methods = any(method in content for method in citation_methods)
            if not has_citation_methods:
                self.validator.add_warning(
                    "Citation-Based Truth",
                    "SearchResult model may lack methods to retrieve citation information",
                )

        # Verify MCP tools return results with citations
        mcp_tools_path = Path("src/python/semantic_search/mcp")
        if mcp_tools_path.exists():
            for tool_file in mcp_tools_path.glob("*.py"):
                if tool_file.name.startswith("__"):
                    continue
                content = tool_file.read_text()
                # Check that responses include citation information
                if "return" in content and "SearchResult" in content:
                    if "file_path" not in content or "line_number" not in content:
                        self.validator.add_warning(
                            "Citation-Based Truth",
                            f"MCP tool {tool_file.name} may not properly include citations",
                        )

        assert self.validator.is_compliant(), (
            f"Citation-Based Truth violations: {self.validator.violations}"
        )

    def test_mcp_first_interface(self):
        """
        Requirement III: MCP-First Interface
        All functionality must be exposed through Model Context Protocol (MCP)
        resources and tools.
        """
        logger.info("Validating MCP-First Interface compliance")

        # Verify that all main functionality is exposed through MCP tools
        expected_mcp_tools = [
            "semantic_search_tool",
            "index_content_tool",
            "get_index_status_tool",
        ]

        mcp_tools_path = Path("src/python/semantic_search/mcp")
        if not mcp_tools_path.exists():
            self.validator.add_violation(
                "MCP-First Interface", "MCP tools directory does not exist"
            )
            return

        for tool_name in expected_mcp_tools:
            tool_file = mcp_tools_path / f"{tool_name.replace('_tool', '_tool')}.py"
            # Try different naming patterns
            possible_files = [
                mcp_tools_path / f"{tool_name}.py",
                mcp_tools_path / f"{tool_name.replace('_tool', '')}_tool.py",
                mcp_tools_path / "search_tool.py" if "search" in tool_name else None,
                mcp_tools_path / "index_tool.py" if "index" in tool_name else None,
                mcp_tools_path / "status_tool.py" if "status" in tool_name else None,
            ]

            tool_exists = any(f and f.exists() for f in possible_files if f)
            if not tool_exists:
                self.validator.add_violation(
                    "MCP-First Interface", f"Required MCP tool not found: {tool_name}"
                )

        # Check that tools follow MCP protocol structure
        for tool_file in mcp_tools_path.glob("*_tool.py"):
            content = tool_file.read_text()
            # Basic MCP tool structure validation
            required_elements = ["async def", "ToolResult", "request"]
            for element in required_elements:
                if element not in content:
                    self.validator.add_warning(
                        "MCP-First Interface",
                        f"Tool {tool_file.name} may not follow MCP protocol structure (missing {element})",
                    )

        # Verify no direct API endpoints outside of MCP
        semantic_search_path = Path("src/python/semantic_search")
        if semantic_search_path.exists():
            for py_file in semantic_search_path.rglob("*.py"):
                if "mcp" in str(py_file):
                    continue  # Skip MCP files
                content = py_file.read_text()
                # Check for FastAPI decorators that bypass MCP
                fastapi_patterns = [
                    "@app.get",
                    "@app.post",
                    "@app.put",
                    "@router.get",
                    "@router.post",
                ]
                for pattern in fastapi_patterns:
                    if pattern in content:
                        self.validator.add_warning(
                            "MCP-First Interface",
                            f"Direct API endpoint found in {py_file.name}, should use MCP interface",
                        )

        assert self.validator.is_compliant(), (
            f"MCP-First Interface violations: {self.validator.violations}"
        )

    @pytest.mark.asyncio
    async def test_performance_boundaries(self):
        """
        Requirement V: Performance Boundaries
        System must meet defined performance targets: query p95 ≤600ms.
        """
        logger.info("Validating Performance Boundaries compliance")

        # This is a basic structural test - full performance testing is in T053
        # Here we just verify that performance monitoring is in place

        # Check that search operations have timeout mechanisms
        vector_search_path = Path(
            "src/python/semantic_search/services/vector_search_service.py"
        )
        if vector_search_path.exists():
            content = vector_search_path.read_text()
            # Look for timeout or performance monitoring
            performance_indicators = [
                "timeout",
                "asyncio.wait_for",
                "time.time()",
                "performance",
            ]
            has_performance_monitoring = any(
                indicator in content for indicator in performance_indicators
            )

            if not has_performance_monitoring:
                self.validator.add_warning(
                    "Performance Boundaries",
                    "Vector search service lacks apparent performance monitoring",
                )

        # Check that MCP tools have performance considerations
        mcp_tools_path = Path("src/python/semantic_search/mcp")
        if mcp_tools_path.exists():
            for tool_file in mcp_tools_path.glob("*_tool.py"):
                content = tool_file.read_text()
                # Look for basic async patterns (performance enabler)
                if "async def" not in content:
                    self.validator.add_warning(
                        "Performance Boundaries",
                        f"MCP tool {tool_file.name} is not async, may impact performance",
                    )

        # Note: Actual performance testing (600ms requirement) is handled in T053
        logger.info(
            "Performance structure validation complete - actual timing tests in T053"
        )

        assert self.validator.is_compliant(), (
            f"Performance Boundaries violations: {self.validator.violations}"
        )

    def test_configuration_awareness(self):
        """
        Requirement IV: Configuration Awareness
        All analysis must be configuration-aware. Results must clearly indicate
        their configuration context.
        """
        logger.info("Validating Configuration Awareness compliance")

        # Check that search results can include configuration context
        search_result_path = Path("src/python/semantic_search/models/search_result.py")
        if search_result_path.exists():
            content = search_result_path.read_text()
            # Look for configuration-related fields or mentions
            config_indicators = ["config", "configuration", "context"]
            has_config_awareness = any(
                indicator in content.lower() for indicator in config_indicators
            )

            if not has_config_awareness:
                self.validator.add_warning(
                    "Configuration Awareness",
                    "SearchResult model may lack configuration context fields",
                )

        # Check that indexing considers configuration context
        index_tool_path = Path("src/python/semantic_search/mcp/index_tool.py")
        if index_tool_path.exists():
            content = index_tool_path.read_text()
            # Look for configuration handling in indexing
            if "config" not in content.lower():
                self.validator.add_warning(
                    "Configuration Awareness",
                    "Index tool may not handle kernel configuration context",
                )

        # For semantic search, configuration awareness might be through content context
        # rather than explicit configuration parameters, so warnings are appropriate

        assert self.validator.is_compliant(), (
            f"Configuration Awareness violations: {self.validator.violations}"
        )

    def test_integration_compliance(self):
        """Test overall integration compliance with KCS architecture."""
        logger.info("Validating overall KCS integration compliance")

        # Check that semantic search integrates with existing KCS structure
        kcs_app_path = Path("src/python/kcs_mcp/app.py")
        if kcs_app_path.exists():
            content = kcs_app_path.read_text()
            if "semantic_search" not in content:
                self.validator.add_violation(
                    "KCS Integration",
                    "Semantic search not integrated into main KCS application",
                )

        # Check that semantic search follows KCS logging patterns
        semantic_search_path = Path("src/python/semantic_search")
        if semantic_search_path.exists():
            found_structlog = False
            for py_file in semantic_search_path.rglob("*.py"):
                content = py_file.read_text()
                if "structlog" in content:
                    found_structlog = True
                    break

            if not found_structlog:
                self.validator.add_warning(
                    "KCS Integration",
                    "Semantic search may not use KCS structured logging patterns",
                )

        assert self.validator.is_compliant(), (
            f"Integration violations: {self.validator.violations}"
        )

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate a comprehensive compliance report."""
        return {
            "timestamp": time.time(),
            "compliant": self.validator.is_compliant(),
            "violations": self.validator.violations,
            "warnings": self.validator.warnings,
            "requirements_checked": [
                "Read-Only Safety",
                "Citation-Based Truth",
                "MCP-First Interface",
                "Performance Boundaries",
                "Configuration Awareness",
            ],
        }


# Standalone compliance checker for CI/CD
async def validate_constitutional_compliance() -> bool:
    """
    Standalone function to validate constitutional compliance.
    Returns True if compliant, False otherwise.
    """
    validator = ConstitutionalComplianceValidator()
    test_instance = TestConstitutionalCompliance()
    test_instance.setup_method()
    test_instance.validator = validator

    try:
        # Run all compliance tests
        test_instance.test_read_only_safety()
        test_instance.test_citation_based_truth()
        test_instance.test_mcp_first_interface()
        await test_instance.test_performance_boundaries()
        test_instance.test_configuration_awareness()
        test_instance.test_integration_compliance()

        # Generate report
        report = test_instance.generate_compliance_report()

        # Log results
        if validator.is_compliant():
            logger.info("✅ Constitutional compliance validation PASSED", report=report)
        else:
            logger.error(
                "❌ Constitutional compliance validation FAILED", report=report
            )

        return validator.is_compliant()

    except Exception as e:
        logger.error("Constitutional compliance validation error", error=str(e))
        return False


if __name__ == "__main__":
    # Run compliance validation directly
    asyncio.run(validate_constitutional_compliance())
