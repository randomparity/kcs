"""
Python-Rust bridge integration for KCS.

Provides high-level Python interface to Rust parsing components
using the compiled PyO3 extension module.
"""

import logging
from typing import Any

try:
    import kcs_python_bridge

    RUST_BRIDGE_AVAILABLE = True
except ImportError:
    RUST_BRIDGE_AVAILABLE = False
    logging.warning(
        "Rust bridge not available, falling back to Python-only implementation"
    )

import structlog

logger = structlog.get_logger(__name__)


class RustParser:
    """
    High-level interface to Rust parser components.

    Wraps the PyO3 bindings and provides error handling and logging.
    """

    def __init__(
        self,
        tree_sitter_enabled: bool = True,
        clang_enabled: bool = False,
        target_arch: str = "x86_64",
        kernel_version: str = "6.1",
    ):
        self.config = {
            "tree_sitter_enabled": tree_sitter_enabled,
            "clang_enabled": clang_enabled,
            "target_arch": target_arch,
            "kernel_version": kernel_version,
        }

        if RUST_BRIDGE_AVAILABLE:
            try:
                self.parser = kcs_python_bridge.PyParser(
                    tree_sitter_enabled, clang_enabled, target_arch, kernel_version
                )
                logger.info("Rust parser initialized successfully", config=self.config)
            except Exception as e:
                logger.error("Failed to initialize Rust parser", error=str(e))
                raise RuntimeError(f"Rust parser initialization failed: {e}") from e
        else:
            self.parser = None
            logger.warning("Using fallback implementation (Rust bridge not available)")

    def parse_file(self, file_path: str, content: str) -> dict[str, Any]:
        """
        Parse a single file and extract symbols.

        Args:
            file_path: Path to the file (for context)
            content: File content to parse

        Returns:
            Dictionary with symbols, call_edges, and errors
        """
        if not RUST_BRIDGE_AVAILABLE or self.parser is None:
            return self._fallback_parse_file(file_path, content)

        try:
            result = self.parser.parse_file(file_path, content)

            return {
                "symbols": [
                    {
                        "name": symbol.name,
                        "kind": symbol.kind,
                        "file_path": symbol.file_path,
                        "start_line": symbol.start_line,
                        "end_line": symbol.end_line,
                        "signature": symbol.signature,
                    }
                    for symbol in result.symbols
                ],
                "call_edges": result.call_edges,
                "errors": result.errors,
            }

        except Exception as e:
            logger.error("Rust parsing failed", file_path=file_path, error=str(e))
            # Fall back to Python implementation
            return self._fallback_parse_file(file_path, content)

    def parse_files(self, files: dict[str, str]) -> dict[str, Any]:
        """
        Parse multiple files in batch.

        Args:
            files: Dictionary mapping file paths to content

        Returns:
            Combined parse results
        """
        if not RUST_BRIDGE_AVAILABLE or self.parser is None:
            return self._fallback_parse_files(files)

        try:
            result = self.parser.parse_files(files)

            return {
                "symbols": [
                    {
                        "name": symbol.name,
                        "kind": symbol.kind,
                        "file_path": symbol.file_path,
                        "start_line": symbol.start_line,
                        "end_line": symbol.end_line,
                        "signature": symbol.signature,
                    }
                    for symbol in result.symbols
                ],
                "call_edges": result.call_edges,
                "errors": result.errors,
            }

        except Exception as e:
            logger.error(
                "Rust batch parsing failed", files_count=len(files), error=str(e)
            )
            return self._fallback_parse_files(files)

    def parse_kernel_tree(self, kernel_path: str, config_name: str) -> dict[str, Any]:
        """
        Parse an entire kernel source tree.

        Args:
            kernel_path: Path to kernel source directory
            config_name: Kernel configuration name

        Returns:
            Parse results for all files in tree
        """
        if not RUST_BRIDGE_AVAILABLE or self.parser is None:
            return self._fallback_parse_kernel_tree(kernel_path, config_name)

        try:
            result = self.parser.parse_kernel_tree(kernel_path, config_name)

            return {
                "symbols": [
                    {
                        "name": symbol.name,
                        "kind": symbol.kind,
                        "file_path": symbol.file_path,
                        "start_line": symbol.start_line,
                        "end_line": symbol.end_line,
                        "signature": symbol.signature,
                    }
                    for symbol in result.symbols
                ],
                "call_edges": result.call_edges,
                "errors": result.errors,
            }

        except Exception as e:
            logger.error(
                "Rust kernel tree parsing failed",
                kernel_path=kernel_path,
                config=config_name,
                error=str(e),
            )
            return self._fallback_parse_kernel_tree(kernel_path, config_name)

    def analyze_kernel_patterns(self, content: str) -> list[str]:
        """
        Analyze content for kernel-specific patterns.

        Args:
            content: Source code content

        Returns:
            List of detected pattern names
        """
        if RUST_BRIDGE_AVAILABLE:
            try:
                return kcs_python_bridge.analyze_kernel_patterns(content)
            except Exception as e:
                logger.error("Rust pattern analysis failed", error=str(e))

        # Fallback pattern detection
        patterns = []
        kernel_patterns = [
            "EXPORT_SYMBOL",
            "EXPORT_SYMBOL_GPL",
            "module_param",
            "MODULE_LICENSE",
            "MODULE_AUTHOR",
            "MODULE_DESCRIPTION",
            "subsys_initcall",
            "module_init",
            "module_exit",
            "__init",
            "__exit",
            "SYSCALL_DEFINE",
            "asmlinkage",
        ]

        for pattern in kernel_patterns:
            if pattern in content:
                patterns.append(pattern)

        return patterns

    def configure(self, config: dict[str, Any]) -> None:
        """
        Update parser configuration.

        Args:
            config: Configuration dictionary
        """
        self.config.update(config)

        if RUST_BRIDGE_AVAILABLE and self.parser is not None:
            try:
                self.parser.configure(config)
                logger.info("Parser reconfigured", config=config)
            except Exception as e:
                logger.error("Parser reconfiguration failed", error=str(e))
                raise

    def _fallback_parse_file(self, file_path: str, content: str) -> dict[str, Any]:
        """
        Fallback Python-only file parsing.

        This provides basic symbol extraction when Rust bridge is unavailable.
        """
        logger.debug("Using fallback file parsing", file_path=file_path)

        # Simple pattern-based symbol extraction
        symbols = []
        errors = []

        try:
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                line = line.strip()

                # Function definitions
                if (
                    line.startswith("int ")
                    or line.startswith("void ")
                    or line.startswith("static ")
                ):
                    if "(" in line and "{" in line:
                        func_name = self._extract_function_name(line)
                        if func_name:
                            symbols.append(
                                {
                                    "name": func_name,
                                    "kind": "function",
                                    "file_path": file_path,
                                    "start_line": i,
                                    "end_line": i,
                                    "signature": line,
                                }
                            )

                # Struct definitions
                elif line.startswith("struct "):
                    if "{" in line:
                        struct_name = self._extract_struct_name(line)
                        if struct_name:
                            symbols.append(
                                {
                                    "name": struct_name,
                                    "kind": "struct",
                                    "file_path": file_path,
                                    "start_line": i,
                                    "end_line": i,
                                    "signature": line,
                                }
                            )

        except Exception as e:
            errors.append(f"Fallback parsing error: {e}")

        return {"symbols": symbols, "call_edges": [], "errors": errors}

    def _fallback_parse_files(self, files: dict[str, str]) -> dict[str, Any]:
        """Fallback batch file parsing."""
        all_symbols = []
        all_errors = []

        for file_path, content in files.items():
            result = self._fallback_parse_file(file_path, content)
            all_symbols.extend(result["symbols"])
            all_errors.extend(result["errors"])

        return {"symbols": all_symbols, "call_edges": [], "errors": all_errors}

    def _fallback_parse_kernel_tree(
        self, kernel_path: str, config_name: str
    ) -> dict[str, Any]:
        """Fallback kernel tree parsing."""
        logger.warning("Fallback kernel tree parsing not fully implemented")
        return {
            "symbols": [],
            "call_edges": [],
            "errors": ["Fallback kernel tree parsing not implemented"],
        }

    def _extract_function_name(self, line: str) -> str | None:
        """Extract function name from function definition line."""
        try:
            if "(" in line:
                before_paren = line.split("(")[0]
                parts = before_paren.split()
                if parts:
                    return parts[-1].strip("*")
        except Exception:
            pass
        return None

    def _extract_struct_name(self, line: str) -> str | None:
        """Extract struct name from struct definition line."""
        try:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "struct":
                return parts[1].strip("{")
        except Exception:
            pass
        return None


# Global parser instance for the application
_rust_parser: RustParser | None = None


def get_rust_parser() -> RustParser:
    """Get or create global Rust parser instance."""
    global _rust_parser
    if _rust_parser is None:
        _rust_parser = RustParser()
    return _rust_parser


def parse_c_file(file_path: str, content: str, arch: str = "x86_64") -> dict[str, Any]:
    """
    Convenience function for parsing a single C file.

    Args:
        file_path: Path to the file
        content: File content
        arch: Target architecture

    Returns:
        Parse results dictionary
    """
    if RUST_BRIDGE_AVAILABLE:
        try:
            return kcs_python_bridge.parse_c_file(file_path, content, arch)
        except Exception as e:
            logger.error("Direct Rust parsing failed", error=str(e))

    # Use global parser instance
    parser = get_rust_parser()
    return parser.parse_file(file_path, content)
