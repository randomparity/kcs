"""
Simplified unit tests for Tree-sitter call pattern queries.

Tests basic Tree-sitter functionality with hand-crafted queries
that avoid problematic patterns in the main query file.
"""

from typing import Any

import pytest
import tree_sitter_c
from tree_sitter import Language, Node, Parser, Query, QueryCursor


class TestSimpleCallPatterns:
    """Test basic Tree-sitter call pattern functionality."""

    @pytest.fixture
    def language(self) -> Language:
        """Get C language for Tree-sitter."""
        return Language(tree_sitter_c.language())

    @pytest.fixture
    def parser(self, language: Language) -> Parser:
        """Create C language parser."""
        parser = Parser()
        parser.language = language
        return parser

    def parse_code(self, parser: Parser, code: str) -> Any:
        """Parse C code and return syntax tree."""
        return parser.parse(bytes(code, "utf8"))

    def get_capture_text(self, node: Node, source: str) -> str:
        """Extract text content from a Tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source[start_byte:end_byte]

    def test_basic_function_calls(self, parser: Parser, language: Language) -> None:
        """Test basic function call detection."""
        query_source = """
        (call_expression
          function: (identifier) @function-name
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = """
        void test_function() {
            printf("Hello, world!");
            strlen("test string");
            malloc(100);
        }
        """

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        function_names = []
        for _pattern_index, captures in matches:
            if "function-name" in captures:
                for node in captures["function-name"]:
                    text = self.get_capture_text(node, code)
                    function_names.append(text)

        assert "printf" in function_names
        assert "strlen" in function_names
        assert "malloc" in function_names

    def test_function_pointer_calls(self, parser: Parser, language: Language) -> None:
        """Test function pointer call detection."""
        query_source = """
        (call_expression
          function: (parenthesized_expression
            (pointer_expression
              argument: (identifier) @pointer-name))
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = """
        void test_pointers() {
            (*func_ptr)(arg1, arg2);
        }
        """

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        pointer_names = []
        for _pattern_index, captures in matches:
            if "pointer-name" in captures:
                for node in captures["pointer-name"]:
                    text = self.get_capture_text(node, code)
                    pointer_names.append(text)

        assert "func_ptr" in pointer_names

    def test_member_function_calls(self, parser: Parser, language: Language) -> None:
        """Test struct member function calls."""
        query_source = """
        (call_expression
          function: (field_expression
            argument: (identifier) @struct-name
            field: (field_identifier) @function-name)
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = """
        void test_members() {
            obj.method(param);
            ptr->func(arg1, arg2);
        }
        """

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        struct_names = []
        function_names = []
        for _pattern_index, captures in matches:
            if "struct-name" in captures:
                for node in captures["struct-name"]:
                    text = self.get_capture_text(node, code)
                    struct_names.append(text)
            if "function-name" in captures:
                for node in captures["function-name"]:
                    text = self.get_capture_text(node, code)
                    function_names.append(text)

        assert "obj" in struct_names
        assert "method" in function_names
        assert "func" in function_names

    def test_uppercase_macro_calls(self, parser: Parser, language: Language) -> None:
        """Test uppercase macro function calls."""
        query_source = """
        (call_expression
          function: (identifier) @macro-name
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = """
        void test_macros() {
            EXPORT_SYMBOL(my_function);
            DEBUG_PRINT("debug message");
        }
        """

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        macro_names = []
        for _pattern_index, captures in matches:
            if "macro-name" in captures:
                for node in captures["macro-name"]:
                    text = self.get_capture_text(node, code)
                    macro_names.append(text)

        assert "EXPORT_SYMBOL" in macro_names
        assert "DEBUG_PRINT" in macro_names

    def test_function_pointer_assignments(
        self, parser: Parser, language: Language
    ) -> None:
        """Test function pointer assignment detection."""
        query_source = """
        (assignment_expression
          left: (field_expression
            argument: (identifier) @struct-name
            field: (field_identifier) @field-name)
          right: (identifier) @function-name) @assignment-site
        """

        query = Query(language, query_source)
        code = """
        void setup() {
            obj.callback = my_callback;
        }
        """

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        struct_names = []
        field_names = []
        function_names = []
        for _pattern_index, captures in matches:
            if "struct-name" in captures:
                for node in captures["struct-name"]:
                    text = self.get_capture_text(node, code)
                    struct_names.append(text)
            if "field-name" in captures:
                for node in captures["field-name"]:
                    text = self.get_capture_text(node, code)
                    field_names.append(text)
            if "function-name" in captures:
                for node in captures["function-name"]:
                    text = self.get_capture_text(node, code)
                    function_names.append(text)

        assert "obj" in struct_names
        assert "callback" in field_names
        assert "my_callback" in function_names

    def test_array_function_calls(self, parser: Parser, language: Language) -> None:
        """Test function calls through array indexing."""
        query_source = """
        (call_expression
          function: (subscript_expression
            argument: (identifier) @array-name
            index: (_) @index)
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = """
        void test_arrays() {
            handlers[0](param);
            callbacks[i](data);
        }
        """

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        array_names = []
        for _pattern_index, captures in matches:
            if "array-name" in captures:
                for node in captures["array-name"]:
                    text = self.get_capture_text(node, code)
                    array_names.append(text)

        assert "handlers" in array_names
        assert "callbacks" in array_names

    def test_complex_call_patterns(self, parser: Parser, language: Language) -> None:
        """Test complex call patterns like chained calls."""
        query_source = """
        (call_expression
          function: (field_expression
            argument: (call_expression) @inner-call
            field: (field_identifier) @function-name)
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = """
        void test_complex() {
            get_device()->read(buffer);
        }
        """

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        function_names = []
        for _pattern_index, captures in matches:
            if "function-name" in captures:
                for node in captures["function-name"]:
                    text = self.get_capture_text(node, code)
                    function_names.append(text)

        assert "read" in function_names

    def test_empty_code(self, parser: Parser, language: Language) -> None:
        """Test handling of empty code."""
        query_source = """
        (call_expression
          function: (identifier) @function-name
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = ""

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = list(cursor.matches(tree.root_node))

        # Should not crash and should return empty results
        assert len(matches) == 0

    def test_malformed_code(self, parser: Parser, language: Language) -> None:
        """Test handling of malformed code."""
        query_source = """
        (call_expression
          function: (identifier) @function-name
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        code = "void incomplete_function( {"

        tree = self.parse_code(parser, code)
        cursor = QueryCursor(query)
        matches = list(cursor.matches(tree.root_node))

        # Should not crash
        assert isinstance(matches, list)

    def test_query_cursor_reuse(self, parser: Parser, language: Language) -> None:
        """Test that query cursors can be reused."""
        query_source = """
        (call_expression
          function: (identifier) @function-name
          arguments: (argument_list) @args) @call-site
        """

        query = Query(language, query_source)
        cursor = QueryCursor(query)

        # First usage
        code1 = "void test1() { func1(); }"
        tree1 = self.parse_code(parser, code1)
        matches1 = list(cursor.matches(tree1.root_node))

        # Second usage - cursor should work fine
        code2 = "void test2() { func2(); }"
        tree2 = self.parse_code(parser, code2)
        matches2 = list(cursor.matches(tree2.root_node))

        assert len(matches1) > 0
        assert len(matches2) > 0
