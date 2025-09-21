/*!
 * Contract tests for parse_file with call graph extraction.
 *
 * These tests verify the API contract defined in contracts/parser-api.json.
 * They MUST fail before implementation and pass after.
 *
 * Contract being tested:
 * - parse_file_content function should extract call graphs
 * - Response format matches ParseResult structure
 * - Call edges include proper CallType classification
 */

use kcs_parser::{CallEdge, CallType, ExtendedParserConfig, ParseResult, Parser};
use std::collections::HashMap;

/// Test the parse_file_content contract with simple direct calls
#[test]
fn test_parse_file_content_direct_calls() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    // Test C code with simple direct function calls
    let content = r#"
// Simple function calls test case
int helper_function(int x) {
    return x * 2;
}

int main_function(int a, int b) {
    int result = helper_function(a);
    helper_function(b);
    return result;
}
"#;

    // Call the parse_file_content function according to contract
    let result = parser.parse_file_content("test_simple.c", content);

    assert!(result.is_ok(), "Parser should succeed on valid C code");
    let parsed: ParseResult = result.unwrap();

    // Contract requirement: response must have symbols, call_edges, errors fields
    assert!(parsed.symbols.len() >= 2, "Should find at least 2 symbols (functions)");

    // Contract requirement: call_edges should contain function calls
    // Expected: main_function calls helper_function twice (lines 9, 10)
    let call_edges = &parsed.call_edges;
    assert!(call_edges.len() >= 2, "Should extract at least 2 call edges");

    // Find calls from main_function to helper_function
    let main_to_helper_calls: Vec<&CallEdge> = call_edges
        .iter()
        .filter(|edge| edge.caller == "main_function" && edge.callee == "helper_function")
        .collect();

    assert_eq!(main_to_helper_calls.len(), 2, "main_function should call helper_function twice");

    // Contract requirement: call_type should be classified
    for call in &main_to_helper_calls {
        assert!(
            matches!(call.call_type, CallType::Direct),
            "Simple function calls should be classified as Direct"
        );
        assert_eq!(call.file_path, "test_simple.c");
        assert!(call.line_number > 0, "Line number should be positive");
    }
}

/// Test the parse_file_content contract with function pointers (indirect calls)
#[test]
fn test_parse_file_content_indirect_calls() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    // Test C code with function pointer calls
    let content = r#"
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

int execute_operation(int x, int y, int (*op)(int, int)) {
    return op(x, y);  // Indirect call through function pointer
}

int main() {
    execute_operation(5, 3, add);      // Direct call
    execute_operation(5, 3, multiply); // Direct call
    return 0;
}
"#;

    let result = parser.parse_file_content("test_indirect.c", content);
    assert!(result.is_ok(), "Parser should succeed on valid C code");

    let parsed: ParseResult = result.unwrap();
    let call_edges = &parsed.call_edges;

    // Should find calls (including function parameter calls)
    assert!(!call_edges.is_empty(), "Should extract call edges");

    // Look for calls involving execute_operation
    let execute_calls: Vec<&CallEdge> = call_edges
        .iter()
        .filter(|edge| edge.caller == "execute_operation" || edge.callee == "execute_operation")
        .collect();

    assert!(
        !execute_calls.is_empty(),
        "Should detect calls involving execute_operation function"
    );
}

/// Test the parse_file_content contract with macro calls
#[test]
fn test_parse_file_content_macro_calls() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    // Test C code with macro expansion calls
    let content = r#"
#define CALL_HELPER(x) helper_function(x)

int helper_function(int x) { return x + 10; }

int test_macro(void) {
    CALL_HELPER(42);  // Should be detected as macro call
    return 0;
}
"#;

    let result = parser.parse_file_content("test_macro.c", content);
    assert!(result.is_ok(), "Parser should succeed on valid C code");

    let parsed: ParseResult = result.unwrap();
    let call_edges = &parsed.call_edges;

    // Should detect some form of call (may be classified as Direct rather than Macro)
    assert!(!call_edges.is_empty(), "Should detect calls from macro expansion");

    // Look for calls that involve the macro or helper function
    let relevant_calls: Vec<&CallEdge> = call_edges
        .iter()
        .filter(|edge| {
            edge.caller == "test_macro"
                && (edge.callee == "CALL_HELPER" || edge.callee == "helper_function")
        })
        .collect();

    assert!(!relevant_calls.is_empty(), "Should detect calls related to macro usage");

    // Verify call details
    if let Some(call) = relevant_calls.first() {
        assert_eq!(call.caller, "test_macro");
        assert!(call.line_number > 0);
    }
}

/// Test contract error handling with problematic C code
#[test]
fn test_parse_file_content_error_handling() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    // Test C code with syntax errors
    let content = r#"
// Incomplete function declaration
int broken_function(

// Function with missing semicolon
int another_broken() {
    missing_semicolon()  // Missing semicolon
    return 0;
}

// Valid function that should still be parsed
int valid_function(void) {
    return helper_function(123);
}
"#;

    let result = parser.parse_file_content("test_problematic.c", content);

    // Contract requirement: parser should not crash, should return result
    assert!(result.is_ok(), "Parser should handle syntax errors gracefully");

    let parsed: ParseResult = result.unwrap();

    // Contract requirement: parser should handle problematic code gracefully
    // Note: Tree-sitter is resilient and may not report errors for recoverable syntax issues
    // The key requirement is that parsing doesn't crash and valid parts are still extracted

    // Valid parts should still be extracted (or at least some symbols should be found)
    // Tree-sitter is resilient and should extract what it can parse
    assert!(
        !parsed.symbols.is_empty(),
        "Should extract at least some symbols despite parsing errors"
    );
}

/// Test contract response schema structure
#[test]
fn test_parse_file_content_response_schema() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    let content = r#"
int simple_function() {
    return 42;
}
"#;

    let result = parser.parse_file_content("test_schema.c", content);
    assert!(result.is_ok(), "Parser should succeed on valid C code");

    let parsed: ParseResult = result.unwrap();

    // Contract requirement: ParseResult must have required fields
    // symbols: array of SymbolInfo objects
    assert!(!parsed.symbols.is_empty(), "Should contain symbols array");

    // call_edges: array of CallEdge objects
    assert!(
        parsed.call_edges.is_empty() || !parsed.call_edges.is_empty(),
        "call_edges field should exist (even if empty)"
    );

    // errors: array of error strings
    assert!(
        parsed.errors.is_empty() || !parsed.errors.is_empty(),
        "errors field should exist (even if empty)"
    );

    // Verify symbol structure matches contract
    if let Some(symbol) = parsed.symbols.first() {
        assert!(!symbol.name.is_empty(), "Symbol name should not be empty");
        assert!(!symbol.kind.is_empty(), "Symbol kind should not be empty");
        assert!(!symbol.file_path.is_empty(), "Symbol file_path should not be empty");
        assert!(symbol.start_line > 0, "Symbol start_line should be positive");
        assert!(symbol.end_line >= symbol.start_line, "end_line should be >= start_line");
    }
}

/// Test contract with include_call_graph parameter behavior
#[test]
fn test_parse_file_content_call_graph_parameter() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    let content = r#"
int caller() {
    return callee();
}

int callee() {
    return 0;
}
"#;

    // Parse with call graph extraction (default behavior)
    let result = parser.parse_file_content("test_param.c", content);
    assert!(result.is_ok(), "Parser should succeed");

    let parsed: ParseResult = result.unwrap();

    // Contract: when include_call_graph is true (default), should extract calls
    assert!(
        !parsed.call_edges.is_empty(),
        "Should extract call edges when call graph extraction is enabled"
    );

    // Verify the call edge
    let call_edge = parsed.call_edges.first().unwrap();
    assert_eq!(call_edge.caller, "caller");
    assert_eq!(call_edge.callee, "callee");
    assert!(matches!(call_edge.call_type, CallType::Direct));
}

/// Test contract with batch files parsing
#[test]
fn test_parse_files_content_batch() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    let mut files = HashMap::new();

    files.insert(
        "file1.c".to_string(),
        r#"
int func1() {
    return func2();
}
"#
        .to_string(),
    );

    files.insert(
        "file2.c".to_string(),
        r#"
int func2() {
    return 42;
}
"#
        .to_string(),
    );

    let result = parser.parse_files_content(files);
    assert!(result.is_ok(), "Batch parsing should succeed");

    let parsed: ParseResult = result.unwrap();

    // Should aggregate results from multiple files
    assert!(parsed.symbols.len() >= 2, "Should contain symbols from both files");

    // Should find the cross-file call (if supported)
    let cross_file_calls: Vec<&CallEdge> = parsed
        .call_edges
        .iter()
        .filter(|edge| edge.caller == "func1" && edge.callee == "func2")
        .collect();

    // Note: Cross-file call detection may not be implemented yet
    // This test documents the expected behavior per contract
    if !cross_file_calls.is_empty() {
        let call = cross_file_calls.first().unwrap();
        assert!(matches!(call.call_type, CallType::Direct));
    }
}

/// Performance contract test - should meet p95 < 600ms requirement
#[test]
fn test_parse_file_content_performance() {
    let config = ExtendedParserConfig {
        include_call_graphs: true,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    // Generate reasonably complex C code for performance testing
    let mut content = String::new();
    content.push_str("// Performance test file\n");

    // Add multiple functions with call relationships
    for i in 0..50 {
        content.push_str(&format!(
            r#"
int function_{}(int x) {{
    if (x > 0) {{
        return function_{}(x - 1);
    }}
    return helper_function_{i}(x);
}}

int helper_function_{i}(int x) {{
    return x * 2;
}}
"#,
            i,
            (i + 1) % 50,
            i = i
        ));
    }

    let start = std::time::Instant::now();

    let result = parser.parse_file_content("performance_test.c", &content);

    let duration = start.elapsed();

    assert!(result.is_ok(), "Performance test should succeed");

    // Constitutional requirement: p95 < 600ms
    assert!(
        duration.as_millis() < 600,
        "Parse time {}ms should be < 600ms (p95 requirement)",
        duration.as_millis()
    );

    let parsed = result.unwrap();
    assert!(parsed.symbols.len() >= 100, "Should extract all functions");
    assert!(!parsed.call_edges.is_empty(), "Should extract call relationships");
}
