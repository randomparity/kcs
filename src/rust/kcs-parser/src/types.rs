//! Type definitions for call graph extraction
//!
//! Defines data structures for representing function call relationships
//! extracted from C source code using tree-sitter AST analysis.

use serde::{Deserialize, Serialize};

/// Type of function call detected during parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallType {
    /// Standard function call: function_name(args)
    Direct,
    /// Function pointer call: (*func_ptr)(args) or func_ptr(args)
    Indirect,
    /// Macro invocation that expands to function call
    Macro,
}

impl std::fmt::Display for CallType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallType::Direct => write!(f, "Direct"),
            CallType::Indirect => write!(f, "Indirect"),
            CallType::Macro => write!(f, "Macro"),
        }
    }
}

/// Represents a function call relationship between two functions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEdge {
    /// Name of the calling function
    pub caller: String,
    /// Name of the called function
    pub callee: String,
    /// Source file path where the call occurs
    pub file_path: String,
    /// Line number of the call site
    pub line_number: u32,
    /// Type of function call
    pub call_type: CallType,
}

impl CallEdge {
    /// Create a new call edge
    pub fn new(
        caller: String,
        callee: String,
        file_path: String,
        line_number: u32,
        call_type: CallType,
    ) -> Self {
        Self {
            caller,
            callee,
            file_path,
            line_number,
            call_type,
        }
    }

    /// Validate that this call edge has valid data
    pub fn is_valid(&self) -> bool {
        !self.caller.is_empty()
            && !self.callee.is_empty()
            && !self.file_path.is_empty()
            && self.line_number > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_type_display() {
        assert_eq!(CallType::Direct.to_string(), "Direct");
        assert_eq!(CallType::Indirect.to_string(), "Indirect");
        assert_eq!(CallType::Macro.to_string(), "Macro");
    }

    #[test]
    fn test_call_edge_new() {
        let edge = CallEdge::new(
            "main_function".to_string(),
            "helper_function".to_string(),
            "/path/to/file.c".to_string(),
            42,
            CallType::Direct,
        );

        assert_eq!(edge.caller, "main_function");
        assert_eq!(edge.callee, "helper_function");
        assert_eq!(edge.file_path, "/path/to/file.c");
        assert_eq!(edge.line_number, 42);
        assert_eq!(edge.call_type, CallType::Direct);
    }

    #[test]
    fn test_call_edge_validation() {
        let valid_edge = CallEdge::new(
            "caller".to_string(),
            "callee".to_string(),
            "file.c".to_string(),
            10,
            CallType::Direct,
        );
        assert!(valid_edge.is_valid());

        let invalid_edge = CallEdge::new(
            "".to_string(),
            "callee".to_string(),
            "file.c".to_string(),
            10,
            CallType::Direct,
        );
        assert!(!invalid_edge.is_valid());

        let invalid_line_edge = CallEdge::new(
            "caller".to_string(),
            "callee".to_string(),
            "file.c".to_string(),
            0,
            CallType::Direct,
        );
        assert!(!invalid_line_edge.is_valid());
    }

    #[test]
    fn test_serialization() {
        let edge = CallEdge::new(
            "main".to_string(),
            "helper".to_string(),
            "test.c".to_string(),
            15,
            CallType::Indirect,
        );

        let json = serde_json::to_string(&edge).unwrap();
        let deserialized: CallEdge = serde_json::from_str(&json).unwrap();

        assert_eq!(edge, deserialized);
    }
}
