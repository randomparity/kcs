//! Type definitions for call graph analysis.
//!
//! This module defines the core types and enums used throughout the call graph
//! extraction and analysis system, including call types, confidence levels,
//! and other classification enums.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Type of function call relationship.
///
/// Represents different ways that functions can call each other in C code,
/// from direct function calls to complex indirect invocations through
/// function pointers and callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallType {
    /// Direct function call by name (e.g., `foo()`)
    Direct,

    /// Indirect call through function pointer (e.g., `(*ptr)()`)
    Indirect,

    /// Macro expansion that results in a function call
    Macro,

    /// Callback function invocation
    Callback,

    /// Conditional call (inside if/switch/loop statements)
    Conditional,

    /// Assembly language function call
    Assembly,

    /// System call invocation
    Syscall,
}

impl CallType {
    /// Returns true if this call type represents a high-confidence call.
    pub fn is_high_confidence_type(self) -> bool {
        matches!(self, CallType::Direct | CallType::Syscall)
    }

    /// Returns true if this call type represents a medium-confidence call.
    pub fn is_medium_confidence_type(self) -> bool {
        matches!(self, CallType::Indirect | CallType::Callback | CallType::Assembly)
    }

    /// Returns true if this call type represents a low-confidence call.
    pub fn is_low_confidence_type(self) -> bool {
        matches!(self, CallType::Macro | CallType::Conditional)
    }

    /// Returns the default confidence level for this call type.
    pub fn default_confidence(self) -> ConfidenceLevel {
        if self.is_high_confidence_type() {
            ConfidenceLevel::High
        } else if self.is_medium_confidence_type() {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }

    /// Returns a human-readable description of the call type.
    pub fn description(self) -> &'static str {
        match self {
            CallType::Direct => "Direct function call",
            CallType::Indirect => "Indirect call through function pointer",
            CallType::Macro => "Macro expansion call",
            CallType::Callback => "Callback function invocation",
            CallType::Conditional => "Conditional function call",
            CallType::Assembly => "Assembly language call",
            CallType::Syscall => "System call",
        }
    }

    /// Returns all possible call types.
    pub fn all() -> &'static [CallType] {
        &[
            CallType::Direct,
            CallType::Indirect,
            CallType::Macro,
            CallType::Callback,
            CallType::Conditional,
            CallType::Assembly,
            CallType::Syscall,
        ]
    }
}

impl fmt::Display for CallType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallType::Direct => write!(f, "direct"),
            CallType::Indirect => write!(f, "indirect"),
            CallType::Macro => write!(f, "macro"),
            CallType::Callback => write!(f, "callback"),
            CallType::Conditional => write!(f, "conditional"),
            CallType::Assembly => write!(f, "assembly"),
            CallType::Syscall => write!(f, "syscall"),
        }
    }
}

impl FromStr for CallType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "direct" => Ok(CallType::Direct),
            "indirect" => Ok(CallType::Indirect),
            "macro" => Ok(CallType::Macro),
            "callback" => Ok(CallType::Callback),
            "conditional" => Ok(CallType::Conditional),
            "assembly" => Ok(CallType::Assembly),
            "syscall" => Ok(CallType::Syscall),
            _ => Err(format!("Unknown call type: {}", s)),
        }
    }
}

/// Confidence level for call detection and analysis.
///
/// Represents how confident the analysis system is about the accuracy
/// of a detected call relationship or function pointer assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Low confidence (50-69% certainty)
    ///
    /// Used for relationships that may be incorrect or incomplete:
    /// - Complex macro expansions
    /// - Assembly language calls
    /// - Heuristic-based function pointer analysis
    Low,

    /// Medium confidence (70-89% certainty)
    ///
    /// Used for relationships that are likely correct but have some ambiguity:
    /// - Indirect calls through function pointers
    /// - Callback registrations
    /// - Function calls in conditional contexts
    Medium,

    /// High confidence (90-100% certainty)
    ///
    /// Used for clear, unambiguous relationships such as:
    /// - Direct function calls by name
    /// - Explicit function pointer assignments
    /// - System calls with known prototypes
    High,
}

impl ConfidenceLevel {
    /// Returns the numeric confidence value (0.0 to 1.0).
    pub fn value(self) -> f64 {
        match self {
            ConfidenceLevel::High => 0.95,
            ConfidenceLevel::Medium => 0.80,
            ConfidenceLevel::Low => 0.60,
        }
    }

    /// Returns the confidence range as a tuple (min, max).
    pub fn range(self) -> (f64, f64) {
        match self {
            ConfidenceLevel::High => (0.90, 1.00),
            ConfidenceLevel::Medium => (0.70, 0.89),
            ConfidenceLevel::Low => (0.50, 0.69),
        }
    }

    /// Creates a confidence level from a numeric value.
    pub fn from_value(value: f64) -> Self {
        if value >= 0.90 {
            ConfidenceLevel::High
        } else if value >= 0.70 {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }

    /// Returns a human-readable description of the confidence level.
    pub fn description(self) -> &'static str {
        match self {
            ConfidenceLevel::High => "High confidence - clear, unambiguous relationship",
            ConfidenceLevel::Medium => "Medium confidence - likely correct with some ambiguity",
            ConfidenceLevel::Low => "Low confidence - uncertain or heuristic-based",
        }
    }

    /// Returns all possible confidence levels.
    pub fn all() -> &'static [ConfidenceLevel] {
        &[ConfidenceLevel::Low, ConfidenceLevel::Medium, ConfidenceLevel::High]
    }
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceLevel::High => write!(f, "high"),
            ConfidenceLevel::Medium => write!(f, "medium"),
            ConfidenceLevel::Low => write!(f, "low"),
        }
    }
}

impl FromStr for ConfidenceLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "high" => Ok(ConfidenceLevel::High),
            "medium" => Ok(ConfidenceLevel::Medium),
            "low" => Ok(ConfidenceLevel::Low),
            _ => Err(format!("Unknown confidence level: {}", s)),
        }
    }
}

/// Analysis scope for function pointer analysis.
///
/// Defines the scope of analysis when examining function pointers
/// and indirect call relationships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisScope {
    /// Analyze all function pointers and indirect calls
    All,

    /// Analyze only structure field assignments
    StructureFields,

    /// Analyze only variable assignments
    Variables,

    /// Analyze only callback patterns
    Callbacks,

    /// Analyze only function pointer arrays
    Arrays,
}

impl fmt::Display for AnalysisScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisScope::All => write!(f, "all"),
            AnalysisScope::StructureFields => write!(f, "structure_fields"),
            AnalysisScope::Variables => write!(f, "variables"),
            AnalysisScope::Callbacks => write!(f, "callbacks"),
            AnalysisScope::Arrays => write!(f, "arrays"),
        }
    }
}

impl FromStr for AnalysisScope {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "all" => Ok(AnalysisScope::All),
            "structure_fields" => Ok(AnalysisScope::StructureFields),
            "variables" => Ok(AnalysisScope::Variables),
            "callbacks" => Ok(AnalysisScope::Callbacks),
            "arrays" => Ok(AnalysisScope::Arrays),
            _ => Err(format!("Unknown analysis scope: {}", s)),
        }
    }
}

/// Function pointer type classification.
///
/// Categorizes different types of function pointer usage patterns
/// found in C code analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PointerType {
    /// Function pointer in structure field
    StructureField,

    /// Function pointer in local/global variable
    Variable,

    /// Function pointer passed as parameter
    Parameter,

    /// Function pointer returned from function
    ReturnValue,

    /// Function pointer in array
    Array,

    /// Function pointer in callback context
    Callback,
}

impl fmt::Display for PointerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PointerType::StructureField => write!(f, "structure_field"),
            PointerType::Variable => write!(f, "variable"),
            PointerType::Parameter => write!(f, "parameter"),
            PointerType::ReturnValue => write!(f, "return_value"),
            PointerType::Array => write!(f, "array"),
            PointerType::Callback => write!(f, "callback"),
        }
    }
}

impl FromStr for PointerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "structure_field" => Ok(PointerType::StructureField),
            "variable" => Ok(PointerType::Variable),
            "parameter" => Ok(PointerType::Parameter),
            "return_value" => Ok(PointerType::ReturnValue),
            "array" => Ok(PointerType::Array),
            "callback" => Ok(PointerType::Callback),
            _ => Err(format!("Unknown pointer type: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_type_confidence() {
        assert!(CallType::Direct.is_high_confidence_type());
        assert!(CallType::Indirect.is_medium_confidence_type());
        assert!(CallType::Macro.is_low_confidence_type());

        assert_eq!(CallType::Direct.default_confidence(), ConfidenceLevel::High);
        assert_eq!(CallType::Indirect.default_confidence(), ConfidenceLevel::Medium);
        assert_eq!(CallType::Macro.default_confidence(), ConfidenceLevel::Low);
    }

    #[test]
    fn test_call_type_from_str() {
        assert_eq!("direct".parse::<CallType>().unwrap(), CallType::Direct);
        assert_eq!("callback".parse::<CallType>().unwrap(), CallType::Callback);
        assert!("invalid".parse::<CallType>().is_err());
    }

    #[test]
    fn test_confidence_level_value() {
        assert_eq!(ConfidenceLevel::High.value(), 0.95);
        assert_eq!(ConfidenceLevel::Medium.value(), 0.80);
        assert_eq!(ConfidenceLevel::Low.value(), 0.60);
    }

    #[test]
    fn test_confidence_level_from_value() {
        assert_eq!(ConfidenceLevel::from_value(0.95), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_value(0.75), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::from_value(0.55), ConfidenceLevel::Low);
    }

    #[test]
    fn test_confidence_level_from_str() {
        assert_eq!("high".parse::<ConfidenceLevel>().unwrap(), ConfidenceLevel::High);
        assert_eq!("medium".parse::<ConfidenceLevel>().unwrap(), ConfidenceLevel::Medium);
        assert_eq!("low".parse::<ConfidenceLevel>().unwrap(), ConfidenceLevel::Low);
        assert!("invalid".parse::<ConfidenceLevel>().is_err());
    }

    #[test]
    fn test_analysis_scope_from_str() {
        assert_eq!("all".parse::<AnalysisScope>().unwrap(), AnalysisScope::All);
        assert_eq!("callbacks".parse::<AnalysisScope>().unwrap(), AnalysisScope::Callbacks);
        assert!("invalid".parse::<AnalysisScope>().is_err());
    }

    #[test]
    fn test_pointer_type_from_str() {
        assert_eq!("structure_field".parse::<PointerType>().unwrap(), PointerType::StructureField);
        assert_eq!("callback".parse::<PointerType>().unwrap(), PointerType::Callback);
        assert!("invalid".parse::<PointerType>().is_err());
    }
}
