//! Call extraction module for different types of function calls.
//!
//! This module provides specialized extractors for different patterns of function calls
//! found in C/kernel code, each optimized for specific call types and patterns.

pub mod direct_calls;
pub mod macro_calls;
pub mod pointer_calls;

pub use direct_calls::{DirectCall, DirectCallConfig, DirectCallDetector, ExtractionStats};
pub use macro_calls::{
    ExtractionStats as MacroExtractionStats, MacroCall, MacroCallConfig, MacroCallDetector,
    MacroCallType,
};
pub use pointer_calls::{
    ExtractionStats as PointerExtractionStats, PointerCall, PointerCallConfig, PointerCallDetector,
    PointerCallType,
};
