//! Call extraction module for different types of function calls.
//!
//! This module provides specialized extractors for different patterns of function calls
//! found in C/kernel code, each optimized for specific call types and patterns.

pub mod direct_calls;

pub use direct_calls::{DirectCall, DirectCallConfig, DirectCallDetector, ExtractionStats};
