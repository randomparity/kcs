//! Call extraction module for different types of function calls.
//!
//! This module provides specialized extractors for different patterns of function calls
//! found in C/kernel code, each optimized for specific call types and patterns.
//!
//! The main entry point is the `CallExtractor` which coordinates all specialized
//! extractors to provide a unified interface for call graph extraction.

pub mod callbacks;
pub mod conditional;
pub mod direct_calls;
pub mod macro_calls;
pub mod pointer_calls;

use anyhow::Result;
use kcs_graph::{CallEdgeModel, CallTypeEnum, ConfidenceLevel};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use tree_sitter::{Language, Tree};

use crate::error_context;
use crate::error_handling::{
    CallGraphError, CallGraphResult, ErrorContext, ErrorHandler, ErrorHandlingConfig,
};

pub use callbacks::{
    CallbackCall, CallbackConfig, CallbackDetector, CallbackType,
    ExtractionStats as CallbackExtractionStats,
};
pub use conditional::{
    ConditionalCall, ConditionalCallConfig, ConditionalCallDetector, ConditionalType,
    ExtractionStats as ConditionalExtractionStats,
};
pub use direct_calls::{DirectCall, DirectCallConfig, DirectCallDetector, ExtractionStats};
pub use macro_calls::{
    ExtractionStats as MacroExtractionStats, MacroCall, MacroCallConfig, MacroCallDetector,
    MacroCallType,
};
pub use pointer_calls::{
    ExtractionStats as PointerExtractionStats, PointerCall, PointerCallConfig, PointerCallDetector,
    PointerCallType,
};

/// Configuration for the main call extraction engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CallExtractionConfig {
    /// Configuration for direct call detection
    pub direct_calls: DirectCallConfig,
    /// Configuration for pointer call detection
    pub pointer_calls: PointerCallConfig,
    /// Configuration for macro call detection
    pub macro_calls: MacroCallConfig,
    /// Configuration for callback detection
    pub callbacks: CallbackConfig,
    /// Configuration for conditional call detection
    pub conditional: ConditionalCallConfig,
    /// Maximum file size to process (in bytes)
    pub max_file_size: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Minimum confidence level to include in results
    pub min_confidence: ConfidenceLevel,
}

impl Default for CallExtractionConfig {
    fn default() -> Self {
        Self {
            direct_calls: DirectCallConfig::default(),
            pointer_calls: PointerCallConfig::default(),
            macro_calls: MacroCallConfig::default(),
            callbacks: CallbackConfig::default(),
            conditional: ConditionalCallConfig::default(),
            max_file_size: 10 * 1024 * 1024, // 10MB
            enable_parallel: true,
            min_confidence: ConfidenceLevel::Low,
        }
    }
}

/// Aggregated extraction statistics from all detectors.
#[derive(Debug, Clone, Default)]
pub struct ExtractorStats {
    /// Statistics from direct call detection
    pub direct_calls: ExtractionStats,
    /// Statistics from pointer call detection
    pub pointer_calls: PointerExtractionStats,
    /// Statistics from macro call detection
    pub macro_calls: MacroExtractionStats,
    /// Statistics from callback detection
    pub callbacks: CallbackExtractionStats,
    /// Statistics from conditional call detection
    pub conditional: ConditionalExtractionStats,
    /// Total number of files processed
    pub files_processed: usize,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
    /// Number of files skipped due to size limits
    pub files_skipped: usize,
    /// Number of processing errors encountered
    pub errors: usize,
}

impl ExtractorStats {
    /// Get the total number of calls extracted across all detectors.
    pub fn total_calls(&self) -> usize {
        self.direct_calls.total_calls_found
            + self.pointer_calls.total_calls
            + self.macro_calls.total_calls
            + self.callbacks.total_callbacks
            + self.conditional.total_calls
    }

    /// Merge statistics from another ExtractorStats.
    pub fn merge(&mut self, other: &ExtractorStats) {
        // Note: Individual ExtractionStats don't have merge methods yet
        // This is a simplified implementation for now
        self.files_processed += other.files_processed;
        self.total_time_ms += other.total_time_ms;
        self.files_skipped += other.files_skipped;
        self.errors += other.errors;
    }
}

/// Main call extraction engine that coordinates all specialized detectors.
///
/// The CallExtractor provides a unified interface for extracting different types
/// of function calls from C source code using Tree-sitter AST analysis.
pub struct CallExtractor {
    /// Extraction configuration
    config: CallExtractionConfig,
    /// Direct call detector
    direct_detector: DirectCallDetector,
    /// Pointer call detector
    pointer_detector: PointerCallDetector,
    /// Macro call detector
    macro_detector: MacroCallDetector,
    /// Callback detector
    callback_detector: CallbackDetector,
    /// Conditional call detector
    conditional_detector: ConditionalCallDetector,
    /// C language parser
    #[allow(dead_code)]
    language: Language,
    /// Error handler for comprehensive error management
    error_handler: ErrorHandler,
}

impl CallExtractor {
    /// Create a new CallExtractor with the given configuration.
    pub fn new(config: CallExtractionConfig) -> Result<Self> {
        let language = tree_sitter_c::language();
        let error_handler = ErrorHandler::new_default();

        Ok(Self {
            direct_detector: DirectCallDetector::new(language, config.direct_calls.clone())?,
            pointer_detector: PointerCallDetector::new(language, config.pointer_calls.clone())?,
            macro_detector: MacroCallDetector::new(language, config.macro_calls.clone())?,
            callback_detector: CallbackDetector::new(language, config.callbacks.clone())?,
            conditional_detector: ConditionalCallDetector::new(config.conditional.clone())?,
            config,
            language,
            error_handler,
        })
    }

    /// Create a new CallExtractor with custom error handling configuration.
    pub fn new_with_error_config(
        config: CallExtractionConfig,
        error_config: ErrorHandlingConfig,
    ) -> Result<Self> {
        let language = tree_sitter_c::language();
        let error_handler = ErrorHandler::new(error_config);

        Ok(Self {
            direct_detector: DirectCallDetector::new(language, config.direct_calls.clone())?,
            pointer_detector: PointerCallDetector::new(language, config.pointer_calls.clone())?,
            macro_detector: MacroCallDetector::new(language, config.macro_calls.clone())?,
            callback_detector: CallbackDetector::new(language, config.callbacks.clone())?,
            conditional_detector: ConditionalCallDetector::new(config.conditional.clone())?,
            config,
            language,
            error_handler,
        })
    }

    /// Create a new CallExtractor with default configuration.
    pub fn new_default() -> Result<Self> {
        Self::new(CallExtractionConfig::default())
    }

    /// Extract all function calls from a single file.
    ///
    /// This method runs all specialized detectors on the given file and returns
    /// a unified set of call edges representing all detected function calls.
    pub fn extract_from_file(
        &mut self,
        file_path: &Path,
        source_code: &str,
        tree: &Tree,
    ) -> CallGraphResult<Vec<CallEdgeModel>> {
        let file_path_str = file_path.to_string_lossy().to_string();
        let start_time = Instant::now();

        // Log extraction start
        self.error_handler.log_progress(
            "Starting call graph extraction",
            &[("file", &file_path_str), ("size_bytes", &source_code.len())],
        );

        // Check file size limit
        if source_code.len() > self.config.max_file_size {
            let error = CallGraphError::ResourceError {
                message: format!(
                    "File {} exceeds size limit: {} bytes (max: {} bytes)",
                    file_path_str,
                    source_code.len(),
                    self.config.max_file_size
                ),
            };
            let context = error_context!(file_path);
            return Err(self.error_handler.handle_error(error, context).unwrap_err());
        }

        let mut all_edges = Vec::new();
        let mut extractor_stats = ExtractorStats {
            files_processed: 1,
            ..Default::default()
        };

        // Extract direct calls with error handling
        match self.extract_direct_calls(tree, source_code, &file_path_str) {
            Ok(edges) => {
                let count = edges.len();
                all_edges.extend(edges);
                self.error_handler.log_progress(
                    "Direct calls extracted",
                    &[("count", &count), ("file", &file_path_str)],
                );
            },
            Err(error) => {
                let context = error_context!(file_path, "direct_calls");
                self.error_handler.handle_error(error, context)?;
                extractor_stats.errors += 1;
            },
        }

        // Extract pointer calls with error handling
        match self.extract_pointer_calls(tree, source_code, &file_path_str) {
            Ok(edges) => {
                let count = edges.len();
                all_edges.extend(edges);
                self.error_handler.log_progress(
                    "Pointer calls extracted",
                    &[("count", &count), ("file", &file_path_str)],
                );
            },
            Err(error) => {
                let context = error_context!(file_path, "pointer_calls");
                self.error_handler.handle_error(error, context)?;
                extractor_stats.errors += 1;
            },
        }

        // Extract macro calls with error handling
        match self.extract_macro_calls(tree, source_code, &file_path_str) {
            Ok(edges) => {
                let count = edges.len();
                all_edges.extend(edges);
                self.error_handler.log_progress(
                    "Macro calls extracted",
                    &[("count", &count), ("file", &file_path_str)],
                );
            },
            Err(error) => {
                let context = error_context!(file_path, "macro_calls");
                self.error_handler.handle_error(error, context)?;
                extractor_stats.errors += 1;
            },
        }

        // Extract callback calls with error handling
        match self.extract_callback_calls(tree, source_code, &file_path_str) {
            Ok(edges) => {
                let count = edges.len();
                all_edges.extend(edges);
                self.error_handler.log_progress(
                    "Callback calls extracted",
                    &[("count", &count), ("file", &file_path_str)],
                );
            },
            Err(error) => {
                let context = error_context!(file_path, "callback_calls");
                self.error_handler.handle_error(error, context)?;
                extractor_stats.errors += 1;
            },
        }

        // Extract conditional calls with error handling
        match self.extract_conditional_calls(tree, source_code, &file_path_str) {
            Ok(edges) => {
                let count = edges.len();
                all_edges.extend(edges);
                self.error_handler.log_progress(
                    "Conditional calls extracted",
                    &[("count", &count), ("file", &file_path_str)],
                );
            },
            Err(error) => {
                let context = error_context!(file_path, "conditional_calls");
                self.error_handler.handle_error(error, context)?;
                extractor_stats.errors += 1;
            },
        }

        // Filter by minimum confidence level
        let initial_count = all_edges.len();
        let filtered_edges: Vec<CallEdgeModel> = all_edges
            .into_iter()
            .filter(|edge| edge.confidence() >= self.config.min_confidence)
            .collect();

        let filtered_count = filtered_edges.len();
        if filtered_count < initial_count {
            self.error_handler.log_progress(
                "Filtered calls by confidence",
                &[
                    ("initial", &initial_count),
                    ("filtered", &filtered_count),
                    ("min_confidence", &format!("{:?}", self.config.min_confidence)),
                ],
            );
        }

        // Log performance metrics
        let duration = start_time.elapsed();
        extractor_stats.total_time_ms = duration.as_millis() as u64;

        self.error_handler.log_performance(
            "call_extraction",
            duration.as_millis() as u64,
            &[
                ("file", &file_path_str),
                ("total_edges", &filtered_count),
                ("errors", &extractor_stats.errors),
            ],
        );

        Ok(filtered_edges)
    }

    /// Extract direct calls with error recovery.
    fn extract_direct_calls(
        &mut self,
        tree: &Tree,
        source_code: &str,
        file_path: &str,
    ) -> CallGraphResult<Vec<CallEdgeModel>> {
        let direct_calls = self
            .direct_detector
            .extract_calls(tree, source_code, file_path)
            .map_err(|e| CallGraphError::DetectionError {
                file: file_path.to_string(),
                line: 0,
                message: format!("Direct call detection failed: {}", e),
            })?;

        Ok(direct_calls
            .into_iter()
            .filter_map(|call| self.direct_call_to_edge(call, file_path))
            .collect())
    }

    /// Extract pointer calls with error recovery.
    fn extract_pointer_calls(
        &mut self,
        tree: &Tree,
        source_code: &str,
        file_path: &str,
    ) -> CallGraphResult<Vec<CallEdgeModel>> {
        let pointer_calls = self
            .pointer_detector
            .extract_calls(tree, source_code, file_path)
            .map_err(|e| CallGraphError::DetectionError {
                file: file_path.to_string(),
                line: 0,
                message: format!("Pointer call detection failed: {}", e),
            })?;

        Ok(pointer_calls
            .into_iter()
            .filter_map(|call| self.pointer_call_to_edge(call, file_path))
            .collect())
    }

    /// Extract macro calls with error recovery.
    fn extract_macro_calls(
        &mut self,
        tree: &Tree,
        source_code: &str,
        file_path: &str,
    ) -> CallGraphResult<Vec<CallEdgeModel>> {
        let macro_calls =
            self.macro_detector.extract_calls(tree, source_code, file_path).map_err(|e| {
                CallGraphError::DetectionError {
                    file: file_path.to_string(),
                    line: 0,
                    message: format!("Macro call detection failed: {}", e),
                }
            })?;

        Ok(macro_calls
            .into_iter()
            .filter_map(|call| self.macro_call_to_edge(call, file_path))
            .collect())
    }

    /// Extract callback calls with error recovery.
    fn extract_callback_calls(
        &mut self,
        tree: &Tree,
        source_code: &str,
        file_path: &str,
    ) -> CallGraphResult<Vec<CallEdgeModel>> {
        let callback_calls = self
            .callback_detector
            .extract_callbacks(tree, source_code, file_path)
            .map_err(|e| CallGraphError::DetectionError {
                file: file_path.to_string(),
                line: 0,
                message: format!("Callback detection failed: {}", e),
            })?;

        Ok(callback_calls
            .into_iter()
            .filter_map(|call| self.callback_call_to_edge(call, file_path))
            .collect())
    }

    /// Extract conditional calls with error recovery.
    fn extract_conditional_calls(
        &mut self,
        tree: &Tree,
        source_code: &str,
        file_path: &str,
    ) -> CallGraphResult<Vec<CallEdgeModel>> {
        let conditional_calls = self
            .conditional_detector
            .extract_calls(tree, source_code, file_path)
            .map_err(|e| CallGraphError::DetectionError {
                file: file_path.to_string(),
                line: 0,
                message: format!("Conditional call detection failed: {}", e),
            })?;

        Ok(conditional_calls
            .into_iter()
            .filter_map(|call| self.conditional_call_to_edge(call, file_path))
            .collect())
    }

    /// Get aggregated statistics from all detectors.
    pub fn get_stats(&self) -> ExtractorStats {
        ExtractorStats {
            direct_calls: self.direct_detector.get_stats(),
            pointer_calls: PointerExtractionStats::default(), // Would need call results to compute
            macro_calls: MacroExtractionStats::default(),     // Would need call results to compute
            callbacks: self.callback_detector.stats().clone(),
            conditional: self.conditional_detector.stats().clone(),
            files_processed: 0,
            total_time_ms: 0,
            files_skipped: 0,
            errors: self.error_handler.get_stats().total_errors,
        }
    }

    /// Get error handling statistics.
    pub fn get_error_stats(&self) -> &crate::error_handling::ErrorStats {
        self.error_handler.get_stats()
    }

    /// Reset error statistics.
    pub fn reset_error_stats(&mut self) {
        self.error_handler.reset_stats();
    }

    /// Reset all detector statistics.
    /// Note: Individual detectors don't expose reset methods yet
    pub fn reset_stats(&mut self) {
        // TODO: Add reset capabilities to individual detectors
    }

    /// Convert a DirectCall to a CallEdgeModel.
    fn direct_call_to_edge(&self, call: DirectCall, file_path: &str) -> Option<CallEdgeModel> {
        Some(CallEdgeModel::new(
            call.caller_function.unwrap_or_else(|| "<unknown>".to_string()),
            call.function_name,
            file_path.to_string(),
            call.line_number,
            CallTypeEnum::Direct,
            ConfidenceLevel::High,
            call.is_conditional,
        ))
    }

    /// Convert a PointerCall to a CallEdgeModel.
    fn pointer_call_to_edge(&self, call: PointerCall, file_path: &str) -> Option<CallEdgeModel> {
        let call_type = match call.call_type {
            PointerCallType::ExplicitDereference => CallTypeEnum::Indirect,
            PointerCallType::Implicit => CallTypeEnum::Indirect,
            PointerCallType::MemberPointer => CallTypeEnum::Indirect,
            PointerCallType::ArrayCallback => CallTypeEnum::Callback,
        };

        Some(CallEdgeModel::new(
            call.caller_function.unwrap_or_else(|| "<unknown>".to_string()),
            call.pointer_expression, // Use pointer_expression as the target
            file_path.to_string(),
            call.line_number,
            call_type,
            call.confidence,
            call.is_conditional,
        ))
    }

    /// Convert a MacroCall to a CallEdgeModel.
    fn macro_call_to_edge(&self, call: MacroCall, file_path: &str) -> Option<CallEdgeModel> {
        Some(CallEdgeModel::new(
            call.caller_function.unwrap_or_else(|| "<unknown>".to_string()),
            call.macro_name, // Use macro name as the target
            file_path.to_string(),
            call.line_number,
            CallTypeEnum::Macro,
            call.confidence,
            call.is_conditional,
        ))
    }

    /// Convert a CallbackCall to a CallEdgeModel.
    fn callback_call_to_edge(&self, call: CallbackCall, file_path: &str) -> Option<CallEdgeModel> {
        Some(CallEdgeModel::new(
            call.caller_function.unwrap_or_else(|| "<unknown>".to_string()),
            call.callback_expression, // Use callback expression as the target
            file_path.to_string(),
            call.line_number,
            CallTypeEnum::Callback,
            call.confidence,
            call.is_conditional,
        ))
    }

    /// Convert a ConditionalCall to a CallEdgeModel.
    fn conditional_call_to_edge(
        &self,
        call: ConditionalCall,
        file_path: &str,
    ) -> Option<CallEdgeModel> {
        Some(CallEdgeModel::new(
            call.caller_function.unwrap_or_else(|| "<unknown>".to_string()),
            call.function_name,
            file_path.to_string(),
            call.line_number,
            CallTypeEnum::Conditional,
            ConfidenceLevel::Medium, // Default confidence for conditional calls
            true,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    #[test]
    fn test_call_extractor_creation() {
        let extractor = CallExtractor::new_default();
        assert!(extractor.is_ok());
    }

    #[test]
    fn test_config_default() {
        let config = CallExtractionConfig::default();
        assert_eq!(config.max_file_size, 10 * 1024 * 1024);
        assert!(config.enable_parallel);
        assert_eq!(config.min_confidence, ConfidenceLevel::Low);
    }

    #[test]
    fn test_extractor_stats_merge() {
        let mut stats1 = ExtractorStats {
            files_processed: 5,
            total_time_ms: 1000,
            ..Default::default()
        };

        let stats2 = ExtractorStats {
            files_processed: 3,
            total_time_ms: 500,
            ..Default::default()
        };

        stats1.merge(&stats2);
        assert_eq!(stats1.files_processed, 8);
        assert_eq!(stats1.total_time_ms, 1500);
    }

    #[test]
    fn test_extract_from_empty_file() {
        let mut extractor = CallExtractor::new_default().unwrap();
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_c::language()).unwrap();

        let source = "";
        let tree = parser.parse(source, None).unwrap();
        let path = Path::new("test.c");

        let result = extractor.extract_from_file(path, source, &tree);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_file_size_limit() {
        let config = CallExtractionConfig {
            max_file_size: 10, // Very small limit
            ..Default::default()
        };

        let mut extractor = CallExtractor::new(config).unwrap();
        let mut parser = Parser::new();
        parser.set_language(tree_sitter_c::language()).unwrap();

        let source = "int main() { return 0; }"; // Exceeds 10 bytes
        let tree = parser.parse(source, None).unwrap();
        let path = Path::new("test.c");

        let result = extractor.extract_from_file(path, source, &tree);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds size limit"));
    }
}
