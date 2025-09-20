//! Call classification logic for categorizing and scoring function calls.
//!
//! This module provides algorithms and heuristics for classifying different
//! types of function calls detected during AST traversal, including confidence
//! scoring and call type determination based on syntactic patterns.

use anyhow::Result;
use kcs_graph::types::{CallType, ConfidenceLevel};
use std::collections::HashMap;
use tree_sitter::{Node, Tree};

/// Configuration for call classification algorithms.
#[derive(Debug, Clone)]
pub struct ClassificationConfig {
    /// Minimum confidence threshold for including calls in results
    pub min_confidence: ConfidenceLevel,
    /// Enable heuristic-based confidence adjustment
    pub enable_heuristics: bool,
    /// Weight for syntactic confidence factors
    pub syntactic_weight: f64,
    /// Weight for contextual confidence factors
    pub contextual_weight: f64,
    /// Apply penalty for calls in complex control flow
    pub control_flow_penalty: f64,
    /// Apply penalty for calls in macro contexts
    pub macro_penalty: f64,
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            min_confidence: ConfidenceLevel::Low,
            enable_heuristics: true,
            syntactic_weight: 0.6,
            contextual_weight: 0.4,
            control_flow_penalty: 0.1,
            macro_penalty: 0.15,
        }
    }
}

/// Context information for call classification.
#[derive(Debug, Clone)]
pub struct CallContext {
    /// Whether the call is inside a conditional compilation block
    pub in_conditional_compilation: bool,
    /// Whether the call is inside control flow (if/while/for/switch)
    pub in_control_flow: bool,
    /// Whether the call is inside a macro definition or expansion
    pub in_macro_context: bool,
    /// Nesting depth of control structures
    pub control_depth: usize,
    /// Whether the call target is a function pointer
    pub is_function_pointer: bool,
    /// Whether the call target name contains known syscall patterns
    pub is_syscall_pattern: bool,
    /// Whether the call is through a struct member function pointer
    pub is_member_pointer: bool,
    /// Number of arguments in the call
    pub argument_count: usize,
    /// Whether all arguments are literals or simple identifiers
    pub has_simple_arguments: bool,
}

impl Default for CallContext {
    fn default() -> Self {
        Self {
            in_conditional_compilation: false,
            in_control_flow: false,
            in_macro_context: false,
            control_depth: 0,
            is_function_pointer: false,
            is_syscall_pattern: false,
            is_member_pointer: false,
            argument_count: 0,
            has_simple_arguments: true,
        }
    }
}

/// Classification result for a function call.
#[derive(Debug, Clone)]
pub struct CallClassification {
    /// Classified call type
    pub call_type: CallType,
    /// Confidence level in the classification
    pub confidence: ConfidenceLevel,
    /// Numeric confidence score (0.0 to 1.0)
    pub confidence_score: f64,
    /// Reasoning for the classification decision
    pub reasoning: Vec<String>,
    /// Whether the call should be included in the final results
    pub should_include: bool,
}

/// Statistics for call classification operations.
#[derive(Debug, Clone, Default)]
pub struct ClassificationStats {
    /// Total calls classified
    pub total_classified: usize,
    /// Calls classified by type
    pub by_type: HashMap<CallType, usize>,
    /// Calls classified by confidence level
    pub by_confidence: HashMap<ConfidenceLevel, usize>,
    /// Calls excluded due to low confidence
    pub excluded_low_confidence: usize,
    /// Average confidence score
    pub average_confidence: f64,
}

impl ClassificationStats {
    /// Update statistics with a new classification result.
    pub fn update(&mut self, classification: &CallClassification) {
        self.total_classified += 1;

        *self.by_type.entry(classification.call_type).or_insert(0) += 1;
        *self.by_confidence.entry(classification.confidence).or_insert(0) += 1;

        if !classification.should_include {
            self.excluded_low_confidence += 1;
        }

        // Update running average
        let total_score = self.average_confidence * (self.total_classified - 1) as f64;
        self.average_confidence =
            (total_score + classification.confidence_score) / self.total_classified as f64;
    }

    /// Get the percentage of calls for a specific type.
    pub fn type_percentage(&self, call_type: CallType) -> f64 {
        if self.total_classified == 0 {
            return 0.0;
        }

        let count = self.by_type.get(&call_type).unwrap_or(&0);
        (*count as f64 / self.total_classified as f64) * 100.0
    }

    /// Get the percentage of calls for a specific confidence level.
    pub fn confidence_percentage(&self, confidence: ConfidenceLevel) -> f64 {
        if self.total_classified == 0 {
            return 0.0;
        }

        let count = self.by_confidence.get(&confidence).unwrap_or(&0);
        (*count as f64 / self.total_classified as f64) * 100.0
    }
}

/// Main call classifier that analyzes and categorizes function calls.
pub struct CallClassifier {
    /// Classification configuration
    config: ClassificationConfig,
    /// Statistics for classification operations
    stats: ClassificationStats,
    /// Known syscall patterns for classification
    syscall_patterns: Vec<String>,
    /// Known callback patterns for classification
    callback_patterns: Vec<String>,
}

impl CallClassifier {
    /// Create a new call classifier with the given configuration.
    pub fn new(config: ClassificationConfig) -> Self {
        let syscall_patterns = vec![
            "sys_".to_string(),
            "__sys_".to_string(),
            "syscall".to_string(),
            "SYSCALL".to_string(),
            "_syscall".to_string(),
        ];

        let callback_patterns = vec![
            "_callback".to_string(),
            "_cb".to_string(),
            "_handler".to_string(),
            "_hook".to_string(),
            "_fn".to_string(),
        ];

        Self {
            config,
            stats: ClassificationStats::default(),
            syscall_patterns,
            callback_patterns,
        }
    }

    /// Create a new call classifier with default configuration.
    pub fn new_default() -> Self {
        Self::new(ClassificationConfig::default())
    }

    /// Classify a function call based on its syntactic and contextual information.
    pub fn classify_call(
        &mut self,
        function_name: &str,
        node: Option<&Node>,
        context: &CallContext,
        _tree: Option<&Tree>,
    ) -> Result<CallClassification> {
        let mut reasoning = Vec::new();

        // Step 1: Determine base call type from syntactic patterns
        let base_call_type =
            self.determine_base_call_type(function_name, node, context, &mut reasoning)?;

        // Step 2: Calculate confidence score
        let confidence_score =
            self.calculate_confidence_score(&base_call_type, context, &mut reasoning);

        // Step 3: Convert score to confidence level
        let confidence_level = ConfidenceLevel::from_value(confidence_score);

        // Step 4: Apply final adjustments
        let (final_call_type, final_confidence) =
            self.apply_final_adjustments(base_call_type, confidence_level, context, &mut reasoning);

        // Step 5: Determine if call should be included
        let should_include = final_confidence >= self.config.min_confidence;

        let classification = CallClassification {
            call_type: final_call_type,
            confidence: final_confidence,
            confidence_score,
            reasoning,
            should_include,
        };

        // Update statistics
        self.stats.update(&classification);

        Ok(classification)
    }

    /// Classify a direct function call by name.
    pub fn classify_direct_call(
        &mut self,
        function_name: &str,
        context: &CallContext,
    ) -> Result<CallClassification> {
        self.classify_call(function_name, None, context, None)
    }

    /// Classify an indirect call through a function pointer.
    pub fn classify_indirect_call(
        &mut self,
        pointer_expression: &str,
        context: &CallContext,
    ) -> Result<CallClassification> {
        let mut modified_context = context.clone();
        modified_context.is_function_pointer = true;

        self.classify_call(pointer_expression, None, &modified_context, None)
    }

    /// Classify a macro call or macro expansion.
    pub fn classify_macro_call(
        &mut self,
        macro_name: &str,
        context: &CallContext,
    ) -> Result<CallClassification> {
        let mut modified_context = context.clone();
        modified_context.in_macro_context = true;

        self.classify_call(macro_name, None, &modified_context, None)
    }

    /// Get current classification statistics.
    pub fn stats(&self) -> &ClassificationStats {
        &self.stats
    }

    /// Reset classification statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ClassificationStats::default();
    }

    /// Determine the base call type from syntactic analysis.
    fn determine_base_call_type(
        &self,
        function_name: &str,
        _node: Option<&Node>,
        context: &CallContext,
        reasoning: &mut Vec<String>,
    ) -> Result<CallType> {
        // Check for syscall patterns
        if context.is_syscall_pattern || self.is_syscall_name(function_name) {
            reasoning.push("Identified syscall pattern in function name".to_string());
            return Ok(CallType::Syscall);
        }

        // Check for macro context
        if context.in_macro_context {
            reasoning.push("Call occurs within macro context".to_string());
            return Ok(CallType::Macro);
        }

        // Check for function pointer patterns
        if context.is_function_pointer {
            reasoning.push("Call through function pointer detected".to_string());

            if context.is_member_pointer {
                reasoning.push("Member function pointer call".to_string());
            }

            // Distinguish between callbacks and regular indirect calls
            if self.is_callback_pattern(function_name) {
                reasoning.push("Callback pattern detected in function name".to_string());
                return Ok(CallType::Callback);
            }

            return Ok(CallType::Indirect);
        }

        // Check for conditional context
        if context.in_conditional_compilation {
            reasoning.push("Call within conditional compilation block".to_string());
            return Ok(CallType::Conditional);
        }

        // Check for callback patterns in direct calls
        if self.is_callback_pattern(function_name) {
            reasoning.push("Callback naming pattern in direct call".to_string());
            return Ok(CallType::Callback);
        }

        // Default to direct call
        reasoning.push("Standard direct function call".to_string());
        Ok(CallType::Direct)
    }

    /// Calculate confidence score based on various factors.
    fn calculate_confidence_score(
        &self,
        call_type: &CallType,
        context: &CallContext,
        reasoning: &mut Vec<String>,
    ) -> f64 {
        if !self.config.enable_heuristics {
            return call_type.default_confidence().value();
        }

        let base_score = call_type.default_confidence().value();
        let mut adjustments = Vec::new();

        // Syntactic confidence factors
        let mut syntactic_score = base_score;

        // Simple arguments increase confidence
        if context.has_simple_arguments {
            syntactic_score += 0.05;
            adjustments.push("Simple arguments (+0.05)".to_string());
        }

        // Assembly context decreases confidence for non-assembly calls
        if matches!(call_type, CallType::Direct | CallType::Indirect) && context.in_macro_context {
            syntactic_score -= self.config.macro_penalty;
            adjustments.push(format!("Macro context (-{})", self.config.macro_penalty));
        }

        // Contextual confidence factors
        let mut contextual_score = base_score;

        // Control flow complexity
        if context.in_control_flow {
            let penalty = self.config.control_flow_penalty * (context.control_depth as f64 * 0.1);
            contextual_score -= penalty;
            adjustments.push(format!("Control flow penalty (-{:.2})", penalty));
        }

        // Very deep nesting is suspicious
        if context.control_depth > 5 {
            contextual_score -= 0.1;
            adjustments.push("Deep nesting penalty (-0.1)".to_string());
        }

        // Combine scores according to configured weights
        let final_score = (syntactic_score * self.config.syntactic_weight)
            + (contextual_score * self.config.contextual_weight);

        // Clamp to valid range
        let clamped_score = final_score.clamp(0.0, 1.0);

        if !adjustments.is_empty() {
            reasoning.push(format!("Confidence adjustments: {}", adjustments.join(", ")));
        }

        reasoning.push(format!(
            "Final confidence score: {:.3} (base: {:.3})",
            clamped_score, base_score
        ));

        clamped_score
    }

    /// Apply final adjustments to call type and confidence.
    fn apply_final_adjustments(
        &self,
        call_type: CallType,
        confidence: ConfidenceLevel,
        context: &CallContext,
        reasoning: &mut Vec<String>,
    ) -> (CallType, ConfidenceLevel) {
        let mut final_type = call_type;
        let mut final_confidence = confidence;

        // Promote indirect calls with high confidence to direct if appropriate
        if matches!(call_type, CallType::Indirect)
            && matches!(confidence, ConfidenceLevel::High)
            && !context.is_function_pointer
            && !context.is_member_pointer
        {
            final_type = CallType::Direct;
            reasoning.push("Promoted indirect to direct call due to high confidence".to_string());
        }

        // Demote calls in very complex contexts
        if context.control_depth > 8 && matches!(confidence, ConfidenceLevel::High) {
            final_confidence = ConfidenceLevel::Medium;
            reasoning.push("Demoted confidence due to excessive complexity".to_string());
        }

        // Syscalls get special treatment
        if matches!(call_type, CallType::Syscall) {
            final_confidence = ConfidenceLevel::High;
            reasoning.push("Syscall confidence boosted to high".to_string());
        }

        (final_type, final_confidence)
    }

    /// Check if a function name matches syscall patterns.
    fn is_syscall_name(&self, name: &str) -> bool {
        self.syscall_patterns.iter().any(|pattern| name.contains(pattern))
    }

    /// Check if a function name matches callback patterns.
    fn is_callback_pattern(&self, name: &str) -> bool {
        self.callback_patterns.iter().any(|pattern| name.contains(pattern))
    }
}

/// Extract call context information from AST node.
pub fn extract_call_context(node: &Node, _tree: &Tree, source: &str) -> Result<CallContext> {
    let mut context = CallContext::default();

    // Traverse up the AST to gather contextual information
    let mut current = Some(*node);
    let mut depth = 0;

    while let Some(n) = current {
        match n.kind() {
            "if_statement" | "while_statement" | "for_statement" | "switch_statement" => {
                context.in_control_flow = true;
                context.control_depth += 1;
            },
            "preproc_ifdef" | "preproc_if" | "preproc_elif" => {
                context.in_conditional_compilation = true;
            },
            "macro_definition" | "preproc_def" => {
                context.in_macro_context = true;
            },
            _ => {},
        }

        current = n.parent();
        depth += 1;

        // Prevent infinite loops
        if depth > 100 {
            break;
        }
    }

    // Analyze the call expression itself
    if node.kind() == "call_expression" {
        let function_node = node.child_by_field_name("function");
        if let Some(func_node) = function_node {
            match func_node.kind() {
                "identifier" => {
                    // Direct call
                },
                "field_expression" => {
                    context.is_member_pointer = true;
                    context.is_function_pointer = true;
                },
                "parenthesized_expression" => {
                    context.is_function_pointer = true;
                },
                _ => {},
            }
        }

        // Count arguments
        if let Some(args_node) = node.child_by_field_name("arguments") {
            context.argument_count = args_node.child_count();

            // Check if arguments are simple
            context.has_simple_arguments =
                (0..args_node.child_count()).filter_map(|i| args_node.child(i)).all(|arg| {
                    matches!(arg.kind(), "identifier" | "number_literal" | "string_literal")
                });
        }
    }

    // Check for syscall patterns in the function name
    if let Some(func_node) = node.child_by_field_name("function") {
        let func_text = func_node.utf8_text(source.as_bytes()).unwrap_or("");
        context.is_syscall_pattern = func_text.contains("sys_")
            || func_text.contains("syscall")
            || func_text.contains("__sys_");
    }

    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_classifier_creation() {
        let classifier = CallClassifier::new_default();
        assert_eq!(classifier.config.min_confidence, ConfidenceLevel::Low);
        assert!(classifier.config.enable_heuristics);
    }

    #[test]
    fn test_direct_call_classification() {
        let mut classifier = CallClassifier::new_default();
        let context = CallContext::default();

        let result = classifier.classify_direct_call("printf", &context);
        assert!(result.is_ok());

        let classification = result.unwrap();
        assert_eq!(classification.call_type, CallType::Direct);
        assert!(classification.should_include);
    }

    #[test]
    fn test_syscall_detection() {
        let mut classifier = CallClassifier::new_default();
        let context = CallContext::default();

        let result = classifier.classify_direct_call("sys_open", &context);
        assert!(result.is_ok());

        let classification = result.unwrap();
        assert_eq!(classification.call_type, CallType::Syscall);
        assert_eq!(classification.confidence, ConfidenceLevel::High);
    }

    #[test]
    fn test_callback_pattern_detection() {
        let mut classifier = CallClassifier::new_default();
        let context = CallContext::default();

        let result = classifier.classify_direct_call("timer_callback", &context);
        assert!(result.is_ok());

        let classification = result.unwrap();
        assert_eq!(classification.call_type, CallType::Callback);
    }

    #[test]
    fn test_indirect_call_classification() {
        let mut classifier = CallClassifier::new_default();
        let context = CallContext::default();

        let result = classifier.classify_indirect_call("(*func_ptr)", &context);
        assert!(result.is_ok());

        let classification = result.unwrap();
        assert_eq!(classification.call_type, CallType::Indirect);
    }

    #[test]
    fn test_macro_call_classification() {
        let mut classifier = CallClassifier::new_default();
        let context = CallContext::default();

        let result = classifier.classify_macro_call("ASSERT", &context);
        assert!(result.is_ok());

        let classification = result.unwrap();
        assert_eq!(classification.call_type, CallType::Macro);
    }

    #[test]
    fn test_control_flow_penalty() {
        let config = ClassificationConfig {
            control_flow_penalty: 0.3, // Higher penalty for more visible effect
            ..Default::default()
        };
        let mut classifier = CallClassifier::new(config);
        let context = CallContext {
            in_control_flow: true,
            control_depth: 5,
            ..Default::default()
        };

        let result = classifier.classify_direct_call("func", &context);
        assert!(result.is_ok());

        let classification = result.unwrap();
        // Should have lower confidence due to control flow complexity
        assert!(classification.confidence_score < CallType::Direct.default_confidence().value());
    }

    #[test]
    fn test_confidence_level_conversion() {
        assert_eq!(ConfidenceLevel::from_value(0.95), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_value(0.75), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::from_value(0.55), ConfidenceLevel::Low);
    }

    #[test]
    fn test_classification_stats() {
        let mut stats = ClassificationStats::default();

        let classification = CallClassification {
            call_type: CallType::Direct,
            confidence: ConfidenceLevel::High,
            confidence_score: 0.95,
            reasoning: vec![],
            should_include: true,
        };

        stats.update(&classification);

        assert_eq!(stats.total_classified, 1);
        assert_eq!(stats.type_percentage(CallType::Direct), 100.0);
        assert_eq!(stats.confidence_percentage(ConfidenceLevel::High), 100.0);
        assert_eq!(stats.average_confidence, 0.95);
    }

    #[test]
    fn test_call_context_default() {
        let context = CallContext::default();
        assert!(!context.in_conditional_compilation);
        assert!(!context.in_control_flow);
        assert!(!context.in_macro_context);
        assert_eq!(context.control_depth, 0);
        assert!(!context.is_function_pointer);
        assert!(context.has_simple_arguments);
    }
}
