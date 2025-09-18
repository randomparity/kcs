//! Query processing module for semantic search
//!
//! Provides query preprocessing, expansion, and result ranking
//! specifically optimized for Linux kernel code search.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    pub max_results: usize,
    pub similarity_threshold: f32,
    pub include_context: bool,
    pub kernel_config: Option<String>,
    pub expand_synonyms: bool,
    pub remove_stop_words: bool,
    pub boost_exact_match: f32,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            max_results: 10,
            similarity_threshold: 0.7,
            include_context: true,
            kernel_config: None,
            expand_synonyms: true,
            remove_stop_words: true,
            boost_exact_match: 1.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: i64,
    pub name: String,
    pub file_path: String,
    pub score: f32,
    pub context: Option<String>,
}

pub struct QueryProcessor {
    config: QueryConfig,
    stop_words: HashSet<String>,
    kernel_synonyms: HashMap<String, Vec<String>>,
    abbreviations: HashMap<String, String>,
}

impl QueryProcessor {
    pub fn new() -> Self {
        Self::with_config(QueryConfig::default())
    }

    pub fn with_config(config: QueryConfig) -> Self {
        let mut processor = Self {
            config,
            stop_words: Self::build_stop_words(),
            kernel_synonyms: Self::build_kernel_synonyms(),
            abbreviations: Self::build_abbreviations(),
        };
        processor.initialize();
        processor
    }

    fn initialize(&mut self) {
        // Additional initialization if needed
    }

    fn build_stop_words() -> HashSet<String> {
        let words = vec![
            "the", "is", "at", "which", "on", "and", "a", "an", "as", "are", "was", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "could", "shall", "to", "of", "in", "for", "with", "by",
            "from", "about", "into", "through",
        ];
        words.into_iter().map(String::from).collect()
    }

    fn build_kernel_synonyms() -> HashMap<String, Vec<String>> {
        let mut synonyms = HashMap::new();

        // File system related
        synonyms.insert(
            "fs".to_string(),
            vec!["filesystem".to_string(), "file_system".to_string()],
        );
        synonyms.insert(
            "vfs".to_string(),
            vec!["virtual_file_system".to_string(), "virtual_fs".to_string()],
        );
        synonyms.insert("inode".to_string(), vec!["index_node".to_string()]);
        synonyms.insert("dentry".to_string(), vec!["directory_entry".to_string()]);

        // System calls
        synonyms.insert(
            "syscall".to_string(),
            vec!["system_call".to_string(), "sys_call".to_string()],
        );
        synonyms.insert(
            "ioctl".to_string(),
            vec!["io_control".to_string(), "device_control".to_string()],
        );

        // Memory management
        synonyms.insert(
            "mm".to_string(),
            vec!["memory_management".to_string(), "mem_mgmt".to_string()],
        );
        synonyms.insert(
            "alloc".to_string(),
            vec!["allocate".to_string(), "allocation".to_string()],
        );
        synonyms.insert(
            "kzalloc".to_string(),
            vec!["kernel_zalloc".to_string(), "zero_alloc".to_string()],
        );
        synonyms.insert(
            "vmalloc".to_string(),
            vec!["virtual_memory_alloc".to_string()],
        );

        // Process management
        synonyms.insert(
            "pid".to_string(),
            vec!["process_id".to_string(), "process_identifier".to_string()],
        );
        synonyms.insert(
            "tid".to_string(),
            vec!["thread_id".to_string(), "thread_identifier".to_string()],
        );
        synonyms.insert(
            "sched".to_string(),
            vec!["scheduler".to_string(), "scheduling".to_string()],
        );

        // Networking
        synonyms.insert(
            "netdev".to_string(),
            vec!["network_device".to_string(), "net_device".to_string()],
        );
        synonyms.insert(
            "skb".to_string(),
            vec!["socket_buffer".to_string(), "sk_buff".to_string()],
        );

        // Drivers
        synonyms.insert("drv".to_string(), vec!["driver".to_string()]);
        synonyms.insert("hw".to_string(), vec!["hardware".to_string()]);
        synonyms.insert(
            "irq".to_string(),
            vec!["interrupt_request".to_string(), "interrupt".to_string()],
        );

        synonyms
    }

    fn build_abbreviations() -> HashMap<String, String> {
        let mut abbrevs = HashMap::new();

        // Common kernel abbreviations
        abbrevs.insert("fs".to_string(), "filesystem".to_string());
        abbrevs.insert("vfs".to_string(), "virtual filesystem".to_string());
        abbrevs.insert("mm".to_string(), "memory management".to_string());
        abbrevs.insert("vm".to_string(), "virtual memory".to_string());
        abbrevs.insert("hw".to_string(), "hardware".to_string());
        abbrevs.insert("drv".to_string(), "driver".to_string());
        abbrevs.insert("dev".to_string(), "device".to_string());
        abbrevs.insert("init".to_string(), "initialize".to_string());
        abbrevs.insert("alloc".to_string(), "allocate".to_string());
        abbrevs.insert("dealloc".to_string(), "deallocate".to_string());
        abbrevs.insert("ref".to_string(), "reference".to_string());
        abbrevs.insert("ptr".to_string(), "pointer".to_string());
        abbrevs.insert("buf".to_string(), "buffer".to_string());
        abbrevs.insert("ctx".to_string(), "context".to_string());
        abbrevs.insert("cfg".to_string(), "configuration".to_string());
        abbrevs.insert("msg".to_string(), "message".to_string());
        abbrevs.insert("pkt".to_string(), "packet".to_string());
        abbrevs.insert("req".to_string(), "request".to_string());
        abbrevs.insert("resp".to_string(), "response".to_string());

        abbrevs
    }

    pub fn preprocess(&self, query: &str) -> Result<String> {
        let mut processed = query.to_lowercase();

        // Expand abbreviations
        for (abbrev, full) in &self.abbreviations {
            let pattern = format!(r"\b{}\b", regex::escape(abbrev));
            if let Ok(re) = regex::Regex::new(&pattern) {
                processed = re.replace_all(&processed, full.as_str()).to_string();
            }
        }

        // Remove stop words if configured
        if self.config.remove_stop_words {
            let tokens: Vec<String> = processed
                .split_whitespace()
                .filter(|word| !self.stop_words.contains(*word))
                .map(String::from)
                .collect();
            processed = tokens.join(" ");
        }

        // Normalize whitespace
        processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");

        Ok(processed)
    }

    pub fn expand_query(&self, query: &str) -> Result<Vec<String>> {
        let mut expansions = vec![query.to_string()];

        if !self.config.expand_synonyms {
            return Ok(expansions);
        }

        // Work with the original query for synonym matching, but keep it lowercase
        let query_lower = query.to_lowercase();
        let tokens: Vec<&str> = query_lower.split_whitespace().collect();

        // Add synonym expansions
        for (i, token) in tokens.iter().enumerate() {
            if let Some(synonyms) = self.kernel_synonyms.get(*token) {
                for synonym in synonyms {
                    let mut expanded = tokens.clone();
                    expanded[i] = synonym;
                    let expanded_query = expanded.join(" ");
                    if !expansions.contains(&expanded_query) {
                        expansions.push(expanded_query);
                    }
                }
            }
        }

        // Add partial matches for compound terms
        if query.contains('_') {
            let parts = query.split('_').collect::<Vec<_>>().join(" ");
            if !expansions.contains(&parts) {
                expansions.push(parts);
            }
        }

        Ok(expansions)
    }

    pub fn rank_results(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
        // Apply similarity threshold
        results.retain(|r| r.score >= self.config.similarity_threshold);

        // Sort by score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(self.config.max_results);

        results
    }

    pub fn boost_exact_matches(&self, results: &mut [SearchResult], query: &str) {
        let query_lower = query.to_lowercase();

        for result in results.iter_mut() {
            // Boost if name contains exact query
            if result.name.to_lowercase().contains(&query_lower) {
                result.score *= self.config.boost_exact_match;
            }
            // Additional boost for exact name match
            if result.name.to_lowercase() == query_lower {
                result.score *= self.config.boost_exact_match * 1.5;
            }
        }
    }

    pub fn extract_keywords(&self, query: &str) -> Result<Vec<String>> {
        let processed = self.preprocess(query)?;
        let tokens: Vec<String> = processed
            .split_whitespace()
            .filter(|t| t.len() > 2) // Filter out very short tokens
            .map(String::from)
            .collect();

        Ok(tokens)
    }

    pub fn is_kernel_term(&self, term: &str) -> bool {
        let term_lower = term.to_lowercase();
        self.kernel_synonyms.contains_key(&term_lower)
            || self.abbreviations.contains_key(&term_lower)
            || term_lower.starts_with("sys_")
            || term_lower.starts_with("__")
            || term_lower.ends_with("_ops")
            || term_lower.ends_with("_operations")
            || term_lower.contains("kernel")
            || term_lower.contains("driver")
    }
}

impl Default for QueryProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_preprocessing() {
        let processor = QueryProcessor::new();
        let result = processor.preprocess("vfs_read function").unwrap();
        assert!(!result.is_empty());
        // Stop word "function" might be removed
        assert!(result.contains("vfs_read") || result.contains("virtual filesystem_read"));
    }

    #[test]
    fn test_abbreviation_expansion() {
        let processor = QueryProcessor::new();
        let result = processor.preprocess("fs and mm modules").unwrap();
        assert!(result.contains("filesystem"));
        assert!(result.contains("memory management"));
    }

    #[test]
    fn test_stop_word_removal() {
        let config = QueryConfig {
            remove_stop_words: true,
            ..Default::default()
        };
        let processor = QueryProcessor::with_config(config);
        let result = processor.preprocess("the kernel is a system").unwrap();
        assert!(!result.contains("the"));
        assert!(!result.contains("is"));
        assert!(!result.contains("a"));
        assert!(result.contains("kernel"));
        assert!(result.contains("system"));
    }

    #[test]
    fn test_query_expansion() {
        let processor = QueryProcessor::new();
        let expansions = processor.expand_query("vfs operations").unwrap();
        assert!(expansions.len() > 1);
        // Should include original and synonym expansions
        assert!(
            expansions.contains(&"vfs operations".to_string())
                || expansions.contains(&"virtual filesystem operations".to_string())
        );
    }

    #[test]
    fn test_compound_term_expansion() {
        let processor = QueryProcessor::new();
        let expansions = processor.expand_query("file_operations").unwrap();
        assert!(expansions.len() >= 2);
        // Should include space-separated version
        assert!(expansions.iter().any(|e| e.contains("file operations")));
    }

    #[test]
    fn test_result_ranking() {
        let processor = QueryProcessor::new();
        let results = vec![
            SearchResult {
                id: 1,
                name: "test1".to_string(),
                file_path: "/path1".to_string(),
                score: 0.9,
                context: None,
            },
            SearchResult {
                id: 2,
                name: "test2".to_string(),
                file_path: "/path2".to_string(),
                score: 0.6,
                context: None,
            },
            SearchResult {
                id: 3,
                name: "test3".to_string(),
                file_path: "/path3".to_string(),
                score: 0.8,
                context: None,
            },
        ];

        let ranked = processor.rank_results(results);
        assert_eq!(ranked.len(), 2); // 0.6 is below threshold
        assert_eq!(ranked[0].score, 0.9);
        assert_eq!(ranked[1].score, 0.8);
    }

    #[test]
    fn test_exact_match_boosting() {
        let processor = QueryProcessor::new();
        let mut results = vec![
            SearchResult {
                id: 1,
                name: "vfs_read".to_string(),
                file_path: "/fs/read.c".to_string(),
                score: 0.8,
                context: None,
            },
            SearchResult {
                id: 2,
                name: "do_read".to_string(),
                file_path: "/kernel/read.c".to_string(),
                score: 0.8,
                context: None,
            },
        ];

        processor.boost_exact_matches(&mut results, "vfs_read");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_keyword_extraction() {
        let processor = QueryProcessor::new();
        let keywords = processor
            .extract_keywords("the Linux kernel memory management")
            .unwrap();
        assert!(keywords.contains(&"linux".to_string()));
        assert!(keywords.contains(&"kernel".to_string()));
        // "the" should be filtered out
        assert!(!keywords.contains(&"the".to_string()));
    }

    #[test]
    fn test_kernel_term_detection() {
        let processor = QueryProcessor::new();
        assert!(processor.is_kernel_term("sys_read"));
        assert!(processor.is_kernel_term("__init"));
        assert!(processor.is_kernel_term("file_operations"));
        assert!(processor.is_kernel_term("vfs"));
        assert!(processor.is_kernel_term("driver"));
        assert!(!processor.is_kernel_term("hello"));
        assert!(!processor.is_kernel_term("world"));
    }
}
