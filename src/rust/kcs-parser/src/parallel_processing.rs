//! Performance optimization with rayon parallelization for call graph extraction.
//!
//! This module provides parallel processing capabilities for the call extraction pipeline,
//! enabling efficient processing of large codebases by leveraging multiple CPU cores.
//! It includes parallel file processing, AST traversal, and call detection optimizations.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tree_sitter::{Language, Parser, Tree};

use crate::ast_traversal::{TraversalConfig, TraversalStats};
use crate::call_extraction::{CallExtractionConfig, CallExtractor, ExtractorStats};
use kcs_graph::CallEdgeModel;

/// Configuration for parallel processing behavior.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use for parallel processing
    /// If None, uses rayon's default (number of CPU cores)
    pub num_threads: Option<usize>,
    /// Number of files to process in each batch
    pub batch_size: usize,
    /// Whether to enable parallel AST traversal within files
    pub enable_parallel_traversal: bool,
    /// Minimum file size (in bytes) to consider for parallel processing
    pub min_file_size_for_parallel: usize,
    /// Maximum number of files to process in parallel
    pub max_concurrent_files: usize,
    /// Whether to use memory mapping for large files
    pub use_memory_mapping: bool,
    /// Memory mapping threshold in bytes
    pub memory_mapping_threshold: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use rayon default
            batch_size: 50,
            enable_parallel_traversal: true,
            min_file_size_for_parallel: 10 * 1024, // 10KB
            max_concurrent_files: 100,
            use_memory_mapping: true,
            memory_mapping_threshold: 1024 * 1024, // 1MB
        }
    }
}

/// Statistics for parallel processing performance.
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    /// Total number of files processed
    pub files_processed: usize,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
    /// Time spent in parallel processing in milliseconds
    pub parallel_time_ms: u64,
    /// Time spent in sequential overhead in milliseconds
    pub sequential_overhead_ms: u64,
    /// Number of threads actually used
    pub threads_used: usize,
    /// Average processing time per file in milliseconds
    pub avg_file_time_ms: f64,
    /// Number of files that used memory mapping
    pub memory_mapped_files: usize,
    /// Total memory mapped (in bytes)
    pub total_memory_mapped: usize,
    /// Aggregated extraction statistics
    pub extraction_stats: ExtractorStats,
    /// Aggregated traversal statistics
    pub traversal_stats: TraversalStats,
}

impl ParallelStats {
    /// Merge statistics from another ParallelStats.
    pub fn merge(&mut self, other: &ParallelStats) {
        self.files_processed += other.files_processed;
        self.total_time_ms += other.total_time_ms;
        self.parallel_time_ms += other.parallel_time_ms;
        self.sequential_overhead_ms += other.sequential_overhead_ms;
        self.memory_mapped_files += other.memory_mapped_files;
        self.total_memory_mapped += other.total_memory_mapped;
        self.extraction_stats.merge(&other.extraction_stats);

        // Update averages
        if self.files_processed > 0 {
            self.avg_file_time_ms = self.total_time_ms as f64 / self.files_processed as f64;
        }
    }
}

/// Result of processing a single file in parallel.
#[derive(Debug)]
pub struct FileProcessingResult {
    /// Path of the processed file
    pub file_path: PathBuf,
    /// Extracted call edges
    pub call_edges: Vec<CallEdgeModel>,
    /// Processing time for this file in milliseconds
    pub processing_time_ms: u64,
    /// Size of the file in bytes
    pub file_size: usize,
    /// Whether memory mapping was used
    pub used_memory_mapping: bool,
    /// Error if processing failed
    pub error: Option<String>,
}

/// Main parallel processing engine for call graph extraction.
pub struct ParallelProcessor {
    /// Configuration for parallel processing
    config: ParallelConfig,
    /// Configuration for call extraction
    extraction_config: CallExtractionConfig,
    /// Configuration for AST traversal
    _traversal_config: TraversalConfig,
    /// Language parser for Tree-sitter
    language: Language,
    /// Statistics collected during processing
    stats: Arc<Mutex<ParallelStats>>,
}

impl ParallelProcessor {
    /// Create a new parallel processor with the given configurations.
    pub fn new(
        config: ParallelConfig,
        extraction_config: CallExtractionConfig,
        traversal_config: TraversalConfig,
    ) -> Result<Self> {
        // Initialize rayon thread pool if specified
        if let Some(num_threads) = config.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .context("Failed to initialize rayon thread pool")?;
        }

        let language = tree_sitter_c::language();
        let stats = Arc::new(Mutex::new(ParallelStats::default()));

        Ok(Self {
            config,
            extraction_config,
            _traversal_config: traversal_config,
            language,
            stats,
        })
    }

    /// Create a new parallel processor with default configurations.
    pub fn new_default() -> Result<Self> {
        Self::new(
            ParallelConfig::default(),
            CallExtractionConfig::default(),
            TraversalConfig::default(),
        )
    }

    /// Process multiple files in parallel and extract call graphs.
    pub fn process_files_parallel(
        &mut self,
        file_paths: &[PathBuf],
    ) -> Result<Vec<FileProcessingResult>> {
        let start_time = Instant::now();

        // Reset stats
        {
            let mut stats = self.stats.lock().unwrap();
            *stats = ParallelStats::default();
            stats.threads_used = rayon::current_num_threads();
        }

        // Process files in batches to avoid overwhelming the system
        let batches: Vec<_> = file_paths.chunks(self.config.batch_size).collect();

        let mut all_results = Vec::new();

        for batch in batches {
            let batch_start = Instant::now();

            // Process this batch in parallel
            let batch_results: Vec<FileProcessingResult> = batch
                .par_iter()
                .filter(|path| self.should_process_file(path))
                .map(|path| self.process_single_file(path))
                .collect();

            all_results.extend(batch_results);

            // Update parallel processing time
            let batch_time = batch_start.elapsed().as_millis() as u64;
            {
                let mut stats = self.stats.lock().unwrap();
                stats.parallel_time_ms += batch_time;
            }
        }

        // Update total processing time
        let total_time = start_time.elapsed().as_millis() as u64;
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_time_ms = total_time;
            stats.sequential_overhead_ms = total_time - stats.parallel_time_ms;
            stats.files_processed = all_results.len();

            if stats.files_processed > 0 {
                stats.avg_file_time_ms = stats.total_time_ms as f64 / stats.files_processed as f64;
            }
        }

        Ok(all_results)
    }

    /// Process a batch of files and merge the results.
    pub fn process_files_batch(
        &mut self,
        file_paths: &[PathBuf],
    ) -> Result<(Vec<CallEdgeModel>, ParallelStats)> {
        let results = self.process_files_parallel(file_paths)?;

        let mut all_edges = Vec::new();
        let mut merged_stats = ParallelStats::default();

        for result in results {
            if result.error.is_none() {
                all_edges.extend(result.call_edges);
            }

            // Update statistics
            merged_stats.files_processed += 1;
            merged_stats.total_time_ms += result.processing_time_ms;

            if result.used_memory_mapping {
                merged_stats.memory_mapped_files += 1;
                merged_stats.total_memory_mapped += result.file_size;
            }
        }

        // Get final stats
        let final_stats = {
            let stats = self.stats.lock().unwrap();
            stats.clone()
        };

        Ok((all_edges, final_stats))
    }

    /// Process files from multiple directories in parallel.
    pub fn process_directories_parallel(
        &mut self,
        directories: &[PathBuf],
        file_extensions: &[&str],
    ) -> Result<(Vec<CallEdgeModel>, ParallelStats)> {
        // Collect all files from directories in parallel
        let all_files: Vec<PathBuf> = directories
            .par_iter()
            .flat_map(|dir| {
                walkdir::WalkDir::new(dir)
                    .into_iter()
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| entry.file_type().is_file())
                    .filter(|entry| {
                        if let Some(ext) = entry.path().extension() {
                            if let Some(ext_str) = ext.to_str() {
                                return file_extensions.contains(&ext_str);
                            }
                        }
                        false
                    })
                    .map(|entry| entry.path().to_path_buf())
                    .collect::<Vec<_>>()
            })
            .collect();

        self.process_files_batch(&all_files)
    }

    /// Get current processing statistics.
    pub fn get_stats(&self) -> ParallelStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Reset processing statistics.
    pub fn reset_stats(&mut self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = ParallelStats::default();
    }

    /// Process a single file and extract its call graph.
    fn process_single_file(&self, file_path: &Path) -> FileProcessingResult {
        let start_time = Instant::now();
        let mut result = FileProcessingResult {
            file_path: file_path.to_path_buf(),
            call_edges: Vec::new(),
            processing_time_ms: 0,
            file_size: 0,
            used_memory_mapping: false,
            error: None,
        };

        // Read file content
        let (source_code, _used_mmap) = match self.read_file_content(file_path) {
            Ok((content, mmap)) => {
                result.file_size = content.len();
                result.used_memory_mapping = mmap;
                (content, mmap)
            },
            Err(e) => {
                result.error = Some(format!("Failed to read file: {}", e));
                result.processing_time_ms = start_time.elapsed().as_millis() as u64;
                return result;
            },
        };

        // Parse the source code
        let tree = match self.parse_source_code(&source_code) {
            Ok(tree) => tree,
            Err(e) => {
                result.error = Some(format!("Failed to parse source: {}", e));
                result.processing_time_ms = start_time.elapsed().as_millis() as u64;
                return result;
            },
        };

        // Extract call graph
        match self.extract_call_graph(file_path, &source_code, &tree) {
            Ok(edges) => {
                result.call_edges = edges;
            },
            Err(e) => {
                result.error = Some(format!("Failed to extract calls: {}", e));
            },
        }

        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        result
    }

    /// Read file content, optionally using memory mapping for large files.
    fn read_file_content(&self, file_path: &Path) -> Result<(String, bool)> {
        let file_size = std::fs::metadata(file_path)
            .with_context(|| format!("Failed to get metadata for {}", file_path.display()))?
            .len() as usize;

        if self.config.use_memory_mapping && file_size >= self.config.memory_mapping_threshold {
            // Use memory mapping for large files
            let file = std::fs::File::open(file_path)
                .with_context(|| format!("Failed to open file {}", file_path.display()))?;

            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .with_context(|| format!("Failed to memory map file {}", file_path.display()))?;

            let content = std::str::from_utf8(&mmap)
                .with_context(|| format!("File {} is not valid UTF-8", file_path.display()))?
                .to_string();

            Ok((content, true))
        } else {
            // Read normally for smaller files
            let content = std::fs::read_to_string(file_path)
                .with_context(|| format!("Failed to read file {}", file_path.display()))?;

            Ok((content, false))
        }
    }

    /// Parse source code using Tree-sitter.
    fn parse_source_code(&self, source_code: &str) -> Result<Tree> {
        let mut parser = Parser::new();
        parser
            .set_language(self.language)
            .context("Failed to set language for parser")?;

        parser.parse(source_code, None).context("Failed to parse source code")
    }

    /// Extract call graph from a parsed AST.
    fn extract_call_graph(
        &self,
        file_path: &Path,
        source_code: &str,
        tree: &Tree,
    ) -> Result<Vec<CallEdgeModel>> {
        let mut extractor = CallExtractor::new(self.extraction_config.clone())
            .context("Failed to create call extractor")?;

        extractor
            .extract_from_file(file_path, source_code, tree)
            .with_context(|| format!("Failed to extract calls from {}", file_path.display()))
    }

    /// Check if a file should be processed based on configuration.
    fn should_process_file(&self, file_path: &Path) -> bool {
        // Check file size
        if let Ok(metadata) = std::fs::metadata(file_path) {
            let file_size = metadata.len() as usize;

            // Skip files that are too small for parallel processing benefit
            if file_size < self.config.min_file_size_for_parallel {
                return false;
            }

            // Skip files that exceed extraction config limits
            if file_size > self.extraction_config.max_file_size {
                return false;
            }
        }

        // Check file extension (basic C/C++ files)
        if let Some(ext) = file_path.extension() {
            if let Some(ext_str) = ext.to_str() {
                return matches!(
                    ext_str.to_lowercase().as_str(),
                    "c" | "h" | "cc" | "cpp" | "cxx" | "hpp"
                );
            }
        }

        false
    }
}

/// Parallel processing utilities for specific operations.
pub mod utils {
    use super::*;

    /// Parallel file discovery with filtering.
    pub fn discover_files_parallel(
        directories: &[PathBuf],
        extensions: &[&str],
        max_file_size: usize,
    ) -> Result<Vec<PathBuf>> {
        let files: Vec<PathBuf> = directories
            .par_iter()
            .flat_map(|dir| {
                walkdir::WalkDir::new(dir)
                    .into_iter()
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| entry.file_type().is_file())
                    .filter_map(|entry| {
                        let path = entry.path();

                        // Check extension
                        if let Some(ext) = path.extension() {
                            if let Some(ext_str) = ext.to_str() {
                                if !extensions.contains(&ext_str) {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }

                        // Check file size
                        if let Ok(metadata) = entry.metadata() {
                            if metadata.len() as usize > max_file_size {
                                return None;
                            }
                        }

                        Some(path.to_path_buf())
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(files)
    }

    /// Parallel hash computation for files (for caching).
    pub fn compute_file_hashes_parallel(
        file_paths: &[PathBuf],
    ) -> Result<HashMap<PathBuf, String>> {
        use sha2::{Digest, Sha256};

        let hashes: Result<HashMap<PathBuf, String>> = file_paths
            .par_iter()
            .map(|path| {
                let content = std::fs::read(path)
                    .with_context(|| format!("Failed to read file {}", path.display()))?;

                let mut hasher = Sha256::new();
                hasher.update(&content);
                let hash = format!("{:x}", hasher.finalize());

                Ok((path.clone(), hash))
            })
            .collect();

        hashes
    }

    /// Parallel content filtering based on patterns.
    pub fn filter_files_by_content_parallel(
        file_paths: &[PathBuf],
        required_patterns: &[regex::Regex],
        excluded_patterns: &[regex::Regex],
    ) -> Result<Vec<PathBuf>> {
        let filtered: Result<Vec<PathBuf>> = file_paths
            .par_iter()
            .filter_map(|path| {
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        // Check required patterns
                        let has_required = required_patterns.is_empty()
                            || required_patterns.iter().any(|pattern| pattern.is_match(&content));

                        // Check excluded patterns
                        let has_excluded =
                            excluded_patterns.iter().any(|pattern| pattern.is_match(&content));

                        if has_required && !has_excluded {
                            Some(Ok(path.clone()))
                        } else {
                            None
                        }
                    },
                    Err(e) => {
                        Some(Err(anyhow::anyhow!("Failed to read file {}: {}", path.display(), e)))
                    },
                }
            })
            .collect();

        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_files(temp_dir: &TempDir) -> Result<Vec<PathBuf>> {
        let files = vec![
            ("test1.c", "int main() { printf(\"hello\"); return 0; }"),
            ("test2.c", "void foo() { bar(); } void bar() { baz(); }"),
            ("test3.h", "#define MAX_SIZE 1024\nvoid init();"),
        ];

        let mut paths = Vec::new();
        for (name, content) in files {
            let path = temp_dir.path().join(name);
            fs::write(&path, content)?;
            paths.push(path);
        }

        Ok(paths)
    }

    #[test]
    fn test_parallel_processor_creation() {
        let processor = ParallelProcessor::new_default();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert_eq!(config.batch_size, 50);
        assert!(config.enable_parallel_traversal);
        assert_eq!(config.min_file_size_for_parallel, 10 * 1024);
    }

    #[test]
    fn test_parallel_stats_merge() {
        let mut stats1 = ParallelStats {
            files_processed: 5,
            total_time_ms: 1000,
            memory_mapped_files: 2,
            ..Default::default()
        };

        let stats2 = ParallelStats {
            files_processed: 3,
            total_time_ms: 500,
            memory_mapped_files: 1,
            ..Default::default()
        };

        stats1.merge(&stats2);
        assert_eq!(stats1.files_processed, 8);
        assert_eq!(stats1.total_time_ms, 1500);
        assert_eq!(stats1.memory_mapped_files, 3);
    }

    #[test]
    fn test_process_files_parallel() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_paths = create_test_files(&temp_dir)?;

        // Use custom config with small file size threshold for test files
        let config = ParallelConfig {
            min_file_size_for_parallel: 1, // Allow processing of small test files
            ..Default::default()
        };

        let mut processor = ParallelProcessor::new(
            config,
            CallExtractionConfig::default(),
            TraversalConfig::default(),
        )?;
        let results = processor.process_files_parallel(&file_paths)?;

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.error.is_none(), "Processing error: {:?}", result.error);
            assert!(result.processing_time_ms > 0);
        }

        Ok(())
    }

    #[test]
    fn test_file_discovery_parallel() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        create_test_files(&temp_dir)?;

        let directories = vec![temp_dir.path().to_path_buf()];
        let extensions = vec!["c", "h"];
        let files = utils::discover_files_parallel(&directories, &extensions, 1024 * 1024)?;

        assert_eq!(files.len(), 3);

        Ok(())
    }

    #[test]
    fn test_file_size_filtering() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let small_file = temp_dir.path().join("small.c");
        fs::write(&small_file, "int x;")?;

        let processor = ParallelProcessor::new_default()?;
        assert!(!processor.should_process_file(&small_file)); // Too small

        Ok(())
    }

    #[test]
    fn test_memory_mapping_threshold() -> Result<()> {
        let config = ParallelConfig {
            memory_mapping_threshold: 100, // Very low threshold
            use_memory_mapping: true,
            ..Default::default()
        };

        let processor = ParallelProcessor::new(
            config,
            CallExtractionConfig::default(),
            TraversalConfig::default(),
        )?;

        let temp_dir = tempfile::tempdir()?;
        let large_file = temp_dir.path().join("large.c");
        let content = "int main() { return 0; }".repeat(10); // Make it larger
        fs::write(&large_file, content)?;

        let (content, used_mmap) = processor.read_file_content(&large_file)?;
        assert!(used_mmap);
        assert!(!content.is_empty());

        Ok(())
    }

    #[test]
    fn test_hash_computation_parallel() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_paths = create_test_files(&temp_dir)?;

        let hashes = utils::compute_file_hashes_parallel(&file_paths)?;
        assert_eq!(hashes.len(), 3);

        // Verify all hashes are valid hex strings
        for (path, hash) in &hashes {
            assert!(hash.len() == 64, "Invalid hash length for {:?}", path);
            assert!(
                hash.chars().all(|c| c.is_ascii_hexdigit()),
                "Invalid hash format for {:?}",
                path
            );
        }

        Ok(())
    }
}
