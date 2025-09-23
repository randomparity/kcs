//! KCS Semantic Search Component
//!
//! This crate provides semantic search functionality for the Kernel Context Server,
//! including embedding generation, query processing, and pgvector integration.

pub mod embeddings;
pub mod query;
pub mod vector_db;

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::collections::HashSet;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub top_k: usize,
    pub threshold: Option<f32>,
    pub config: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub symbol_id: i64,
    pub name: String,
    pub file_path: String,
    pub line_number: i32,
    pub score: f32,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    pub model_name: String,
    pub dimension: usize,
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self {
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            dimension: 384,
        }
    }
}

/// Main search engine that coordinates embedding generation and vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub symbol_id: i64,
    pub name: String,
    pub file_path: String,
    pub line_number: i32,
    pub text: String,
}

pub struct SearchEngine {
    _model: EmbeddingModel,
    index_path: Option<PathBuf>,
}

impl SearchEngine {
    pub fn new() -> Result<Self> {
        Ok(Self { _model: EmbeddingModel::default(), index_path: None })
    }

    pub fn with_model(model: EmbeddingModel) -> Result<Self> {
        Ok(Self { _model: model, index_path: None })
    }

    pub fn with_index_file(mut self, path: PathBuf) -> Self {
        self.index_path = Some(path);
        self
    }

    fn load_index(&self) -> Result<Vec<IndexEntry>> {
        if let Some(path) = &self.index_path {
            if path.exists() {
                let data = fs::read_to_string(path)
                    .with_context(|| format!("failed to read index file {}", path.display()))?;
                let entries: Vec<IndexEntry> = serde_json::from_str(&data)
                    .with_context(|| format!("failed to parse index file {}", path.display()))?;
                Ok(entries)
            } else {
                Ok(Vec::new())
            }
        } else {
            Ok(Vec::new())
        }
    }

    fn save_index(&self, entries: &[IndexEntry]) -> Result<()> {
        if let Some(path) = &self.index_path {
            let json = serde_json::to_string_pretty(entries)?;
            fs::write(path, json)
                .with_context(|| format!("failed to write index file {}", path.display()))?;
        }
        Ok(())
    }

    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        // If index file is configured, perform a simple token-overlap search
        if self.index_path.is_some() {
            let entries = self.load_index()?;
            let q_tokens = tokenize(&query.text);

            let mut scored: Vec<SearchResult> = entries
                .into_iter()
                .map(|e| {
                    let score = jaccard_similarity(&q_tokens, &tokenize(&e.text));
                    SearchResult {
                        symbol_id: e.symbol_id,
                        name: e.name,
                        file_path: e.file_path,
                        line_number: e.line_number,
                        score,
                        snippet: String::new(),
                    }
                })
                .collect();

            if let Some(th) = query.threshold {
                scored.retain(|r| r.score >= th);
            }
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(query.top_k);
            return Ok(scored);
        }

        // Default placeholder behavior
        Ok(Vec::new())
    }

    pub async fn index_symbol(&self, symbol_id: i64, text: &str) -> Result<()> {
        if self.index_path.is_some() {
            let mut entries = self.load_index()?;
            entries.push(IndexEntry {
                symbol_id,
                name: format!("symbol_{}", symbol_id),
                file_path: String::new(),
                line_number: 0,
                text: text.to_string(),
            });
            self.save_index(&entries)?;
        }
        Ok(())
    }
}

fn tokenize(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() { return 0.0; }
    let inter = a.intersection(b).count() as f32;
    let uni = a.union(b).count() as f32;
    if uni == 0.0 { 0.0 } else { inter / uni }
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default search engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_query_creation() {
        let query = SearchQuery {
            text: "test query".to_string(),
            top_k: 5,
            threshold: Some(0.8),
            config: Some("x86_64:defconfig".to_string()),
        };

        assert_eq!(query.text, "test query");
        assert_eq!(query.top_k, 5);
        assert_eq!(query.threshold, Some(0.8));
        assert_eq!(query.config.as_deref(), Some("x86_64:defconfig"));
    }

    #[test]
    fn test_search_result_structure() {
        let result = SearchResult {
            symbol_id: 123,
            name: "vfs_read".to_string(),
            file_path: "/fs/read_write.c".to_string(),
            line_number: 456,
            score: 0.95,
            snippet: "Function for reading from VFS".to_string(),
        };

        assert_eq!(result.symbol_id, 123);
        assert_eq!(result.name, "vfs_read");
        assert_eq!(result.file_path, "/fs/read_write.c");
        assert_eq!(result.line_number, 456);
        assert_eq!(result.score, 0.95);
        assert_eq!(result.snippet, "Function for reading from VFS");
    }

    #[test]
    fn test_embedding_model_default() {
        let model = EmbeddingModel::default();
        assert_eq!(model.model_name, "BAAI/bge-small-en-v1.5");
        assert_eq!(model.dimension, 384);
    }

    #[test]
    fn test_search_engine_creation() {
        let engine = SearchEngine::new();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_search_engine_with_custom_model() {
        let model = EmbeddingModel {
            model_name: "custom-model".to_string(),
            dimension: 512,
        };

        let engine = SearchEngine::with_model(model);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_empty_search() {
        let engine = SearchEngine::new().unwrap();
        let query = SearchQuery {
            text: String::new(),
            top_k: 10,
            threshold: None,
            config: None,
        };

        let results = engine.search(query).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_index_symbol() {
        let engine = SearchEngine::new().unwrap();
        let result = engine.index_symbol(1, "test symbol").await;
        assert!(result.is_ok());
    }
}
