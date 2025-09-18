//! Embedding generation module for semantic search
//!
//! This module provides embedding generation capabilities for semantic search.
//! Currently implements a deterministic hash-based approach for testing,
//! with structure ready for ML model integration.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub vector: Vec<f32>,
    pub dimension: usize,
}

impl Embedding {
    /// Create a new embedding with the given vector
    pub fn new(vector: Vec<f32>) -> Self {
        let dimension = vector.len();
        Self { vector, dimension }
    }

    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, other: &Embedding) -> Result<f32> {
        if self.dimension != other.dimension {
            anyhow::bail!(
                "Dimension mismatch: {} vs {}",
                self.dimension,
                other.dimension
            );
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Normalize the embedding vector to unit length
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            self.vector.iter_mut().for_each(|x| *x /= norm);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Dimension of the embedding vectors
    pub dimension: usize,
    /// Model name or type
    pub model_name: String,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Maximum text length to process
    pub max_text_length: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            model_name: "hash-based".to_string(),
            normalize: true,
            max_text_length: 512,
        }
    }
}

pub struct EmbeddingGenerator {
    config: EmbeddingConfig,
}

impl EmbeddingGenerator {
    pub fn new() -> Result<Self> {
        Self::with_config(EmbeddingConfig::default())
    }

    pub fn with_config(config: EmbeddingConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn generate(&self, text: &str) -> Result<Embedding> {
        // Preprocess text
        let processed = self.preprocess_text(text);

        // Generate embedding based on model type
        let mut embedding = match self.config.model_name.as_str() {
            "hash-based" => self.generate_hash_embedding(&processed)?,
            _ => {
                // Placeholder for future ML models
                self.generate_hash_embedding(&processed)?
            }
        };

        // Normalize if configured
        if self.config.normalize {
            embedding.normalize();
        }

        Ok(embedding)
    }

    /// Preprocess text for embedding generation
    fn preprocess_text(&self, text: &str) -> String {
        // Convert to lowercase and truncate if necessary
        let mut processed = text.to_lowercase();

        // Truncate to max length
        if processed.len() > self.config.max_text_length {
            processed.truncate(self.config.max_text_length);
        }

        // Remove extra whitespace
        processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");

        processed
    }

    /// Generate deterministic hash-based embeddings for testing
    fn generate_hash_embedding(&self, text: &str) -> Result<Embedding> {
        let mut vector = vec![0.0; self.config.dimension];

        // Create multiple hash values from text with different seeds
        for (i, chunk) in text.as_bytes().chunks(8).enumerate() {
            let mut hasher = DefaultHasher::new();
            hasher.write_usize(i); // Use index as seed
            chunk.hash(&mut hasher);
            let hash = hasher.finish();

            // Distribute hash bits across embedding dimensions
            for j in 0..64.min(self.config.dimension - (i * 64)) {
                let idx = (i * 64 + j) % self.config.dimension;
                let bit = ((hash >> j) & 1) as f32;
                // Map to [-1, 1] range with some variation
                vector[idx] = (bit * 2.0 - 1.0) * (1.0 + (j as f32 * 0.01));
            }
        }

        // Add some deterministic "semantic" features
        if text.contains("kernel") {
            vector[0] += 0.5;
        }
        if text.contains("syscall") || text.contains("sys_") {
            vector[1] += 0.5;
        }
        if text.contains("driver") {
            vector[2] += 0.5;
        }
        if text.contains("memory") || text.contains("alloc") {
            vector[3] += 0.5;
        }
        if text.contains("file") || text.contains("vfs") {
            vector[4] += 0.5;
        }

        Ok(Embedding::new(vector))
    }

    pub fn batch_generate(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        texts
            .iter()
            .map(|t| self.generate(t))
            .collect::<Result<Vec<_>>>()
            .context("Failed to generate batch embeddings")
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

impl Default for EmbeddingGenerator {
    fn default() -> Self {
        Self::new().expect("Failed to create default embedding generator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_generation() {
        let generator = EmbeddingGenerator::new().unwrap();
        let embedding = generator.generate("test text").unwrap();
        assert_eq!(embedding.dimension, 384);
        assert_eq!(embedding.vector.len(), 384);
    }

    #[test]
    fn test_embedding_deterministic() {
        let generator = EmbeddingGenerator::new().unwrap();
        let text = "Linux kernel VFS subsystem";
        let emb1 = generator.generate(text).unwrap();
        let emb2 = generator.generate(text).unwrap();

        // Should generate same embedding for same text
        assert_eq!(emb1.vector, emb2.vector);
    }

    #[test]
    fn test_embedding_different_texts() {
        let generator = EmbeddingGenerator::new().unwrap();
        let emb1 = generator.generate("syscall handler").unwrap();
        let emb2 = generator.generate("memory allocator").unwrap();

        // Different texts should produce different embeddings
        assert_ne!(emb1.vector, emb2.vector);
    }

    #[test]
    fn test_cosine_similarity() {
        let generator = EmbeddingGenerator::new().unwrap();
        let emb1 = generator.generate("kernel syscall").unwrap();
        let emb2 = generator.generate("kernel syscall").unwrap();
        let emb3 = generator.generate("completely different").unwrap();

        // Same text should have similarity close to 1
        let sim_same = emb1.cosine_similarity(&emb2).unwrap();
        assert!((sim_same - 1.0).abs() < 0.001);

        // Different texts should have lower similarity
        let sim_diff = emb1.cosine_similarity(&emb3).unwrap();
        assert!(sim_diff < sim_same);
    }

    #[test]
    fn test_batch_generation() {
        let generator = EmbeddingGenerator::new().unwrap();
        let texts = vec![
            "kernel module".to_string(),
            "device driver".to_string(),
            "file system".to_string(),
        ];

        let embeddings = generator.batch_generate(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);

        for emb in &embeddings {
            assert_eq!(emb.dimension, 384);
        }
    }

    #[test]
    fn test_normalization() {
        let mut embedding = Embedding::new(vec![3.0, 4.0, 0.0]);
        embedding.normalize();

        // After normalization, magnitude should be 1
        let magnitude: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_custom_config() {
        let config = EmbeddingConfig {
            dimension: 512,
            model_name: "custom".to_string(),
            normalize: false,
            max_text_length: 1024,
        };

        let generator = EmbeddingGenerator::with_config(config).unwrap();
        let embedding = generator.generate("test").unwrap();

        assert_eq!(embedding.dimension, 512);
        assert_eq!(generator.dimension(), 512);
        assert_eq!(generator.model_name(), "custom");
    }

    #[test]
    fn test_text_preprocessing() {
        let generator = EmbeddingGenerator::new().unwrap();

        // Test case insensitivity
        let emb1 = generator.generate("Linux KERNEL").unwrap();
        let emb2 = generator.generate("linux kernel").unwrap();
        assert_eq!(emb1.vector, emb2.vector);

        // Test whitespace normalization
        let emb3 = generator.generate("linux  \t kernel\n").unwrap();
        assert_eq!(emb1.vector, emb3.vector);
    }

    #[test]
    fn test_text_truncation() {
        let config = EmbeddingConfig {
            dimension: 384,
            model_name: "test".to_string(),
            normalize: false,
            max_text_length: 10, // Very short limit for testing
        };

        let generator = EmbeddingGenerator::with_config(config).unwrap();
        let long_text = "This is a very long text that should be truncated";
        let embedding = generator.generate(long_text);

        assert!(embedding.is_ok());
        // Should still generate valid embedding even with truncation
        assert_eq!(embedding.unwrap().dimension, 384);
    }

    #[test]
    fn test_empty_text_handling() {
        let generator = EmbeddingGenerator::new().unwrap();

        let emb_empty = generator.generate("").unwrap();
        let emb_spaces = generator.generate("   ").unwrap();

        // Both should generate valid embeddings
        assert_eq!(emb_empty.dimension, 384);
        assert_eq!(emb_spaces.dimension, 384);

        // Empty strings should produce the same embedding
        assert_eq!(emb_empty.vector, emb_spaces.vector);
    }

    #[test]
    fn test_semantic_features() {
        let generator = EmbeddingGenerator::new().unwrap();

        // Test that semantic features affect embeddings
        let kernel_emb = generator.generate("kernel").unwrap();
        let syscall_emb = generator.generate("syscall").unwrap();
        let driver_emb = generator.generate("driver").unwrap();

        // First few dimensions should be affected by semantic features
        assert!(kernel_emb.vector[0] != 0.0);
        assert!(syscall_emb.vector[1] != 0.0);
        assert!(driver_emb.vector[2] != 0.0);
    }

    #[test]
    fn test_similarity_symmetry() {
        let generator = EmbeddingGenerator::new().unwrap();

        let emb1 = generator.generate("test1").unwrap();
        let emb2 = generator.generate("test2").unwrap();

        let sim12 = emb1.cosine_similarity(&emb2).unwrap();
        let sim21 = emb2.cosine_similarity(&emb1).unwrap();

        // Cosine similarity should be symmetric
        assert!((sim12 - sim21).abs() < 0.0001);
    }

    #[test]
    fn test_batch_generate_consistency() {
        let generator = EmbeddingGenerator::new().unwrap();

        let texts = vec!["text1".to_string(), "text2".to_string()];

        let batch_embeddings = generator.batch_generate(&texts).unwrap();
        let individual_emb1 = generator.generate("text1").unwrap();
        let individual_emb2 = generator.generate("text2").unwrap();

        // Batch generation should produce same results as individual
        assert_eq!(batch_embeddings[0].vector, individual_emb1.vector);
        assert_eq!(batch_embeddings[1].vector, individual_emb2.vector);
    }

    #[test]
    fn test_similarity_dimension_mismatch() {
        let emb1 = Embedding::new(vec![1.0, 2.0, 3.0]);
        let emb2 = Embedding::new(vec![1.0, 2.0]);

        // Should return an error due to dimension mismatch
        let result = emb1.cosine_similarity(&emb2);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Dimension mismatch"));
    }

    #[test]
    fn test_zero_vector_similarity() {
        let emb1 = Embedding::new(vec![0.0, 0.0, 0.0]);
        let emb2 = Embedding::new(vec![1.0, 2.0, 3.0]);

        let similarity = emb1.cosine_similarity(&emb2).unwrap();

        // Similarity with zero vector should be 0
        assert_eq!(similarity, 0.0);
    }
}
