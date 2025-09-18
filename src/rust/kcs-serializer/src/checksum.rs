//! Checksum calculation module for chunk integrity verification
//!
//! This module provides SHA256 hash calculation with optional caching,
//! streaming support, and verification capabilities for chunk files.

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// Configuration for checksum calculation
#[derive(Debug, Clone)]
pub struct ChecksumConfig {
    /// Hash algorithm to use
    pub algorithm: HashAlgorithm,
    /// Whether to verify checksums on read operations
    pub verify_on_read: bool,
    /// Buffer size for file I/O operations
    pub buffer_size: usize,
    /// Enable parallel processing for large files
    pub parallel_processing: bool,
    /// Enable checksum caching
    pub cache_checksums: bool,
}

impl Default for ChecksumConfig {
    fn default() -> Self {
        Self {
            algorithm: HashAlgorithm::Sha256,
            verify_on_read: false,
            buffer_size: 64 * 1024, // 64KB buffer
            parallel_processing: false,
            cache_checksums: true,
        }
    }
}

/// Supported hash algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HashAlgorithm {
    /// SHA-256 algorithm
    Sha256,
    /// SHA-1 algorithm (legacy support)
    Sha1,
    /// MD5 algorithm (legacy support)
    Md5,
}

/// Errors that can occur during checksum operations
#[derive(Debug)]
pub enum ChecksumError {
    /// Invalid input provided
    InvalidInput(String),
    /// I/O error occurred
    IoError(std::io::Error),
    /// Checksum verification failed
    VerificationFailed {
        /// Expected checksum value
        expected: String,
        /// Actual calculated checksum
        actual: String,
    },
    /// Unsupported algorithm requested
    UnsupportedAlgorithm(String),
}

impl std::fmt::Display for ChecksumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::VerificationFailed { expected, actual } => {
                write!(
                    f,
                    "Checksum verification failed: expected {}, got {}",
                    expected, actual
                )
            }
            Self::UnsupportedAlgorithm(algo) => write!(f, "Unsupported algorithm: {}", algo),
        }
    }
}

impl std::error::Error for ChecksumError {}

impl From<std::io::Error> for ChecksumError {
    fn from(error: std::io::Error) -> Self {
        ChecksumError::IoError(error)
    }
}

/// Checksum calculation result with metadata
#[derive(Debug, Clone)]
pub struct ChecksumResult {
    /// Algorithm used for calculation
    pub algorithm: HashAlgorithm,
    /// Calculated hash value (hex string)
    pub hash: String,
    /// Size of data processed
    pub data_size: usize,
    /// Time taken for calculation in milliseconds
    pub calculation_time_ms: u64,
    /// Whether result was retrieved from cache
    pub from_cache: bool,
}

/// Main checksum calculator with caching support
#[derive(Debug)]
pub struct ChecksumCalculator {
    config: ChecksumConfig,
    checksum_cache: HashMap<String, String>,
}

impl ChecksumCalculator {
    /// Create a new checksum calculator with the given configuration
    pub fn new(config: ChecksumConfig) -> Result<Self, ChecksumError> {
        Ok(Self {
            config,
            checksum_cache: HashMap::new(),
        })
    }

    /// Calculate SHA256 checksum for data in memory
    pub fn calculate_sha256(&mut self, data: &[u8]) -> Result<String, ChecksumError> {
        if self.config.algorithm != HashAlgorithm::Sha256 {
            return Err(ChecksumError::UnsupportedAlgorithm(format!(
                "{:?}",
                self.config.algorithm
            )));
        }

        // Generate cache key if caching is enabled
        let cache_key = if self.config.cache_checksums {
            Some(format!("sha256:{}:{}", data.len(), self.quick_hash(data)))
        } else {
            None
        };

        // Check cache first
        if let Some(ref key) = cache_key {
            if let Some(cached_hash) = self.checksum_cache.get(key) {
                return Ok(cached_hash.clone());
            }
        }

        // Calculate SHA256
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hex::encode(hasher.finalize()).to_lowercase();

        // Store in cache if enabled
        if let Some(key) = cache_key {
            self.checksum_cache.insert(key, hash.clone());
        }

        Ok(hash)
    }

    /// Calculate SHA256 checksum for a file
    pub fn calculate_sha256_file(&mut self, file_path: &Path) -> Result<String, ChecksumError> {
        if self.config.algorithm != HashAlgorithm::Sha256 {
            return Err(ChecksumError::UnsupportedAlgorithm(format!(
                "{:?}",
                self.config.algorithm
            )));
        }

        // Generate cache key based on file path and metadata
        let cache_key = if self.config.cache_checksums {
            match std::fs::metadata(file_path) {
                Ok(metadata) => {
                    let modified = metadata
                        .modified()
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    Some(format!(
                        "sha256_file:{}:{}:{}",
                        file_path.to_string_lossy(),
                        metadata.len(),
                        modified
                    ))
                }
                Err(_) => None,
            }
        } else {
            None
        };

        // Check cache first
        if let Some(ref key) = cache_key {
            if let Some(cached_hash) = self.checksum_cache.get(key) {
                return Ok(cached_hash.clone());
            }
        }

        // Calculate hash from file
        let file = File::open(file_path)?;
        let reader = BufReader::with_capacity(self.config.buffer_size, file);
        let hash = self.calculate_streaming(Box::new(reader))?;

        // Store in cache if enabled
        if let Some(key) = cache_key {
            self.checksum_cache.insert(key, hash.clone());
        }

        Ok(hash)
    }

    /// Calculate checksum with additional metadata
    pub fn calculate_with_metadata(
        &mut self,
        data: &[u8],
    ) -> Result<ChecksumResult, ChecksumError> {
        let start_time = Instant::now();

        // Generate cache key if caching is enabled
        let cache_key = if self.config.cache_checksums {
            Some(format!("sha256:{}:{}", data.len(), self.quick_hash(data)))
        } else {
            None
        };

        // Check cache first
        let (hash, from_cache) = if let Some(ref key) = cache_key {
            if let Some(cached_hash) = self.checksum_cache.get(key) {
                (cached_hash.clone(), true)
            } else {
                let hash = self.calculate_sha256(data)?;
                (hash, false)
            }
        } else {
            let hash = self.calculate_sha256(data)?;
            (hash, false)
        };

        let elapsed = start_time.elapsed();

        Ok(ChecksumResult {
            algorithm: self.config.algorithm,
            hash,
            data_size: data.len(),
            calculation_time_ms: elapsed.as_millis() as u64,
            from_cache,
        })
    }

    /// Verify that data matches the expected checksum
    pub fn verify_checksum(&mut self, data: &[u8], expected: &str) -> Result<bool, ChecksumError> {
        // Validate expected checksum format first
        if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(ChecksumError::InvalidInput(format!(
                "Invalid checksum format: expected 64 hex characters, got '{}'",
                expected
            )));
        }

        let actual = self.calculate_sha256(data)?;

        // Normalize case for comparison
        let expected_lower = expected.to_lowercase();
        let actual_lower = actual.to_lowercase();

        Ok(expected_lower == actual_lower)
    }

    /// Verify that a file matches the expected checksum
    pub fn verify_file_checksum(
        &mut self,
        file_path: &Path,
        expected: &str,
    ) -> Result<bool, ChecksumError> {
        // Validate expected checksum format first
        if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(ChecksumError::InvalidInput(format!(
                "Invalid checksum format: expected 64 hex characters, got '{}'",
                expected
            )));
        }

        let actual = self.calculate_sha256_file(file_path)?;

        // Normalize case for comparison
        let expected_lower = expected.to_lowercase();
        let actual_lower = actual.to_lowercase();

        Ok(expected_lower == actual_lower)
    }

    /// Clear the checksum cache
    pub fn clear_cache(&mut self) {
        self.checksum_cache.clear();
    }

    /// Get the number of entries in the cache
    pub fn get_cache_size(&self) -> usize {
        self.checksum_cache.len()
    }

    /// Calculate checksum from a streaming reader
    pub fn calculate_streaming(
        &mut self,
        mut reader: Box<dyn Read>,
    ) -> Result<String, ChecksumError> {
        if self.config.algorithm != HashAlgorithm::Sha256 {
            return Err(ChecksumError::UnsupportedAlgorithm(format!(
                "{:?}",
                self.config.algorithm
            )));
        }

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; self.config.buffer_size];

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hex::encode(hasher.finalize()).to_lowercase())
    }

    // Helper methods

    /// Generate a quick hash for cache key generation
    fn quick_hash(&self, data: &[u8]) -> u64 {
        // Simple FNV-1a hash for cache key generation
        let mut hash = 14695981039346656037u64;
        for &byte in data.iter().take(1024) {
            // Only hash first 1KB for speed
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        hash
    }
}

// Utility functions for testing
/// Create test data with a specific pattern
pub fn create_test_data(size: usize, pattern: u8) -> Vec<u8> {
    vec![pattern; size]
}

/// Create pseudorandom test data
pub fn create_random_data(size: usize) -> Vec<u8> {
    // Simple pseudorandom data for testing
    (0..size).map(|i| (i * 37 + 13) as u8).collect()
}

/// Known test vector for SHA256 validation
pub struct KnownTestVector {
    /// Input string to hash
    pub input: &'static str,
    /// Expected SHA256 hash (lowercase hex)
    pub expected_sha256: &'static str,
}

/// Standard SHA256 test vectors for validation
pub const SHA256_TEST_VECTORS: &[KnownTestVector] = &[
    KnownTestVector {
        input: "",
        expected_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    KnownTestVector {
        input: "abc",
        expected_sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    },
    KnownTestVector {
        input: "The quick brown fox jumps over the lazy dog",
        expected_sha256: "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
    },
    KnownTestVector {
        input: "The quick brown fox jumps over the lazy dog.",
        expected_sha256: "ef537f25c895bfa782526529a9b63d97aa631564d5d789c2b765448c8635fb6c",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checksum_calculator_creation() -> Result<(), ChecksumError> {
        let config = ChecksumConfig::default();
        let _calculator = ChecksumCalculator::new(config)?;
        Ok(())
    }

    #[test]
    fn test_sha256_basic() -> Result<(), ChecksumError> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let data = b"test";
        let hash = calculator.calculate_sha256(data)?;

        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

        Ok(())
    }

    #[test]
    fn test_known_vectors() -> Result<(), ChecksumError> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        for vector in SHA256_TEST_VECTORS {
            let result = calculator.calculate_sha256(vector.input.as_bytes())?;
            assert_eq!(result.to_lowercase(), vector.expected_sha256.to_lowercase());
        }

        Ok(())
    }

    #[test]
    fn test_caching() -> Result<(), ChecksumError> {
        let config = ChecksumConfig {
            cache_checksums: true,
            ..Default::default()
        };
        let mut calculator = ChecksumCalculator::new(config)?;

        let data = b"test data for caching";

        // First calculation
        let hash1 = calculator.calculate_sha256(data)?;
        assert_eq!(calculator.get_cache_size(), 1);

        // Second calculation should use cache
        let hash2 = calculator.calculate_sha256(data)?;
        assert_eq!(hash1, hash2);
        assert_eq!(calculator.get_cache_size(), 1);

        // Clear cache
        calculator.clear_cache();
        assert_eq!(calculator.get_cache_size(), 0);

        Ok(())
    }

    #[test]
    fn test_verification() -> Result<(), ChecksumError> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let data = b"test data";
        let hash = calculator.calculate_sha256(data)?;

        // Verify with correct hash
        assert!(calculator.verify_checksum(data, &hash)?);

        // Verify with incorrect hash should fail
        let result = calculator.verify_checksum(data, "invalid_hash");
        assert!(result.is_err());

        Ok(())
    }
}
