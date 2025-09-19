//! Unit tests for SHA256 checksum calculation
//!
//! These tests verify the checksum module provides accurate and reliable
//! SHA256 hash calculation for chunk integrity verification.

use crate::checksum::*;
use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic SHA256 calculation for small data
    #[test]
    fn test_sha256_basic_calculation() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let test_data = b"Hello, World!";
        let result = calculator.calculate_sha256(test_data)?;

        // Verify format (64 character hex string)
        assert_eq!(result.len(), 64);
        assert!(result.chars().all(|c| c.is_ascii_hexdigit()));

        // Test should produce consistent results
        let result2 = calculator.calculate_sha256(test_data)?;
        assert_eq!(result, result2);

        Ok(())
    }

    /// Test SHA256 calculation against known test vectors
    #[test]
    fn test_sha256_known_vectors() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        for test_vector in SHA256_TEST_VECTORS {
            let result = calculator.calculate_sha256(test_vector.input.as_bytes())?;
            assert_eq!(
                result.to_lowercase(),
                test_vector.expected_sha256.to_lowercase(),
                "Failed for input: '{}'",
                test_vector.input
            );
        }

        Ok(())
    }

    /// Test SHA256 calculation for large data (chunk-sized)
    #[test]
    fn test_sha256_large_data() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        // Test with 10MB of data (typical chunk size)
        let large_data = create_test_data(10 * 1024 * 1024, 0xAB);
        let result = calculator.calculate_sha256(&large_data)?;

        assert_eq!(result.len(), 64);
        assert!(result.chars().all(|c| c.is_ascii_hexdigit()));

        // Test with different pattern should give different result
        let different_data = create_test_data(10 * 1024 * 1024, 0xCD);
        let result2 = calculator.calculate_sha256(&different_data)?;
        assert_ne!(result, result2);

        Ok(())
    }

    /// Test file-based SHA256 calculation
    #[test]
    fn test_sha256_file_calculation() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test_file.txt");

        let test_content = b"This is a test file for checksum calculation";
        std::fs::write(&file_path, test_content)?;

        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let file_result = calculator.calculate_sha256_file(&file_path)?;
        let memory_result = calculator.calculate_sha256(test_content)?;

        assert_eq!(file_result, memory_result);
        assert_eq!(file_result.len(), 64);

        Ok(())
    }

    /// Test checksum verification functionality
    #[test]
    fn test_checksum_verification() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let test_data = b"verification test data";
        let correct_checksum = calculator.calculate_sha256(test_data)?;

        // Verify with correct checksum
        let is_valid = calculator.verify_checksum(test_data, &correct_checksum)?;
        assert!(is_valid);

        // Verify with incorrect checksum
        let wrong_checksum = "0".repeat(64);
        let is_invalid = calculator.verify_checksum(test_data, &wrong_checksum)?;
        assert!(!is_invalid);

        Ok(())
    }

    /// Test file checksum verification
    #[test]
    fn test_file_checksum_verification() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("verify_test.txt");

        let test_content = b"File verification test content";
        std::fs::write(&file_path, test_content)?;

        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let file_checksum = calculator.calculate_sha256_file(&file_path)?;
        let is_valid = calculator.verify_file_checksum(&file_path, &file_checksum)?;
        assert!(is_valid);

        // Test with wrong checksum
        let wrong_checksum = "a".repeat(64);
        let is_invalid = calculator.verify_file_checksum(&file_path, &wrong_checksum)?;
        assert!(!is_invalid);

        Ok(())
    }

    /// Test checksum calculation with metadata
    #[test]
    fn test_checksum_with_metadata() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let test_data = create_random_data(1024);
        let result = calculator.calculate_with_metadata(&test_data)?;

        assert_eq!(result.algorithm, HashAlgorithm::Sha256);
        assert_eq!(result.hash.len(), 64);
        assert_eq!(result.data_size, 1024);
        // calculation_time_ms is unsigned, so this assertion is always true and unnecessary
        assert!(!result.from_cache); // First calculation shouldn't be from cache

        Ok(())
    }

    /// Test checksum caching functionality
    #[test]
    fn test_checksum_caching() -> Result<()> {
        let config = ChecksumConfig {
            cache_checksums: true,
            ..Default::default()
        };
        let mut calculator = ChecksumCalculator::new(config)?;

        let test_data = b"cache test data";

        // First calculation
        let result1 = calculator.calculate_with_metadata(test_data)?;
        assert!(!result1.from_cache);
        assert_eq!(calculator.get_cache_size(), 1);

        // Second calculation should use cache
        let result2 = calculator.calculate_with_metadata(test_data)?;
        assert!(result2.from_cache);
        assert_eq!(result1.hash, result2.hash);

        // Clear cache and verify
        calculator.clear_cache();
        assert_eq!(calculator.get_cache_size(), 0);

        Ok(())
    }

    /// Test streaming checksum calculation
    #[test]
    fn test_streaming_checksum() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        let test_data = b"streaming checksum test data";
        let cursor = std::io::Cursor::new(test_data);

        let streaming_result = calculator.calculate_streaming(Box::new(cursor))?;
        let memory_result = calculator.calculate_sha256(test_data)?;

        assert_eq!(streaming_result, memory_result);

        Ok(())
    }

    /// Test edge cases for checksum calculation
    #[test]
    fn test_checksum_edge_cases() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        // Test empty data
        let empty_result = calculator.calculate_sha256(&[])?;
        assert_eq!(
            empty_result.to_lowercase(),
            SHA256_TEST_VECTORS[0].expected_sha256.to_lowercase()
        );

        // Test single byte
        let single_byte = calculator.calculate_sha256(&[0xFF])?;
        assert_eq!(single_byte.len(), 64);

        // Test maximum chunk size (50MB)
        let max_chunk = create_test_data(50 * 1024 * 1024, 0x42);
        let max_result = calculator.calculate_sha256(&max_chunk)?;
        assert_eq!(max_result.len(), 64);

        Ok(())
    }

    /// Test performance with large data sets
    #[test]
    fn test_checksum_performance() -> Result<()> {
        let config = ChecksumConfig {
            buffer_size: 1024 * 1024, // 1MB buffer
            ..Default::default()
        };
        let mut calculator = ChecksumCalculator::new(config)?;

        // Test 10MB data performance
        let large_data = create_random_data(10 * 1024 * 1024);

        let start_time = std::time::Instant::now();
        let result = calculator.calculate_with_metadata(&large_data)?;
        let duration = start_time.elapsed();

        assert_eq!(result.data_size, 10 * 1024 * 1024);
        assert_eq!(result.hash.len(), 64);

        // Should complete within reasonable time (adjust threshold as needed)
        assert!(
            duration.as_secs() < 5,
            "Checksum calculation took too long: {:?}",
            duration
        );

        Ok(())
    }

    /// Test error handling for invalid inputs
    #[test]
    fn test_checksum_error_handling() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        // Test verification with invalid checksum format
        let test_data = b"error test data";
        let invalid_checksum = "not_a_valid_checksum";

        let result = calculator.verify_checksum(test_data, invalid_checksum);
        assert!(result.is_err());

        match result.unwrap_err() {
            ChecksumError::InvalidInput(_) => (), // Expected
            _ => panic!("Expected InvalidInput error"),
        }

        Ok(())
    }

    /// Test parallel checksum calculation (if supported)
    #[test]
    fn test_parallel_checksum_calculation() -> Result<()> {
        let config = ChecksumConfig {
            parallel_processing: true,
            ..Default::default()
        };
        let mut calculator = ChecksumCalculator::new(config)?;

        // Create multiple data sets
        let data_sets: Vec<Vec<u8>> = (0..5)
            .map(|i| create_test_data(1024 * 1024, i as u8)) // 1MB each
            .collect();

        let mut results = Vec::new();
        for data in &data_sets {
            results.push(calculator.calculate_sha256(data)?);
        }

        // Verify all results are unique (different input patterns)
        for i in 0..results.len() {
            for j in i + 1..results.len() {
                assert_ne!(results[i], results[j]);
            }
        }

        Ok(())
    }

    /// Test checksum validation against chunk manifest requirements
    #[test]
    fn test_manifest_checksum_compatibility() -> Result<()> {
        let config = ChecksumConfig::default();
        let mut calculator = ChecksumCalculator::new(config)?;

        // Test chunk-like data structure
        let chunk_data = serde_json::json!({
            "manifest_version": "1.0.0",
            "chunk_id": "kernel_001",
            "subsystem": "kernel",
            "symbols": [],
            "entrypoints": [],
            "call_graph": []
        });

        let json_bytes = serde_json::to_vec(&chunk_data)?;
        let checksum = calculator.calculate_sha256(&json_bytes)?;

        // Verify format matches OpenAPI schema requirement (64 char hex)
        assert_eq!(checksum.len(), 64);
        assert!(checksum.chars().all(|c| c.is_ascii_hexdigit()));
        // Verify lowercase format (no uppercase letters A-F)
        assert!(checksum.chars().all(|c| !c.is_ascii_uppercase()));

        // Verify checksum can be used in ChunkMetadata
        let _metadata = format!(
            r#"{{
                "id": "kernel_001",
                "sequence": 1,
                "file": "chunks/kernel_001.json",
                "subsystem": "kernel",
                "size_bytes": {},
                "checksum_sha256": "{}",
                "symbol_count": 0
            }}"#,
            json_bytes.len(),
            checksum
        );

        Ok(())
    }
}
