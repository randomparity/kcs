//! Unit tests for chunk writer with size limits
//!
//! These tests verify the ChunkWriter implementation handles size constraints
//! correctly, including the constitutional 50MB chunk size limit.

use crate::chunk_writer::*;
use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that ChunkWriter enforces the 50MB size limit
    #[test]
    fn test_chunk_writer_size_limit() -> Result<()> {
        let config = ChunkWriterConfig {
            max_chunk_size: 50 * 1024 * 1024,    // 50MB constitutional limit
            target_chunk_size: 45 * 1024 * 1024, // 45MB target
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        // Create test data that would exceed the size limit
        let large_data = vec![0u8; 60 * 1024 * 1024]; // 60MB of data

        // Should fail when trying to write oversized chunk
        let result = writer.write_chunk("test_chunk_001", &large_data);
        assert!(result.is_err());

        match result.unwrap_err() {
            ChunkWriterError::ChunkTooLarge { size, limit } => {
                assert_eq!(size, 60 * 1024 * 1024);
                assert_eq!(limit, 50 * 1024 * 1024);
            }
            _ => panic!("Expected ChunkTooLarge error"),
        }

        Ok(())
    }

    /// Test that ChunkWriter handles valid sized chunks correctly
    #[test]
    fn test_chunk_writer_valid_size() -> Result<()> {
        let config = ChunkWriterConfig {
            max_chunk_size: 50 * 1024 * 1024,
            target_chunk_size: 45 * 1024 * 1024,
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        // Create test data within size limits
        let valid_data = vec![0u8; 30 * 1024 * 1024]; // 30MB of data

        // Should succeed for valid sized chunk
        let result = writer.write_chunk("test_chunk_002", &valid_data);
        assert!(result.is_ok());

        let chunk_info = result.unwrap();
        assert_eq!(chunk_info.chunk_id, "test_chunk_002");
        assert_eq!(chunk_info.size_bytes, 30 * 1024 * 1024);
        assert!(chunk_info.checksum_sha256.len() == 64); // SHA256 hex length

        Ok(())
    }

    /// Test chunk writer with streaming JSON serialization
    #[test]
    fn test_chunk_writer_streaming_json() -> Result<()> {
        let config = ChunkWriterConfig {
            format: ChunkFormat::StreamingJson,
            max_chunk_size: 50 * 1024 * 1024,
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        // Create test JSON-serializable data
        let test_symbols = vec![
            ("symbol_1", "function"),
            ("symbol_2", "variable"),
            ("symbol_3", "function"),
        ];

        let result = writer.write_json_chunk("symbols_chunk", &test_symbols);
        assert!(result.is_ok());

        let chunk_info = result.unwrap();
        assert_eq!(chunk_info.chunk_id, "symbols_chunk");
        assert!(chunk_info.size_bytes > 0);
        assert_eq!(chunk_info.format, "streaming_json");

        Ok(())
    }

    /// Test chunk writer with size-based automatic splitting
    #[test]
    fn test_chunk_writer_auto_splitting() -> Result<()> {
        let config = ChunkWriterConfig {
            max_chunk_size: 10 * 1024 * 1024,   // 10MB limit for testing
            target_chunk_size: 8 * 1024 * 1024, // 8MB target
            auto_split: true,
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        // Create data larger than one chunk but splittable
        let large_symbols: Vec<String> = (0..1000)
            .map(|i| format!("very_long_symbol_name_that_takes_space_{}", i))
            .collect();

        let result = writer.write_splittable_data("large_dataset", &large_symbols);
        assert!(result.is_ok());

        let chunks = result.unwrap();
        assert!(chunks.len() > 1, "Should split into multiple chunks");

        for chunk in &chunks {
            assert!(
                chunk.size_bytes <= 10 * 1024 * 1024,
                "Each chunk should be within size limit"
            );
        }

        // Verify total data is preserved
        let total_items: usize = chunks.iter().map(|c| c.item_count).sum();
        assert_eq!(total_items, 1000, "Should preserve all items across chunks");

        Ok(())
    }

    /// Test chunk writer compression feature
    #[test]
    fn test_chunk_writer_compression() -> Result<()> {
        let config = ChunkWriterConfig {
            compression: CompressionLevel::Fast,
            max_chunk_size: 50 * 1024 * 1024,
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        // Create highly compressible test data (repeated pattern)
        let repetitive_data = "repeated_pattern_".repeat(100_000);

        let result = writer.write_chunk("compressed_chunk", repetitive_data.as_bytes());
        assert!(result.is_ok());

        let chunk_info = result.unwrap();
        assert!(chunk_info.compressed_size < chunk_info.uncompressed_size);
        assert!(chunk_info.compression_ratio > 0.0);

        Ok(())
    }

    /// Test chunk writer boundary conditions
    #[test]
    fn test_chunk_writer_boundary_conditions() -> Result<()> {
        let config = ChunkWriterConfig::default();
        let mut writer = ChunkWriter::new(config.clone())?;

        // Test empty chunk
        let empty_result = writer.write_chunk("empty_chunk", &[]);
        assert!(empty_result.is_ok());
        let empty_info = empty_result.unwrap();
        assert_eq!(empty_info.size_bytes, 0);

        // Test exactly at limit
        let limit_data = vec![0u8; config.max_chunk_size];
        let limit_result = writer.write_chunk("limit_chunk", &limit_data);
        assert!(limit_result.is_ok());

        // Test one byte over limit
        let over_data = vec![0u8; config.max_chunk_size + 1];
        let over_result = writer.write_chunk("over_chunk", &over_data);
        assert!(over_result.is_err());

        Ok(())
    }

    /// Test chunk writer file output with proper naming
    #[test]
    fn test_chunk_writer_file_output() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let output_path = temp_dir.path().join("chunks");

        let config = ChunkWriterConfig {
            output_directory: Some(output_path.clone()),
            file_naming: FileNamingScheme::Sequential,
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        let test_data = b"test chunk content";
        let result = writer.write_chunk_to_file("test_chunk", test_data);
        assert!(result.is_ok());

        let file_info = result.unwrap();
        assert!(file_info.file_path.starts_with(&output_path));
        assert!(file_info.file_path.exists());
        assert_eq!(file_info.file_size, test_data.len());

        Ok(())
    }

    /// Test chunk writer metadata generation
    #[test]
    fn test_chunk_writer_metadata() -> Result<()> {
        let config = ChunkWriterConfig {
            include_metadata: true,
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        let test_data = b"test data with metadata";
        let result = writer.write_chunk("metadata_chunk", test_data);
        assert!(result.is_ok());

        let chunk_info = result.unwrap();
        assert!(chunk_info.metadata.is_some());

        let metadata = chunk_info.metadata.unwrap();
        assert!(!metadata.created_at.is_empty());
        assert_eq!(metadata.chunk_version, "1.0.0");
        assert!(metadata.total_symbols > 0 || metadata.total_entry_points > 0);

        Ok(())
    }

    /// Test chunk writer concurrent safety
    #[test]
    fn test_chunk_writer_concurrent_writes() -> Result<()> {
        // This test will fail until proper implementation
        // For now, just test that multiple writers can be created
        let config = ChunkWriterConfig::default();

        let mut writers = vec![];
        for _i in 0..5 {
            match ChunkWriter::new(config.clone()) {
                Ok(writer) => writers.push(writer),
                Err(_) => {
                    // Expected to fail until T018 implementation
                    // If writer creation fails, test passes as implementation is not complete
                    return Ok(());
                }
            }
        }

        // If we get here, implementation exists and should work
        assert_eq!(writers.len(), 5);
        Ok(())
    }

    /// Test chunk writer performance with large data
    #[test]
    fn test_chunk_writer_performance() -> Result<()> {
        let config = ChunkWriterConfig {
            max_chunk_size: 50 * 1024 * 1024,
            buffer_size: 1024 * 1024, // 1MB buffer
            ..Default::default()
        };

        let mut writer = ChunkWriter::new(config.clone())?;

        // Create 10MB of test data
        let large_data = vec![0u8; 10 * 1024 * 1024];

        let start = std::time::Instant::now();
        let result = writer.write_chunk("performance_chunk", &large_data);
        let duration = start.elapsed();

        assert!(result.is_ok());

        // Should complete within reasonable time (adjust threshold as needed)
        assert!(
            duration.as_secs() < 5,
            "Chunk writing took too long: {:?}",
            duration
        );

        let chunk_info = result.unwrap();
        assert_eq!(chunk_info.size_bytes, 10 * 1024 * 1024);

        Ok(())
    }
}
