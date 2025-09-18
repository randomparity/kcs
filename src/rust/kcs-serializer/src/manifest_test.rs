//! Unit tests for manifest builder functionality
//!
//! These tests verify the ManifestBuilder implementation creates valid
//! chunk manifests following the OpenAPI schema and data model constraints.

use crate::manifest::*;
use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that ManifestBuilder creates valid manifest with required fields
    #[test]
    fn test_manifest_builder_basic_creation() -> Result<()> {
        let config = ManifestBuilderConfig {
            version: "1.0.0".to_string(),
            kernel_version: Some("6.7.0".to_string()),
            kernel_path: Some("/home/user/src/linux".to_string()),
            config: Some("x86_64:defconfig".to_string()),
            ..Default::default()
        };

        let mut builder = ManifestBuilder::new(config.clone())?;

        // Add some test chunks
        let chunk_input = ChunkInput {
            file_path: std::path::PathBuf::from("chunks/kernel_001.json"),
            subsystem: "kernel".to_string(),
            symbol_count: 1250,
            entry_point_count: 45,
            file_count: 23,
        };

        let chunk_id = builder.add_chunk(chunk_input)?;
        assert_eq!(chunk_id, "kernel_001");

        let manifest = builder.build()?;

        // Verify required fields per OpenAPI schema
        assert_eq!(manifest.version, "1.0.0");
        assert!(!manifest.created.is_empty());
        assert_eq!(manifest.kernel_version, Some("6.7.0".to_string()));
        assert_eq!(manifest.total_chunks, 1);
        assert_eq!(manifest.chunks.len(), 1);

        // Verify chunk metadata
        let chunk = &manifest.chunks[0];
        assert_eq!(chunk.id, "kernel_001");
        assert_eq!(chunk.sequence, 1);
        assert_eq!(chunk.subsystem, "kernel");
        assert_eq!(chunk.symbol_count, Some(1250));

        Ok(())
    }

    /// Test manifest validation per OpenAPI schema constraints
    #[test]
    fn test_manifest_schema_validation() -> Result<()> {
        let config = ManifestBuilderConfig::default();
        let builder = ManifestBuilder::new(config)?;

        // Create manifest with missing required fields
        let manifest = ChunkManifest {
            version: "invalid_version".to_string(), // Should match ^\d+\.\d+\.\d+$
            created: "invalid_datetime".to_string(), // Should be ISO 8601
            kernel_version: None,
            kernel_path: None,
            config: None,
            total_chunks: 0, // Should be minimum 1
            total_size_bytes: 0,
            chunks: vec![],
        };

        let result = builder.validate_manifest(&manifest);
        assert!(result.is_err());

        match result.unwrap_err() {
            ManifestError::ValidationError(_) => (), // Expected
            _ => panic!("Expected ValidationError"),
        }

        Ok(())
    }

    /// Test chunk ID uniqueness validation
    #[test]
    fn test_chunk_id_uniqueness() -> Result<()> {
        let config = ManifestBuilderConfig::default();
        let mut builder = ManifestBuilder::new(config)?;

        let chunk_input = ChunkInput {
            file_path: std::path::PathBuf::from("chunks/kernel_001.json"),
            subsystem: "kernel".to_string(),
            symbol_count: 100,
            entry_point_count: 5,
            file_count: 10,
        };

        // Add first chunk successfully
        let chunk_id1 = builder.add_chunk(chunk_input.clone())?;
        assert_eq!(chunk_id1, "kernel_001");

        // Try to add duplicate chunk ID
        let result = builder.add_chunk(chunk_input);
        assert!(result.is_err());

        match result.unwrap_err() {
            ManifestError::DuplicateChunkId(id) => assert_eq!(id, "kernel_001"),
            _ => panic!("Expected DuplicateChunkId error"),
        }

        Ok(())
    }

    /// Test total size calculation and validation
    #[test]
    fn test_total_size_calculation() -> Result<()> {
        let config = ManifestBuilderConfig::default();
        let mut builder = ManifestBuilder::new(config)?;

        // Add chunks with known sizes
        let chunk1 = ChunkInput {
            file_path: std::path::PathBuf::from("chunks/kernel_001.json"),
            subsystem: "kernel".to_string(),
            symbol_count: 500,
            entry_point_count: 20,
            file_count: 15,
        };

        let chunk2 = ChunkInput {
            file_path: std::path::PathBuf::from("chunks/drivers_001.json"),
            subsystem: "drivers".to_string(),
            symbol_count: 750,
            entry_point_count: 30,
            file_count: 25,
        };

        builder.add_chunk(chunk1)?;
        builder.add_chunk(chunk2)?;

        let manifest = builder.build()?;

        // Verify total_chunks matches chunks array length
        assert_eq!(manifest.total_chunks, manifest.chunks.len());

        // Verify total_size_bytes equals sum of chunk sizes
        let calculated_size: u64 = manifest.chunks.iter().map(|c| c.size_bytes).sum();
        assert_eq!(manifest.total_size_bytes, calculated_size);

        Ok(())
    }

    /// Test chunk sequence ordering
    #[test]
    fn test_chunk_sequence_ordering() -> Result<()> {
        let config = ManifestBuilderConfig {
            sort_chunks: true,
            ..Default::default()
        };
        let mut builder = ManifestBuilder::new(config)?;

        // Add chunks in random order
        let chunks = vec![
            ("drivers_003", "drivers"),
            ("kernel_001", "kernel"),
            ("fs_002", "fs"),
            ("drivers_001", "drivers"),
            ("kernel_002", "kernel"),
        ];

        for (chunk_id, subsystem) in chunks {
            let chunk_input = ChunkInput {
                file_path: std::path::PathBuf::from(format!("chunks/{}.json", chunk_id)),
                subsystem: subsystem.to_string(),
                symbol_count: 100,
                entry_point_count: 5,
                file_count: 10,
            };
            builder.add_chunk(chunk_input)?;
        }

        let manifest = builder.build()?;

        // Verify chunks are sorted by subsystem, then by sequence
        let mut prev_subsystem = "";
        let mut prev_sequence = 0;

        for chunk in &manifest.chunks {
            if chunk.subsystem != prev_subsystem {
                prev_subsystem = &chunk.subsystem;
                prev_sequence = 0;
            }
            assert!(chunk.sequence > prev_sequence);
            prev_sequence = chunk.sequence;
        }

        Ok(())
    }

    /// Test manifest serialization to JSON
    #[test]
    fn test_manifest_json_serialization() -> Result<()> {
        let config = ManifestBuilderConfig::default();
        let mut builder = ManifestBuilder::new(config)?;

        let chunk_input = ChunkInput {
            file_path: std::path::PathBuf::from("chunks/kernel_001.json"),
            subsystem: "kernel".to_string(),
            symbol_count: 250,
            entry_point_count: 12,
            file_count: 8,
        };

        builder.add_chunk(chunk_input)?;
        let manifest = builder.build()?;

        // Serialize to JSON
        let json_str = serde_json::to_string_pretty(&manifest)?;
        assert!(json_str.contains("\"version\""));
        assert!(json_str.contains("\"total_chunks\""));
        assert!(json_str.contains("\"chunks\""));

        // Deserialize back
        let deserialized: ChunkManifest = serde_json::from_str(&json_str)?;
        assert_eq!(manifest, deserialized);

        Ok(())
    }

    /// Test manifest file I/O operations
    #[test]
    fn test_manifest_file_operations() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let manifest_path = temp_dir.path().join("manifest.json");

        let config = ManifestBuilderConfig {
            output_directory: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };
        let mut builder = ManifestBuilder::new(config)?;

        let chunk_input = ChunkInput {
            file_path: std::path::PathBuf::from("chunks/kernel_001.json"),
            subsystem: "kernel".to_string(),
            symbol_count: 300,
            entry_point_count: 15,
            file_count: 12,
        };

        builder.add_chunk(chunk_input)?;

        // Write manifest to file
        let manifest = builder.build_and_write(&manifest_path)?;

        // Verify file exists and is readable
        assert!(manifest_path.exists());
        let file_size = std::fs::metadata(&manifest_path)?.len();
        assert!(file_size > 0);

        // Read and verify content
        let content = std::fs::read_to_string(&manifest_path)?;
        let parsed: ChunkManifest = serde_json::from_str(&content)?;
        assert_eq!(parsed, manifest);

        Ok(())
    }

    /// Test chunk metadata updates
    #[test]
    fn test_chunk_metadata_updates() -> Result<()> {
        let config = ManifestBuilderConfig::default();
        let mut builder = ManifestBuilder::new(config)?;

        let chunk_input = ChunkInput {
            file_path: std::path::PathBuf::from("chunks/kernel_001.json"),
            subsystem: "kernel".to_string(),
            symbol_count: 100,
            entry_point_count: 5,
            file_count: 10,
        };

        let chunk_id = builder.add_chunk(chunk_input)?;

        // Update chunk metadata
        let updated_metadata = ChunkMetadata {
            id: chunk_id.clone(),
            sequence: 1,
            file: "chunks/kernel_001.json".to_string(),
            subsystem: "kernel".to_string(),
            size_bytes: 2048,
            checksum_sha256: "abcd1234".repeat(8), // 64 char hex string
            symbol_count: Some(150),               // Updated count
            entry_point_count: Some(8),            // Updated count
            file_count: Some(12),                  // Updated count
        };

        builder.update_chunk_metadata(&chunk_id, updated_metadata)?;

        let manifest = builder.build()?;
        let chunk = &manifest.chunks[0];
        assert_eq!(chunk.symbol_count, Some(150));
        assert_eq!(chunk.entry_point_count, Some(8));

        Ok(())
    }

    /// Test chunk removal functionality
    #[test]
    fn test_chunk_removal() -> Result<()> {
        let config = ManifestBuilderConfig::default();
        let mut builder = ManifestBuilder::new(config)?;

        // Add multiple chunks
        for i in 1..=3 {
            let chunk_input = ChunkInput {
                file_path: std::path::PathBuf::from(format!("chunks/kernel_{:03}.json", i)),
                subsystem: "kernel".to_string(),
                symbol_count: 100 * i,
                entry_point_count: 5 * i,
                file_count: 10 * i,
            };
            builder.add_chunk(chunk_input)?;
        }

        assert_eq!(builder.get_chunk_count(), 3);

        // Remove middle chunk
        builder.remove_chunk("kernel_002")?;
        assert_eq!(builder.get_chunk_count(), 2);

        let manifest = builder.build()?;
        assert_eq!(manifest.total_chunks, 2);

        // Verify remaining chunks
        let chunk_ids: Vec<&str> = manifest.chunks.iter().map(|c| c.id.as_str()).collect();
        assert!(chunk_ids.contains(&"kernel_001"));
        assert!(!chunk_ids.contains(&"kernel_002"));
        assert!(chunk_ids.contains(&"kernel_003"));

        Ok(())
    }

    /// Test large manifest with many chunks
    #[test]
    fn test_large_manifest_performance() -> Result<()> {
        let config = ManifestBuilderConfig::default();
        let mut builder = ManifestBuilder::new(config)?;

        let start_time = std::time::Instant::now();

        // Add 100 chunks to test performance
        for i in 1..=100 {
            let subsystem = match i % 4 {
                0 => "kernel",
                1 => "drivers",
                2 => "fs",
                _ => "net",
            };

            let chunk_input = ChunkInput {
                file_path: std::path::PathBuf::from(format!("chunks/{}_{:03}.json", subsystem, i)),
                subsystem: subsystem.to_string(),
                symbol_count: 50 + i,
                entry_point_count: 2 + (i % 10),
                file_count: 5 + (i % 20),
            };

            builder.add_chunk(chunk_input)?;
        }

        let manifest = builder.build()?;
        let elapsed = start_time.elapsed();

        assert_eq!(manifest.total_chunks, 100);
        assert!(
            elapsed.as_millis() < 1000,
            "Manifest build took too long: {:?}",
            elapsed
        );

        // Verify subsystem grouping if sorting is enabled
        let subsystems: Vec<&str> = manifest
            .chunks
            .iter()
            .map(|c| c.subsystem.as_str())
            .collect();
        let mut unique_subsystems = subsystems.clone();
        unique_subsystems.sort();
        unique_subsystems.dedup();
        assert_eq!(unique_subsystems, vec!["drivers", "fs", "kernel", "net"]);

        Ok(())
    }

    /// Test checksum validation for chunk metadata
    #[test]
    fn test_checksum_validation() -> Result<()> {
        let config = ManifestBuilderConfig {
            validate_schema: true,
            ..Default::default()
        };
        let mut builder = ManifestBuilder::new(config)?;

        // Test invalid checksum length
        let invalid_metadata = ChunkMetadata {
            id: "test_chunk".to_string(),
            sequence: 1,
            file: "chunks/test.json".to_string(),
            subsystem: "test".to_string(),
            size_bytes: 1024,
            checksum_sha256: "invalid_checksum".to_string(), // Not 64 chars
            symbol_count: Some(10),
            entry_point_count: Some(2),
            file_count: Some(5),
        };

        let result = builder.update_chunk_metadata("test_chunk", invalid_metadata);
        assert!(result.is_err());

        // Test valid checksum
        let valid_metadata = ChunkMetadata {
            id: "test_chunk".to_string(),
            sequence: 1,
            file: "chunks/test.json".to_string(),
            subsystem: "test".to_string(),
            size_bytes: 1024,
            checksum_sha256: "a".repeat(64), // Valid 64 char hex string
            symbol_count: Some(10),
            entry_point_count: Some(2),
            file_count: Some(5),
        };

        // This should fail because the chunk doesn't exist
        let result = builder.update_chunk_metadata("test_chunk", valid_metadata);
        assert!(result.is_err());

        match result.unwrap_err() {
            ManifestError::InvalidChunkData { chunk_id, reason } => {
                assert_eq!(chunk_id, "test_chunk");
                assert!(reason.contains("not found"));
            }
            other => panic!("Expected InvalidChunkData error, got: {:?}", other),
        }

        Ok(())
    }
}
