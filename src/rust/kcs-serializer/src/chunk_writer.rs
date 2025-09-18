//! ChunkWriter implementation for managing large data chunks with size limits
//!
//! This module provides streaming JSON output with configurable chunk sizes,
//! including the constitutional 50MB chunk size limit for memory-efficient processing.

use anyhow::Result;
use flate2::{write::GzEncoder, Compression};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Configuration for ChunkWriter behavior
#[derive(Debug, Clone)]
pub struct ChunkWriterConfig {
    /// Maximum chunk size in bytes (constitutional limit: 50MB)
    pub max_chunk_size: usize,
    /// Target chunk size for auto-splitting (typically 90% of max)
    pub target_chunk_size: usize,
    /// Whether to automatically split large datasets
    pub auto_split: bool,
    /// Output format for chunks
    pub format: ChunkFormat,
    /// Compression level for output
    pub compression: CompressionLevel,
    /// Directory for chunk file output (None for in-memory only)
    pub output_directory: Option<PathBuf>,
    /// File naming scheme for chunk files
    pub file_naming: FileNamingScheme,
    /// Whether to include metadata in chunk info
    pub include_metadata: bool,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
}

impl Default for ChunkWriterConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 50 * 1024 * 1024,    // 50MB constitutional limit
            target_chunk_size: 45 * 1024 * 1024, // 45MB target (90% of max)
            auto_split: false,
            format: ChunkFormat::Json,
            compression: CompressionLevel::None,
            output_directory: None,
            file_naming: FileNamingScheme::ChunkId,
            include_metadata: false,
            buffer_size: 64 * 1024, // 64KB default buffer
        }
    }
}

/// Supported chunk output formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkFormat {
    /// Standard JSON format
    Json,
    /// Streaming JSON format for large arrays
    StreamingJson,
    /// Binary format (for future use)
    Binary,
}

/// Compression levels for chunk output
#[derive(Debug, Clone, Copy)]
pub enum CompressionLevel {
    /// No compression
    None,
    /// Fast compression (level 1)
    Fast,
    /// Balanced compression (level 6)
    Balanced,
    /// Best compression (level 9)
    Best,
}

impl CompressionLevel {
    fn to_flate2_level(self) -> Compression {
        match self {
            CompressionLevel::None => Compression::none(),
            CompressionLevel::Fast => Compression::fast(),
            CompressionLevel::Balanced => Compression::default(),
            CompressionLevel::Best => Compression::best(),
        }
    }
}

/// File naming schemes for chunk output
#[derive(Debug, Clone, Copy)]
pub enum FileNamingScheme {
    /// Use chunk ID as filename
    ChunkId,
    /// Use sequential numbers
    Sequential,
    /// Use timestamp-based names
    Timestamped,
}

/// Errors that can occur during chunk writing
#[derive(Debug)]
pub enum ChunkWriterError {
    /// Chunk size exceeds the configured limit
    ChunkTooLarge {
        /// Actual size of the chunk in bytes
        size: usize,
        /// Maximum allowed size limit in bytes
        limit: usize,
    },
    /// I/O error occurred
    IoError(std::io::Error),
    /// Serialization error
    SerializationError(String),
    /// Compression error
    CompressionError(String),
}

impl std::fmt::Display for ChunkWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChunkTooLarge { size, limit } => {
                write!(f, "Chunk size {} exceeds limit {}", size, limit)
            }
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
            Self::CompressionError(e) => write!(f, "Compression error: {}", e),
        }
    }
}

impl std::error::Error for ChunkWriterError {}

impl From<std::io::Error> for ChunkWriterError {
    fn from(error: std::io::Error) -> Self {
        ChunkWriterError::IoError(error)
    }
}

impl From<serde_json::Error> for ChunkWriterError {
    fn from(error: serde_json::Error) -> Self {
        ChunkWriterError::SerializationError(error.to_string())
    }
}

/// Main chunk writer implementation
#[derive(Debug)]
pub struct ChunkWriter {
    config: ChunkWriterConfig,
    chunks_written: usize,
}

/// Information about a written chunk
#[derive(Debug)]
pub struct ChunkInfo {
    /// Unique identifier for the chunk
    pub chunk_id: String,
    /// Size of the chunk in bytes
    pub size_bytes: usize,
    /// SHA256 checksum of the chunk content
    pub checksum_sha256: String,
    /// Format used for the chunk
    pub format: String,
    /// Compressed size (if compression is used)
    pub compressed_size: usize,
    /// Uncompressed size
    pub uncompressed_size: usize,
    /// Compression ratio (compressed/uncompressed)
    pub compression_ratio: f64,
    /// Optional metadata about the chunk
    pub metadata: Option<ChunkWriterMetadata>,
    /// Number of items in the chunk
    pub item_count: usize,
}

/// Metadata associated with a chunk (internal to ChunkWriter)
#[derive(Debug)]
pub struct ChunkWriterMetadata {
    /// Timestamp when chunk was created
    pub created_at: String,
    /// Version of the chunk format
    pub chunk_version: String,
    /// Total symbols in the chunk
    pub total_symbols: usize,
    /// Total entry points in the chunk
    pub total_entry_points: usize,
}

/// File information for chunks written to disk
#[derive(Debug)]
pub struct FileInfo {
    /// Path to the written file
    pub file_path: PathBuf,
    /// Size of the file in bytes
    pub file_size: usize,
}

impl ChunkWriter {
    /// Create a new ChunkWriter with the given configuration
    pub fn new(config: ChunkWriterConfig) -> Result<Self, ChunkWriterError> {
        // Create output directory if specified
        if let Some(ref output_dir) = config.output_directory {
            fs::create_dir_all(output_dir)?;
        }

        Ok(Self {
            config,
            chunks_written: 0,
        })
    }

    /// Write a raw chunk of data
    pub fn write_chunk(
        &mut self,
        chunk_id: &str,
        data: &[u8],
    ) -> Result<ChunkInfo, ChunkWriterError> {
        // Check size limits
        if data.len() > self.config.max_chunk_size {
            return Err(ChunkWriterError::ChunkTooLarge {
                size: data.len(),
                limit: self.config.max_chunk_size,
            });
        }

        let uncompressed_size = data.len();
        let (processed_data, compressed_size) = self.apply_compression(data)?;

        // Calculate checksum
        let checksum = self.calculate_checksum(&processed_data);

        // Generate metadata if requested
        let metadata = if self.config.include_metadata {
            Some(ChunkWriterMetadata {
                created_at: chrono::Utc::now().to_rfc3339(),
                chunk_version: "1.0.0".to_string(),
                total_symbols: self.estimate_symbols(&processed_data),
                total_entry_points: self.estimate_entry_points(&processed_data),
            })
        } else {
            None
        };

        let compression_ratio = if uncompressed_size > 0 {
            compressed_size as f64 / uncompressed_size as f64
        } else {
            1.0
        };

        self.chunks_written += 1;

        Ok(ChunkInfo {
            chunk_id: chunk_id.to_string(),
            size_bytes: processed_data.len(),
            checksum_sha256: checksum,
            format: self.format_string(),
            compressed_size,
            uncompressed_size,
            compression_ratio,
            metadata,
            item_count: 1, // Single chunk = 1 item
        })
    }

    /// Write a JSON-serializable chunk
    pub fn write_json_chunk<T: Serialize + ?Sized>(
        &mut self,
        chunk_id: &str,
        data: &T,
    ) -> Result<ChunkInfo, ChunkWriterError> {
        let json_data = match self.config.format {
            ChunkFormat::Json => serde_json::to_vec(data)?,
            ChunkFormat::StreamingJson => serde_json::to_vec(data)?, // For now, same as JSON
            ChunkFormat::Binary => {
                return Err(ChunkWriterError::SerializationError(
                    "Binary format not supported for JSON data".to_string(),
                ));
            }
        };

        self.write_chunk(chunk_id, &json_data)
    }

    /// Write splittable data that can be automatically divided into chunks
    pub fn write_splittable_data<T: Serialize>(
        &mut self,
        dataset_id: &str,
        data: &[T],
    ) -> Result<Vec<ChunkInfo>, ChunkWriterError> {
        if !self.config.auto_split {
            // Write as single chunk
            return Ok(vec![self.write_json_chunk(dataset_id, data)?]);
        }

        let mut chunks = Vec::new();
        let mut start_idx = 0;

        while start_idx < data.len() {
            // Find optimal chunk size
            let end_idx = self.find_optimal_split_point(data, start_idx)?;
            let chunk_data = &data[start_idx..end_idx];

            let chunk_id = format!("{}_{:03}", dataset_id, chunks.len() + 1);
            let mut chunk_info = self.write_json_chunk(&chunk_id, chunk_data)?;
            chunk_info.item_count = chunk_data.len();

            chunks.push(chunk_info);
            start_idx = end_idx;
        }

        Ok(chunks)
    }

    /// Write chunk to file and return file information
    pub fn write_chunk_to_file(
        &mut self,
        chunk_id: &str,
        data: &[u8],
    ) -> Result<FileInfo, ChunkWriterError> {
        let output_dir = self.config.output_directory.as_ref().ok_or_else(|| {
            ChunkWriterError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No output directory configured",
            ))
        })?;

        let filename = self.generate_filename(chunk_id);
        let file_path = output_dir.join(&filename);

        // Write the chunk data
        let _chunk_info = self.write_chunk(chunk_id, data)?;

        // Apply compression if configured and write to file
        let (final_data, _) = self.apply_compression(data)?;

        let mut file = BufWriter::new(File::create(&file_path)?);
        file.write_all(&final_data)?;
        file.flush()?;

        Ok(FileInfo {
            file_path,
            file_size: final_data.len(),
        })
    }

    // Helper methods

    fn apply_compression(&self, data: &[u8]) -> Result<(Vec<u8>, usize), ChunkWriterError> {
        match self.config.compression {
            CompressionLevel::None => Ok((data.to_vec(), data.len())),
            compression_level => {
                let mut encoder = GzEncoder::new(Vec::new(), compression_level.to_flate2_level());
                encoder
                    .write_all(data)
                    .map_err(|e| ChunkWriterError::CompressionError(e.to_string()))?;
                let compressed_data = encoder
                    .finish()
                    .map_err(|e| ChunkWriterError::CompressionError(e.to_string()))?;
                Ok((compressed_data.clone(), compressed_data.len()))
            }
        }
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    fn format_string(&self) -> String {
        match self.config.format {
            ChunkFormat::Json => "json".to_string(),
            ChunkFormat::StreamingJson => "streaming_json".to_string(),
            ChunkFormat::Binary => "binary".to_string(),
        }
    }

    fn estimate_symbols(&self, _data: &[u8]) -> usize {
        // Simple heuristic: estimate based on data size
        // In real implementation, this would parse JSON to count symbols
        (_data.len() / 100).max(1)
    }

    fn estimate_entry_points(&self, _data: &[u8]) -> usize {
        // Simple heuristic: estimate based on data size
        // In real implementation, this would parse JSON to count entry points
        (_data.len() / 1000).max(1)
    }

    fn find_optimal_split_point<T: Serialize>(
        &self,
        data: &[T],
        start_idx: usize,
    ) -> Result<usize, ChunkWriterError> {
        let remaining = data.len() - start_idx;
        if remaining == 0 {
            return Ok(start_idx);
        }

        // Binary search for optimal split point
        let mut low = start_idx + 1;
        let mut high = (start_idx + remaining).min(start_idx + 1000); // Reasonable upper bound
        let mut best_split = high;

        while low <= high && low < data.len() {
            let mid = (low + high) / 2;
            let chunk_data = &data[start_idx..mid];

            // Estimate serialized size
            let estimated_size = self.estimate_serialized_size(chunk_data)?;

            if estimated_size <= self.config.target_chunk_size {
                best_split = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        Ok(best_split.min(data.len()))
    }

    fn estimate_serialized_size<T: Serialize>(
        &self,
        data: &[T],
    ) -> Result<usize, ChunkWriterError> {
        // For performance, we estimate rather than fully serialize
        // This is a heuristic based on the assumption that each item averages ~1KB when serialized
        Ok(data.len() * 1024)
    }

    fn generate_filename(&self, chunk_id: &str) -> String {
        match self.config.file_naming {
            FileNamingScheme::ChunkId => format!("{}.json", chunk_id),
            FileNamingScheme::Sequential => format!("chunk_{:06}.json", self.chunks_written),
            FileNamingScheme::Timestamped => {
                let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f");
                format!("chunk_{}_{}.json", timestamp, chunk_id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_writer_creation() -> Result<(), ChunkWriterError> {
        let config = ChunkWriterConfig::default();
        let _writer = ChunkWriter::new(config)?;
        Ok(())
    }

    #[test]
    fn test_compression_none() -> Result<(), ChunkWriterError> {
        let config = ChunkWriterConfig {
            compression: CompressionLevel::None,
            ..Default::default()
        };
        let writer = ChunkWriter::new(config)?;

        let data = b"test data";
        let (compressed, size) = writer.apply_compression(data)?;

        assert_eq!(compressed, data);
        assert_eq!(size, data.len());
        Ok(())
    }

    #[test]
    fn test_checksum_calculation() -> Result<(), ChunkWriterError> {
        let config = ChunkWriterConfig::default();
        let writer = ChunkWriter::new(config)?;

        let data = b"test data";
        let checksum = writer.calculate_checksum(data);

        // Should be a valid SHA256 hex string
        assert_eq!(checksum.len(), 64);
        assert!(checksum.chars().all(|c| c.is_ascii_hexdigit()));
        Ok(())
    }

    #[test]
    fn test_format_string() -> Result<(), ChunkWriterError> {
        let configs = [
            (ChunkFormat::Json, "json"),
            (ChunkFormat::StreamingJson, "streaming_json"),
            (ChunkFormat::Binary, "binary"),
        ];

        for (format, expected) in configs {
            let config = ChunkWriterConfig {
                format,
                ..Default::default()
            };
            let writer = ChunkWriter::new(config)?;
            assert_eq!(writer.format_string(), expected);
        }
        Ok(())
    }
}
