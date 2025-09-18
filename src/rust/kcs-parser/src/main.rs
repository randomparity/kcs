//! KCS Parser CLI
//!
//! Command-line interface for parsing kernel source code.

use anyhow::{Context, Result};
use clap::{Parser as ClapParser, Subcommand};
use kcs_parser::{ExtendedParserConfig, ParsedFile, Parser};
use kcs_serializer::{
    ChunkInput, ChunkWriter, ChunkWriterConfig, ManifestBuilder, ManifestBuilderConfig,
};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(ClapParser)]
#[command(name = "kcs-parser")]
#[command(about = "Parse Linux kernel source code")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format
    #[arg(long, default_value = "json")]
    format: OutputFormat,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Number of parallel workers
    #[arg(long, default_value = "1")]
    workers: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse a kernel repository
    Parse {
        /// Path to kernel repository
        #[arg(short, long)]
        repo: PathBuf,

        /// Configuration to parse (e.g., x86_64:defconfig)
        #[arg(short, long, default_value = "x86_64:defconfig")]
        config: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Use clang for semantic analysis
        #[arg(long, default_value = "true")]
        clang: bool,

        /// Path to compile_commands.json
        #[arg(long)]
        compile_commands: Option<PathBuf>,

        /// Additional include paths
        #[arg(long)]
        include: Vec<PathBuf>,

        /// Preprocessor defines (KEY=VALUE)
        #[arg(long)]
        define: Vec<String>,

        /// Include function call graphs in output
        #[arg(long)]
        include_calls: bool,

        /// Split output into chunks of specified size (e.g., "50MB", "100KB")
        #[arg(long)]
        chunk_size: Option<String>,

        /// Output directory for chunk files (required when using --chunk-size)
        #[arg(long)]
        chunk_output_dir: Option<PathBuf>,
    },

    /// Parse a single file
    File {
        /// Path to source file
        file: PathBuf,

        /// Configuration context
        #[arg(short, long, default_value = "x86_64:defconfig")]
        config: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Include function call graphs in output
        #[arg(long)]
        include_calls: bool,
    },

    /// List available configurations
    ListConfigs {
        /// Path to kernel repository
        repo: PathBuf,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum OutputFormat {
    Json,
    JsonPretty,
    Ndjson,
    Csv,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new(format!("kcs_parser={}", log_level))
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    match cli.command {
        Commands::Parse {
            repo,
            config,
            output,
            clang,
            compile_commands,
            include,
            define,
            include_calls,
            chunk_size,
            chunk_output_dir,
        } => {
            parse_repository(
                repo,
                config,
                output,
                clang,
                compile_commands,
                include,
                define,
                include_calls,
                chunk_size,
                chunk_output_dir,
                cli.format,
            )
            .await
        }
        Commands::File {
            file,
            config,
            output,
            include_calls,
        } => parse_single_file(file, config, output, include_calls, cli.format).await,
        Commands::ListConfigs { repo } => list_configs(repo).await,
    }
}

#[allow(clippy::too_many_arguments)]
async fn parse_repository(
    repo: PathBuf,
    config: String,
    output: Option<PathBuf>,
    use_clang: bool,
    compile_commands: Option<PathBuf>,
    include_paths: Vec<PathBuf>,
    defines: Vec<String>,
    include_calls: bool,
    chunk_size: Option<String>,
    chunk_output_dir: Option<PathBuf>,
    format: OutputFormat,
) -> Result<()> {
    tracing::info!("Parsing repository: {}", repo.display());
    tracing::info!("Configuration: {}", config);

    // Validate chunk configuration if provided
    if chunk_size.is_some() && chunk_output_dir.is_none() {
        anyhow::bail!("--chunk-output-dir is required when using --chunk-size");
    }

    // Parse config string (e.g., "x86_64:defconfig")
    let (arch, config_name) = parse_config_string(&config)?;

    // Parse defines
    let defines_map = parse_defines(&defines)?;

    // Create parser configuration
    let parser_config = ExtendedParserConfig {
        use_clang,
        compile_commands_path: compile_commands,
        include_paths,
        defines: defines_map,
        arch: arch.to_string(),
        config_name: config_name.to_string(),
        include_call_graphs: include_calls,
    };

    // Create parser
    let mut parser = Parser::new(parser_config)?;

    // Parse the repository
    let parsed_files = parser
        .parse_directory(&repo)
        .context("Failed to parse repository")?;

    tracing::info!("Parsed {} files", parsed_files.len());

    // Output results with optional chunking
    if let Some(chunk_size_str) = chunk_size {
        let chunk_dir = chunk_output_dir.unwrap(); // Already validated above
        output_chunked_results(&parsed_files, chunk_size_str, chunk_dir, format).await?;
    } else {
        output_results(&parsed_files, output, format)?;
    }

    Ok(())
}

async fn parse_single_file(
    file: PathBuf,
    config: String,
    output: Option<PathBuf>,
    include_calls: bool,
    format: OutputFormat,
) -> Result<()> {
    tracing::debug!("Parsing file: {}", file.display());

    let (arch, config_name) = parse_config_string(&config)?;

    let parser_config = ExtendedParserConfig {
        arch: arch.to_string(),
        config_name: config_name.to_string(),
        include_call_graphs: include_calls,
        ..Default::default()
    };

    let mut parser = Parser::new(parser_config)?;
    let parsed_file = parser.parse_file(&file)?;

    output_results(&[parsed_file], output, format)?;

    Ok(())
}

async fn list_configs(repo: PathBuf) -> Result<()> {
    tracing::info!("Listing configurations for: {}", repo.display());

    // This would typically read arch/*/configs/* and Kconfig files
    // For now, return common configurations
    let configs = vec![
        "x86_64:defconfig",
        "x86_64:allmodconfig",
        "arm64:defconfig",
        "arm64:allmodconfig",
    ];

    for config in configs {
        println!("{}", config);
    }

    Ok(())
}

fn parse_config_string(config: &str) -> Result<(&str, &str)> {
    let parts: Vec<&str> = config.split(':').collect();
    if parts.len() != 2 {
        anyhow::bail!("Config must be in format 'arch:config' (e.g., x86_64:defconfig)");
    }
    Ok((parts[0], parts[1]))
}

fn parse_defines(defines: &[String]) -> Result<HashMap<String, String>> {
    let mut defines_map = HashMap::new();

    for define in defines {
        let parts: Vec<&str> = define.splitn(2, '=').collect();
        match parts.len() {
            1 => {
                defines_map.insert(parts[0].to_string(), "1".to_string());
            }
            2 => {
                defines_map.insert(parts[0].to_string(), parts[1].to_string());
            }
            _ => anyhow::bail!("Invalid define format: {}", define),
        }
    }

    Ok(defines_map)
}

fn output_results(
    parsed_files: &[ParsedFile],
    output: Option<PathBuf>,
    format: OutputFormat,
) -> Result<()> {
    let output_string = match format {
        OutputFormat::Json => serde_json::to_string(parsed_files)?,
        OutputFormat::JsonPretty => serde_json::to_string_pretty(parsed_files)?,
        OutputFormat::Ndjson => {
            // Output each file as a separate JSON line
            parsed_files
                .iter()
                .map(serde_json::to_string)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("JSON serialization error: {}", e))?
                .join("\n")
        }
        OutputFormat::Csv => {
            // Simple CSV output with symbol information
            let mut csv = String::new();
            csv.push_str("file,symbol,kind,start_line,end_line\n");

            for file in parsed_files {
                for symbol in &file.symbols {
                    let kind_str = format!("{:?}", symbol.kind);
                    csv.push_str(&format!(
                        "{},{},{},{},{}\n",
                        file.path.display(),
                        symbol.name,
                        kind_str,
                        symbol.start_line,
                        symbol.end_line
                    ));
                }
            }
            csv
        }
    };

    match output {
        Some(output_path) => {
            std::fs::write(&output_path, output_string)
                .with_context(|| format!("Failed to write output to {}", output_path.display()))?;
            tracing::info!("Output written to: {}", output_path.display());
        }
        None => {
            print!("{}", output_string);
        }
    }

    Ok(())
}

/// Output results with chunking support
async fn output_chunked_results(
    parsed_files: &[ParsedFile],
    chunk_size_str: String,
    chunk_output_dir: PathBuf,
    _format: OutputFormat,
) -> Result<()> {
    // Parse chunk size string (e.g., "50MB", "100KB")
    let chunk_size_bytes = parse_chunk_size(&chunk_size_str)?;

    tracing::info!(
        "Chunking output with max size: {} bytes ({})",
        chunk_size_bytes,
        chunk_size_str
    );

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&chunk_output_dir).with_context(|| {
        format!(
            "Failed to create chunk output directory: {}",
            chunk_output_dir.display()
        )
    })?;

    // Configure chunk writer
    let chunk_config = ChunkWriterConfig {
        max_chunk_size: chunk_size_bytes,
        target_chunk_size: (chunk_size_bytes as f64 * 0.9) as usize, // 90% of max
        auto_split: true,
        output_directory: Some(chunk_output_dir.clone()),
        include_metadata: true,
        ..Default::default()
    };

    let mut chunk_writer = ChunkWriter::new(chunk_config)?;

    // Write parsed files in chunks using splittable data API
    let chunk_infos = chunk_writer.write_splittable_data("kernel_data", parsed_files)?;

    // Create manifest builder
    let manifest_config = ManifestBuilderConfig {
        version: "1.0.0".to_string(),
        kernel_version: None,
        kernel_path: None,
        config: None,
        output_directory: Some(chunk_output_dir.clone()),
        chunk_prefix: "kernel".to_string(),
        validate_schema: true,
        sort_chunks: true,
    };

    let mut manifest_builder = ManifestBuilder::new(manifest_config)?;

    // Add chunks to manifest
    for (i, chunk_info) in chunk_infos.iter().enumerate() {
        let chunk_input = ChunkInput {
            file_path: chunk_output_dir.join(format!("kernel_{:03}.json", i + 1)),
            subsystem: "kernel".to_string(),
            symbol_count: chunk_info
                .metadata
                .as_ref()
                .map(|m| m.total_symbols)
                .unwrap_or(0),
            entry_point_count: chunk_info
                .metadata
                .as_ref()
                .map(|m| m.total_entry_points)
                .unwrap_or(0),
            file_count: parsed_files.len(),
        };
        manifest_builder.add_chunk(chunk_input)?;
    }

    // Build and write manifest
    let manifest_path = chunk_output_dir.join("manifest.json");
    let manifest = manifest_builder.build_and_write(&manifest_path)?;

    tracing::info!(
        "Created {} chunks in directory: {}",
        manifest.total_chunks,
        chunk_output_dir.display()
    );
    tracing::info!("Manifest written to: {}", manifest_path.display());

    Ok(())
}

/// Parse chunk size string into bytes
fn parse_chunk_size(size_str: &str) -> Result<usize> {
    let size_str = size_str.trim().to_uppercase();

    // Extract number and unit
    let (number_part, unit_part) = if let Some(pos) = size_str.find(|c: char| c.is_alphabetic()) {
        size_str.split_at(pos)
    } else {
        // No unit, assume bytes
        return size_str
            .parse::<usize>()
            .with_context(|| format!("Invalid chunk size format: {}", size_str));
    };

    let number: f64 = number_part
        .trim()
        .parse()
        .with_context(|| format!("Invalid number in chunk size: {}", number_part))?;

    let multiplier = match unit_part.trim() {
        "B" | "BYTES" => 1,
        "KB" | "K" => 1024,
        "MB" | "M" => 1024 * 1024,
        "GB" | "G" => 1024 * 1024 * 1024,
        _ => anyhow::bail!(
            "Unsupported chunk size unit: {} (supported: B, KB, MB, GB)",
            unit_part
        ),
    };

    let bytes = (number * multiplier as f64) as usize;

    // Validate reasonable limits
    const MIN_CHUNK_SIZE: usize = 1024; // 1KB minimum
    const MAX_CHUNK_SIZE: usize = 1024 * 1024 * 1024; // 1GB maximum

    if bytes < MIN_CHUNK_SIZE {
        anyhow::bail!(
            "Chunk size too small: {} (minimum: {})",
            bytes,
            MIN_CHUNK_SIZE
        );
    }

    if bytes > MAX_CHUNK_SIZE {
        anyhow::bail!(
            "Chunk size too large: {} (maximum: {})",
            bytes,
            MAX_CHUNK_SIZE
        );
    }

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config_string() {
        let (arch, config) = parse_config_string("x86_64:defconfig").unwrap();
        assert_eq!(arch, "x86_64");
        assert_eq!(config, "defconfig");

        assert!(parse_config_string("invalid").is_err());
    }

    #[test]
    fn test_parse_defines() {
        let defines = vec!["CONFIG_DEBUG=1".to_string(), "CONFIG_FEATURE".to_string()];
        let defines_map = parse_defines(&defines).unwrap();

        assert_eq!(defines_map.get("CONFIG_DEBUG"), Some(&"1".to_string()));
        assert_eq!(defines_map.get("CONFIG_FEATURE"), Some(&"1".to_string()));
    }

    #[test]
    fn test_parse_chunk_size() {
        // Test basic units
        assert_eq!(parse_chunk_size("1024").unwrap(), 1024);
        assert_eq!(parse_chunk_size("1KB").unwrap(), 1024);
        assert_eq!(parse_chunk_size("1MB").unwrap(), 1024 * 1024);
        assert_eq!(parse_chunk_size("1GB").unwrap(), 1024 * 1024 * 1024);

        // Test case insensitive
        assert_eq!(parse_chunk_size("50mb").unwrap(), 50 * 1024 * 1024);
        assert_eq!(parse_chunk_size("50MB").unwrap(), 50 * 1024 * 1024);

        // Test fractional values
        assert_eq!(parse_chunk_size("0.5MB").unwrap(), 512 * 1024);

        // Test alternative units
        assert_eq!(parse_chunk_size("1K").unwrap(), 1024);
        assert_eq!(parse_chunk_size("1M").unwrap(), 1024 * 1024);

        // Test whitespace handling
        assert_eq!(parse_chunk_size(" 10 MB ").unwrap(), 10 * 1024 * 1024);

        // Test error cases
        assert!(parse_chunk_size("invalid").is_err());
        assert!(parse_chunk_size("10XB").is_err());
        assert!(parse_chunk_size("100").unwrap() < 1024); // Too small should be caught in validation
    }
}
