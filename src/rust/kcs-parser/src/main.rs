//! KCS Parser CLI
//!
//! Command-line interface for parsing kernel source code.

use anyhow::{Context, Result};
use clap::{Parser as ClapParser, Subcommand};
use kcs_parser::{ExtendedParserConfig, ParsedFile, Parser};
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
    format: OutputFormat,
) -> Result<()> {
    tracing::info!("Parsing repository: {}", repo.display());
    tracing::info!("Configuration: {}", config);

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

    // Output results
    output_results(&parsed_files, output, format)?;

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
}
