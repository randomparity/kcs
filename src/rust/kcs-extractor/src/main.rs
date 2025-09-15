use anyhow::Result;
use clap::{Parser, Subcommand};
use kcs_extractor::{ExtractionConfig, Extractor};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kcs-extractor")]
#[command(about = "Extract kernel entry points")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract entry points from kernel source directory
    Extract {
        /// Input kernel source directory
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for extracted entry points (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Entry point types to extract
        #[arg(long, value_delimiter = ',')]
        types: Option<Vec<String>>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Extract entry points from parser index file
    Index {
        /// Input index file (JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for extracted entry points (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Entry point types to extract
        #[arg(long, value_delimiter = ',')]
        types: Option<Vec<String>>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Extract {
            input,
            output,
            types,
            verbose,
        } => {
            if verbose {
                tracing_subscriber::fmt::init();
            }

            let config = build_extraction_config(types);
            let extractor = Extractor::new(config);

            println!("Extracting entry points from: {}", input.display());
            let entry_points = extractor.extract_from_directory(&input)?;

            println!("Found {} entry points", entry_points.len());

            // Print summary by type
            let mut type_counts = std::collections::HashMap::new();
            for ep in &entry_points {
                *type_counts
                    .entry(format!("{:?}", ep.entry_type))
                    .or_insert(0) += 1;
            }

            for (entry_type, count) in type_counts {
                println!("  {}: {}", entry_type, count);
            }

            // Output results
            let json_output = serde_json::to_string_pretty(&entry_points)?;

            if let Some(output_path) = output {
                std::fs::write(&output_path, &json_output)?;
                println!("Results written to: {}", output_path.display());
            } else {
                println!("\nExtracted entry points:");
                println!("{}", json_output);
            }
        }

        Commands::Index {
            input,
            output,
            types,
        } => {
            let config = build_extraction_config(types);
            let extractor = Extractor::new(config);

            println!("Extracting entry points from index: {}", input.display());
            let index_content = std::fs::read_to_string(&input)?;
            let entry_points = extractor.extract_from_index(&index_content)?;

            println!("Found {} entry points from index", entry_points.len());

            let json_output = serde_json::to_string_pretty(&entry_points)?;

            if let Some(output_path) = output {
                std::fs::write(&output_path, &json_output)?;
                println!("Results written to: {}", output_path.display());
            } else {
                println!("{}", json_output);
            }
        }
    }

    Ok(())
}

fn build_extraction_config(types: Option<Vec<String>>) -> ExtractionConfig {
    let mut config = ExtractionConfig {
        include_syscalls: false,
        include_ioctls: false,
        include_file_ops: false,
        include_sysfs: false,
        include_procfs: false,
        include_debugfs: false,
        include_modules: false,
    };

    if let Some(types) = types {
        for type_str in types {
            match type_str.to_lowercase().as_str() {
                "syscalls" | "syscall" => config.include_syscalls = true,
                "ioctls" | "ioctl" => config.include_ioctls = true,
                "fileops" | "file_ops" => config.include_file_ops = true,
                "sysfs" => config.include_sysfs = true,
                "procfs" => config.include_procfs = true,
                "debugfs" => config.include_debugfs = true,
                "modules" | "module" => config.include_modules = true,
                "all" => return ExtractionConfig::default(),
                _ => eprintln!("Warning: Unknown entry point type: {}", type_str),
            }
        }
    } else {
        // Default to all types
        return ExtractionConfig::default();
    }

    config
}
