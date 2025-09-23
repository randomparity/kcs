//! kcs-search CLI interface

use anyhow::Result;
use clap::{Parser, Subcommand};
use kcs_search::{SearchEngine, SearchQuery};
use std::io::{self, BufRead};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Database URL for pgvector connection
    #[arg(long)]
    database_url: Option<String>,

    /// Path to local JSON index file for offline search
    #[arg(long)]
    index_file: Option<String>,

    /// Output format
    #[arg(long, default_value = "json")]
    format: String,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Search for kernel symbols semantically
    Search {
        /// Search query
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Similarity threshold (0.0 to 1.0)
        #[arg(short = 't', long)]
        threshold: Option<f32>,

        /// Kernel configuration filter
        #[arg(short = 'c', long)]
        config: Option<String>,
    },

    /// Index kernel symbols for semantic search
    Index {
        /// Read symbols from stdin in JSON format
        #[arg(long)]
        stdin: bool,

        /// Symbol ID to index
        #[arg(long, conflicts_with = "stdin")]
        symbol_id: Option<i64>,

        /// Text content to index
        #[arg(long, required_unless_present = "stdin")]
        text: Option<String>,
    },

    /// Generate embeddings for given text
    Embed {
        /// Text to generate embeddings for
        text: String,

        /// Output raw vector values
        #[arg(long)]
        raw: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Search {
            query,
            top_k,
            threshold,
            config,
        } => {
            info!("Searching for: {}", query);

            let search_query = SearchQuery {
                text: query,
                top_k,
                threshold,
                config,
            };

            let mut engine = SearchEngine::new()?;
            if let Some(path) = cli.index_file.clone() { engine = engine.with_index_file(path.into()); }
            let results = engine.search(search_query).await?;

            // Output results
            if cli.format == "json" {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                for result in results {
                    println!(
                        "{}: {} (score: {:.3})",
                        result.file_path, result.name, result.score
                    );
                }
            }
        }

        Commands::Index {
            stdin,
            symbol_id,
            text,
        } => {
            let mut engine = SearchEngine::new()?;
            if let Some(path) = cli.index_file.clone() { engine = engine.with_index_file(path.into()); }

            if stdin {
                info!("Reading symbols from stdin");
                let stdin = io::stdin();
                use serde_json::Value;
                let mut entries: Vec<kcs_search::IndexEntry> = Vec::new();
                for line in stdin.lock().lines() {
                    let line = line?;
                    let v: Value = serde_json::from_str(&line)?;
                    let entry = kcs_search::IndexEntry {
                        symbol_id: v.get("symbol_id").and_then(|x| x.as_i64()).unwrap_or(0),
                        name: v.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                        file_path: v.get("file_path").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                        line_number: v.get("line_number").and_then(|x| x.as_i64()).unwrap_or(0) as i32,
                        text: v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    };
                    entries.push(entry);
                }
                // Persist entries
                for e in entries {
                    engine.index_symbol(e.symbol_id, &e.text).await?;
                }
                println!("Indexed entries to local file");
            } else if let (Some(id), Some(text)) = (symbol_id, text) {
                info!("Indexing symbol {}", id);
                engine.index_symbol(id, &text).await?;
                println!("Symbol {} indexed successfully", id);
            }
        }

        Commands::Embed { text, raw } => {
            info!("Generating embeddings for text");

            // TODO: Generate actual embeddings
            let placeholder_embedding = vec![0.0f32; 384];

            if raw {
                for value in &placeholder_embedding[..10] {
                    print!("{:.6} ", value);
                }
                println!("... ({} total dimensions)", placeholder_embedding.len());
            } else {
                let output = serde_json::json!({
                    "text": text,
                    "dimension": placeholder_embedding.len(),
                    "embedding": placeholder_embedding,
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
        }
    }

    Ok(())
}
