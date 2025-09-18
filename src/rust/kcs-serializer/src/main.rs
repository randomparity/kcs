//! Command-line interface for kcs-serializer
//!
//! This binary provides CLI access to graph serialization functionality,
//! allowing conversion between different graph formats.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use kcs_serializer::{
    ChunkOptions, GraphChunker, GraphExporter, GraphImporter, GraphMLExporter, JsonGraphExporter,
};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "kcs-serializer")]
#[command(about = "Kernel call graph serialization tool")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Export a kernel graph to various formats
    Export {
        /// Input graph file path
        #[arg(short, long)]
        input: String,

        /// Output file path
        #[arg(short, long)]
        output: String,

        /// Output format
        #[arg(short, long, value_enum)]
        format: ExportFormat,

        /// Enable chunking for large graphs
        #[arg(long)]
        chunked: bool,

        /// Maximum nodes per chunk (if chunking is enabled)
        #[arg(long, default_value_t = 1000)]
        max_nodes: usize,
    },

    /// Import a graph from various formats
    Import {
        /// Input file path
        #[arg(short, long)]
        input: String,

        /// Output graph file path
        #[arg(short, long)]
        output: String,

        /// Input format
        #[arg(short, long, value_enum)]
        format: ExportFormat,
    },

    /// Convert between graph formats
    Convert {
        /// Input file path
        #[arg(short, long)]
        input: String,

        /// Output file path
        #[arg(short, long)]
        output: String,

        /// Input format
        #[arg(long, value_enum)]
        from: ExportFormat,

        /// Output format
        #[arg(long, value_enum)]
        to: ExportFormat,
    },

    /// Get information about a serialized graph
    Info {
        /// Graph file path
        #[arg(short, long)]
        file: String,

        /// File format
        #[arg(long, value_enum)]
        format: ExportFormat,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExportFormat {
    /// JSON Graph format
    Json,
    /// GraphML XML format
    Graphml,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Export {
            input,
            output,
            format,
            chunked,
            max_nodes,
        } => handle_export(input, output, format, chunked, max_nodes),
        Commands::Import {
            input,
            output,
            format,
        } => handle_import(input, output, format),
        Commands::Convert {
            input,
            output,
            from,
            to,
        } => handle_convert(input, output, from, to),
        Commands::Info { file, format } => handle_info(file, format),
    }
}

fn handle_export(
    input: String,
    output: String,
    format: ExportFormat,
    chunked: bool,
    max_nodes: usize,
) -> Result<()> {
    // For export, we assume the input is already a serialized graph in some format
    // and we want to export it to a different format or with chunking
    println!("Reading graph from: {}", input);

    // Try to detect input format based on file extension
    let input_format = detect_format(&input)?;
    let graph = import_graph(&input, input_format)?;

    println!(
        "Graph loaded: {} nodes, {} edges",
        graph.symbol_count(),
        graph.call_count()
    );

    if chunked {
        let chunk_options = ChunkOptions {
            max_nodes_per_chunk: max_nodes,
            max_edges_per_chunk: max_nodes * 5, // heuristic
            preserve_components: true,
            include_boundary_edges: true,
        };
        let chunker = GraphChunker::with_options(chunk_options);
        let chunks = chunker.chunk_graph(&graph)?;

        println!("Chunking graph into {} chunks", chunks.len());

        // Save chunks as separate files
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_file = format!("{}.chunk_{:03}", output, i);
            let chunk_json = serde_json::to_string_pretty(chunk)?;
            fs::write(&chunk_file, chunk_json)?;
            println!("Saved chunk {}: {}", i, chunk_file);
        }
    } else {
        export_graph(&graph, &output, format)?;
        println!("Graph exported to: {}", output);
    }

    Ok(())
}

fn handle_import(input: String, output: String, format: ExportFormat) -> Result<()> {
    println!("Importing graph from: {} (format: {:?})", input, format);

    let graph = import_graph(&input, format)?;

    println!(
        "Graph imported: {} nodes, {} edges",
        graph.symbol_count(),
        graph.call_count()
    );

    // For simplicity, export as JSON format for the graph representation
    let json_exporter = JsonGraphExporter::new().with_metadata(true);
    json_exporter.export_to_file(&graph, &output)?;

    println!("Graph saved to: {}", output);
    Ok(())
}

fn handle_convert(
    input: String,
    output: String,
    from: ExportFormat,
    to: ExportFormat,
) -> Result<()> {
    println!(
        "Converting {} to {} (from {:?} to {:?})",
        input, output, from, to
    );

    let graph = import_graph(&input, from)?;

    println!(
        "Graph loaded: {} nodes, {} edges",
        graph.symbol_count(),
        graph.call_count()
    );

    export_graph(&graph, &output, to)?;

    println!("Conversion complete: {}", output);
    Ok(())
}

fn handle_info(file: String, format: ExportFormat) -> Result<()> {
    println!("Analyzing graph file: {} (format: {:?})", file, format);

    let graph = import_graph(&file, format)?;

    // Display graph statistics
    println!("\nGraph Information:");
    println!("├─ Nodes: {}", graph.symbol_count());
    println!("├─ Edges: {}", graph.call_count());

    // Note: Symbol type analysis would require iterating over all symbols
    // which is not currently exposed by the KernelGraph API
    if graph.symbol_count() > 0 {
        println!("├─ Symbol Details: API enhancement needed for detailed analysis");
    }

    // File size information
    let metadata = fs::metadata(&file)?;
    let file_size = metadata.len();
    println!(
        "└─ File Size: {} bytes ({:.2} KB)",
        file_size,
        file_size as f64 / 1024.0
    );

    Ok(())
}

fn detect_format(file_path: &str) -> Result<ExportFormat> {
    let path = Path::new(file_path);
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => Ok(ExportFormat::Json),
        Some("graphml") | Some("xml") => Ok(ExportFormat::Graphml),
        _ => anyhow::bail!(
            "Cannot detect format from file extension. Supported: .json, .graphml, .xml"
        ),
    }
}

fn import_graph(file_path: &str, format: ExportFormat) -> Result<kcs_graph::KernelGraph> {
    match format {
        ExportFormat::Json => {
            let json_exporter = JsonGraphExporter::new();
            json_exporter.import_from_file(file_path)
        }
        ExportFormat::Graphml => {
            let graphml_exporter = GraphMLExporter::new();
            graphml_exporter.import_from_file(file_path)
        }
    }
}

fn export_graph(
    graph: &kcs_graph::KernelGraph,
    file_path: &str,
    format: ExportFormat,
) -> Result<()> {
    match format {
        ExportFormat::Json => {
            let json_exporter = JsonGraphExporter::new().with_metadata(true);
            json_exporter.export_to_file(graph, file_path)
        }
        ExportFormat::Graphml => {
            let graphml_exporter = GraphMLExporter::new().with_attributes(true);
            graphml_exporter.export_to_file(graph, file_path)
        }
    }
}
