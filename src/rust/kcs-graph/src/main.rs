use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kcs_graph::builder::GraphBuilder;
use kcs_graph::config::ConfigParser;
use kcs_graph::{
    queries::{ChangeType, QueryEngine},
    KernelGraph,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kcs-graph")]
#[command(about = "Kernel call graph analysis")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build call graph from parser output
    Build {
        /// Input parser output file (JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output graph file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Kernel config file (.config)
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Query the call graph
    Query {
        /// Graph file to query
        #[arg(short, long)]
        graph: PathBuf,

        /// Query type
        #[command(subcommand)]
        query_type: QueryType,

        /// Kernel config for filtering
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum QueryType {
    /// Find who calls a function
    WhoCalls {
        /// Symbol name to find callers for
        symbol: String,

        /// Maximum search depth
        #[arg(short, long, default_value = "5")]
        depth: u32,
    },

    /// List dependencies of a function
    Dependencies {
        /// Symbol name to find dependencies for
        symbol: String,

        /// Maximum search depth
        #[arg(short, long, default_value = "5")]
        depth: u32,
    },

    /// Find path between two symbols
    Path {
        /// Source symbol
        from: String,

        /// Target symbol
        to: String,

        /// Maximum path length
        #[arg(short, long, default_value = "10")]
        max_length: u32,
    },

    /// Analyze impact of changes
    Impact {
        /// Symbol to analyze
        symbol: String,

        /// Type of change
        #[arg(value_enum)]
        change_type: ChangeTypeArg,
    },
}

#[derive(clap::ValueEnum, Clone)]
enum ChangeTypeArg {
    Signature,
    Behavior,
    Deletion,
}

impl From<ChangeTypeArg> for ChangeType {
    fn from(arg: ChangeTypeArg) -> Self {
        match arg {
            ChangeTypeArg::Signature => ChangeType::SignatureChange,
            ChangeTypeArg::Behavior => ChangeType::BehaviorChange,
            ChangeTypeArg::Deletion => ChangeType::Deletion,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            input,
            output,
            config,
            verbose,
        } => {
            if verbose {
                tracing_subscriber::fmt::init();
            }

            println!("Building call graph from: {}", input.display());

            let mut builder = GraphBuilder::new();

            // Load kernel config if provided
            if let Some(config_path) = config {
                let mut config_parser = ConfigParser::new();
                config_parser.parse_dotconfig(&config_path)?;
                let config_context = config_parser.get_enabled_options();
                builder = builder.with_config_context(config_context);
                println!("Loaded kernel config: {}", config_path.display());
            }

            // Load parser output
            let parser_content = std::fs::read_to_string(&input)?;
            builder.load_from_parser_output(&parser_content)?;

            let graph = builder.finalize();

            println!(
                "Built graph with {} symbols and {} calls",
                graph.symbol_count(),
                graph.call_count()
            );

            // Save graph if output specified
            if let Some(output_path) = output {
                // For now, just save basic statistics
                // TODO: Implement proper graph serialization
                let stats = serde_json::json!({
                    "symbol_count": graph.symbol_count(),
                    "call_count": graph.call_count(),
                    "built_at": chrono::Utc::now().to_rfc3339()
                });

                std::fs::write(&output_path, serde_json::to_string_pretty(&stats)?)?;
                println!("Graph stats saved to: {}", output_path.display());
            }
        },

        Commands::Query {
            graph,
            query_type,
            config,
        } => {
            // Strict behavior: require a valid graph file and fail otherwise.
            let loaded_graph: KernelGraph = detect_and_load_graph(&graph)
                .with_context(|| format!("failed to load graph from {}", graph.display()))?;
            println!("Loaded graph from: {}", graph.display());

            let mut query_engine = QueryEngine::new(&loaded_graph);

            // Load config if provided
            if let Some(config_path) = config {
                let mut config_parser = ConfigParser::new();
                config_parser.parse_dotconfig(&config_path)?;
                let config_context = config_parser.get_enabled_options();
                query_engine = query_engine.with_config_context(config_context);
            }

            match query_type {
                QueryType::WhoCalls { symbol, depth } => {
                    let result = query_engine.who_calls(&symbol, Some(depth))?;
                    println!("{}", serde_json::to_string_pretty(&result)?);
                },

                QueryType::Dependencies { symbol, depth } => {
                    let result = query_engine.list_dependencies(&symbol, Some(depth))?;
                    println!("{}", serde_json::to_string_pretty(&result)?);
                },

                QueryType::Path {
                    from,
                    to,
                    max_length,
                } => {
                    let path = query_engine.find_path(&from, &to, Some(max_length))?;
                    if let Some(path) = path {
                        println!("Path found: {}", path.join(" -> "));
                    } else {
                        println!("No path found between {} and {}", from, to);
                    }
                },

                QueryType::Impact {
                    symbol,
                    change_type,
                } => {
                    let analysis = query_engine.analyze_impact(&symbol, change_type.into())?;
                    println!("{}", serde_json::to_string_pretty(&analysis)?);
                },
            }
        },
    }

    Ok(())
}

#[derive(serde::Deserialize)]
struct JsonGraph {
    nodes: Vec<JsonNode>,
    edges: Vec<JsonEdge>,
}

#[derive(serde::Deserialize)]
struct JsonNode {
    id: String,
    label: String,
    #[allow(dead_code)]
    attributes: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct JsonEdge {
    source: String,
    target: String,
    label: Option<String>,
    attributes: Option<serde_json::Value>,
}

fn detect_and_load_graph(path: &std::path::Path) -> anyhow::Result<KernelGraph> {
    use kcs_graph::{CallEdge, CallType, Symbol, SymbolType};
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ext != "json" {
        anyhow::bail!(
            "unsupported graph format: {:?}. Only JSON Graph (.json) is supported in kcs-graph CLI",
            ext
        );
    }

    let data = std::fs::read_to_string(path)?;
    let parsed: JsonGraph = serde_json::from_str(&data)?;

    // Build KernelGraph from JSON Graph
    let mut graph = KernelGraph::new();
    use std::collections::HashMap;
    let mut id_to_name: HashMap<String, String> = HashMap::new();

    // Nodes
    for node in parsed.nodes {
        let mut symbol = Symbol {
            name: node.label.clone(),
            file_path: String::new(),
            line_number: 0,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        };

        if let Some(attrs) = &node.attributes {
            if let Some(fp) = attrs.get("file_path").and_then(|v| v.as_str()) {
                symbol.file_path = fp.to_string();
            }
            if let Some(ln) = attrs.get("line_number").and_then(|v| v.as_u64()) {
                symbol.line_number = ln as u32;
            }
            if let Some(sig) = attrs.get("signature").and_then(|v| v.as_str()) {
                symbol.signature = Some(sig.to_string());
            }
            if let Some(st) = attrs.get("symbol_type").and_then(|v| v.as_str()) {
                symbol.symbol_type = match st {
                    "Variable" => SymbolType::Variable,
                    "Macro" => SymbolType::Macro,
                    "Type" => SymbolType::Type,
                    "Constant" => SymbolType::Constant,
                    _ => SymbolType::Function,
                };
            }
            if let Some(deps) = attrs.get("config_dependencies").and_then(|v| v.as_array()) {
                symbol.config_dependencies =
                    deps.iter().filter_map(|d| d.as_str().map(|s| s.to_string())).collect();
            }
        }

        graph.add_symbol(symbol);
        id_to_name.insert(node.id, node.label);
    }

    // Edges
    for edge in parsed.edges {
        let Some(source_name) = id_to_name.get(&edge.source) else {
            continue;
        };
        let Some(target_name) = id_to_name.get(&edge.target) else {
            continue;
        };

        let mut call_edge = CallEdge {
            call_type: CallType::Direct,
            call_site_line: 0,
            conditional: false,
            config_guard: None,
        };
        if let Some(label) = &edge.label {
            call_edge.call_type = match label.as_str() {
                "Indirect" => CallType::Indirect,
                "FunctionPointer" => CallType::FunctionPointer,
                "Macro" => CallType::Macro,
                _ => CallType::Direct,
            };
        }
        if let Some(attrs) = &edge.attributes {
            if let Some(ln) = attrs.get("call_site_line").and_then(|v| v.as_u64()) {
                call_edge.call_site_line = ln as u32;
            }
            if let Some(cond) = attrs.get("conditional").and_then(|v| v.as_bool()) {
                call_edge.conditional = cond;
            }
            if let Some(guard) = attrs.get("config_guard").and_then(|v| v.as_str()) {
                call_edge.config_guard = Some(guard.to_string());
            }
        }

        // Ignore errors for missing symbols (already filtered by id_to_name)
        let _ = graph.add_call(source_name, target_name, call_edge);
    }

    Ok(graph)
}
