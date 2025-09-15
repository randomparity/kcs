use anyhow::Result;
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
        }

        Commands::Query {
            graph: _,
            query_type,
            config,
        } => {
            // For now, create a simple test graph since we don't have serialization yet
            let test_graph = create_test_graph();

            let mut query_engine = QueryEngine::new(&test_graph);

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
                }

                QueryType::Dependencies { symbol, depth } => {
                    let result = query_engine.list_dependencies(&symbol, Some(depth))?;
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }

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
                }

                QueryType::Impact {
                    symbol,
                    change_type,
                } => {
                    let analysis = query_engine.analyze_impact(&symbol, change_type.into())?;
                    println!("{}", serde_json::to_string_pretty(&analysis)?);
                }
            }
        }
    }

    Ok(())
}

fn create_test_graph() -> KernelGraph {
    use kcs_graph::{CallEdge, CallType, Symbol, SymbolType};

    let mut graph = KernelGraph::new();

    // Add some test symbols
    let symbols = vec![
        Symbol {
            name: "vfs_read".to_string(),
            file_path: "fs/read_write.c".to_string(),
            line_number: 450,
            symbol_type: SymbolType::Function,
            signature: Some(
                "ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)"
                    .to_string(),
            ),
            config_dependencies: vec!["CONFIG_VFS".to_string()],
        },
        Symbol {
            name: "sys_read".to_string(),
            file_path: "fs/read_write.c".to_string(),
            line_number: 600,
            symbol_type: SymbolType::Function,
            signature: Some(
                "SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)"
                    .to_string(),
            ),
            config_dependencies: vec![],
        },
        Symbol {
            name: "generic_file_read_iter".to_string(),
            file_path: "mm/filemap.c".to_string(),
            line_number: 2500,
            symbol_type: SymbolType::Function,
            signature: None,
            config_dependencies: vec![],
        },
    ];

    for symbol in symbols {
        graph.add_symbol(symbol);
    }

    // Add call relationships
    let edge1 = CallEdge {
        call_type: CallType::Direct,
        call_site_line: 605,
        conditional: false,
        config_guard: None,
    };
    graph.add_call("sys_read", "vfs_read", edge1).unwrap();

    let edge2 = CallEdge {
        call_type: CallType::Indirect,
        call_site_line: 455,
        conditional: true,
        config_guard: Some("CONFIG_VFS".to_string()),
    };
    graph
        .add_call("vfs_read", "generic_file_read_iter", edge2)
        .unwrap();

    graph
}
