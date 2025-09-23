use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kcs_graph::KernelGraph;
use kcs_impact::analyzer::AdvancedAnalyzer;
use kcs_impact::diff_analyzer::{GitDiffAnalyzer, SemanticDiffAnalyzer};
use kcs_impact::{ChangeType, ImpactAnalyzer};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kcs-impact")]
#[command(about = "Kernel change impact analysis")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze impact of a patch file
    Patch {
        /// Patch file to analyze
        #[arg(short, long)]
        patch: PathBuf,

        /// Graph file for call graph analysis
        #[arg(short, long)]
        graph: Option<PathBuf>,

        /// Output file for analysis results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Include detailed recommendations
        #[arg(short, long)]
        detailed: bool,
    },

    /// Analyze impact of git commit(s)
    Git {
        /// Git repository path
        #[arg(short, long)]
        repo: PathBuf,

        /// Commit SHA or range (e.g., HEAD~5..HEAD)
        #[arg(short, long)]
        commit: String,

        /// Graph file for call graph analysis
        #[arg(short, long)]
        graph: Option<PathBuf>,

        /// Output file for analysis results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Analyze impact of specific symbol changes
    Symbol {
        /// Symbol name to analyze
        symbol: String,

        /// Type of change
        #[arg(value_enum)]
        change_type: ChangeTypeArg,

        /// Graph file for call graph analysis
        #[arg(short, long)]
        graph: PathBuf,

        /// Maximum analysis depth
        #[arg(short, long, default_value = "5")]
        depth: u32,
    },

    /// Analyze blast radius of changes
    BlastRadius {
        /// Symbol name to analyze
        symbol: String,

        /// Graph file for call graph analysis
        #[arg(short, long)]
        graph: PathBuf,

        /// Maximum radius depth
        #[arg(short, long, default_value = "3")]
        depth: u32,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum ChangeTypeArg {
    FunctionAdded,
    FunctionRemoved,
    FunctionModified,
    SignatureChanged,
    StructChanged,
    MacroChanged,
}

impl From<ChangeTypeArg> for ChangeType {
    fn from(arg: ChangeTypeArg) -> Self {
        match arg {
            ChangeTypeArg::FunctionAdded => ChangeType::FunctionAdded,
            ChangeTypeArg::FunctionRemoved => ChangeType::FunctionRemoved,
            ChangeTypeArg::FunctionModified => ChangeType::FunctionModified,
            ChangeTypeArg::SignatureChanged => ChangeType::SignatureChanged,
            ChangeTypeArg::StructChanged => ChangeType::StructChanged,
            ChangeTypeArg::MacroChanged => ChangeType::MacroChanged,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Patch {
            patch,
            graph,
            output,
            detailed,
        } => {
            println!("Analyzing patch: {}", patch.display());

            if let Some(graph_path) = graph {
                // Load graph from file and perform full analysis
                let graph = detect_and_load_graph(&graph_path).with_context(|| {
                    format!("failed to load graph from {}", graph_path.display())
                })?;
                let analyzer = ImpactAnalyzer::new(graph);

                let analysis = analyzer.analyze_patch(&patch)?;

                println!("Impact Analysis Results:");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("Total changes: {}", analysis.target_changes.len());
                println!("Affected symbols: {}", analysis.affected_symbols.len());
                println!("Overall risk: {:?}", analysis.risk_assessment.overall_risk);

                if detailed {
                    println!("\nDetailed Analysis:");
                    for change in &analysis.target_changes {
                        println!(
                            "  {} ({}:{}): {:?}",
                            change.description,
                            change.file_path,
                            change.line_number,
                            change.change_type
                        );
                    }

                    println!("\nAffected Symbols:");
                    for affected in &analysis.affected_symbols {
                        println!(
                            "  {} - {:?} (distance: {})",
                            affected.symbol.name,
                            affected.impact_level,
                            affected.call_chain_distance
                        );
                    }

                    println!("\nRecommendations:");
                    for rec in &analysis.recommendations {
                        println!("  • {}", rec);
                    }
                }

                // Save results if output specified
                if let Some(output_path) = output {
                    let json_output = serde_json::to_string_pretty(&analysis)?;
                    std::fs::write(&output_path, json_output)?;
                    println!("\nResults saved to: {}", output_path.display());
                }
            } else {
                // Basic patch analysis without graph
                let semantic_analyzer = SemanticDiffAnalyzer::new();
                let patch_content = std::fs::read_to_string(&patch)?;
                let changes = kcs_impact::patch_parser::parse_patch(&patch_content)?;

                println!("Basic Patch Analysis:");
                println!("Changes detected: {}", changes.len());

                for change in &changes {
                    let risk = semantic_analyzer.classify_change_risk(change)?;
                    let impacts = semantic_analyzer.analyze_change_semantic_impact(change)?;

                    println!(
                        "\n{} ({}:{})",
                        change.description, change.file_path, change.line_number
                    );
                    println!("  Risk: {:?}", risk);
                    if !impacts.is_empty() {
                        println!("  Impacts:");
                        for impact in impacts {
                            println!("    • {}", impact);
                        }
                    }
                }
            }
        }

        Commands::Git {
            repo,
            commit,
            graph: _,
            output,
        } => {
            println!("Analyzing git changes: {} in {}", commit, repo.display());

            let git_analyzer = GitDiffAnalyzer::new(&repo)?;

            let changes = if commit.contains("..") {
                let parts: Vec<&str> = commit.split("..").collect();
                if parts.len() == 2 {
                    git_analyzer.analyze_range(parts[0], parts[1])?
                } else {
                    return Err(anyhow::anyhow!("Invalid commit range format"));
                }
            } else {
                git_analyzer.analyze_commit(&commit)?
            };

            println!("Git Analysis Results:");
            println!("━━━━━━━━━━━━━━━━━━━━━━");
            println!("Files changed: {}", changes.len());

            for change in &changes {
                println!("  {} - {:?}", change.file_path, change.change_type);
            }

            if let Some(output_path) = output {
                let json_output = serde_json::to_string_pretty(&changes)?;
                std::fs::write(&output_path, json_output)?;
                println!("\nResults saved to: {}", output_path.display());
            }
        }

        Commands::Symbol {
            symbol,
            change_type,
            graph,
            depth: _depth,
        } => {
            println!("Analyzing symbol change: {} ({:?})", symbol, change_type);

            // Load graph strictly from provided file
            let graph = detect_and_load_graph(&graph)
                .with_context(|| format!("failed to load graph from {}", graph.display()))?;
            let analyzer = AdvancedAnalyzer::new(graph);

            let affected = match change_type {
                ChangeTypeArg::SignatureChanged => {
                    analyzer.analyze_function_signature_change(&symbol)?
                }
                ChangeTypeArg::StructChanged => analyzer.analyze_struct_change(&symbol, &[])?,
                ChangeTypeArg::MacroChanged => analyzer.analyze_macro_change(&symbol)?,
                _ => {
                    // Use the same loaded graph for the basic analyzer
                    let basic_analyzer = ImpactAnalyzer::new(analyzer.graph().clone());
                    basic_analyzer.analyze_symbol_change(&symbol, change_type.into())?
                }
            };

            println!("Symbol Analysis Results:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Affected symbols: {}", affected.len());

            for affect in &affected {
                println!(
                    "  {} - {:?} (distance: {})",
                    affect.symbol.name, affect.impact_level, affect.call_chain_distance
                );
                println!("    Reason: {}", affect.impact_reason);
                if affect.requires_recompilation {
                    println!("    ⚠️  Requires recompilation");
                }
                if affect.requires_testing {
                    println!("    🧪 Requires testing");
                }
            }
        }

        Commands::BlastRadius {
            symbol,
            graph,
            depth,
        } => {
            println!("Analyzing blast radius for: {} (depth: {})", symbol, depth);

            let graph = detect_and_load_graph(&graph)
                .with_context(|| format!("failed to load graph from {}", graph.display()))?;
            let analyzer = AdvancedAnalyzer::new(graph);

            let blast_radius = analyzer.analyze_blast_radius(&symbol, depth)?;

            println!("Blast Radius Analysis:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━");

            for (distance, symbols) in blast_radius {
                println!("Distance {}: {} symbols", distance, symbols.len());
                for sym in &symbols {
                    println!("  {} ({}:{})", sym.name, sym.file_path, sym.line_number);
                }
            }
        }
    }

    Ok(())
}

fn detect_and_load_graph(path: &std::path::Path) -> anyhow::Result<KernelGraph> {
    use kcs_serializer::{GraphImporter, GraphMLExporter, JsonGraphExporter};
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "json" {
        let importer = JsonGraphExporter::new();
        importer.import_from_file(&path.to_string_lossy())
    } else if ext == "graphml" || ext == "xml" {
        let importer = GraphMLExporter::new();
        importer.import_from_file(&path.to_string_lossy())
    } else {
        // Try JSON first, then GraphML
        let importer_json = JsonGraphExporter::new();
        if let Ok(g) = importer_json.import_from_file(&path.to_string_lossy()) {
            return Ok(g);
        }
        let importer_graphml = GraphMLExporter::new();
        importer_graphml.import_from_file(&path.to_string_lossy())
    }
}
