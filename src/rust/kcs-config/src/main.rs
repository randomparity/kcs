use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use kcs_config::{ConfigOption, KernelConfig};
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use tracing::debug;

#[derive(Parser)]
#[clap(version, about = "Linux kernel configuration parser")]
struct Args {
    /// Path to config file (.config or Kconfig)
    #[clap(help = "Path to kernel configuration file (use - for stdin)")]
    config_path: Option<PathBuf>,

    /// Output format
    #[clap(value_enum, short = 'f', long, default_value = "json")]
    format: OutputFormat,

    /// Parse as Kconfig file instead of .config
    #[clap(long)]
    kconfig: bool,

    /// Enable verbose logging
    #[clap(short, long)]
    verbose: bool,

    /// Filter options by name prefix (e.g., CONFIG_NET)
    #[clap(long)]
    filter: Option<String>,
}

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Json,
    Toml,
    Summary,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("kcs_config=debug")
            .init();
    }

    let content = read_input(&args.config_path)?;

    if args.kconfig {
        let options =
            KernelConfig::parse_kconfig(&content).context("Failed to parse Kconfig file")?;

        let filtered = filter_options(options, &args.filter);
        output_kconfig_options(&filtered, args.format)?;
    } else {
        let mut config = KernelConfig::parse(&content).context("Failed to parse config file")?;

        if let Some(prefix) = &args.filter {
            config.options.retain(|name, _| name.starts_with(prefix));
        }

        output_config(&config, args.format)?;
    }

    Ok(())
}

fn read_input(path: &Option<PathBuf>) -> Result<String> {
    match path {
        Some(p) if p.to_str() == Some("-") => {
            debug!("Reading from stdin");
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            Ok(buffer)
        }
        Some(p) => {
            debug!("Reading from file: {:?}", p);
            fs::read_to_string(p).with_context(|| format!("Failed to read file: {:?}", p))
        }
        None => {
            debug!("Reading from stdin (no path specified)");
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            Ok(buffer)
        }
    }
}

fn filter_options(options: Vec<ConfigOption>, filter: &Option<String>) -> Vec<ConfigOption> {
    match filter {
        Some(prefix) => options
            .into_iter()
            .filter(|opt| opt.name.starts_with(prefix))
            .collect(),
        None => options,
    }
}

fn output_config(config: &KernelConfig, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(config)?;
            println!("{}", json);
        }
        OutputFormat::Toml => {
            let toml = toml::to_string_pretty(config)?;
            println!("{}", toml);
        }
        OutputFormat::Summary => {
            println!("Kernel Configuration Summary");
            println!("============================");
            println!("Architecture: {}", config.arch);
            println!("Version: {}", config.version);
            println!("Config Name: {}", config.config_name);
            println!("Total Options: {}", config.options.len());

            let enabled = config
                .options
                .values()
                .filter(|opt| matches!(opt.value, kcs_config::ConfigValue::Bool(true)))
                .count();
            let modules = config
                .options
                .values()
                .filter(|opt| matches!(opt.value, kcs_config::ConfigValue::Module))
                .count();
            let not_set = config
                .options
                .values()
                .filter(|opt| matches!(opt.value, kcs_config::ConfigValue::NotSet))
                .count();

            println!("\nOption Statistics:");
            println!("  Enabled (y): {}", enabled);
            println!("  Modules (m): {}", modules);
            println!("  Not Set: {}", not_set);
            println!(
                "  Other: {}",
                config.options.len() - enabled - modules - not_set
            );
        }
    }
    Ok(())
}

fn output_kconfig_options(options: &[ConfigOption], format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(options)?;
            println!("{}", json);
        }
        OutputFormat::Toml => {
            #[derive(serde::Serialize)]
            struct OptionsWrapper {
                options: Vec<ConfigOption>,
            }
            let wrapper = OptionsWrapper {
                options: options.to_vec(),
            };
            let toml = toml::to_string_pretty(&wrapper)?;
            println!("{}", toml);
        }
        OutputFormat::Summary => {
            println!("Kconfig Options Summary");
            println!("======================");
            println!("Total Options: {}", options.len());

            let with_help = options.iter().filter(|opt| opt.help_text.is_some()).count();
            let with_deps = options
                .iter()
                .filter(|opt| !opt.depends_on.is_empty())
                .count();
            let with_defaults = options
                .iter()
                .filter(|opt| !matches!(opt.value, kcs_config::ConfigValue::NotSet))
                .count();

            println!("\nOption Features:");
            println!("  With help text: {}", with_help);
            println!("  With dependencies: {}", with_deps);
            println!("  With default values: {}", with_defaults);

            println!("\nConfig Types:");
            for opt in options {
                println!("  {}: {:?}", opt.name, opt.config_type);
            }
        }
    }
    Ok(())
}
