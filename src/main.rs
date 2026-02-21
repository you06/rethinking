mod agent;
mod config;
mod memory;
mod subprocess;
mod types;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "rethinking", about = "AI iterative data analysis tool")]
struct Cli {
    /// Config file path
    #[arg(short, long, default_value = "rethinking.toml")]
    config: String,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Run in MCP server mode
    #[arg(long)]
    mcp_server: bool,

    /// Working directory
    #[arg(short, long)]
    work_dir: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing with explicit stderr writer to keep stdout free for MCP server mode
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log_level.parse().unwrap_or_default()),
        )
        .init();

    tracing::info!("rethinking starting");

    // Load config
    let config = config::Config::load(&cli.config)?;
    tracing::debug!(?config, "loaded config");

    if cli.mcp_server {
        tracing::info!("MCP server mode");
        // TODO: WORK16
        todo!("MCP server mode")
    }

    // TODO: WORK15 - Main flow
    tracing::info!("analysis complete");
    Ok(())
}
