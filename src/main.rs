use clap::Parser;
use rethinking::{agent, config, iteration, mcp_server, memory, subprocess, tools, types};

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

    if cli.mcp_server {
        tracing::info!("MCP server mode");

        let config = config::Config::load(&cli.config)?;

        let work_dir = cli
            .work_dir
            .clone()
            .or(Some(config.output.work_dir.clone()))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| ".rethinking_work".to_string());
        tokio::fs::create_dir_all(&work_dir).await?;

        let memory_dsn = memory::resolve_memory_dsn(&config.memory).await?;
        let memory_db = std::sync::Arc::new(
            memory::MemoryDB::connect(&memory_dsn, config.memory.size_limit_mb).await?,
        );

        return mcp_server::run_mcp_server(
            memory_db,
            config.goal.data_dsn.clone(),
            work_dir,
        )
        .await;
    }

    run(&cli).await
}

async fn run(cli: &Cli) -> anyhow::Result<()> {
    // 1. Load config
    let config = config::Config::load(&cli.config)?;
    tracing::debug!(?config, "loaded config");

    // 2. Determine working directory
    let work_dir = cli
        .work_dir
        .clone()
        .or(Some(config.output.work_dir.clone()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| ".rethinking_work".to_string());
    tokio::fs::create_dir_all(&work_dir).await?;

    // 3. Connect to memory database
    tracing::info!("connecting to memory database");
    let memory_dsn = memory::resolve_memory_dsn(&config.memory).await?;
    let memory_db = std::sync::Arc::new(
        memory::MemoryDB::connect(&memory_dsn, config.memory.size_limit_mb).await?,
    );
    tracing::info!("memory database connected");

    // 4. Create agent
    let agent: Box<dyn agent::Agent> = Box::new(subprocess::SubprocessAgent::new(
        config.agent.command.clone(),
        config.agent.model.clone(),
        cli.config.clone(),
        work_dir.clone(),
    )?);

    // 5. Create tool sets
    let forward_tools = tools::ForwardTools::new(
        config.goal.data_dsn.clone(),
        memory_db.clone(),
        work_dir.clone(),
    );
    let backward_tools = tools::BackwardTools::new(memory_db.clone());

    // 6. Build goal
    let goal = types::Goal {
        description: config.goal.description.clone(),
        loss_prompt: config.goal.loss_prompt.clone(),
        data_dsn: config.goal.data_dsn.clone(),
        data_files: config.goal.data_files.clone(),
    };

    // 7. Run iterations
    let result = iteration::run_iterations(
        agent.as_ref(),
        &forward_tools,
        &backward_tools,
        &goal,
        &config.iteration,
        &config.goal.initial_prompt,
    )
    .await?;

    // 8. Output results
    tracing::info!(
        iterations = result.iterations.len(),
        final_score = result.final_score.value,
        "analysis complete"
    );
    println!("{result}");

    // 9. Save results to JSON file
    let result_path = if config.output.result_file.is_empty() {
        format!("{}/result.json", work_dir)
    } else {
        config.output.result_file.clone()
    };
    let json = serde_json::to_string_pretty(&result)?;
    tokio::fs::write(&result_path, &json).await?;
    tracing::info!(path = %result_path, "results saved");

    Ok(())
}
