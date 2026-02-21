# Rethinking - AI Iterative Data Analysis Tool

## Project Overview
Rust CLI tool that implements AI-driven iterative data analysis optimization loop:
Forward Pass (analyze) → Loss Computation (evaluate) → Stop Check (convergence) → Backward Pass (optimize prompt)

## Architecture
```
src/
  main.rs           -- CLI entry, arg parsing, main orchestration
  config.rs          -- Config types and loading (TOML + CLI + env vars)
  types.rs           -- Core domain types: State, Goal, Score, IterationResult
  agent.rs           -- Agent trait + ToolExecutor trait + message types
  subprocess.rs      -- Subprocess agent (calls local CLI tool like codex/claude-code)
  memory.rs          -- Memory DB (TiDB Zero auto-provision or custom MySQL DSN) + SQL ops + size management
  tools.rs           -- Tool definitions (JSON Schema) + dispatch logic
  iteration.rs       -- Forward / Loss / Stop / Backward iteration logic
  script.rs          -- Python script creation, execution, result collection
  mcp_server.rs      -- MCP server exposing all tools (memory DB, data source, Python scripts, file access)
```

## Build & Test
```bash
cargo build          # Build
cargo test           # Run tests
cargo run -- --help  # Show CLI help
cargo run -- -c rethinking.toml  # Run with config
cargo run -- --mcp-server -c rethinking.toml  # MCP server mode
```

## Key Dependencies
- tokio (async runtime), serde/serde_json/toml (serialization), reqwest (HTTP, for TiDB Zero API)
- sqlx with mysql (any MySQL-compatible DB), clap (CLI), tracing (logging)
- async-trait, anyhow, rmcp (MCP server)

## Development Conventions
- Edition 2024
- All struct fields are `pub`
- Use `anyhow::Result` for error handling
- Use `tracing` for logging (not println! for diagnostics)
- Use `Arc<MemoryDB>` for shared database access
- Agent implementations use `#[async_trait]`
- Tool definitions use `serde_json::json!` macro for JSON Schema

## Completed Steps
- WORK01: Core domain types (`src/types.rs`)
- WORK02: Configuration system (`src/config.rs`) - TOML config loading with defaults
- WORK03: Async runtime + logging + CLI skeleton (`src/main.rs`) - tokio, clap, tracing-subscriber
- WORK04: Agent trait + tool abstraction (`src/agent.rs`) - Agent/ToolExecutor traits, message types, run_agent_loop
- WORK05: Subprocess agent (`src/subprocess.rs`) - SubprocessAgent calling CLI tools (codex/claude-code) via subprocess with MCP server config discovery
- WORK06: Memory database instance creation (`src/memory.rs`) - TiDB Zero auto-provisioning + custom DSN support, resolve_memory_dsn entry point
- WORK07: Connection pool + SQL operations + size management (`src/memory.rs`) - MemoryDB struct with connect/query/execute/get_size_mb/check_size, row-to-JSON conversion, result truncation at 1000 rows
- WORK08: Python script management (`src/script.rs`) - run_python for writing/executing Python scripts, ScriptResult with stdout/stderr truncation (10KB), to_agent_string formatting
- WORK09: Tool definitions + dispatch executor (`src/tools.rs`) - ForwardTools (query_data_db, read_file, query_memory, run_python) and BackwardTools (execute_memory_sql, query_memory) with ToolExecutor implementations, path safety checks for read_file

## Work Plan
See `work/WORK01.md` through `work/WORK18.md` for step-by-step implementation plan.
Each step should pass `cargo build` and `cargo test` before proceeding.

## Environment Variables
- `RUST_LOG` - Log level filter (e.g., `info`, `debug`, `rethinking=debug`)

Note: No API keys are needed by rethinking itself. The subprocess agent (e.g., codex, claude-code)
manages its own credentials. TiDB Zero auto-provisioning does NOT require an API token.
