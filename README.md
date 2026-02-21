# Rethinking

AI-powered iterative data analysis tool. Inspired by neural network training loops, Rethinking automatically improves data analysis quality through repeated cycles of analysis, evaluation, and prompt optimization.

## How It Works

Rethinking implements a training-like loop for data analysis:

```
┌─────────────────────────────────────────────────┐
│                 Iteration Loop                  │
│                                                 │
│  1. Forward Pass     - AI analyzes data         │
│  2. Loss Computation - AI evaluates quality     │
│  3. Stop Check       - Check for convergence    │
│  4. Backward Pass    - AI optimizes the prompt  │
│                                                 │
│  Repeat until convergence or max iterations     │
└─────────────────────────────────────────────────┘
```

**Forward Pass**: The AI agent examines data sources, queries databases, reads files, and runs Python scripts to perform analysis.

**Loss Computation**: A separate AI call evaluates the analysis quality using a user-defined loss prompt, producing a score between 0.0 and 1.0.

**Stop Check**: The system checks convergence criteria — score threshold, patience (plateau detection), or iteration limits.

**Backward Pass**: The AI updates the analysis prompt and stores findings in a persistent memory database for future iterations.

## Features

- **Iterative optimization**: Automatically improves analysis through multiple passes
- **Subprocess AI agent**: Calls any local CLI tool (codex, claude, etc.) as a subprocess
- **Persistent memory**: Any MySQL-compatible database (TiDB Zero auto-provisioned, or bring your own MySQL/TiDB/MariaDB)
- **Python scripting**: AI creates and runs Python scripts for complex analysis
- **MCP server mode**: Expose memory database as an MCP server for external tools
- **Configurable convergence**: Min/max iterations, score thresholds, patience-based stopping

## Quick Start

### Prerequisites

- Rust (edition 2024)
- Python 3 (for script execution)
- A local AI CLI tool installed (e.g., [codex](https://github.com/openai/codex), [claude](https://docs.anthropic.com/en/docs/claude-code))
- A MySQL-compatible database, **or** internet access for TiDB Zero auto-provisioning (no token needed)

### Installation

```bash
git clone <repo-url>
cd rethinking
cargo build --release
```

### Configuration

Create a `rethinking.toml`:

```toml
[goal]
description = "Analyze sales data to identify top-performing regions and seasonal trends"
loss_prompt = "Rate the analysis on completeness, accuracy, and actionability of insights"
data_dsn = "mysql://user:pass@host:port/database"
data_files = ["data/sales.csv"]
initial_prompt = "Analyze the sales data focusing on regional performance and time-based patterns"

[agent]
command = "claude"  # CLI command: "claude" or "codex"
model = "o4-mini"   # model name passed to the CLI tool

[memory]
# Option 1: Provide your own MySQL-compatible database
# dsn = "mysql://user:pass@localhost:3306/rethinking_memory"
# Option 2: Leave dsn empty to auto-provision a TiDB Zero instance (no token needed)
# tag = "rethinking"  # Optional tag for the TiDB Zero instance
size_limit_mb = 512

[iteration]
min_iterations = 1
max_iterations = 10
convergence_threshold = 0.9
patience = 3
convergence_delta = 0.01

[output]
work_dir = "./work_output"
result_file = "./results.json"
```

### Run

```bash
# Run analysis
rethinking -c rethinking.toml

# With debug logging
rethinking -c rethinking.toml -l debug

# MCP server mode (auto-spawned by the CLI tool; can also run manually for debugging)
rethinking --mcp-server -c rethinking.toml
```

## Architecture

```
src/
  main.rs           -- CLI entry point and orchestration
  config.rs         -- TOML configuration
  types.rs          -- Core types: Goal, State, Score, IterationResult, RunResult
  agent.rs          -- Agent trait, ToolExecutor trait, tool-use loop
  subprocess.rs     -- Subprocess agent (calls local CLI tool)
  memory.rs         -- Memory database management (TiDB Zero or custom MySQL) + SQL operations
  tools.rs          -- Tool definitions and dispatch (Forward + Backward tool sets)
  iteration.rs      -- Forward/Loss/Stop/Backward pass implementations
  script.rs         -- Python script lifecycle management
  mcp_server.rs     -- MCP server for memory database access
```

### How Tools Reach the AI Agent

The AI agent (claude/codex) accesses tools via the **MCP server**, not via direct function calls. When rethinking starts an analysis run:

1. `SubprocessAgent` writes an MCP config file into the working directory:
   - For **claude**: `.mcp.json`
   - For **codex**: `.codex/config.toml`
2. The config tells the CLI tool to spawn `rethinking --mcp-server -c <config>` as an MCP server child process
3. The CLI tool discovers the MCP server on startup and gains access to all 5 tools via MCP protocol (stdio transport)

This means `--mcp-server` mode is primarily spawned automatically by the CLI tool. You can also run it manually for debugging (e.g., with the [MCP Inspector](https://github.com/modelcontextprotocol/inspector)).

### Tool Sets

**Forward Pass tools** (read-only):
- `query_data_db` — Query the data source database
- `read_file` — Read data files from work directory
- `query_memory` — Query the memory database
- `run_python` — Create and execute Python scripts

**Backward Pass tools** (read-write memory):
- `execute_memory_sql` — Write to memory database (CREATE/INSERT/UPDATE/DELETE)
- `query_memory` — Query the memory database

## Memory System

Rethinking uses a MySQL-compatible database as persistent memory across iterations. You have two options:

**Option 1: Bring your own database** — Set `memory.dsn` in config to use any MySQL-compatible database (MySQL, TiDB, MariaDB, etc.). This gives you full control over the database lifecycle.

**Option 2: TiDB Zero (auto-provisioned)** — Leave `memory.dsn` empty. Rethinking will automatically create an ephemeral [TiDB Zero](https://zero.tidbapi.com) instance per run. No API token needed.

In both cases:
- The AI agent decides the schema and manages the data
- Size-limited (default 512MB) with monitoring
- Accessible via MCP server for external tools

## License

MIT
