use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub goal: GoalConfig,
    pub agent: AgentConfig,
    pub memory: MemoryConfig,
    pub iteration: IterationConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalConfig {
    pub description: String,
    pub loss_prompt: String,
    pub data_dsn: String,
    #[serde(default)]
    pub data_files: Vec<String>,
    #[serde(default)]
    pub initial_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// CLI command to invoke (e.g., "codex", "claude-code")
    #[serde(default = "default_command")]
    pub command: String,
    /// Model name passed to the CLI tool
    #[serde(default = "default_model")]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Directly provide a MySQL DSN (skips TiDB Zero auto-creation if set)
    /// Supports any MySQL protocol-compatible database (TiDB, MySQL, MariaDB, etc.)
    #[serde(default)]
    pub dsn: String,
    /// TiDB Zero instance tag (only used when dsn is empty, auto-creates temporary instance)
    #[serde(default = "default_memory_tag")]
    pub tag: String,
    /// Database size limit (MB), default 512
    #[serde(default = "default_size_limit")]
    pub size_limit_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationConfig {
    #[serde(default = "default_min_iterations")]
    pub min_iterations: u32,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
    /// Convergence threshold, score >= this value is considered converged
    #[serde(default = "default_convergence_threshold")]
    pub convergence_threshold: f64,
    /// Convergence by patience: stop when consecutive `patience` adjacent iteration
    /// score differences are all less than delta (requires patience+1 scores)
    #[serde(default = "default_patience")]
    pub patience: u32,
    #[serde(default = "default_convergence_delta")]
    pub convergence_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub work_dir: String,
    #[serde(default)]
    pub result_file: String,
}

fn default_command() -> String {
    "codex".to_string()
}

fn default_model() -> String {
    "o4-mini".to_string()
}

fn default_memory_tag() -> String {
    "rethinking".to_string()
}

fn default_size_limit() -> u64 {
    512
}

fn default_min_iterations() -> u32 {
    1
}

fn default_max_iterations() -> u32 {
    10
}

fn default_convergence_threshold() -> f64 {
    0.9
}

fn default_patience() -> u32 {
    3
}

fn default_convergence_delta() -> f64 {
    0.01
}

impl Config {
    pub fn load(path: &str) -> Result<Config> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file: {}", path))?;
        let config: Config = toml::from_str(&content)
            .with_context(|| format!("failed to parse config file: {}", path))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_config() {
        let toml_str = r#"
[goal]
description = "Analyze sales data"
loss_prompt = "Evaluate analysis quality"
data_dsn = "mysql://localhost:4000/sales"
data_files = ["data/sales.csv"]
initial_prompt = "Start analysis"

[agent]
command = "claude-code"
model = "sonnet"

[memory]
dsn = "mysql://user:pass@localhost:3306/memdb"
tag = "test-tag"
size_limit_mb = 256

[iteration]
min_iterations = 2
max_iterations = 20
convergence_threshold = 0.95
patience = 5
convergence_delta = 0.005

[output]
work_dir = "/tmp/rethinking"
result_file = "result.json"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.goal.description, "Analyze sales data");
        assert_eq!(config.goal.loss_prompt, "Evaluate analysis quality");
        assert_eq!(config.goal.data_dsn, "mysql://localhost:4000/sales");
        assert_eq!(config.goal.data_files, vec!["data/sales.csv"]);
        assert_eq!(config.goal.initial_prompt, "Start analysis");
        assert_eq!(config.agent.command, "claude-code");
        assert_eq!(config.agent.model, "sonnet");
        assert_eq!(config.memory.dsn, "mysql://user:pass@localhost:3306/memdb");
        assert_eq!(config.memory.tag, "test-tag");
        assert_eq!(config.memory.size_limit_mb, 256);
        assert_eq!(config.iteration.min_iterations, 2);
        assert_eq!(config.iteration.max_iterations, 20);
        assert_eq!(config.iteration.convergence_threshold, 0.95);
        assert_eq!(config.iteration.patience, 5);
        assert_eq!(config.iteration.convergence_delta, 0.005);
        assert_eq!(config.output.work_dir, "/tmp/rethinking");
        assert_eq!(config.output.result_file, "result.json");
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml_str = r#"
[goal]
description = "Analyze data"
loss_prompt = "Evaluate"
data_dsn = "mysql://localhost:4000/db"

[agent]

[memory]

[iteration]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.goal.description, "Analyze data");
        assert_eq!(config.goal.data_files.len(), 0);
        assert_eq!(config.goal.initial_prompt, "");
        assert_eq!(config.agent.command, "codex");
        assert_eq!(config.agent.model, "o4-mini");
        assert_eq!(config.memory.dsn, "");
        assert_eq!(config.memory.tag, "rethinking");
        assert_eq!(config.memory.size_limit_mb, 512);
        assert_eq!(config.iteration.min_iterations, 1);
        assert_eq!(config.iteration.max_iterations, 10);
        assert_eq!(config.iteration.convergence_threshold, 0.9);
        assert_eq!(config.iteration.patience, 3);
        assert_eq!(config.iteration.convergence_delta, 0.01);
        assert_eq!(config.output.work_dir, "");
        assert_eq!(config.output.result_file, "");
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = Config::load("/nonexistent/path.toml");
        assert!(result.is_err());
    }
}
