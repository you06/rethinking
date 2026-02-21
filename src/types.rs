use serde::{Deserialize, Serialize};

/// Analysis goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// User-described analysis goal
    pub description: String,
    /// Loss function prompt (for evaluating analysis quality)
    pub loss_prompt: String,
    /// Database connection string (data source)
    pub data_dsn: String,
    /// Data file paths in the working directory
    pub data_files: Vec<String>,
}

/// Iteration state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Current prompt (optimized during backward pass)
    pub prompt: String,
    /// Current iteration round (starting from 1)
    pub iteration: u32,
    /// Previous round's Python script path (None for first round)
    pub last_script: Option<String>,
    /// Previous round's script output
    pub last_output: Option<String>,
    /// Accumulated script feedback
    pub script_feedback: Vec<String>,
}

/// Iteration score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    /// Score between 0.0 and 1.0
    pub value: f64,
    /// Evaluation reasoning
    pub reasoning: String,
}

/// Single iteration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationResult {
    pub iteration: u32,
    pub score: Score,
    pub script_path: Option<String>,
    pub script_output: Option<String>,
    pub updated_prompt: String,
    pub converged: bool,
}

/// Overall run result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub goal: Goal,
    pub iterations: Vec<IterationResult>,
    pub final_prompt: String,
    pub final_score: Score,
}
