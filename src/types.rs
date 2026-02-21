use serde::{Deserialize, Serialize};
use std::fmt;

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

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} - {}", self.value, self.reasoning)
    }
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

impl fmt::Display for RunResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Rethinking Analysis Results ===")?;
        writeln!(f, "Goal: {}", self.goal.description)?;
        writeln!(f, "Iterations: {}", self.iterations.len())?;
        writeln!(f, "Final Score: {:.2}/1.0", self.final_score.value)?;
        writeln!(f, "Reasoning: {}", self.final_score.reasoning)?;
        writeln!(f)?;
        for iter in &self.iterations {
            writeln!(f, "--- Iteration {} ---", iter.iteration)?;
            writeln!(f, "  Score: {:.2}", iter.score.value)?;
            writeln!(f, "  Converged: {}", iter.converged)?;
            if let Some(ref path) = iter.script_path {
                writeln!(f, "  Script: {}", path)?;
            }
        }
        writeln!(f)?;
        writeln!(f, "=== Final Prompt ===")?;
        write!(f, "{}", self.final_prompt)?;
        Ok(())
    }
}
