use anyhow::Result;

use crate::agent::{run_agent_loop, Agent, ContentBlock, Message, Role};
use crate::config::IterationConfig;
use crate::tools::{BackwardTools, ForwardTools};
use crate::types::{Goal, IterationResult, RunResult, Score, State};

/// Run the complete iteration loop.
///
/// Orchestrates Forward Pass -> Loss Computation -> Stop Check -> Backward Pass
/// until convergence or max iterations is reached.
pub async fn run_iterations(
    agent: &dyn Agent,
    forward_tools: &ForwardTools,
    backward_tools: &BackwardTools,
    goal: &Goal,
    config: &IterationConfig,
    initial_prompt: &str,
) -> Result<RunResult> {
    let mut state = State {
        prompt: initial_prompt.to_string(),
        iteration: 0,
        last_script: None,
        last_output: None,
        script_feedback: Vec::new(),
    };

    let mut iterations = Vec::new();
    let mut scores = Vec::new();

    loop {
        state.iteration += 1;
        tracing::info!(iteration = state.iteration, "starting iteration");

        // 1. Forward Pass
        tracing::info!("forward pass");
        let forward_result = forward_pass(agent, forward_tools, &state, goal).await?;

        // 2. Loss Computation
        tracing::info!("computing loss");
        let score = compute_loss(agent, goal, &forward_result).await?;
        tracing::info!(score = score.value, reasoning = %score.reasoning, "loss computed");

        scores.push(score.clone());

        // 3. Stop Check
        let converged = check_stop(config, state.iteration, &scores);

        // 4. Backward Pass (unless already converged)
        let updated_prompt = if !converged {
            tracing::info!("backward pass");
            let backward =
                backward_pass(agent, backward_tools, &state, goal, &forward_result, &score)
                    .await?;
            state.script_feedback.push(backward.script_feedback.clone());
            backward.updated_prompt
        } else {
            state.prompt.clone()
        };

        // Record iteration result
        let iter_result = IterationResult {
            iteration: state.iteration,
            score: score.clone(),
            script_path: forward_result.script_path.clone(),
            script_output: forward_result.script_output.clone(),
            updated_prompt: updated_prompt.clone(),
            converged,
        };
        iterations.push(iter_result);

        // Update state
        state.prompt = updated_prompt;
        state.last_script = forward_result.script_path;
        state.last_output = forward_result.script_output;

        if converged {
            tracing::info!(iteration = state.iteration, "converged");
            break;
        }
    }

    let final_score = scores.last().cloned().unwrap_or(Score {
        value: 0.0,
        reasoning: "no iterations completed".into(),
    });

    Ok(RunResult {
        goal: goal.clone(),
        iterations,
        final_prompt: state.prompt,
        final_score,
    })
}

/// Forward Pass result
pub struct ForwardResult {
    pub analysis: String,
    pub script_path: Option<String>,
    pub script_output: Option<String>,
    pub messages: Vec<Message>,
}

/// Execute Forward Pass
///
/// Builds a system prompt from the current state and goal, sends an initial
/// user message to the agent, runs the tool-use loop (with ForwardTools),
/// and returns the analysis along with any script info extracted from the
/// message history.
pub async fn forward_pass(
    agent: &dyn Agent,
    tools: &ForwardTools,
    state: &State,
    goal: &Goal,
) -> Result<ForwardResult> {
    let system = build_forward_system_prompt(state, goal);

    let initial_message = Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: format!(
                "Analyze the data according to the goal. This is iteration {}.\n\n\
                 Goal: {}\n\n\
                 Current prompt:\n{}",
                state.iteration, goal.description, state.prompt
            ),
        }],
    };

    let (messages, final_text) = run_agent_loop(
        agent,
        tools as &dyn crate::agent::ToolExecutor,
        vec![initial_message],
        &system,
    )
    .await?;

    let (script_path, script_output) = extract_script_info(&messages);

    Ok(ForwardResult {
        analysis: final_text,
        script_path,
        script_output,
        messages,
    })
}

fn build_forward_system_prompt(state: &State, goal: &Goal) -> String {
    let mut prompt = format!(
        "You are a data analyst AI. Your task is to analyze data and produce insights.\n\n\
         Goal: {}\n\n\
         Available tools:\n\
         - query_data_db: Execute read-only SQL queries against the data source\n\
         - read_file: Read data files from the working directory\n\
         - query_memory: Query the memory database for previous findings\n\
         - run_python: Write and execute Python scripts for complex analysis\n\n\
         Guidelines:\n\
         - Query the data source to understand the data\n\
         - Use the memory database to check previous findings\n\
         - Write Python scripts for complex analysis\n\
         - Provide clear, structured analysis results",
        goal.description
    );

    if let Some(feedback) = state.script_feedback.last() {
        prompt.push_str(&format!("\n\nPrevious feedback:\n{}", feedback));
    }
    if let Some(ref last_output) = state.last_output {
        prompt.push_str(&format!("\n\nPrevious script output:\n{}", last_output));
    }

    prompt
}

/// Execute Loss Computation
///
/// Evaluates the quality of a forward pass analysis by asking the agent to
/// score it on a 0.0–1.0 scale according to the goal's loss_prompt criteria.
/// No tools are used — this is a pure text evaluation.
pub async fn compute_loss(
    agent: &dyn Agent,
    goal: &Goal,
    forward_result: &ForwardResult,
) -> Result<Score> {
    let system = format!(
        "You are an evaluator. Score the quality of a data analysis on a scale of 0.0 to 1.0.\n\n\
         Evaluation criteria:\n{}\n\n\
         Respond with EXACTLY this JSON format:\n\
         {{\"value\": <score>, \"reasoning\": \"<your reasoning>\"}}\n\n\
         The score must be a decimal between 0.0 and 1.0.",
        goal.loss_prompt
    );

    let mut content = format!("## Analysis Result\n\n{}", forward_result.analysis);
    if let Some(ref output) = forward_result.script_output {
        content.push_str(&format!("\n\n## Script Output\n\n{}", output));
    }

    let message = Message {
        role: Role::User,
        content: vec![ContentBlock::Text { text: content }],
    };

    let response = agent.chat(&[message], &[], &system).await?;

    let text = response
        .content
        .iter()
        .filter_map(|b| {
            if let ContentBlock::Text { text } = b {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("");

    let score = parse_score_from_text(&text)?;
    let value = score.value.clamp(0.0, 1.0);

    Ok(Score {
        value,
        reasoning: score.reasoning,
    })
}

fn parse_score_from_text(text: &str) -> Result<Score> {
    // Try parsing the entire text as JSON directly
    if let Ok(score) = serde_json::from_str::<Score>(text) {
        return Ok(score);
    }
    // Try extracting JSON block from text
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if let Ok(score) = serde_json::from_str::<Score>(&text[start..=end]) {
                return Ok(score);
            }
        }
    }
    anyhow::bail!("Failed to parse score from response: {text}")
}

/// Traverse messages to find the last run_python tool call and its result.
///
/// Returns (script_path, script_output) extracted from the message history.
/// script_path is the filename from the tool input (defaults to "script.py").
/// script_output is the content of the corresponding ToolResult.
fn extract_script_info(messages: &[Message]) -> (Option<String>, Option<String>) {
    // Find the last run_python ToolUse by scanning in reverse
    let mut last_tool_use: Option<(String, String)> = None; // (id, filename)

    for msg in messages.iter().rev() {
        for block in msg.content.iter().rev() {
            if let ContentBlock::ToolUse { id, name, input } = block {
                if name == "run_python" {
                    let filename = input["filename"]
                        .as_str()
                        .unwrap_or("script.py")
                        .to_string();
                    last_tool_use = Some((id.clone(), filename));
                    break;
                }
            }
        }
        if last_tool_use.is_some() {
            break;
        }
    }

    let (tool_use_id, filename) = match last_tool_use {
        Some(v) => v,
        None => return (None, None),
    };

    // Find the corresponding ToolResult
    for msg in messages {
        for block in &msg.content {
            if let ContentBlock::ToolResult {
                tool_use_id: id,
                content,
                ..
            } = block
            {
                if id == &tool_use_id {
                    return (Some(filename), Some(content.clone()));
                }
            }
        }
    }

    // ToolUse found but no matching result (shouldn't happen in normal flow)
    (Some(filename), None)
}

/// Check whether iteration should stop.
///
/// Returns `true` if the loop should terminate based on iteration count,
/// score convergence threshold, or patience-based convergence detection.
pub fn check_stop(config: &IterationConfig, iteration: u32, scores: &[Score]) -> bool {
    // Have not reached minimum iterations
    if iteration < config.min_iterations {
        return false;
    }

    // Reached maximum iterations
    if iteration >= config.max_iterations {
        tracing::info!("reached max iterations ({})", config.max_iterations);
        return true;
    }

    // Check if score has reached threshold
    if let Some(last) = scores.last() {
        if last.value >= config.convergence_threshold {
            tracing::info!(
                score = last.value,
                threshold = config.convergence_threshold,
                "convergence threshold reached"
            );
            return true;
        }
    }

    // Check patience (consecutive patience adjacent score differences all less than delta)
    if scores.len() > config.patience as usize {
        let recent = &scores[scores.len() - config.patience as usize - 1..];
        let all_small_delta = recent
            .windows(2)
            .all(|w| (w[1].value - w[0].value).abs() < config.convergence_delta);
        if all_small_delta {
            tracing::info!(patience = config.patience, "converged (small delta)");
            return true;
        }
    }

    false
}

/// Backward Pass result
pub struct BackwardResult {
    pub updated_prompt: String,
    pub script_feedback: String,
    pub messages: Vec<Message>,
}

/// Execute Backward Pass
///
/// Uses the current analysis results, score, and feedback to have the agent
/// optimize the prompt. The agent can use BackwardTools (execute_memory_sql,
/// query_memory) to store findings in the memory database.
pub async fn backward_pass(
    agent: &dyn Agent,
    tools: &BackwardTools,
    state: &State,
    goal: &Goal,
    forward_result: &ForwardResult,
    score: &Score,
) -> Result<BackwardResult> {
    let system = format!(
        "You are an optimization AI. Your job is to improve the data analysis process.\n\n\
         Goal: {}\n\n\
         The current analysis scored {:.2}/1.0.\n\
         Reasoning: {}\n\n\
         Your tasks:\n\
         1. Store useful findings in the memory database (execute_memory_sql)\n\
         2. Review the memory for patterns (query_memory)\n\
         3. Provide an improved prompt for the next iteration\n\
         4. Provide feedback on the Python script (if any)\n\n\
         Respond with EXACTLY this JSON at the end:\n\
         {{\"updated_prompt\": \"<improved prompt>\", \"script_feedback\": \"<feedback for script improvement>\"}}",
        goal.description, score.value, score.reasoning
    );

    let user_content = format!(
        "## Current State\n\
         Iteration: {}\n\
         Current prompt: {}\n\n\
         ## Analysis Result\n{}\n\n\
         ## Score\n{:.2} - {}\n\n\
         Please optimize and provide the updated prompt.",
        state.iteration, state.prompt, forward_result.analysis, score.value, score.reasoning
    );

    let initial_message = Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: user_content,
        }],
    };

    let (messages, final_text) = run_agent_loop(
        agent,
        tools as &dyn crate::agent::ToolExecutor,
        vec![initial_message],
        &system,
    )
    .await?;

    let (updated_prompt, script_feedback) =
        parse_backward_result(&final_text, &state.prompt);

    Ok(BackwardResult {
        updated_prompt,
        script_feedback,
        messages,
    })
}

/// Parse backward pass JSON result from agent text.
///
/// Extracts `{"updated_prompt": "...", "script_feedback": "..."}` from the
/// agent's response. On parse failure, falls back to the original prompt and
/// empty feedback.
fn parse_backward_result(text: &str, fallback_prompt: &str) -> (String, String) {
    // Try the entire text as JSON
    if let Some(result) = try_parse_backward_json(text) {
        return result;
    }
    // Try extracting JSON object from surrounding text
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if let Some(result) = try_parse_backward_json(&text[start..=end]) {
                return result;
            }
        }
    }
    // Fallback
    (fallback_prompt.to_string(), String::new())
}

fn try_parse_backward_json(text: &str) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let prompt = v.get("updated_prompt")?.as_str()?;
    if prompt.is_empty() {
        return None;
    }
    let feedback = v
        .get("script_feedback")
        .and_then(|f| f.as_str())
        .unwrap_or("");
    Some((prompt.to_string(), feedback.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_forward_system_prompt_basic() {
        let state = State {
            prompt: "Analyze sales data".into(),
            iteration: 1,
            last_script: None,
            last_output: None,
            script_feedback: vec![],
        };
        let goal = Goal {
            description: "Find top-selling products".into(),
            loss_prompt: "Evaluate analysis quality".into(),
            data_dsn: "mysql://localhost/test".into(),
            data_files: vec![],
        };

        let system = build_forward_system_prompt(&state, &goal);
        assert!(system.contains("Find top-selling products"));
        assert!(system.contains("query_data_db"));
        assert!(system.contains("read_file"));
        assert!(system.contains("query_memory"));
        assert!(system.contains("run_python"));
        assert!(!system.contains("Previous feedback"));
        assert!(!system.contains("Previous script output"));
    }

    #[test]
    fn test_build_forward_system_prompt_with_feedback() {
        let state = State {
            prompt: "Analyze sales data".into(),
            iteration: 2,
            last_script: Some("script.py".into()),
            last_output: Some("Total sales: 1000".into()),
            script_feedback: vec!["Need more detail".into()],
        };
        let goal = Goal {
            description: "Find top-selling products".into(),
            loss_prompt: "Evaluate analysis quality".into(),
            data_dsn: "mysql://localhost/test".into(),
            data_files: vec![],
        };

        let system = build_forward_system_prompt(&state, &goal);
        assert!(system.contains("Previous feedback:\nNeed more detail"));
        assert!(system.contains("Previous script output:\nTotal sales: 1000"));
    }

    #[test]
    fn test_extract_script_info_no_python() {
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "world".into(),
                }],
            },
        ];
        let (path, output) = extract_script_info(&messages);
        assert!(path.is_none());
        assert!(output.is_none());
    }

    #[test]
    fn test_extract_script_info_with_python() {
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "analyze data".into(),
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "run_python".into(),
                    input: serde_json::json!({
                        "code": "print('hello')",
                        "filename": "analysis.py"
                    }),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "--- stdout ---\nhello\n".into(),
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "Done".into(),
                }],
            },
        ];

        let (path, output) = extract_script_info(&messages);
        assert_eq!(path.as_deref(), Some("analysis.py"));
        assert_eq!(output.as_deref(), Some("--- stdout ---\nhello\n"));
    }

    #[test]
    fn test_extract_script_info_default_filename() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "run_python".into(),
                    input: serde_json::json!({
                        "code": "print('hi')"
                    }),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "--- stdout ---\nhi\n".into(),
                    is_error: false,
                }],
            },
        ];

        let (path, output) = extract_script_info(&messages);
        assert_eq!(path.as_deref(), Some("script.py"));
        assert_eq!(output.as_deref(), Some("--- stdout ---\nhi\n"));
    }

    #[test]
    fn test_extract_script_info_multiple_python_calls() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "run_python".into(),
                    input: serde_json::json!({
                        "code": "print('first')",
                        "filename": "first.py"
                    }),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "first output".into(),
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_2".into(),
                    name: "run_python".into(),
                    input: serde_json::json!({
                        "code": "print('second')",
                        "filename": "second.py"
                    }),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_2".into(),
                    content: "second output".into(),
                    is_error: false,
                }],
            },
        ];

        let (path, output) = extract_script_info(&messages);
        assert_eq!(path.as_deref(), Some("second.py"));
        assert_eq!(output.as_deref(), Some("second output"));
    }

    #[test]
    fn test_extract_script_info_mixed_tools() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "query_data_db".into(),
                    input: serde_json::json!({"sql": "SELECT 1"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "[{\"1\":1}]".into(),
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_2".into(),
                    name: "run_python".into(),
                    input: serde_json::json!({
                        "code": "print('result')",
                        "filename": "compute.py"
                    }),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_2".into(),
                    content: "--- stdout ---\nresult\n".into(),
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_3".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "data.csv"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_3".into(),
                    content: "a,b\n1,2".into(),
                    is_error: false,
                }],
            },
        ];

        // Should find the run_python call, not the read_file call
        let (path, output) = extract_script_info(&messages);
        assert_eq!(path.as_deref(), Some("compute.py"));
        assert_eq!(output.as_deref(), Some("--- stdout ---\nresult\n"));
    }

    #[test]
    fn test_parse_score_pure_json() {
        let input = r#"{"value": 0.85, "reasoning": "Good analysis"}"#;
        let score = parse_score_from_text(input).unwrap();
        assert!((score.value - 0.85).abs() < f64::EPSILON);
        assert_eq!(score.reasoning, "Good analysis");
    }

    #[test]
    fn test_parse_score_json_in_text() {
        let input = r#"Here is my evaluation: {"value": 0.42, "reasoning": "Needs more detail"} That's it."#;
        let score = parse_score_from_text(input).unwrap();
        assert!((score.value - 0.42).abs() < f64::EPSILON);
        assert_eq!(score.reasoning, "Needs more detail");
    }

    #[test]
    fn test_parse_score_invalid_input() {
        let input = "This is not valid JSON at all";
        assert!(parse_score_from_text(input).is_err());
    }

    #[test]
    fn test_parse_score_incomplete_json() {
        let input = r#"{"value": 0.5}"#;
        assert!(parse_score_from_text(input).is_err());
    }

    #[test]
    fn test_parse_score_with_markdown_code_block() {
        let input = "```json\n{\"value\": 0.73, \"reasoning\": \"Solid work\"}\n```";
        let score = parse_score_from_text(input).unwrap();
        assert!((score.value - 0.73).abs() < f64::EPSILON);
        assert_eq!(score.reasoning, "Solid work");
    }

    fn make_config(
        min: u32,
        max: u32,
        threshold: f64,
        patience: u32,
        delta: f64,
    ) -> IterationConfig {
        IterationConfig {
            min_iterations: min,
            max_iterations: max,
            convergence_threshold: threshold,
            patience,
            convergence_delta: delta,
        }
    }

    fn make_scores(values: &[f64]) -> Vec<Score> {
        values
            .iter()
            .map(|&v| Score {
                value: v,
                reasoning: String::new(),
            })
            .collect()
    }

    #[test]
    fn test_check_stop_below_min_iterations() {
        let config = make_config(3, 10, 0.9, 3, 0.01);
        // iteration 1 and 2 are below min_iterations=3, should not stop
        assert!(!check_stop(&config, 1, &make_scores(&[0.95])));
        assert!(!check_stop(&config, 2, &make_scores(&[0.95, 0.99])));
    }

    #[test]
    fn test_check_stop_max_iterations() {
        let config = make_config(1, 5, 0.9, 3, 0.01);
        // iteration >= max_iterations should stop
        assert!(check_stop(&config, 5, &make_scores(&[0.1, 0.2, 0.3, 0.4, 0.5])));
        assert!(check_stop(&config, 10, &make_scores(&[0.1])));
    }

    #[test]
    fn test_check_stop_convergence_threshold() {
        let config = make_config(1, 10, 0.9, 3, 0.01);
        // last score >= threshold should stop
        assert!(check_stop(&config, 2, &make_scores(&[0.5, 0.9])));
        assert!(check_stop(&config, 3, &make_scores(&[0.5, 0.8, 0.95])));
        // last score below threshold should not stop
        assert!(!check_stop(&config, 2, &make_scores(&[0.5, 0.89])));
    }

    #[test]
    fn test_check_stop_patience_convergence() {
        // patience=3, delta=0.01 -> need 4 scores where last 3 diffs are all < 0.01
        let config = make_config(1, 10, 0.9, 3, 0.01);
        let scores = make_scores(&[0.3, 0.5, 0.505, 0.508, 0.509]);
        // Last 4 scores: 0.5, 0.505, 0.508, 0.509 -> diffs: 0.005, 0.003, 0.001 all < 0.01
        assert!(check_stop(&config, 5, &scores));
    }

    #[test]
    fn test_check_stop_patience_not_converged() {
        let config = make_config(1, 10, 0.9, 3, 0.01);
        // diffs include one >= 0.01
        let scores = make_scores(&[0.3, 0.5, 0.52, 0.525, 0.526]);
        // Last 4: 0.5, 0.52, 0.525, 0.526 -> diffs: 0.02, 0.005, 0.001 -> first diff >= 0.01
        assert!(!check_stop(&config, 5, &scores));
    }

    #[test]
    fn test_check_stop_patience_not_enough_scores() {
        let config = make_config(1, 10, 0.9, 3, 0.01);
        // Need patience+1=4 scores, only have 3
        let scores = make_scores(&[0.5, 0.505, 0.508]);
        assert!(!check_stop(&config, 3, &scores));
    }

    #[test]
    fn test_check_stop_normal_case() {
        let config = make_config(1, 10, 0.9, 3, 0.01);
        // Not at max, not at threshold, not converged by patience
        let scores = make_scores(&[0.3, 0.5, 0.65]);
        assert!(!check_stop(&config, 3, &scores));
    }

    #[test]
    fn test_check_stop_empty_scores() {
        let config = make_config(1, 10, 0.9, 3, 0.01);
        assert!(!check_stop(&config, 1, &[]));
    }

    #[test]
    fn test_check_stop_exact_threshold() {
        let config = make_config(1, 10, 0.9, 3, 0.01);
        // score == threshold should stop (>=)
        assert!(check_stop(&config, 1, &make_scores(&[0.9])));
    }

    #[test]
    fn test_parse_backward_result_pure_json() {
        let input = r#"{"updated_prompt": "Analyze sales by region", "script_feedback": "Add error bars"}"#;
        let (prompt, feedback) = parse_backward_result(input, "fallback");
        assert_eq!(prompt, "Analyze sales by region");
        assert_eq!(feedback, "Add error bars");
    }

    #[test]
    fn test_parse_backward_result_json_in_text() {
        let input = r#"Here is the optimized result: {"updated_prompt": "Better prompt", "script_feedback": "Use pandas"} Done."#;
        let (prompt, feedback) = parse_backward_result(input, "fallback");
        assert_eq!(prompt, "Better prompt");
        assert_eq!(feedback, "Use pandas");
    }

    #[test]
    fn test_parse_backward_result_missing_feedback() {
        let input = r#"{"updated_prompt": "New prompt"}"#;
        let (prompt, feedback) = parse_backward_result(input, "fallback");
        assert_eq!(prompt, "New prompt");
        assert_eq!(feedback, "");
    }

    #[test]
    fn test_parse_backward_result_invalid_json() {
        let input = "This is not valid JSON";
        let (prompt, feedback) = parse_backward_result(input, "original prompt");
        assert_eq!(prompt, "original prompt");
        assert_eq!(feedback, "");
    }

    #[test]
    fn test_parse_backward_result_empty_prompt_uses_fallback() {
        let input = r#"{"updated_prompt": "", "script_feedback": "some feedback"}"#;
        let (prompt, feedback) = parse_backward_result(input, "fallback");
        assert_eq!(prompt, "fallback");
        assert_eq!(feedback, "");
    }

    #[test]
    fn test_parse_backward_result_with_markdown_code_block() {
        let input = "```json\n{\"updated_prompt\": \"Improved analysis\", \"script_feedback\": \"Add charts\"}\n```";
        let (prompt, feedback) = parse_backward_result(input, "fallback");
        assert_eq!(prompt, "Improved analysis");
        assert_eq!(feedback, "Add charts");
    }
}
