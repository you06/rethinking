use anyhow::Result;

use crate::agent::{run_agent_loop, Agent, ContentBlock, Message, Role};
use crate::tools::ForwardTools;
use crate::types::{Goal, Score, State};

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
}
