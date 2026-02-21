use anyhow::Result;
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};

use rethinking::agent::*;
use rethinking::config::*;
use rethinking::iteration::*;
use rethinking::script;
use rethinking::types::*;

// ---------------------------------------------------------------------------
// MockAgent
// ---------------------------------------------------------------------------

struct MockAgent {
    responses: Vec<Message>,
    call_count: AtomicUsize,
}

impl MockAgent {
    fn new(responses: Vec<Message>) -> Self {
        Self {
            responses,
            call_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Agent for MockAgent {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _system: &str,
    ) -> Result<Message> {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(self.responses[idx % self.responses.len()].clone())
    }
}

// ---------------------------------------------------------------------------
// MockToolExecutor
// ---------------------------------------------------------------------------

struct MockToolExecutor {
    tools: Vec<ToolDefinition>,
}

impl MockToolExecutor {
    fn empty() -> Self {
        Self { tools: vec![] }
    }
}

#[async_trait]
impl ToolExecutor for MockToolExecutor {
    fn available_tools(&self) -> Vec<ToolDefinition> {
        self.tools.clone()
    }

    async fn execute(&self, tool_name: &str, _input: &serde_json::Value) -> Result<String> {
        anyhow::bail!("mock executor: unexpected tool call: {tool_name}")
    }
}

// ---------------------------------------------------------------------------
// Test: Config parsing
// ---------------------------------------------------------------------------

#[test]
fn test_config_parsing() {
    let toml_str = r#"
[goal]
description = "Analyze sales data"
loss_prompt = "Evaluate completeness"
data_dsn = "mysql://localhost:4000/sales"
data_files = ["sales.csv"]
initial_prompt = "Start here"

[agent]
command = "claude-code"
model = "sonnet"

[memory]
dsn = "mysql://user:pass@localhost:3306/mem"
tag = "test"
size_limit_mb = 128

[iteration]
min_iterations = 2
max_iterations = 15
convergence_threshold = 0.95
patience = 4
convergence_delta = 0.005

[output]
work_dir = "/tmp/work"
result_file = "out.json"
"#;

    let config: Config = toml::from_str(toml_str).unwrap();

    assert_eq!(config.goal.description, "Analyze sales data");
    assert_eq!(config.goal.loss_prompt, "Evaluate completeness");
    assert_eq!(config.goal.data_dsn, "mysql://localhost:4000/sales");
    assert_eq!(config.goal.data_files, vec!["sales.csv"]);
    assert_eq!(config.goal.initial_prompt, "Start here");
    assert_eq!(config.agent.command, "claude-code");
    assert_eq!(config.agent.model, "sonnet");
    assert_eq!(config.memory.dsn, "mysql://user:pass@localhost:3306/mem");
    assert_eq!(config.memory.tag, "test");
    assert_eq!(config.memory.size_limit_mb, 128);
    assert_eq!(config.iteration.min_iterations, 2);
    assert_eq!(config.iteration.max_iterations, 15);
    assert_eq!(config.iteration.convergence_threshold, 0.95);
    assert_eq!(config.iteration.patience, 4);
    assert_eq!(config.iteration.convergence_delta, 0.005);
    assert_eq!(config.output.work_dir, "/tmp/work");
    assert_eq!(config.output.result_file, "out.json");
}

#[test]
fn test_config_defaults() {
    let toml_str = r#"
[goal]
description = "d"
loss_prompt = "l"
data_dsn = "mysql://localhost/db"

[agent]
[memory]
[iteration]
"#;

    let config: Config = toml::from_str(toml_str).unwrap();

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
}

// ---------------------------------------------------------------------------
// Test: Score parsing
// ---------------------------------------------------------------------------

#[test]
fn test_score_parsing_pure_json() {
    let json = r#"{"value": 0.85, "reasoning": "Thorough analysis"}"#;
    let score: Score = serde_json::from_str(json).unwrap();
    assert!((score.value - 0.85).abs() < f64::EPSILON);
    assert_eq!(score.reasoning, "Thorough analysis");
}

#[test]
fn test_score_parsing_embedded_in_text() {
    // Score embedded in surrounding text — parse the JSON substring
    let text = r#"My evaluation: {"value": 0.72, "reasoning": "Decent"} end."#;
    let start = text.find('{').unwrap();
    let end = text.rfind('}').unwrap();
    let score: Score = serde_json::from_str(&text[start..=end]).unwrap();
    assert!((score.value - 0.72).abs() < f64::EPSILON);
    assert_eq!(score.reasoning, "Decent");
}

#[test]
fn test_score_parsing_invalid() {
    let bad = "not json at all";
    assert!(serde_json::from_str::<Score>(bad).is_err());
}

#[test]
fn test_score_display() {
    let score = Score {
        value: 0.87,
        reasoning: "Good work".into(),
    };
    let display = format!("{score}");
    assert!(display.contains("0.87"));
    assert!(display.contains("Good work"));
}

// ---------------------------------------------------------------------------
// Test: Stop check — min iterations
// ---------------------------------------------------------------------------

fn make_iter_config(
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
fn test_stop_check_min_iterations() {
    let config = make_iter_config(3, 10, 0.9, 3, 0.01);
    // Even with a perfect score, should NOT stop before min_iterations
    assert!(!check_stop(&config, 1, &make_scores(&[1.0])));
    assert!(!check_stop(&config, 2, &make_scores(&[1.0, 1.0])));
}

// ---------------------------------------------------------------------------
// Test: Stop check — max iterations
// ---------------------------------------------------------------------------

#[test]
fn test_stop_check_max_iterations() {
    let config = make_iter_config(1, 5, 0.9, 3, 0.01);
    assert!(check_stop(
        &config,
        5,
        &make_scores(&[0.1, 0.2, 0.3, 0.4, 0.5])
    ));
    assert!(check_stop(&config, 10, &make_scores(&[0.1])));
}

// ---------------------------------------------------------------------------
// Test: Stop check — convergence threshold
// ---------------------------------------------------------------------------

#[test]
fn test_stop_check_convergence() {
    let config = make_iter_config(1, 10, 0.9, 3, 0.01);
    // Score >= threshold should stop (iteration >= min)
    assert!(check_stop(&config, 1, &make_scores(&[0.9])));
    assert!(check_stop(&config, 2, &make_scores(&[0.5, 0.95])));
    // Score below threshold should NOT stop
    assert!(!check_stop(&config, 2, &make_scores(&[0.5, 0.89])));
}

// ---------------------------------------------------------------------------
// Test: Stop check — patience convergence
// ---------------------------------------------------------------------------

#[test]
fn test_stop_check_patience() {
    let config = make_iter_config(1, 10, 0.99, 3, 0.01);
    // patience=3 needs 4 scores where last 3 diffs are all < 0.01
    let scores = make_scores(&[0.3, 0.5, 0.505, 0.508, 0.509]);
    assert!(check_stop(&config, 5, &scores));

    // Large jump in middle: should NOT converge by patience
    let scores2 = make_scores(&[0.3, 0.5, 0.52, 0.525, 0.526]);
    assert!(!check_stop(&config, 5, &scores2));
}

// ---------------------------------------------------------------------------
// Test: Message serialization round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_message_serialization() {
    let msg = Message {
        role: Role::Assistant,
        content: vec![
            ContentBlock::Text {
                text: "Hello".into(),
            },
            ContentBlock::ToolUse {
                id: "t1".into(),
                name: "query_memory".into(),
                input: serde_json::json!({"sql": "SELECT 1"}),
            },
            ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: "[{\"1\":1}]".into(),
                is_error: false,
            },
        ],
    };

    let json = serde_json::to_string(&msg).unwrap();
    let deser: Message = serde_json::from_str(&json).unwrap();

    assert_eq!(deser.role, Role::Assistant);
    assert_eq!(deser.content.len(), 3);

    // Verify Text block
    match &deser.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "Hello"),
        other => panic!("expected Text, got {other:?}"),
    }
    // Verify ToolUse block
    match &deser.content[1] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "t1");
            assert_eq!(name, "query_memory");
            assert_eq!(input["sql"], "SELECT 1");
        }
        other => panic!("expected ToolUse, got {other:?}"),
    }
    // Verify ToolResult block
    match &deser.content[2] {
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            assert_eq!(tool_use_id, "t1");
            assert_eq!(content, "[{\"1\":1}]");
            assert!(!is_error);
        }
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test: Python script execution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_run_python_script() {
    let tmp = tempfile::tempdir().unwrap();
    let work_dir = tmp.path().to_str().unwrap();

    let result = script::run_python("print('hello from integration test')", None, work_dir)
        .await
        .unwrap();

    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("hello from integration test"));
    assert!(result.stderr.is_empty());
    assert!(result.script_path.ends_with("script.py"));
}

#[tokio::test]
async fn test_run_python_script_with_error() {
    let tmp = tempfile::tempdir().unwrap();
    let work_dir = tmp.path().to_str().unwrap();

    let result = script::run_python("raise ValueError('boom')", None, work_dir)
        .await
        .unwrap();

    assert_eq!(result.exit_code, 1);
    assert!(result.stderr.contains("ValueError"));
    assert!(result.stderr.contains("boom"));
}

// ---------------------------------------------------------------------------
// Test: Tool definitions produce valid JSON
// ---------------------------------------------------------------------------

#[test]
fn test_tool_definitions_valid_json() {
    // ForwardTools and BackwardTools tool definitions should have valid schemas
    let forward_defs = rethinking::tools::ForwardTools::tool_definitions();
    let backward_defs = rethinking::tools::BackwardTools::tool_definitions();

    for tool in forward_defs.iter().chain(backward_defs.iter()) {
        // Name should be non-empty
        assert!(!tool.name.is_empty(), "tool name should not be empty");
        // Description should be non-empty
        assert!(
            !tool.description.is_empty(),
            "tool description should not be empty for {}",
            tool.name
        );
        // input_schema should be a valid JSON object
        assert!(
            tool.input_schema.is_object(),
            "input_schema should be an object for {}",
            tool.name
        );
        // Should have "type": "object"
        assert_eq!(
            tool.input_schema["type"], "object",
            "input_schema.type should be 'object' for {}",
            tool.name
        );
        // Should have "properties"
        assert!(
            tool.input_schema["properties"].is_object(),
            "input_schema should have 'properties' for {}",
            tool.name
        );
        // Should have "required" as array
        assert!(
            tool.input_schema["required"].is_array(),
            "input_schema should have 'required' array for {}",
            tool.name
        );
        // The whole schema should round-trip through JSON
        let json_str = serde_json::to_string(&tool.input_schema).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed, tool.input_schema);
    }
}

// ---------------------------------------------------------------------------
// Test: run_agent_loop with MockAgent
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_agent_loop_simple() {
    // MockAgent that returns a simple text response (no tool calls)
    let agent = MockAgent::new(vec![Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text {
            text: "Analysis complete.".into(),
        }],
    }]);

    let executor = MockToolExecutor::empty();
    let initial = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "Analyze data".into(),
        }],
    }];

    let (messages, final_text) =
        run_agent_loop(&agent, &executor, initial, "You are a test agent.").await.unwrap();

    assert_eq!(final_text, "Analysis complete.");
    // Should have 2 messages: initial user + assistant response
    assert_eq!(messages.len(), 2);
}

// ---------------------------------------------------------------------------
// Test: RunResult display
// ---------------------------------------------------------------------------

#[test]
fn test_run_result_display() {
    let result = RunResult {
        goal: Goal {
            description: "Test goal".into(),
            loss_prompt: "Evaluate".into(),
            data_dsn: "mysql://localhost/db".into(),
            data_files: vec![],
        },
        iterations: vec![IterationResult {
            iteration: 1,
            score: Score {
                value: 0.85,
                reasoning: "Good".into(),
            },
            script_path: Some("analysis.py".into()),
            script_output: Some("output".into()),
            updated_prompt: "Better prompt".into(),
            converged: true,
        }],
        final_prompt: "Better prompt".into(),
        final_score: Score {
            value: 0.85,
            reasoning: "Good".into(),
        },
    };

    let display = format!("{result}");
    assert!(display.contains("Rethinking Analysis Results"));
    assert!(display.contains("Test goal"));
    assert!(display.contains("0.85"));
    assert!(display.contains("Iteration 1"));
    assert!(display.contains("analysis.py"));
    assert!(display.contains("Better prompt"));
}

// ---------------------------------------------------------------------------
// Test: RunResult JSON serialization
// ---------------------------------------------------------------------------

#[test]
fn test_run_result_serialization() {
    let result = RunResult {
        goal: Goal {
            description: "Test".into(),
            loss_prompt: "Eval".into(),
            data_dsn: "mysql://localhost/db".into(),
            data_files: vec!["data.csv".into()],
        },
        iterations: vec![],
        final_prompt: "prompt".into(),
        final_score: Score {
            value: 0.5,
            reasoning: "ok".into(),
        },
    };

    let json = serde_json::to_string_pretty(&result).unwrap();
    let deser: RunResult = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.goal.description, "Test");
    assert_eq!(deser.final_score.value, 0.5);
    assert_eq!(deser.final_prompt, "prompt");
}

// ---------------------------------------------------------------------------
// Test: Full iteration with MockAgent (ignored — requires careful mock setup)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore] // Requires external services or more elaborate mocking
async fn test_full_iteration_with_mock() {
    // This test simulates a complete iteration loop using MockAgent.
    // In a real environment, ForwardTools/BackwardTools need a MemoryDB connection.
    // For now, this serves as a compile-check and can be extended when a test DB is available.

    let forward_response = Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text {
            text: "Analysis: The data shows sales trends increasing over Q3.".into(),
        }],
    };

    let loss_response = Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text {
            text: r#"{"value": 0.92, "reasoning": "Comprehensive analysis with clear trends"}"#
                .into(),
        }],
    };

    let backward_response = Message {
        role: Role::Assistant,
        content: vec![ContentBlock::Text {
            text: r#"{"updated_prompt": "Analyze sales data with focus on regional breakdown", "script_feedback": "Add visualization for trends"}"#.into(),
        }],
    };

    // The iteration loop calls: forward_pass (1 chat), compute_loss (1 chat), backward_pass (1 chat)
    // With score >= threshold (0.9), iteration 2 would converge.
    // But min_iterations=1, so first iteration: forward + loss + (score 0.92 >= 0.9 → converge)
    // Converged: no backward pass needed. Total calls = 2 (forward + loss).
    let _agent = MockAgent::new(vec![forward_response, loss_response, backward_response]);

    let config = IterationConfig {
        min_iterations: 1,
        max_iterations: 5,
        convergence_threshold: 0.9,
        patience: 3,
        convergence_delta: 0.01,
    };

    let _goal = Goal {
        description: "Analyze sales data".into(),
        loss_prompt: "Evaluate completeness and accuracy".into(),
        data_dsn: "mysql://localhost/test".into(),
        data_files: vec![],
    };

    // NOTE: Cannot call run_iterations without real ForwardTools/BackwardTools (need MemoryDB).
    // This test validates that mock setup compiles and types align.
    // To run a real iteration loop, provide a MySQL/TiDB connection via environment.
    let _ = config;
}
