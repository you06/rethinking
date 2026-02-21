use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Messages sent/received by the Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Agent trait - core abstraction
#[async_trait]
pub trait Agent: Send + Sync {
    /// Send messages and get a response (may include tool_use)
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system: &str,
    ) -> Result<Message>;
}

/// Tool executor trait
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool call, return result string
    async fn execute(&self, tool_name: &str, input: &serde_json::Value) -> Result<String>;

    /// Get list of available tools
    fn available_tools(&self) -> Vec<ToolDefinition>;
}

/// Complete conversation with Agent including tool loop.
///
/// Continuously calls agent.chat(), executes tools on tool_use, and continues
/// conversation until the agent returns plain text (no tool calls).
pub async fn run_agent_loop(
    agent: &dyn Agent,
    executor: &dyn ToolExecutor,
    initial_messages: Vec<Message>,
    system: &str,
) -> Result<(Vec<Message>, String)> {
    let tools = executor.available_tools();
    let mut messages = initial_messages;

    loop {
        let response = agent.chat(&messages, &tools, system).await?;
        messages.push(response.clone());

        // Check for tool_use
        let tool_uses: Vec<_> = response
            .content
            .iter()
            .filter_map(|b| {
                if let ContentBlock::ToolUse { id, name, input } = b {
                    Some((id.clone(), name.clone(), input.clone()))
                } else {
                    None
                }
            })
            .collect();

        if tool_uses.is_empty() {
            // Extract final text
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
                .join("\n");
            return Ok((messages, text));
        }

        // Execute tools and build tool_result messages
        let mut results = Vec::new();
        for (id, name, input) in &tool_uses {
            match executor.execute(name, input).await {
                Ok(content) => results.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content,
                    is_error: false,
                }),
                Err(e) => results.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: format!("Error: {e}"),
                    is_error: true,
                }),
            }
        }

        messages.push(Message {
            role: Role::User,
            content: results,
        });
    }
}
