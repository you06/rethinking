use anyhow::Result;
use async_trait::async_trait;
use tokio::io::AsyncWriteExt;

use crate::agent::{Agent, ContentBlock, Message, Role, ToolDefinition};

pub struct SubprocessAgent {
    pub command: String,
    pub model: String,
    pub config_path: String,
    pub work_dir: String,
}

impl SubprocessAgent {
    pub fn new(
        command: String,
        model: String,
        config_path: String,
        work_dir: String,
    ) -> Result<Self> {
        let agent = Self {
            command,
            model,
            config_path,
            work_dir,
        };
        agent.setup_mcp_config()?;
        Ok(agent)
    }

    /// Write MCP server config so the CLI tool can discover and spawn the rethinking MCP server.
    /// The config tells the CLI tool to run `<current_exe> --mcp-server -c <config_path>`.
    fn setup_mcp_config(&self) -> Result<()> {
        let exe = std::env::current_exe()
            .unwrap_or_else(|_| std::path::PathBuf::from("rethinking"))
            .to_string_lossy()
            .to_string();

        if self.command.contains("codex") {
            // Codex reads MCP config from .codex/config.toml in project dir
            let codex_dir = std::path::Path::new(&self.work_dir).join(".codex");
            std::fs::create_dir_all(&codex_dir)?;
            let config = format!(
                "[mcp_servers.rethinking]\ncommand = \"{exe}\"\nargs = [\"--mcp-server\", \"-c\", \"{}\"]\n",
                self.config_path
            );
            std::fs::write(codex_dir.join("config.toml"), config)?;
        } else {
            // Claude Code reads MCP config from .mcp.json in project dir
            let mcp_config = serde_json::json!({
                "mcpServers": {
                    "rethinking": {
                        "command": exe,
                        "args": ["--mcp-server", "-c", &self.config_path]
                    }
                }
            });
            std::fs::write(
                std::path::Path::new(&self.work_dir).join(".mcp.json"),
                serde_json::to_string_pretty(&mcp_config)?,
            )?;
        }

        tracing::info!(command = %self.command, "MCP server config written to work_dir");
        Ok(())
    }

    /// Build CLI arguments based on the command type.
    fn build_args(&self) -> Vec<String> {
        if self.command.contains("codex") {
            vec![
                "--model".into(),
                self.model.clone(),
                "--full-auto".into(),
                "-a".into(),
                "never".into(),
                "--cd".into(),
                self.work_dir.clone(),
                "exec".into(),
                "-".into(),
            ]
        } else {
            // claude-code style
            vec![
                "-p".into(),
                "-".into(),
                "--model".into(),
                self.model.clone(),
                "--quiet".into(),
                "--full-auto".into(),
            ]
        }
    }
}

#[async_trait]
impl Agent for SubprocessAgent {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        system: &str,
    ) -> Result<Message> {
        // Concatenate messages into a text prompt
        let mut prompt = String::new();
        if !system.is_empty() {
            prompt.push_str(system);
            prompt.push_str("\n\n");
        }
        for msg in messages {
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    prompt.push_str(text);
                    prompt.push('\n');
                }
            }
        }

        let args = self.build_args();
        tracing::debug!(command = %self.command, ?args, "spawning subprocess");

        let mut cmd = tokio::process::Command::new(&self.command);
        cmd.args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // For claude-code, set current_dir to work_dir so it discovers .mcp.json
        if !self.command.contains("codex") {
            cmd.current_dir(&self.work_dir);
        }

        let mut child = cmd.spawn()?;

        // Write prompt to stdin then close it
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
            // stdin is dropped here, closing the pipe
        }

        let output = child.wait_with_output().await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} CLI failed: {stderr}", self.command);
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();

        Ok(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: stdout }],
        })
    }
}
