use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::{Column, Row};
use std::sync::Arc;

use crate::memory::MemoryDB;
use crate::script;

// --- Parameter types ---

#[derive(Deserialize, Serialize, JsonSchema)]
struct SqlParams {
    /// SQL query or statement
    sql: String,
}

#[derive(Deserialize, Serialize, JsonSchema)]
struct RunPythonParams {
    /// Python script content
    code: String,
    /// Script filename (optional, defaults to script.py)
    filename: Option<String>,
}

#[derive(Deserialize, Serialize, JsonSchema)]
struct ReadFileParams {
    /// File path relative to working directory
    path: String,
}

// --- Server ---

#[derive(Clone)]
pub struct RethinkingMcpServer {
    pub memory_db: Arc<MemoryDB>,
    pub data_dsn: String,
    pub work_dir: String,
    tool_router: ToolRouter<Self>,
}

impl RethinkingMcpServer {
    pub fn new(memory_db: Arc<MemoryDB>, data_dsn: String, work_dir: String) -> Self {
        Self {
            memory_db,
            data_dsn,
            work_dir,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl RethinkingMcpServer {
    #[tool(description = "Execute a read-only SQL query against the memory database")]
    async fn query_memory(
        &self,
        Parameters(params): Parameters<SqlParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        match self.memory_db.query(&params.sql).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("Error: {e}"))])),
        }
    }

    #[tool(description = "Execute a write SQL statement (INSERT/UPDATE/DELETE/CREATE/ALTER) against the memory database")]
    async fn execute_memory_sql(
        &self,
        Parameters(params): Parameters<SqlParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        match self.memory_db.execute(&params.sql).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("Error: {e}"))])),
        }
    }

    #[tool(description = "Execute a read-only SQL query against the data source database")]
    async fn query_data_db(
        &self,
        Parameters(params): Parameters<SqlParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let pool = match sqlx::MySqlPool::connect(&self.data_dsn).await {
            Ok(pool) => pool,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Error connecting to data source: {e}"
                ))]));
            }
        };
        let result = match sqlx::query(&params.sql).fetch_all(&pool).await {
            Ok(rows) => {
                let mut results: Vec<serde_json::Value> = Vec::with_capacity(rows.len());
                for row in &rows {
                    let columns = row.columns();
                    let mut obj = serde_json::Map::new();
                    for (i, col) in columns.iter().enumerate() {
                        let name = col.name().to_string();
                        let value = data_row_value_to_json(row, i);
                        obj.insert(name, value);
                    }
                    results.push(serde_json::Value::Object(obj));
                }
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&results)
                        .unwrap_or_else(|e| format!("Error: {e}")),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("Error: {e}"))])),
        };
        pool.close().await;
        result
    }

    #[tool(description = "Create and run a Python script in the working directory. Returns stdout and stderr.")]
    async fn run_python(
        &self,
        Parameters(params): Parameters<RunPythonParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        match script::run_python(&params.code, params.filename.as_deref(), &self.work_dir).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                result.to_agent_string(),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("Error: {e}"))])),
        }
    }

    #[tool(description = "Read a file from the working directory")]
    async fn read_file(
        &self,
        Parameters(params): Parameters<ReadFileParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let full_path = std::path::Path::new(&self.work_dir).join(&params.path);
        let canonical = match full_path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Error: {e}"
                ))]));
            }
        };
        let work_canonical = match std::path::Path::new(&self.work_dir).canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Error: {e}"
                ))]));
            }
        };
        if !canonical.starts_with(&work_canonical) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Error: path escapes working directory",
            )]));
        }
        match tokio::fs::read_to_string(&canonical).await {
            Ok(content) => Ok(CallToolResult::success(vec![Content::text(content)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("Error: {e}"))])),
        }
    }
}

/// Extract a column value from a MySQL row as a JSON value.
fn data_row_value_to_json(row: &sqlx::mysql::MySqlRow, index: usize) -> serde_json::Value {
    if let Ok(v) = row.try_get::<String, _>(index) {
        return serde_json::Value::String(v);
    }
    if let Ok(v) = row.try_get::<i64, _>(index) {
        return serde_json::Value::Number(v.into());
    }
    if let Ok(v) = row.try_get::<f64, _>(index) {
        if let Some(n) = serde_json::Number::from_f64(v) {
            return serde_json::Value::Number(n);
        }
        return serde_json::Value::String(v.to_string());
    }
    if let Ok(v) = row.try_get::<bool, _>(index) {
        return serde_json::Value::Bool(v);
    }
    serde_json::Value::Null
}

#[tool_handler]
impl ServerHandler for RethinkingMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "rethinking".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                ..Default::default()
            },
            instructions: Some(
                "Rethinking AI data analysis tools. Provides memory database, data source queries, Python script execution, and file access.".into(),
            ),
        }
    }
}

/// Run MCP server (stdio mode)
pub async fn run_mcp_server(
    memory_db: Arc<MemoryDB>,
    data_dsn: String,
    work_dir: String,
) -> Result<()> {
    let server = RethinkingMcpServer::new(memory_db, data_dsn, work_dir);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
