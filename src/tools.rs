use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use crate::agent::{ToolDefinition, ToolExecutor};
use crate::memory::MemoryDB;
use crate::script;

/// Forward Pass tool set (read-only data access + Python execution)
pub struct ForwardTools {
    pub data_dsn: String,
    pub memory_db: Arc<MemoryDB>,
    pub work_dir: String,
}

/// Backward Pass tool set (read-write memory database)
pub struct BackwardTools {
    pub memory_db: Arc<MemoryDB>,
}

impl ForwardTools {
    pub fn new(data_dsn: String, memory_db: Arc<MemoryDB>, work_dir: String) -> Self {
        Self {
            data_dsn,
            memory_db,
            work_dir,
        }
    }
}

impl BackwardTools {
    pub fn new(memory_db: Arc<MemoryDB>) -> Self {
        Self { memory_db }
    }
}

// --- Tool Definitions ---

fn tool_query_data_db() -> ToolDefinition {
    ToolDefinition {
        name: "query_data_db".into(),
        description: "Execute a read-only SQL query against the data source database".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "sql": { "type": "string", "description": "SQL SELECT query" }
            },
            "required": ["sql"]
        }),
    }
}

fn tool_read_file() -> ToolDefinition {
    ToolDefinition {
        name: "read_file".into(),
        description: "Read a data file from the working directory".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path relative to work dir" }
            },
            "required": ["path"]
        }),
    }
}

fn tool_query_memory() -> ToolDefinition {
    ToolDefinition {
        name: "query_memory".into(),
        description: "Execute a read-only SQL query against the memory database".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "sql": { "type": "string", "description": "SQL SELECT query" }
            },
            "required": ["sql"]
        }),
    }
}

fn tool_run_python() -> ToolDefinition {
    ToolDefinition {
        name: "run_python".into(),
        description: "Create and run a Python script. Returns stdout+stderr.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "code": { "type": "string", "description": "Python script content" },
                "filename": { "type": "string", "description": "Script filename (saved in work dir)" }
            },
            "required": ["code"]
        }),
    }
}

fn tool_execute_memory_sql() -> ToolDefinition {
    ToolDefinition {
        name: "execute_memory_sql".into(),
        description: "Execute a write SQL (INSERT/UPDATE/DELETE/CREATE/ALTER) against the memory database".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "sql": { "type": "string", "description": "SQL statement to execute" }
            },
            "required": ["sql"]
        }),
    }
}

// --- Tool Execution Helpers ---

/// Query the data source database using a temporary connection.
async fn exec_query_data_db(data_dsn: &str, sql: &str) -> Result<String> {
    use sqlx::mysql::MySqlPoolOptions;
    use sqlx::{Column, Row};

    let pool = MySqlPoolOptions::new()
        .max_connections(1)
        .connect(data_dsn)
        .await
        .context("failed to connect to data source database")?;

    let rows = sqlx::query(sql)
        .fetch_all(&pool)
        .await
        .context("failed to execute data source query")?;

    let mut results: Vec<serde_json::Value> = Vec::with_capacity(rows.len());
    for row in &rows {
        let columns = row.columns();
        let mut map = serde_json::Map::new();
        for (i, col) in columns.iter().enumerate() {
            let name = col.name().to_string();
            let value = data_row_value_to_json(row, i);
            map.insert(name, value);
        }
        results.push(serde_json::Value::Object(map));
    }

    pool.close().await;
    Ok(serde_json::to_string(&results)?)
}

/// Extract a column value from a MySQL row as a JSON value (for data source queries).
fn data_row_value_to_json(row: &sqlx::mysql::MySqlRow, index: usize) -> serde_json::Value {
    use sqlx::Row;

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

/// Read a file, ensuring the path stays within the working directory.
async fn exec_read_file(work_dir: &str, relative_path: &str) -> Result<String> {
    let work_dir_canon = tokio::fs::canonicalize(work_dir)
        .await
        .context("failed to resolve working directory")?;

    let target = Path::new(work_dir).join(relative_path);
    let target_canon = tokio::fs::canonicalize(&target)
        .await
        .context("failed to resolve file path")?;

    if !target_canon.starts_with(&work_dir_canon) {
        anyhow::bail!("path escapes working directory");
    }

    let content = tokio::fs::read_to_string(&target_canon)
        .await
        .context("failed to read file")?;

    Ok(content)
}

// --- ToolExecutor Implementations ---

#[async_trait]
impl ToolExecutor for ForwardTools {
    fn available_tools(&self) -> Vec<ToolDefinition> {
        vec![
            tool_query_data_db(),
            tool_read_file(),
            tool_query_memory(),
            tool_run_python(),
        ]
    }

    async fn execute(&self, tool_name: &str, input: &serde_json::Value) -> Result<String> {
        match tool_name {
            "query_data_db" => {
                let sql = input["sql"]
                    .as_str()
                    .context("missing required parameter: sql")?;
                exec_query_data_db(&self.data_dsn, sql).await
            }
            "read_file" => {
                let path = input["path"]
                    .as_str()
                    .context("missing required parameter: path")?;
                exec_read_file(&self.work_dir, path).await
            }
            "query_memory" => {
                let sql = input["sql"]
                    .as_str()
                    .context("missing required parameter: sql")?;
                self.memory_db.query(sql).await
            }
            "run_python" => {
                let code = input["code"]
                    .as_str()
                    .context("missing required parameter: code")?;
                let filename = input["filename"].as_str();
                let result = script::run_python(code, filename, &self.work_dir).await?;
                Ok(result.to_agent_string())
            }
            _ => anyhow::bail!("unknown tool: {tool_name}"),
        }
    }
}

#[async_trait]
impl ToolExecutor for BackwardTools {
    fn available_tools(&self) -> Vec<ToolDefinition> {
        vec![tool_execute_memory_sql(), tool_query_memory()]
    }

    async fn execute(&self, tool_name: &str, input: &serde_json::Value) -> Result<String> {
        match tool_name {
            "execute_memory_sql" => {
                let sql = input["sql"]
                    .as_str()
                    .context("missing required parameter: sql")?;
                self.memory_db.execute(sql).await
            }
            "query_memory" => {
                let sql = input["sql"]
                    .as_str()
                    .context("missing required parameter: sql")?;
                self.memory_db.query(sql).await
            }
            _ => anyhow::bail!("unknown tool: {tool_name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_tool_definitions() {
        let tools = vec![
            tool_query_data_db(),
            tool_read_file(),
            tool_query_memory(),
            tool_run_python(),
        ];
        assert_eq!(tools.len(), 4);
        assert_eq!(tools[0].name, "query_data_db");
        assert_eq!(tools[1].name, "read_file");
        assert_eq!(tools[2].name, "query_memory");
        assert_eq!(tools[3].name, "run_python");

        // All schemas should have "type": "object"
        for tool in &tools {
            assert_eq!(tool.input_schema["type"], "object");
            assert!(tool.input_schema["properties"].is_object());
            assert!(tool.input_schema["required"].is_array());
        }
    }

    #[test]
    fn test_backward_tool_definitions() {
        let tools = vec![tool_execute_memory_sql(), tool_query_memory()];
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "execute_memory_sql");
        assert_eq!(tools[1].name, "query_memory");
    }

    #[test]
    fn test_forward_tools_available() {
        // We can't construct MemoryDB without a real connection, so just test the definitions
        let tools = vec![
            tool_query_data_db(),
            tool_read_file(),
            tool_query_memory(),
            tool_run_python(),
        ];
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"query_data_db"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"query_memory"));
        assert!(names.contains(&"run_python"));
    }

    #[test]
    fn test_backward_tools_available() {
        let tools = vec![tool_execute_memory_sql(), tool_query_memory()];
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"execute_memory_sql"));
        assert!(names.contains(&"query_memory"));
    }

    #[tokio::test]
    async fn test_read_file_within_work_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path().to_str().unwrap();

        // Create a test file
        let test_file = tmp.path().join("data.txt");
        tokio::fs::write(&test_file, "hello data")
            .await
            .unwrap();

        let content = exec_read_file(work_dir, "data.txt").await.unwrap();
        assert_eq!(content, "hello data");
    }

    #[tokio::test]
    async fn test_read_file_path_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path().to_str().unwrap();

        // Try to escape work dir
        let result = exec_read_file(work_dir, "../../../etc/passwd").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("path escapes working directory") || err.contains("failed to resolve"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_read_file_nonexistent() {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path().to_str().unwrap();

        let result = exec_read_file(work_dir, "nonexistent.txt").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_schemas_have_required_fields() {
        let query_data = tool_query_data_db();
        assert_eq!(query_data.input_schema["required"][0], "sql");

        let read_file = tool_read_file();
        assert_eq!(read_file.input_schema["required"][0], "path");

        let query_mem = tool_query_memory();
        assert_eq!(query_mem.input_schema["required"][0], "sql");

        let run_py = tool_run_python();
        assert_eq!(run_py.input_schema["required"][0], "code");

        let exec_sql = tool_execute_memory_sql();
        assert_eq!(exec_sql.input_schema["required"][0], "sql");
    }
}
