use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::mysql::MySqlPoolOptions;
use sqlx::{Column, MySqlPool, Row};

const TIDB_ZERO_API: &str = "https://zero.tidbapi.com/v1alpha1/instances";

/// Maximum number of rows returned by a query before truncation.
const MAX_QUERY_ROWS: usize = 1000;

/// Memory database manager
pub struct MemoryDB {
    pub pool: MySqlPool,
    pub size_limit_mb: u64,
}

impl MemoryDB {
    /// Create connection pool from DSN (supports any MySQL protocol-compatible database)
    pub async fn connect(dsn: &str, size_limit_mb: u64) -> Result<Self> {
        let pool = MySqlPoolOptions::new()
            .max_connections(5)
            .connect(dsn)
            .await
            .context("failed to connect to memory database")?;

        Ok(Self {
            pool,
            size_limit_mb,
        })
    }

    /// Execute a query SQL (SELECT), return results in JSON format
    pub async fn query(&self, sql: &str) -> Result<String> {
        let rows = sqlx::query(sql)
            .fetch_all(&self.pool)
            .await
            .context("failed to execute query")?;

        let truncated = rows.len() > MAX_QUERY_ROWS;
        let total_rows = rows.len();
        let rows_to_process = if truncated { &rows[..MAX_QUERY_ROWS] } else { &rows[..] };

        let mut results: Vec<serde_json::Value> = Vec::with_capacity(rows_to_process.len());

        for row in rows_to_process {
            let columns = row.columns();
            let mut map = serde_json::Map::new();
            for (i, col) in columns.iter().enumerate() {
                let name = col.name().to_string();
                let value = row_value_to_json(row, i);
                map.insert(name, value);
            }
            results.push(serde_json::Value::Object(map));
        }

        if truncated {
            let mut output = serde_json::to_string(&results)?;
            output.push_str(&format!(
                "\n[Notice: results truncated, showing {MAX_QUERY_ROWS} of {total_rows} rows]"
            ));
            Ok(output)
        } else {
            Ok(serde_json::to_string(&results)?)
        }
    }

    /// Execute a write SQL (INSERT/UPDATE/DELETE/CREATE/ALTER), return affected rows count
    pub async fn execute(&self, sql: &str) -> Result<String> {
        let result = sqlx::query(sql)
            .execute(&self.pool)
            .await
            .context("failed to execute SQL")?;

        Ok(format!("Affected rows: {}", result.rows_affected()))
    }

    /// Get database size (MB)
    pub async fn get_size_mb(&self) -> Result<f64> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(data_length + index_length) / 1024 / 1024, 0) AS size_mb \
             FROM information_schema.tables WHERE table_schema = DATABASE()",
        )
        .fetch_one(&self.pool)
        .await
        .context("failed to query database size")?;

        let size: f64 = row.try_get("size_mb").unwrap_or(0.0);
        Ok(size)
    }

    /// Check if database size is within limit
    pub async fn check_size(&self) -> Result<bool> {
        let size = self.get_size_mb().await?;
        Ok(size < self.size_limit_mb as f64)
    }

    /// Get pool reference (for MCP server use)
    pub fn pool(&self) -> &MySqlPool {
        &self.pool
    }
}

/// Extract a column value from a MySQL row as a JSON value.
/// Tries String first, then falls back to i64, f64, and finally null.
fn row_value_to_json(row: &sqlx::mysql::MySqlRow, index: usize) -> serde_json::Value {
    // Try String first
    if let Ok(v) = row.try_get::<String, _>(index) {
        return serde_json::Value::String(v);
    }
    // Try i64
    if let Ok(v) = row.try_get::<i64, _>(index) {
        return serde_json::Value::Number(v.into());
    }
    // Try f64
    if let Ok(v) = row.try_get::<f64, _>(index) {
        if let Some(n) = serde_json::Number::from_f64(v) {
            return serde_json::Value::Number(n);
        }
        return serde_json::Value::String(v.to_string());
    }
    // Try bool
    if let Ok(v) = row.try_get::<bool, _>(index) {
        return serde_json::Value::Bool(v);
    }
    // NULL or unsupported type
    serde_json::Value::Null
}

/// TiDB Zero instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiDBInstance {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub connection_string: String,
    pub expires_at: String,
}

// --- API request/response types (private) ---

#[derive(Debug, Serialize)]
struct CreateInstanceRequest {
    tag: String,
}

#[derive(Debug, Deserialize)]
struct CreateInstanceResponse {
    instance: InstanceInfo,
    #[serde(rename = "remainingDatabaseQuota")]
    remaining_database_quota: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct InstanceInfo {
    connection: ConnectionInfo,
    #[serde(rename = "connectionString")]
    connection_string: String,
    #[serde(rename = "expiresAt")]
    expires_at: String,
}

#[derive(Debug, Deserialize)]
struct ConnectionInfo {
    host: String,
    #[serde(default = "default_port")]
    port: u16,
    username: String,
    password: String,
}

fn default_port() -> u16 {
    4000
}

/// Create a TiDB Zero instance (no authentication required)
pub async fn create_instance(tag: &str) -> Result<TiDBInstance> {
    let client = reqwest::Client::new();
    let response = client
        .post(TIDB_ZERO_API)
        .json(&CreateInstanceRequest {
            tag: tag.to_string(),
        })
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await
        .context("failed to create TiDB Zero instance")?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("TiDB Zero API error ({status}): {body}");
    }

    let resp: CreateInstanceResponse = response
        .json()
        .await
        .context("failed to parse TiDB Zero response")?;

    let conn = resp.instance.connection;
    tracing::info!(
        host = %conn.host,
        port = conn.port,
        expires_at = %resp.instance.expires_at,
        remaining_quota = ?resp.remaining_database_quota,
        "TiDB Zero instance provisioned"
    );

    Ok(TiDBInstance {
        host: conn.host,
        port: conn.port,
        username: conn.username,
        password: conn.password,
        connection_string: resp.instance.connection_string,
        expires_at: resp.instance.expires_at,
    })
}

impl TiDBInstance {
    /// Return the connection string provided by the API
    pub fn dsn(&self) -> &str {
        &self.connection_string
    }
}

/// Resolve memory database DSN based on config.
/// - If `config.dsn` is non-empty, use it directly.
/// - Otherwise call TiDB Zero API to auto-create an instance (no authentication required).
pub async fn resolve_memory_dsn(config: &crate::config::MemoryConfig) -> Result<String> {
    if !config.dsn.is_empty() {
        tracing::info!("using user-provided memory database DSN");
        Ok(config.dsn.clone())
    } else {
        tracing::info!("creating TiDB Zero instance for memory database");
        let instance = create_instance(&config.tag).await?;
        Ok(instance.connection_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_api_response() {
        let json = r#"{
            "instance": {
                "connection": {
                    "host": "gateway01.us-east-1.shared.aws.tidbcloud.com",
                    "port": 4000,
                    "username": "2e6sMMi1HKPBSag.root",
                    "password": "testpassword123"
                },
                "connectionString": "mysql://2e6sMMi1HKPBSag.root:testpassword123@gateway01.us-east-1.shared.aws.tidbcloud.com:4000/test?ssl-mode=required",
                "expiresAt": "2026-02-22T12:00:00Z"
            },
            "remainingDatabaseQuota": 5
        }"#;

        let resp: CreateInstanceResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.instance.connection.host,
            "gateway01.us-east-1.shared.aws.tidbcloud.com"
        );
        assert_eq!(resp.instance.connection.port, 4000);
        assert_eq!(resp.instance.connection.username, "2e6sMMi1HKPBSag.root");
        assert_eq!(resp.instance.connection.password, "testpassword123");
        assert!(resp
            .instance
            .connection_string
            .starts_with("mysql://"));
        assert_eq!(resp.instance.expires_at, "2026-02-22T12:00:00Z");
        assert_eq!(resp.remaining_database_quota, Some(5));
    }

    #[test]
    fn test_parse_response_default_port() {
        let json = r#"{
            "instance": {
                "connection": {
                    "host": "example.com",
                    "username": "user",
                    "password": "pass"
                },
                "connectionString": "mysql://user:pass@example.com:4000/test",
                "expiresAt": "2026-02-22T12:00:00Z"
            },
            "remainingDatabaseQuota": null
        }"#;

        let resp: CreateInstanceResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.instance.connection.port, 4000);
        assert_eq!(resp.remaining_database_quota, None);
    }

    #[test]
    fn test_tidb_instance_dsn() {
        let instance = TiDBInstance {
            host: "example.com".to_string(),
            port: 4000,
            username: "user".to_string(),
            password: "pass".to_string(),
            connection_string: "mysql://user:pass@example.com:4000/test".to_string(),
            expires_at: "2026-02-22T12:00:00Z".to_string(),
        };
        assert_eq!(instance.dsn(), "mysql://user:pass@example.com:4000/test");
    }

    #[tokio::test]
    async fn test_resolve_memory_dsn_with_provided_dsn() {
        let config = crate::config::MemoryConfig {
            dsn: "mysql://localhost:3306/testdb".to_string(),
            tag: "test".to_string(),
            size_limit_mb: 512,
        };
        let dsn = resolve_memory_dsn(&config).await.unwrap();
        assert_eq!(dsn, "mysql://localhost:3306/testdb");
    }
}
