use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const TIDB_ZERO_API: &str = "https://zero.tidbapi.com/v1alpha1/instances";

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
