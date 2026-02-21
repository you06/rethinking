use anyhow::{Context, Result};
use std::path::Path;

/// Maximum length (in bytes) for stdout/stderr passed to agent
const MAX_OUTPUT_LEN: usize = 10000;

/// Python script execution result
#[derive(Debug)]
pub struct ScriptResult {
    pub script_path: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Truncate a string to approximately max_len bytes, respecting UTF-8 char boundaries
fn truncate_output(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}\n... (truncated, {} total bytes)", &s[..end], s.len())
    }
}

/// Run a Python script
/// - Write code to work_dir/filename
/// - Execute python3 <script_path>
/// - Return execution result (stdout/stderr truncated to MAX_OUTPUT_LEN)
pub async fn run_python(
    code: &str,
    filename: Option<&str>,
    work_dir: &str,
) -> Result<ScriptResult> {
    let work_path = Path::new(work_dir);
    tokio::fs::create_dir_all(work_path)
        .await
        .context("failed to create work directory")?;

    let filename = filename.unwrap_or("script.py");
    let script_path = work_path.join(filename);

    tokio::fs::write(&script_path, code)
        .await
        .context("failed to write Python script")?;

    tracing::info!(path = %script_path.display(), "running Python script");

    let output = tokio::process::Command::new("python3")
        .arg(&script_path)
        .current_dir(work_path)
        .output()
        .await
        .context("failed to execute Python script")?;

    let stdout_raw = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr_raw = String::from_utf8_lossy(&output.stderr).to_string();

    let result = ScriptResult {
        script_path: script_path.to_string_lossy().to_string(),
        stdout: truncate_output(&stdout_raw, MAX_OUTPUT_LEN),
        stderr: truncate_output(&stderr_raw, MAX_OUTPUT_LEN),
        exit_code: output.status.code().unwrap_or(-1),
    };

    tracing::debug!(
        exit_code = result.exit_code,
        stdout_len = stdout_raw.len(),
        stderr_len = stderr_raw.len(),
        "script completed"
    );

    Ok(result)
}

impl ScriptResult {
    /// Format as a string to pass to the agent
    pub fn to_agent_string(&self) -> String {
        let mut s = String::new();
        if self.exit_code != 0 {
            s.push_str(&format!("Exit code: {}\n", self.exit_code));
        }
        if !self.stdout.is_empty() {
            s.push_str("--- stdout ---\n");
            s.push_str(&self.stdout);
            s.push('\n');
        }
        if !self.stderr.is_empty() {
            s.push_str("--- stderr ---\n");
            s.push_str(&self.stderr);
            s.push('\n');
        }
        if s.is_empty() {
            s.push_str("(no output)");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_output_short() {
        let s = "hello world";
        let result = truncate_output(s, 100);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_truncate_output_exact() {
        let s = "hello";
        let result = truncate_output(s, 5);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_output_long() {
        let s = "hello world, this is a long string";
        let result = truncate_output(s, 11);
        assert!(result.starts_with("hello world"));
        assert!(result.contains("truncated"));
        assert!(result.contains(&format!("{} total bytes", s.len())));
    }

    #[test]
    fn test_truncate_output_multibyte() {
        // Each CJK character is 3 bytes in UTF-8
        let s = "abcde\u{4e16}\u{754c}"; // "abcde世界" = 5 + 3 + 3 = 11 bytes
        // Truncate at byte 7: lands in the middle of '世' (bytes 5..8)
        let result = truncate_output(s, 7);
        // Should walk back to byte 5 (boundary before '世')
        assert!(result.starts_with("abcde"));
        assert!(result.contains("truncated"));
        assert!(result.contains("11 total bytes"));
    }

    #[tokio::test]
    async fn test_run_python_hello() {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path().to_str().unwrap();

        let result = run_python("print('hello')", None, work_dir).await.unwrap();

        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello"));
        assert!(result.stderr.is_empty());
        assert!(result.script_path.ends_with("script.py"));
    }

    #[tokio::test]
    async fn test_run_python_custom_filename() {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path().to_str().unwrap();

        let result = run_python("print('test')", Some("custom.py"), work_dir)
            .await
            .unwrap();

        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("test"));
        assert!(result.script_path.ends_with("custom.py"));
    }

    #[tokio::test]
    async fn test_run_python_error() {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path().to_str().unwrap();

        let result = run_python("import sys; sys.exit(1)", None, work_dir)
            .await
            .unwrap();

        assert_eq!(result.exit_code, 1);
    }

    #[test]
    fn test_to_agent_string_success() {
        let result = ScriptResult {
            script_path: "test.py".to_string(),
            stdout: "hello\n".to_string(),
            stderr: String::new(),
            exit_code: 0,
        };
        let s = result.to_agent_string();
        assert!(s.contains("--- stdout ---"));
        assert!(s.contains("hello"));
        assert!(!s.contains("Exit code"));
    }

    #[test]
    fn test_to_agent_string_error() {
        let result = ScriptResult {
            script_path: "test.py".to_string(),
            stdout: String::new(),
            stderr: "error occurred\n".to_string(),
            exit_code: 1,
        };
        let s = result.to_agent_string();
        assert!(s.contains("Exit code: 1"));
        assert!(s.contains("--- stderr ---"));
        assert!(s.contains("error occurred"));
    }

    #[test]
    fn test_to_agent_string_no_output() {
        let result = ScriptResult {
            script_path: "test.py".to_string(),
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
        };
        let s = result.to_agent_string();
        assert_eq!(s, "(no output)");
    }
}
