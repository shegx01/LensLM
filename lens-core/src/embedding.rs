//! Embedding-model installation via Ollama's `POST /api/pull`.
//!
//! `lens-core` stays Tauri-free, so this module owns the pure pieces: the
//! [`InstallProgress`] IPC type and the NDJSON-streaming pull routine that
//! reports progress through a caller-supplied closure. The Tauri command layer
//! adapts that closure onto a `tauri::ipc::Channel`.

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::LensError;

/// Progress for an embedding-model pull. Frozen IPC contract — mirrored in the
/// Svelte client as `{ status, completed, total }`.
///
/// `status` is Ollama's own status string (e.g. `"pulling manifest"`,
/// `"downloading <digest>"`, `"success"`). `completed`/`total` are the byte
/// counters Ollama attaches to layer-download lines (absent on status-only
/// lines), so the UI can render a percentage when they're present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallProgress {
    /// Ollama's status string for this NDJSON line.
    pub status: String,
    /// Bytes pulled so far for the current layer, if reported.
    pub completed: Option<u64>,
    /// Total bytes for the current layer, if reported.
    pub total: Option<u64>,
}

/// One NDJSON line from Ollama's `POST /api/pull` stream. Ollama also emits an
/// `error` field on failure lines; we surface that as a [`LensError::Model`].
#[derive(Debug, Deserialize)]
struct PullLine {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    completed: Option<u64>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

/// Pulls `model` from the Ollama runtime at `base_url` via `POST /api/pull`
/// (`{name, stream:true}`), parsing the NDJSON response line-by-line and
/// invoking `on_progress` for each status line.
///
/// On a clean finish a final `InstallProgress { status: "success", .. }` is
/// emitted (Ollama's own terminal line) and `Ok(())` returned. A connection
/// failure → [`LensError::Network`]; an `error` line in the stream →
/// [`LensError::Model`].
///
/// `base_url` is a parameter (resolved from config by the command layer) so the
/// tests can point it at a mock NDJSON server.
pub async fn pull_embedding_model<F>(
    base_url: &str,
    model: &str,
    mut on_progress: F,
) -> Result<(), LensError>
where
    F: FnMut(InstallProgress),
{
    let base_url = base_url.trim_end_matches('/');
    let url = format!("{base_url}/api/pull");
    let client = reqwest::Client::new();

    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "name": model, "stream": true }))
        .send()
        .await
        .map_err(|e| LensError::Network(format!("Ollama pull request failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(LensError::Network(format!(
            "Ollama pull failed with status {}",
            resp.status()
        )));
    }

    // NDJSON arrives in arbitrary byte chunks; buffer and split on newlines so a
    // line straddling two chunks is still parsed whole.
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| LensError::Network(format!("Ollama stream error: {e}")))?;
        buf.extend_from_slice(&chunk);
        emit_complete_lines(&mut buf, &mut on_progress)?;
    }
    // Flush any trailing line not terminated by a newline.
    if !buf.is_empty() {
        process_line(&buf, &mut on_progress)?;
    }
    Ok(())
}

/// Drains every newline-terminated line from `buf` (leaving any partial trailing
/// line in place) and processes each one.
fn emit_complete_lines<F>(buf: &mut Vec<u8>, on_progress: &mut F) -> Result<(), LensError>
where
    F: FnMut(InstallProgress),
{
    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = buf.drain(..=pos).collect();
        process_line(&line[..pos], on_progress)?;
    }
    Ok(())
}

/// Parses one NDJSON line and emits a progress event. Blank lines are skipped;
/// a line carrying an `error` field aborts with [`LensError::Model`].
fn process_line<F>(line: &[u8], on_progress: &mut F) -> Result<(), LensError>
where
    F: FnMut(InstallProgress),
{
    let trimmed = line
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .map(|start| &line[start..])
        .unwrap_or(&[]);
    if trimmed.is_empty() {
        return Ok(());
    }
    let parsed: PullLine = serde_json::from_slice(trimmed)
        .map_err(|e| LensError::Parse(format!("Ollama pull line: {e}")))?;
    if let Some(err) = parsed.error {
        return Err(LensError::Model(format!("Ollama pull error: {err}")));
    }
    on_progress(InstallProgress {
        status: parsed.status.unwrap_or_default(),
        completed: parsed.completed,
        total: parsed.total,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn pull_emits_progress_from_ndjson_stream() {
        let ndjson = concat!(
            "{\"status\":\"pulling manifest\"}\n",
            "{\"status\":\"downloading\",\"completed\":1000,\"total\":5000}\n",
            "{\"status\":\"downloading\",\"completed\":5000,\"total\":5000}\n",
            "{\"status\":\"success\"}\n",
        );
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/pull"))
            .respond_with(ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;

        let mut events = Vec::new();
        pull_embedding_model(&server.uri(), "nomic-embed-text", |p| events.push(p))
            .await
            .unwrap();

        assert_eq!(events.len(), 4);
        assert_eq!(events[0].status, "pulling manifest");
        assert_eq!(events[1].completed, Some(1000));
        assert_eq!(events[1].total, Some(5000));
        assert_eq!(events.last().unwrap().status, "success");
    }

    #[tokio::test]
    async fn pull_handles_lines_split_across_chunks() {
        // A single NDJSON line with no trailing newline must still be flushed.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/pull"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{\"status\":\"success\"}"))
            .mount(&server)
            .await;

        let mut events = Vec::new();
        pull_embedding_model(&server.uri(), "all-minilm", |p| events.push(p))
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].status, "success");
    }

    #[tokio::test]
    async fn pull_surfaces_error_line_as_model_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/pull"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("{\"error\":\"model 'bogus' not found\"}\n"),
            )
            .mount(&server)
            .await;

        let err = pull_embedding_model(&server.uri(), "bogus", |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Model(_)));
    }

    #[tokio::test]
    async fn pull_errors_when_ollama_unreachable() {
        let err = pull_embedding_model("http://127.0.0.1:1", "nomic-embed-text", |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Network(_)));
    }
}
