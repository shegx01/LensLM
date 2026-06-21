//! System / diagnostic commands.

use lens_core::{CheckResult, LensEngine, LensError, LlmDetection};
use serde::Serialize;

#[cfg(debug_assertions)]
use tauri::ipc::Channel;

#[cfg(debug_assertions)]
use crate::stream::{StreamEvent, send_event};

/// Result of a [`health_check`]: DB reachable + applied migration count.
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    /// Whether the database query succeeded.
    pub db_ok: bool,
    /// Number of migrations recorded in `_sqlx_migrations`.
    pub migration_count: i64,
}

/// Verifies the database is reachable and reports the applied migration count.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn health_check(engine: tauri::State<'_, LensEngine>) -> Result<HealthStatus, LensError> {
    let migration_count = engine.migration_count().await?;
    Ok(HealthStatus {
        db_ok: true,
        migration_count,
    })
}

/// Runs all onboarding system probes (engine/DB health, LLM runtime,
/// embedding model, vector database, disk permissions) and returns the
/// ordered results for the system-check screen.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn run_system_check(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<CheckResult>, LensError> {
    engine.run_system_check().await
}

/// Probes `base_url` for both Ollama-style and OpenAI-compatible endpoints,
/// returning a [`LlmDetection`] that summarizes reachability, server version,
/// and the list of available model names/ids.
///
/// Never returns an `Err` for "not reachable"; `LensError` is reserved for
/// genuine internal faults. The frontend should invoke this command as:
/// `invoke("detect_llm", { base_url: "http://..." })`.
///
/// We log a SANITIZED target (scheme + host[:port] only) rather than the raw
/// `base_url`: a user could paste a URL embedding `user:password@` userinfo, and
/// `%base_url` would leak those credentials into the trace/log stream.
#[tracing::instrument(skip_all, fields(target = %sanitize_url_for_log(&base_url)))]
#[tauri::command]
pub async fn detect_llm(base_url: String) -> Result<LlmDetection, LensError> {
    Ok(lens_core::detect_llm(&base_url).await)
}

/// Reduces a URL to `scheme://host[:port]` for safe logging, stripping any
/// `userinfo` (`user:pass@`), path, query, and fragment. Falls back to just the
/// scheme (or `<redacted>`) when the URL can't be parsed, so we never echo a raw
/// string that might carry credentials.
fn sanitize_url_for_log(raw: &str) -> String {
    let Some((scheme, rest)) = raw.split_once("://") else {
        return "<redacted>".to_string();
    };
    // Authority ends at the first '/', '?' or '#'.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    // Drop any userinfo before an '@'.
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    format!("{scheme}://{host_port}")
}

/// Demonstrator that exercises the streaming primitive end to end: emits
/// `Started`, three `Progress` updates, then `Done` over the channel.
///
/// Gated behind `debug_assertions` so it never appears on the release command
/// surface — it exists only to validate the streaming plumbing during dev.
#[cfg(debug_assertions)]
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn stream_demo(channel: Channel<StreamEvent<String>>) -> Result<(), LensError> {
    let total = 3u64;
    send_event(&channel, StreamEvent::Started)?;
    for done in 1..=total {
        send_event(
            &channel,
            StreamEvent::Progress {
                done,
                total: Some(total),
            },
        )?;
    }
    send_event(&channel, StreamEvent::Done)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lens_core::{CheckId, CheckStatus};
    #[cfg(debug_assertions)]
    use std::sync::{Arc, Mutex};
    use tauri::Manager;
    #[cfg(debug_assertions)]
    use tauri::ipc::Channel;

    #[test]
    fn sanitize_url_for_log_strips_userinfo_and_path() {
        // Embedded credentials must never survive into the log field.
        assert_eq!(
            sanitize_url_for_log("http://user:secret@localhost:11434/api/version"),
            "http://localhost:11434"
        );
        // Plain URL: keep scheme + host + port, drop path/query.
        assert_eq!(
            sanitize_url_for_log("https://api.example.com/v1/models?x=1"),
            "https://api.example.com"
        );
        // No port, no path.
        assert_eq!(
            sanitize_url_for_log("http://localhost:1234"),
            "http://localhost:1234"
        );
        // Unparseable (no scheme separator) → redacted, never echoed raw.
        assert_eq!(sanitize_url_for_log("not-a-url"), "<redacted>");
    }

    #[tokio::test]
    async fn health_check_reports_db_ok_and_migrations() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let status = health_check(engine).await.unwrap();
        assert!(status.db_ok);
        // The single 0001_init migration is recorded.
        assert_eq!(status.migration_count, 1);
    }

    #[tokio::test]
    async fn run_system_check_returns_five_ordered_checks() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let checks = run_system_check(engine).await.unwrap();

        // The fixed row order matches the design / engine contract.
        let ids: Vec<CheckId> = checks.iter().map(|c| c.id).collect();
        assert_eq!(
            ids,
            vec![
                CheckId::LocalBackend,
                CheckId::LlmRuntime,
                CheckId::EmbeddingModel,
                CheckId::VectorDatabase,
                CheckId::DiskPermissions,
            ]
        );

        let status_of = |id: CheckId| checks.iter().find(|c| c.id == id).unwrap().status;

        // Real probes that must pass in the test environment.
        assert_eq!(status_of(CheckId::LocalBackend), CheckStatus::Pass);
        assert_eq!(status_of(CheckId::DiskPermissions), CheckStatus::Pass);

        // The LLM runtime probe is network-dependent; with no Ollama/LM Studio
        // running it is `Fail`, but we only assert it resolved (not `Pending`)
        // to stay robust if a runtime happens to be present on the host.
        let llm = status_of(CheckId::LlmRuntime);
        assert!(llm == CheckStatus::Fail || llm == CheckStatus::Pass);

        // Embedding + vector are intentionally not wired yet.
        assert_eq!(status_of(CheckId::EmbeddingModel), CheckStatus::Pending);
        assert_eq!(status_of(CheckId::VectorDatabase), CheckStatus::Pending);
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn stream_demo_emits_started_progress_done_in_order() {
        let collected = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&collected);
        let channel = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
            // The mock receives the already-serialized IPC body; deserialize it
            // back into the typed envelope to assert ordering/content.
            let event = body.deserialize::<StreamEvent<String>>().unwrap();
            sink.lock().unwrap().push(event);
            Ok(())
        });

        stream_demo(channel).await.unwrap();

        let events = collected.lock().unwrap().clone();
        assert_eq!(
            events,
            vec![
                StreamEvent::Started,
                StreamEvent::Progress {
                    done: 1,
                    total: Some(3)
                },
                StreamEvent::Progress {
                    done: 2,
                    total: Some(3)
                },
                StreamEvent::Progress {
                    done: 3,
                    total: Some(3)
                },
                StreamEvent::Done,
            ]
        );
    }
}
