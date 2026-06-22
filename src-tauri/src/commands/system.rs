//! System / diagnostic commands.

use lens_core::{
    ALLOWED_EMBEDDING_MODELS, CheckResult, DownloadProgress, InstallProgress, LensEngine,
    LensError, LlmDetection, TtsVoice,
};
use serde::Serialize;
use tauri::Manager;
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

/// Runs the three onboarding readiness gates (LLM runtime, embedding model,
/// text-to-speech) and returns the ordered results for the system-check screen.
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

/// Returns the static Kokoro voice catalog (5 female + 5 male) for the TTS
/// onboarding panel. Invoked as `invoke("list_tts_voices")`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_tts_voices() -> Result<Vec<TtsVoice>, LensError> {
    Ok(lens_core::list_tts_voices())
}

/// Installs an embedding model by streaming Ollama's `POST /api/pull`.
///
/// Each NDJSON status line from Ollama is forwarded to the frontend as an
/// [`InstallProgress`] over the `on_progress` channel. The target runtime is the
/// configured Ollama base URL (same resolution as the system-check probe). If
/// Ollama is unreachable the command returns an `Err` for the UI to surface.
///
/// `model` is validated against [`ALLOWED_EMBEDDING_MODELS`]; anything else is
/// rejected with a [`LensError::Validation`] before any network call.
///
/// Invoked as `invoke("install_embedding_model", { model, onProgress })` where
/// `onProgress` is a `Channel<InstallProgress>`.
#[tracing::instrument(skip(on_progress, engine), fields(model = %model))]
#[tauri::command]
pub async fn install_embedding_model(
    model: String,
    on_progress: Channel<InstallProgress>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    if !ALLOWED_EMBEDDING_MODELS.contains(&model.as_str()) {
        return Err(LensError::Validation(format!(
            "unsupported embedding model: {model}"
        )));
    }
    let base_url = lens_core::ollama_base_url(&engine.config().await);
    lens_core::pull_embedding_model(&base_url, &model, |progress| {
        // A send failure means the frontend dropped the channel; log and keep
        // going (the pull itself is unaffected and will still complete).
        if let Err(e) = on_progress.send(progress) {
            tracing::warn!("install_embedding_model: progress channel send failed: {e}");
        }
    })
    .await
}

/// Downloads the Kokoro ONNX engine to `{app_data_dir}/models/kokoro/`,
/// streaming [`DownloadProgress`] over the `on_progress` channel. Idempotent: a
/// complete file already on disk emits a single `done` event without
/// re-downloading.
///
/// Invoked as `invoke("download_tts_engine", { onProgress })` where `onProgress`
/// is a `Channel<DownloadProgress>`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn download_tts_engine(
    on_progress: Channel<DownloadProgress>,
    app: tauri::AppHandle,
) -> Result<(), LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    let dest = data_dir.join(lens_core::KOKORO_MODEL_RELPATH);
    lens_core::download_kokoro_model(lens_core::KOKORO_MODEL_URL, &dest, |progress| {
        if let Err(e) = on_progress.send(progress) {
            tracing::warn!("download_tts_engine: progress channel send failed: {e}");
        }
    })
    .await
}

/// Returns whether the Kokoro ONNX model is already on disk at
/// `{app_data_dir}/models/kokoro/...`. Lets the TTS panel skip the download step
/// and show voice selection when the engine is already installed, instead of
/// always offering "Download Kokoro".
///
/// Invoked as `invoke("kokoro_downloaded")`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn kokoro_downloaded(app: tauri::AppHandle) -> Result<bool, LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    Ok(data_dir.join(lens_core::KOKORO_MODEL_RELPATH).is_file())
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
    async fn run_system_check_returns_three_ordered_checks() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let checks = run_system_check(engine).await.unwrap();

        // The fixed row order matches the design / engine contract.
        let ids: Vec<CheckId> = checks.iter().map(|c| c.id).collect();
        assert_eq!(
            ids,
            vec![
                CheckId::LlmRuntime,
                CheckId::EmbeddingModel,
                CheckId::TextToSpeech,
            ]
        );

        let status_of = |id: CheckId| checks.iter().find(|c| c.id == id).unwrap().status;

        // All three are now real readiness gates whose outcome depends on the
        // host (a local runtime / installed embed model / downloaded Kokoro model
        // may or may not be present). We assert each resolved to a binary
        // Pass/Fail — never `Pending` (the old advisory state is gone).
        for id in [
            CheckId::LlmRuntime,
            CheckId::EmbeddingModel,
            CheckId::TextToSpeech,
        ] {
            let status = status_of(id);
            assert!(
                status == CheckStatus::Pass || status == CheckStatus::Fail,
                "{id:?} must be a binary gate, got {status:?}"
            );
        }
    }

    #[tokio::test]
    async fn install_embedding_model_rejects_unlisted_model() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        // A model id outside the allowlist must be rejected with a Validation
        // error BEFORE any network call (no Ollama is running in the test env).
        let channel = Channel::new(|_: tauri::ipc::InvokeResponseBody| Ok(()));
        let err = install_embedding_model("rm -rf /".to_string(), channel, engine)
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[tokio::test]
    async fn list_tts_voices_returns_male_and_female_sets() {
        use lens_core::Gender;
        let voices = list_tts_voices().await.unwrap();
        assert_eq!(voices.len(), 10);
        assert_eq!(
            voices.iter().filter(|v| v.gender == Gender::Female).count(),
            5
        );
        assert_eq!(
            voices.iter().filter(|v| v.gender == Gender::Male).count(),
            5
        );
        assert!(voices.iter().any(|v| v.id == "af_heart"));
        assert!(voices.iter().any(|v| v.id == "am_onyx"));
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
