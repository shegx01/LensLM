//! System / diagnostic commands.

use lens_core::{
    CheckResult, DownloadProgress, InstallProgress, LensEngine, LensError, LlmDetection, TtsVoice,
    WHISPER_REGISTRY,
};
use serde::Serialize;
use tauri::Manager;
use tauri::ipc::Channel;

#[cfg(debug_assertions)]
use crate::stream::{StreamEvent, send_event};

/// A recent document suggested for the onboarding "Add sources" step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecentDocument {
    /// Absolute file path.
    pub path: String,
    /// File name (with extension).
    pub name: String,
    /// Lowercased extension without the dot (`pdf` | `docx` | `txt` | `md`).
    pub ext: String,
    /// File size in bytes.
    pub size: u64,
    /// Last-modified time as a Unix timestamp (seconds), or `0` if unavailable.
    pub mtime: u64,
}

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

/// Probes `base_url` for Ollama-style and OpenAI-compatible endpoints. Never returns
/// `Err` for "not reachable". Logs a sanitized target (scheme+host only) to avoid
/// leaking `user:password@` userinfo that could appear in a pasted URL.
#[tracing::instrument(skip_all, fields(target = %sanitize_url_for_log(&base_url)))]
// `rename_all = "snake_case"`: Tauri v2 defaults to camelCase; without this, `base_url`
// silently fails to bind and auto-detect no-ops.
#[tauri::command(rename_all = "snake_case")]
pub async fn detect_llm(base_url: String) -> Result<LlmDetection, LensError> {
    Ok(lens_core::detect_llm(&base_url).await)
}

/// Reduces a URL to `scheme://host[:port]`, stripping userinfo/path/query/fragment.
/// Falls back to `<redacted>` on parse failure so credentials are never echoed.
fn sanitize_url_for_log(raw: &str) -> String {
    let Some((scheme, rest)) = raw.split_once("://") else {
        return "<redacted>".to_string();
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    format!("{scheme}://{host_port}")
}

/// Returns the currently selected TTS backend's named-voice catalog, adapter-driven
/// via `TtsProvider::voices()`. Empty only when no provider resolves for the backend
/// (e.g. cloud, or the sidecar-backed Qwen3Local without a sidecar).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_tts_voices(
    engine: tauri::State<'_, LensEngine>,
    app: tauri::AppHandle,
) -> Result<Vec<TtsVoice>, LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    let config = engine.config().await;
    Ok(resolve_voices(&config.tts, &data_dir))
}

/// Testable core of [`list_tts_voices`] (no `AppHandle`/engine state).
fn resolve_voices(cfg: &lens_core::TtsConfig, data_dir: &std::path::Path) -> Vec<TtsVoice> {
    lens_core::resolve_tts_provider(cfg.backend, cfg, data_dir)
        .map(|provider| provider.voices())
        .unwrap_or_default()
}

/// Returns the static per-engine TTS capability catalog (#194) for the Settings
/// engine selector: platform, needs_key, availability (Qwen off Apple Silicon,
/// Cloud without a key), supported languages, preset voices, model size, and the
/// language-capability label. This is the selector's single source of truth —
/// distinct from `list_tts_voices` (reserved for runtime synthesis).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn tts_engine_catalog(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<lens_core::EngineCatalogEntry>, LensError> {
    let config = engine.config().await;
    let cloud_key_present = config
        .tts
        .cloud
        .as_ref()
        .map(|c| !c.api_key.is_empty())
        .unwrap_or(false);
    Ok(lens_core::tts_catalog_serialized(cloud_key_present))
}

/// Installs an embedding model via Ollama `POST /api/pull`, streaming NDJSON progress.
/// `model` is validated against the registry allowlist before any network call.
/// RESERVED FOR FUTURE USE (M5+): registered but has no frontend caller — onboarding
/// moved to the fastembed warm path. Kept because removing a registered Tauri command
/// is higher-risk churn than documenting its dormant status.
#[tracing::instrument(skip(on_progress, engine), fields(model = %model))]
#[tauri::command]
pub async fn install_embedding_model(
    model: String,
    on_progress: Channel<InstallProgress>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    if !lens_core::is_allowlisted_embedding_id(&model) {
        return Err(LensError::Validation(format!(
            "unsupported embedding model: {model}"
        )));
    }
    let base_url = lens_core::ollama_base_url(&engine.config().await);
    lens_core::pull_embedding_model(&base_url, &model, |progress| {
        if let Err(e) = on_progress.send(progress) {
            tracing::warn!("install_embedding_model: progress channel send failed: {e}");
        }
    })
    .await
}

/// Downloads a TTS model artifact (registry id such as `"orpheus"`/`"snac"`) to
/// `{app_data_dir}/models/<id>/`. Parallels `download_whisper_model`; the extra
/// `engine` arg is for #194 backend routing / log context — the registry is keyed by `model`.
#[tracing::instrument(skip_all, fields(engine = %engine, model = %model))]
#[tauri::command(rename_all = "snake_case")]
pub async fn download_tts_model(
    engine: String,
    model: String,
    on_progress: Channel<DownloadProgress>,
    app: tauri::AppHandle,
) -> Result<(), LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    lens_core::download_tts_model(&data_dir, &model, |progress| {
        if let Err(e) = on_progress.send(progress) {
            tracing::warn!("download_tts_model: progress channel send failed: {e}");
        }
    })
    .await
    .map(|_| ())
}

/// Returns whether the given TTS model artifact is already on disk, so the
/// onboarding/Settings UI can skip the download step. Mirrors `whisper_model_downloaded`.
///
/// `engine == "qwen3_local"` (#194) is special-cased: Qwen's MLX weights live in
/// the huggingface_hub cache (not the `models/<id>/` registry), so presence is the
/// HF-snapshot check under `HF_HOME` and `model` is ignored. Apple-Silicon only;
/// off-target the engine does not exist, so it falls through to the registry probe.
#[tracing::instrument(skip_all, fields(engine = %engine, model = %model))]
#[tauri::command(rename_all = "snake_case")]
pub async fn tts_model_downloaded(
    engine: String,
    model: String,
    app: tauri::AppHandle,
) -> Result<bool, LensError> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if engine == "qwen3_local" {
        let paths = crate::qwen::sidecar_paths(&app)?;
        return Ok(crate::qwen::qwen_snapshot_present(&paths.hf_cache_dir));
    }
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    Ok(lens_core::tts_model_downloaded(&data_dir, &model))
}

/// Explicitly downloads the Qwen3-TTS MLX model (~4.5 GB) via a one-shot sidecar
/// `--prepare` process, streaming byte progress on `on_progress` (#194). Mirrors
/// `download_tts_model`'s Channel contract; Qwen's weights are otherwise fetched
/// lazily by the sidecar on first synth. Apple-Silicon only (the Qwen target).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn prepare_qwen_model(
    on_progress: Channel<DownloadProgress>,
    app: tauri::AppHandle,
) -> Result<(), LensError> {
    let paths = crate::qwen::sidecar_paths(&app)?;
    let resolver = crate::qwen::spawn_resolver(&paths);
    crate::qwen::run_prepare(resolver, move |progress| {
        if let Err(e) = on_progress.send(progress) {
            tracing::warn!("prepare_qwen_model: progress channel send failed: {e}");
        }
    })
    .await
}

/// UI representation of a Whisper model entry from the registry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WhisperModelInfo {
    /// Short id (`"tiny"` | `"base"` | `"small"`).
    pub id: String,
    /// Approximate size in MiB for the onboarding size label.
    pub approx_mb: u32,
    /// Whether this is the default recommended model.
    pub is_default: bool,
}

/// Returns the Whisper model registry (tiny / base / small) with size labels,
/// matching the onboarding UI convention of `list_tts_voices`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_whisper_models() -> Result<Vec<WhisperModelInfo>, LensError> {
    Ok(WHISPER_REGISTRY
        .iter()
        .map(|spec| WhisperModelInfo {
            id: spec.id.to_string(),
            approx_mb: spec.approx_mb,
            is_default: spec.id == lens_core::DEFAULT_WHISPER_MODEL_ID,
        })
        .collect())
}

/// Downloads the requested Whisper ggml model to `{app_data_dir}/models/whisper/`.
/// Idempotent: a complete file on disk emits a single `done` event without re-downloading.
/// Mirrors `download_tts_model` exactly: same channel type, same progress reporting.
#[tracing::instrument(skip_all, fields(model = %model))]
#[tauri::command(rename_all = "snake_case")]
pub async fn download_whisper_model(
    model: String,
    on_progress: Channel<DownloadProgress>,
    app: tauri::AppHandle,
) -> Result<(), LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    lens_core::download_whisper_model(&data_dir, &model, |progress| {
        if let Err(e) = on_progress.send(progress) {
            tracing::warn!("download_whisper_model: progress channel send failed: {e}");
        }
    })
    .await
    .map(|_| ())
}

/// Returns whether the given Whisper model is already on disk, so the
/// onboarding UI can skip the download step. Mirrors `tts_model_downloaded`.
#[tracing::instrument(skip_all, fields(model = %model))]
#[tauri::command(rename_all = "snake_case")]
pub async fn whisper_model_downloaded(
    model: String,
    app: tauri::AppHandle,
) -> Result<bool, LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    Ok(lens_core::whisper_model_downloaded(&data_dir, &model))
}

/// Returns `true` when Apple-native ASR is available on this device:
/// compiled with the `apple-native-asr` feature AND running on macOS >= 26.
/// Used by the onboarding UI to skip the Whisper download step — this is a
/// UI signal only, NOT a router input (backend selection is `select_asr_backend`).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn asr_apple_native_available() -> Result<bool, LensError> {
    // Compile-time gate: the feature must be present and the target must be
    // aarch64-apple-darwin; this is false on every other platform/feature.
    if !cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        feature = "apple-native-asr"
    )) {
        return Ok(false);
    }
    // Runtime gate: macOS >= 26 is required for SpeechAnalyzer/SpeechTranscriber.
    Ok(macos_major_version()? >= lens_core::MIN_MACOS_FOR_APPLE_ASR)
}

/// Parses the macOS major version from `sw_vers -productVersion`.
/// Returns `LensError::Internal` on parse failure (never panics).
/// `pub` so the `main.rs` `.setup` block can use the same runtime gate.
pub fn macos_major_version() -> Result<u32, LensError> {
    let out = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .map_err(|e| LensError::Internal(format!("sw_vers failed: {e}")))?;
    let version = String::from_utf8_lossy(&out.stdout);
    version
        .trim()
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
        .ok_or_else(|| {
            LensError::Internal(format!(
                "could not parse macOS major version from: {version:?}"
            ))
        })
}

/// Returns registry model ids whose fastembed weights are cached under
/// `{app_data_dir}/models/fastembed/`. Uses the same predicate as the readiness gate
/// so the card state and the gate can never disagree. Best-effort: empty on failure.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn fastembed_models_cached(app: tauri::AppHandle) -> Result<Vec<String>, LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    Ok(fastembed_cached_ids(&data_dir))
}

/// Testable core of [`fastembed_models_cached`] (no `AppHandle`). Ollama-only models
/// are skipped: they are served by the daemon, never downloaded by fastembed (#80).
fn fastembed_cached_ids(data_dir: &std::path::Path) -> Vec<String> {
    lens_core::REGISTRY
        .iter()
        .filter(|spec| spec.supports(lens_core::EmbeddingBackend::Fastembed))
        .filter(|spec| lens_core::fastembed_weights_cached(data_dir, spec.id))
        .map(|spec| spec.id.to_string())
        .collect()
}

/// Warms (downloads + caches) a fastembed model's weights so it passes the readiness
/// gate without a first ingest. Unknown ids are rejected. No byte-level progress
/// (fastembed init is synchronous/opaque); the UI shows an indeterminate spinner.
#[tracing::instrument(skip(engine), fields(model = %model))]
#[tauri::command]
pub async fn warm_fastembed_model(
    model: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    if lens_core::resolve_opt(&model).is_none() {
        return Err(LensError::Validation(format!(
            "unsupported embedding model: {model}"
        )));
    }
    engine.warm_fastembed_model(&model).await
}

/// Returns model ids that run on the Apple GPU (candle + Metal) on this build (#91).
/// `["nomic-embed-text-v1.5"]` on Apple Silicon; `[]` elsewhere. Used by the
/// Embeddings UI to badge models "Apple GPU" and surface the "best performance" hint.
#[tauri::command]
pub fn gpu_accelerated_models() -> Vec<String> {
    lens_core::embedder::gpu_accelerated_model_ids()
        .into_iter()
        .map(String::from)
        .collect()
}

/// Allowed document extensions (lowercased, no dot) for recent-doc suggestions.
const RECENT_DOC_EXTS: [&str; 4] = ["pdf", "docx", "txt", "md"];

/// Maximum number of recent-document suggestions returned.
const RECENT_DOC_CAP: usize = 8;

/// Shallowly scans `~/Documents`, `~/Downloads`, `~/Desktop` for `pdf|docx|txt|md`,
/// returning up to [`RECENT_DOC_CAP`] sorted by mtime descending. Best-effort: any
/// failure (missing `$HOME`, unreadable dir, TCC denial) yields fewer/zero results;
/// never returns an `Err`. NOTE: Unix/macOS path assumptions; revisit for Windows.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_recent_documents() -> Result<Vec<RecentDocument>, LensError> {
    let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) else {
        return Ok(Vec::new());
    };
    let dirs = [
        home.join("Documents"),
        home.join("Downloads"),
        home.join("Desktop"),
    ];
    Ok(scan_recent_documents(&dirs))
}

/// Testable core of [`list_recent_documents`]. Errors are best-effort-ignored.
fn scan_recent_documents(dirs: &[std::path::PathBuf]) -> Vec<RecentDocument> {
    let mut docs: Vec<RecentDocument> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
            else {
                continue;
            };
            if !RECENT_DOC_EXTS.contains(&ext.as_str()) {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            docs.push(RecentDocument {
                path: path.display().to_string(),
                name: name.to_string(),
                ext,
                size: meta.len(),
                mtime,
            });
        }
    }
    docs.sort_by(|a, b| b.mtime.cmp(&a.mtime));
    docs.truncate(RECENT_DOC_CAP);
    docs
}

/// Dev-only streaming primitive demonstrator. Gated behind `debug_assertions`;
/// never appears on the release command surface.
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
    fn fastembed_cached_ids_excludes_ollama_only_models() {
        let dir = tempfile::tempdir().unwrap();
        // Seed a real, non-empty fastembed cache dir for a fastembed model.
        let nomic_subdir = lens_core::resolve("nomic-embed-text-v1.5")
            .fastembed_cache_subdir()
            .expect("nomic has a fastembed cache subdir");
        let model_dir = dir
            .path()
            .join("models")
            .join("fastembed")
            .join(&nomic_subdir);
        std::fs::create_dir_all(model_dir.join("snapshots")).unwrap();
        std::fs::write(model_dir.join("snapshots").join("model.onnx"), b"x").unwrap();

        let ids = fastembed_cached_ids(dir.path());
        assert!(
            ids.contains(&"nomic-embed-text-v1.5".to_string()),
            "the seeded fastembed model is reported cached: {ids:?}"
        );
        for ollama_only in [
            "embeddinggemma",
            "qwen3-embedding:4b",
            "nomic-embed-text-v2-moe",
            "snowflake-arctic-embed2",
        ] {
            assert!(
                !ids.contains(&ollama_only.to_string()),
                "ollama-only {ollama_only} must not be fastembed-cached: {ids:?}"
            );
        }
    }

    #[test]
    fn sanitize_url_for_log_strips_userinfo_and_path() {
        assert_eq!(
            sanitize_url_for_log("http://user:secret@localhost:11434/api/version"),
            "http://localhost:11434"
        );
        assert_eq!(
            sanitize_url_for_log("https://api.example.com/v1/models?x=1"),
            "https://api.example.com"
        );
        assert_eq!(
            sanitize_url_for_log("http://localhost:1234"),
            "http://localhost:1234"
        );
        assert_eq!(sanitize_url_for_log("not-a-url"), "<redacted>");
    }

    #[test]
    fn scan_recent_documents_filters_sorts_and_caps() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(root.join("a.pdf"), b"a").unwrap();
        std::fs::write(root.join("b.MD"), b"b").unwrap();
        std::fs::write(root.join("c.docx"), b"c").unwrap();
        std::fs::write(root.join("d.txt"), b"d").unwrap();
        std::fs::write(root.join("ignore.zip"), b"z").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();

        let docs = scan_recent_documents(&[root.to_path_buf()]);
        assert_eq!(docs.len(), 4);
        let exts: std::collections::HashSet<String> = docs.iter().map(|d| d.ext.clone()).collect();
        assert_eq!(
            exts,
            ["pdf", "md", "docx", "txt"]
                .iter()
                .map(|s| s.to_string())
                .collect::<std::collections::HashSet<String>>()
        );
        assert!(docs.iter().any(|d| d.name == "b.MD" && d.ext == "md"));
        for w in docs.windows(2) {
            assert!(w[0].mtime >= w[1].mtime);
        }
    }

    #[test]
    fn scan_recent_documents_missing_dir_is_empty_not_error() {
        let docs = scan_recent_documents(&[std::path::PathBuf::from(
            "/nonexistent/lens/recent/scan/path",
        )]);
        assert!(docs.is_empty());
    }

    #[tokio::test]
    async fn health_check_reports_db_ok_and_migrations() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let status = health_check(engine).await.unwrap();
        assert!(status.db_ok);
        assert_eq!(status.migration_count, 21);
    }

    #[tokio::test]
    async fn run_system_check_returns_three_ordered_checks() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let checks = run_system_check(engine).await.unwrap();
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

        let channel = Channel::new(|_: tauri::ipc::InvokeResponseBody| Ok(()));
        let err = install_embedding_model("rm -rf /".to_string(), channel, engine)
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[test]
    fn resolve_voices_default_orpheus_returns_named_catalog() {
        let cfg = lens_core::TtsConfig::default();
        let voices = resolve_voices(&cfg, std::path::Path::new("/data"));
        assert_eq!(voices.len(), 8);
        assert!(voices.iter().any(|v| v.id == "tara"));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn resolve_voices_sidecar_backend_is_empty_without_sidecar() {
        // Qwen3Local resolves a provider only with an injected sidecar; the
        // no-sidecar `resolve_voices` path yields an empty catalog.
        let cfg = lens_core::TtsConfig {
            backend: lens_core::TtsBackend::Qwen3Local,
            ..lens_core::TtsConfig::default()
        };
        let voices = resolve_voices(&cfg, std::path::Path::new("/data"));
        assert!(voices.is_empty());
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn stream_demo_emits_started_progress_done_in_order() {
        let collected = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&collected);
        let channel = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
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
