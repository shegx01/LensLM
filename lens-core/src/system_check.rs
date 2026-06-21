//! First-run system check: honest probes of the local intelligence stack.
//!
//! This module defines the FROZEN IPC contract ([`CheckResult`]) returned by
//! [`crate::LensEngine::run_system_check`], plus the five probes that populate
//! it. The contract is consumed verbatim by the Tauri command layer and mirrored
//! in the Svelte UI; do not reshape it without updating every mirror.
//!
//! Design honesty rule: a probe NEVER paints a green check for a subsystem that
//! does not exist yet. Subsystems that are wired in a later milestone (embeddings,
//! the vector database) report [`CheckStatus::Pending`] with product-facing copy.
//! Internal milestone vocabulary (e.g. "M4") lives ONLY in code comments — never
//! in a user-facing `detail` string. (Embeddings + the vector DB are wired in M4.)
//!
//! Probes never surface an expected-absent subsystem as a [`crate::LensError`]:
//! absence is a `Fail`/`Pending` status. `LensError` is reserved for genuinely
//! unexpected failures.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::config::AppConfig;

/// Connect timeout for a single runtime-detection HTTP request.
const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// Overall (read) timeout for a single runtime-detection HTTP request.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// Default Ollama base URL when none is configured.
const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
/// Default LM Studio OpenAI-compatible base URL.
const DEFAULT_LMSTUDIO_BASE_URL: &str = "http://localhost:1234";
/// Sentinel file used to verify the app data directory is writable.
const WRITE_SENTINEL_NAME: &str = ".lens_write_test";

/// Status of a single system-check row.
///
/// Serializes lowercase: `pass` | `fail` | `pending`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    /// The subsystem is present and healthy.
    Pass,
    /// The subsystem is expected but absent / unhealthy.
    Fail,
    /// The subsystem is intentionally not wired yet (set up later, automatically).
    Pending,
}

/// Stable identifier for each system-check row. Drives UI row ordering/mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckId {
    /// In-process engine + database health.
    LocalBackend,
    /// Local LLM runtime (Ollama / LM Studio) detection.
    LlmRuntime,
    /// Embedding model availability (advisory).
    EmbeddingModel,
    /// Vector database (built-in, set up automatically later).
    VectorDatabase,
    /// App-data-directory write permissions.
    DiskPermissions,
}

/// Optional UI affordance attached to a check row.
///
/// Absence of an action is expressed ONLY by `Option::None` on
/// [`CheckResult::action`] — there is deliberately NO `None` variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckAction {
    /// Open configuration for this subsystem (e.g. set an LLM endpoint).
    Configure,
    /// Choose among options (e.g. pick an embedding model).
    Choose,
    /// Retry the probe.
    Retry,
}

/// One row in the system-check screen.
///
/// THIS IS THE FROZEN IPC CONTRACT. It crosses the Tauri boundary verbatim and
/// is mirrored in the Svelte client; field names and the serde shape are locked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CheckResult {
    /// Stable row identifier.
    pub id: CheckId,
    /// Human-readable row label, e.g. "Local backend".
    pub label: String,
    /// Pass / fail / pending.
    pub status: CheckStatus,
    /// Product-facing detail copy. NO internal milestone vocabulary.
    pub detail: String,
    /// Optional UI affordance; absence is `None` (no `CheckAction::None`).
    pub action: Option<CheckAction>,
}

/// Shape of Ollama's `GET /api/version` response.
#[derive(Debug, Deserialize)]
struct OllamaVersion {
    version: String,
}

/// One model entry from Ollama's `GET /api/tags`.
#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    #[serde(default)]
    name: String,
    #[serde(default)]
    details: Option<OllamaModelDetails>,
}

/// Model details carried by an Ollama tag entry.
#[derive(Debug, Deserialize)]
struct OllamaModelDetails {
    #[serde(default)]
    family: Option<String>,
}

/// Shape of Ollama's `GET /api/tags` response.
#[derive(Debug, Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaTagModel>,
}

/// Outcome of probing the local LLM runtime, shared between the LLM-runtime row
/// and the (advisory) embedding-model row so the latter can reuse the tags fetch.
struct LlmRuntimeProbe {
    /// The completed LLM-runtime check row.
    result: CheckResult,
    /// Whether a local Ollama runtime answered (gates the embedding probe).
    ollama_up: bool,
    /// The Ollama base URL we probed (for the embedding tags fetch).
    ollama_base_url: String,
}

/// Builds a short-timeout HTTP client for runtime detection.
///
/// Both connect and read timeouts are bounded so a closed port or a black-hole
/// host fails fast rather than hanging the onboarding screen.
/// The shared probe-client builder: bounded connect/read timeouts plus SSRF
/// hardening (never follow a redirect — a malicious / misconfigured endpoint
/// could 30x a probe toward an internal host; a probe only ever inspects the
/// directly-addressed service). Centralized so the primary build and its
/// fallback can never drift apart.
fn probe_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .connect_timeout(PROBE_CONNECT_TIMEOUT)
        .timeout(PROBE_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
}

fn probe_client() -> reqwest::Client {
    // The builder only fails if the TLS backend can't initialize. Retry the
    // identical (timeout + no-redirect) builder once; the final
    // `unwrap_or_default` is a last-resort guard that can realistically never
    // run (rustls is pure Rust with no system deps) so a probe degrades to a
    // clean Fail, never a panic.
    probe_builder()
        .build()
        .unwrap_or_else(|_| probe_builder().build().unwrap_or_default())
}

/// Resolves the configured Ollama base URL, defaulting to localhost.
fn ollama_base_url(config: &AppConfig) -> String {
    provider_base_url(config, "ollama").unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string())
}

/// Resolves the configured LM Studio base URL, defaulting to localhost:1234.
///
/// Mirrors [`ollama_base_url`] so the LM Studio probe target is configurable
/// rather than hard-coded, and so the aggregate fallback can be tested via the
/// seam (point both seams at a mock server).
fn lmstudio_base_url(config: &AppConfig) -> String {
    provider_base_url(config, "lmstudio")
        .or_else(|| provider_base_url(config, "lm_studio"))
        .or_else(|| provider_base_url(config, "lm studio"))
        .unwrap_or_else(|| DEFAULT_LMSTUDIO_BASE_URL.to_string())
}

/// Finds the first configured model for `provider` with a non-empty base URL,
/// returning its trailing-slash-trimmed URL.
fn provider_base_url(config: &AppConfig, provider: &str) -> Option<String> {
    config
        .models
        .iter()
        .find(|m| m.provider.eq_ignore_ascii_case(provider) && !m.base_url.is_empty())
        .map(|m| m.base_url.trim_end_matches('/').to_string())
}

/// Probe 1 — in-process engine / database health.
///
/// The engine is already constructed, so we only verify the database answers.
/// No port string is reported: there is no separate service. Takes a cloned
/// pool (cheap `Arc` clone) so the caller can drop the engine read guard before
/// running this against the clone.
async fn probe_local_backend(db: &SqlitePool) -> CheckResult {
    let healthy = sqlx::query_scalar::<_, i64>("SELECT 1")
        .fetch_one(db)
        .await
        .map(|one| one == 1)
        .unwrap_or(false);

    if healthy {
        CheckResult {
            id: CheckId::LocalBackend,
            label: "Local backend".to_string(),
            status: CheckStatus::Pass,
            detail: "In-process engine ready".to_string(),
            action: None,
        }
    } else {
        CheckResult {
            id: CheckId::LocalBackend,
            label: "Local backend".to_string(),
            status: CheckStatus::Fail,
            detail: "Engine database unavailable".to_string(),
            action: Some(CheckAction::Retry),
        }
    }
}

/// Detects Ollama via `GET {base}/api/version`. Returns the parsed version on a
/// 200, `None` on a clean connect/timeout failure.
async fn detect_ollama(client: &reqwest::Client, base_url: &str) -> Option<String> {
    let url = format!("{base_url}/api/version");
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            resp.json::<OllamaVersion>().await.ok().map(|v| v.version)
        }
        // A non-2xx response or a parse miss means "not a healthy Ollama here".
        _ => None,
    }
}

/// Detects LM Studio via `GET {base}/v1/models`. Returns `true` on a 200.
async fn detect_lmstudio(client: &reqwest::Client, base_url: &str) -> bool {
    let url = format!("{base_url}/v1/models");
    matches!(client.get(&url).send().await, Ok(resp) if resp.status().is_success())
}

/// Probe 2 — local LLM runtime detection.
///
/// Probes Ollama (`/api/version`) and LM Studio (`/v1/models`) CONCURRENTLY via
/// [`tokio::join!`] so the total wall-clock is one timeout window (connect 1s +
/// read 2s ⇒ ≤ 2.5s budget), NOT the ~4s of two sequential probes. A clean
/// connect/timeout failure on both is a `Fail` "not detected", never a
/// [`crate::LensError`].
async fn probe_llm_runtime(config: &AppConfig) -> LlmRuntimeProbe {
    let client = probe_client();
    let ollama_base = ollama_base_url(config);
    let lmstudio_base = lmstudio_base_url(config);

    let (ollama_version, lmstudio_up) = tokio::join!(
        detect_ollama(&client, &ollama_base),
        detect_lmstudio(&client, &lmstudio_base),
    );

    let ollama_up = ollama_version.is_some();

    let result = match (ollama_version, lmstudio_up) {
        (Some(version), _) => CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Pass,
            detail: format!("Ollama {version} detected"),
            action: Some(CheckAction::Configure),
        },
        (None, true) => CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Pass,
            detail: "LM Studio detected".to_string(),
            action: Some(CheckAction::Configure),
        },
        (None, false) => CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Fail,
            detail: "No local LLM runtime detected".to_string(),
            action: Some(CheckAction::Configure),
        },
    };

    LlmRuntimeProbe {
        result,
        ollama_up,
        ollama_base_url: ollama_base,
    }
}

/// Probe 3 — embedding model availability (advisory).
///
/// If Ollama is up, fetch `/api/tags` and look for an embed-capable model (name
/// contains "embed", or details.family is a known embedding family). Present ⇒
/// `Pass`. Absent-but-Ollama-up ⇒ `Pending` (set up automatically later). Ollama
/// down ⇒ `Pending` (connect a runtime first). NEVER `Fail` — embeddings are not
/// an M1 deliverable. (Internal: wired in M4; keep "M4" out of `detail`.)
async fn probe_embedding_model(client: &reqwest::Client, runtime: &LlmRuntimeProbe) -> CheckResult {
    let label = "Embedding model".to_string();

    if !runtime.ollama_up {
        return CheckResult {
            id: CheckId::EmbeddingModel,
            label,
            status: CheckStatus::Pending,
            detail: "Set up automatically after connecting an LLM runtime".to_string(),
            action: None,
        };
    }

    let url = format!("{}/api/tags", runtime.ollama_base_url);
    let found = match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => resp
            .json::<OllamaTags>()
            .await
            .ok()
            .map(|tags| {
                tags.models.iter().any(|m| {
                    let name = m.name.to_ascii_lowercase();
                    let family = m
                        .details
                        .as_ref()
                        .and_then(|d| d.family.as_deref())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    name.contains("embed") || matches!(family.as_str(), "bert" | "nomic-bert")
                })
            })
            .unwrap_or(false),
        _ => false,
    };

    if found {
        CheckResult {
            id: CheckId::EmbeddingModel,
            label,
            status: CheckStatus::Pass,
            detail: "Embedding model available".to_string(),
            action: None,
        }
    } else {
        CheckResult {
            id: CheckId::EmbeddingModel,
            label,
            status: CheckStatus::Pending,
            detail: "Set up automatically when you add your first source".to_string(),
            action: Some(CheckAction::Choose),
        }
    }
}

/// Probe 4 — vector database (static `Pending`).
///
/// The built-in vector store is set up automatically with the first source; we
/// do NOT pull or initialize it during onboarding. Always `Pending`, no action.
/// (Internal: M4 flips this to `Pass`; keep "M4" out of `detail`.)
fn probe_vector_database() -> CheckResult {
    CheckResult {
        id: CheckId::VectorDatabase,
        label: "Vector database".to_string(),
        status: CheckStatus::Pending,
        detail: "Built-in (LanceDB) · set up automatically when you add your first source"
            .to_string(),
        action: None,
    }
}

/// Probe 5 — app-data-directory write permissions.
///
/// Writes then deletes a sentinel file in the configured data directory. Success
/// ⇒ `Pass` with the resolved path; failure ⇒ `Fail` with a retry affordance.
async fn probe_disk_permissions(config: &AppConfig) -> CheckResult {
    let label = "Disk permissions".to_string();
    let data_dir = config.paths.data_dir.clone();

    match write_test_sentinel(Path::new(&data_dir)) {
        Ok(()) => CheckResult {
            id: CheckId::DiskPermissions,
            label,
            status: CheckStatus::Pass,
            detail: data_dir,
            action: None,
        },
        Err(_) => CheckResult {
            id: CheckId::DiskPermissions,
            label,
            status: CheckStatus::Fail,
            detail: "Cannot write to app data directory".to_string(),
            action: Some(CheckAction::Retry),
        },
    }
}

/// Writes and removes a sentinel file in `dir`, proving it is writable.
fn write_test_sentinel(dir: &Path) -> std::io::Result<()> {
    let path = dir.join(WRITE_SENTINEL_NAME);
    std::fs::write(&path, b"lens-write-test")?;
    std::fs::remove_file(&path)
}

/// Runs all five system-check probes concurrently and returns them in the fixed
/// row order: LocalBackend, LlmRuntime, EmbeddingModel, VectorDatabase,
/// DiskPermissions.
///
/// Probes run via [`tokio::join!`] so the wall-clock cost is roughly the slowest
/// probe (the bounded LLM timeout window), not the sum of all probes.
///
/// Takes a `&AppConfig` + `&SqlitePool` rather than the engine handle: the
/// caller clones both cheaply under the engine read guard and DROPS the guard
/// before calling here, so the multi-second HTTP probes never hold the engine
/// lock (which would block concurrent `get_config`/`set_config`).
pub(crate) async fn run_system_check(config: &AppConfig, db: &SqlitePool) -> Vec<CheckResult> {
    let embed_client = probe_client();

    // The embedding probe reuses the LLM-runtime outcome, so it is awaited after
    // the LLM probe within this future; the other three run truly concurrently.
    let llm_and_embed = async {
        let runtime = probe_llm_runtime(config).await;
        let embedding = probe_embedding_model(&embed_client, &runtime).await;
        (runtime.result, embedding)
    };

    let (local_backend, (llm_runtime, embedding_model), disk_permissions) = tokio::join!(
        probe_local_backend(db),
        llm_and_embed,
        probe_disk_permissions(config),
    );

    vec![
        local_backend,
        llm_runtime,
        embedding_model,
        probe_vector_database(),
        disk_permissions,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use crate::config::ModelConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_with_ollama(base_url: &str) -> AppConfig {
        AppConfig {
            models: vec![ModelConfig {
                provider: "ollama".to_string(),
                base_url: base_url.to_string(),
                ..ModelConfig::default()
            }],
            ..AppConfig::default()
        }
    }

    /// Builds a config carrying both an Ollama and an LM Studio model entry so
    /// both probe seams can be pointed at mock servers (or dead URLs).
    fn config_with_runtimes(ollama_url: &str, lmstudio_url: &str) -> AppConfig {
        AppConfig {
            models: vec![
                ModelConfig {
                    provider: "ollama".to_string(),
                    base_url: ollama_url.to_string(),
                    ..ModelConfig::default()
                },
                ModelConfig {
                    provider: "lmstudio".to_string(),
                    base_url: lmstudio_url.to_string(),
                    ..ModelConfig::default()
                },
            ],
            ..AppConfig::default()
        }
    }

    #[test]
    fn lmstudio_base_url_defaults_then_reads_config() {
        // No lmstudio entry ⇒ the default seam.
        assert_eq!(
            lmstudio_base_url(&AppConfig::default()),
            DEFAULT_LMSTUDIO_BASE_URL
        );
        // A configured entry wins, trailing slash trimmed.
        let cfg = config_with_runtimes("", "http://127.0.0.1:9999/");
        assert_eq!(lmstudio_base_url(&cfg), "http://127.0.0.1:9999");
    }

    #[tokio::test]
    async fn aggregate_falls_back_to_lmstudio_via_seam() {
        // Ollama is down (dead URL), but a configured LM Studio seam answers 200
        // on /v1/models ⇒ the aggregate LLM probe reports Pass via the fallback.
        let ollama = MockServer::start().await;
        let dead_ollama = ollama.uri();
        drop(ollama);

        let lmstudio = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": []
            })))
            .mount(&lmstudio)
            .await;

        let config = config_with_runtimes(&dead_ollama, &lmstudio.uri());
        let probe = probe_llm_runtime(&config).await;

        assert_eq!(probe.result.status, CheckStatus::Pass);
        assert!(!probe.ollama_up);
        assert_eq!(probe.result.detail, "LM Studio detected");
        assert_eq!(probe.result.action, Some(CheckAction::Configure));
    }

    #[tokio::test]
    async fn llm_runtime_pass_when_ollama_responds() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.3.2"
            })))
            .mount(&server)
            .await;

        let config = config_with_ollama(&server.uri());
        let probe = probe_llm_runtime(&config).await;

        assert_eq!(probe.result.status, CheckStatus::Pass);
        assert!(probe.ollama_up);
        assert_eq!(probe.result.detail, "Ollama 0.3.2 detected");
    }

    #[tokio::test]
    async fn llm_runtime_fail_when_nothing_responds() {
        // Reserve a port then drop the server so the address refuses connections.
        let server = MockServer::start().await;
        let dead_url = server.uri();
        drop(server);

        let config = config_with_ollama(&dead_url);
        let probe = probe_llm_runtime(&config).await;

        assert_eq!(probe.result.status, CheckStatus::Fail);
        assert!(!probe.ollama_up);
        assert_eq!(probe.result.detail, "No local LLM runtime detected");
        assert_eq!(probe.result.action, Some(CheckAction::Configure));
    }

    #[tokio::test]
    async fn llm_runtime_falls_back_to_lmstudio() {
        // Ollama path 404s (present server, wrong endpoint ⇒ no version), but the
        // LM Studio endpoint answers 200. Because the LM Studio probe targets a
        // fixed default port, this asserts the fallback branch via the server
        // responding 200 on /v1/models while /api/version is absent.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": []
            })))
            .mount(&server)
            .await;
        // No /api/version mock ⇒ wiremock returns 404 ⇒ Ollama treated as absent.

        // Point the (fixed-port) LM Studio detector at this server by detecting
        // directly; the aggregate probe uses the default port, so we assert the
        // building blocks compose into the fallback Pass.
        let client = probe_client();
        let ollama = detect_ollama(&client, &server.uri()).await;
        let lmstudio = detect_lmstudio(&client, &server.uri()).await;

        assert!(ollama.is_none(), "no /api/version ⇒ Ollama absent");
        assert!(lmstudio, "LM Studio /v1/models answered 200");
    }

    #[tokio::test]
    async fn llm_probe_stays_within_time_budget_when_offline() {
        let server = MockServer::start().await;
        let dead_url = server.uri();
        drop(server);

        let config = config_with_ollama(&dead_url);
        let start = Instant::now();
        let _ = probe_llm_runtime(&config).await;
        let elapsed = start.elapsed();

        // Concurrent (not sequential) probing keeps the wall-clock to roughly
        // ONE timeout window: PROBE_CONNECT_TIMEOUT (1s) + PROBE_TIMEOUT (2s) =
        // 3s for the slowest single probe, NOT the ~6s of two sequential ones.
        // The 3500ms budget is that 3s window plus 500ms of slack for CI
        // scheduling jitter; bump it only if those two constants change.
        assert!(
            elapsed < Duration::from_millis(3_500),
            "llm probe took {elapsed:?}, exceeding the concurrent budget"
        );
    }

    #[tokio::test]
    async fn embedding_pass_when_embed_model_present() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{ "name": "nomic-embed-text:latest" }]
            })))
            .mount(&server)
            .await;

        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: probe_local_backend_placeholder(),
            ollama_up: true,
            ollama_base_url: server.uri(),
        };
        let result = probe_embedding_model(&client, &runtime).await;

        assert_eq!(result.status, CheckStatus::Pass);
        assert!(!result.detail.contains("M4"));
    }

    #[tokio::test]
    async fn embedding_pending_when_no_embed_model() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{ "name": "llama3:latest" }]
            })))
            .mount(&server)
            .await;

        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: probe_local_backend_placeholder(),
            ollama_up: true,
            ollama_base_url: server.uri(),
        };
        let result = probe_embedding_model(&client, &runtime).await;

        assert_eq!(result.status, CheckStatus::Pending);
        assert_eq!(result.action, Some(CheckAction::Choose));
        assert!(!result.detail.contains("M4"));
    }

    #[tokio::test]
    async fn embedding_pending_when_ollama_down() {
        let client = probe_client();
        let runtime = LlmRuntimeProbe {
            result: probe_local_backend_placeholder(),
            ollama_up: false,
            ollama_base_url: DEFAULT_OLLAMA_BASE_URL.to_string(),
        };
        let result = probe_embedding_model(&client, &runtime).await;

        assert_eq!(result.status, CheckStatus::Pending);
        assert!(!result.detail.contains("M4"));
    }

    #[test]
    fn vector_database_is_always_pending() {
        let result = probe_vector_database();
        assert_eq!(result.status, CheckStatus::Pending);
        assert_eq!(result.action, None);
        assert!(!result.detail.contains("M4"));
        assert!(result.detail.contains("LanceDB"));
    }

    #[tokio::test]
    async fn disk_permissions_pass_on_writable_dir() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = AppConfig::default();
        config.paths.data_dir = dir.path().display().to_string();

        let result = probe_disk_permissions(&config).await;
        assert_eq!(result.status, CheckStatus::Pass);
        assert_eq!(result.detail, dir.path().display().to_string());
    }

    #[tokio::test]
    async fn disk_permissions_fail_on_missing_dir() {
        let mut config = AppConfig::default();
        config.paths.data_dir = "/nonexistent/lens/data/dir/that/should/not/exist".to_string();

        let result = probe_disk_permissions(&config).await;
        assert_eq!(result.status, CheckStatus::Fail);
        assert_eq!(result.action, Some(CheckAction::Retry));
    }

    #[tokio::test]
    async fn run_system_check_returns_five_rows_in_order() {
        let engine = crate::LensEngine::for_test().await;
        let dir = tempfile::tempdir().unwrap();
        {
            let mut guard = engine.write().await;
            guard.config.paths.data_dir = dir.path().display().to_string();
        }
        let results = engine.run_system_check().await.unwrap();

        let ids: Vec<CheckId> = results.iter().map(|r| r.id).collect();
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
        // No user-facing detail leaks internal milestone vocabulary.
        for r in &results {
            assert!(
                !r.detail.contains("M4"),
                "detail for {:?} leaked milestone vocab: {}",
                r.id,
                r.detail
            );
        }
        // Local backend is healthy for a migrated test engine.
        assert_eq!(results[0].status, CheckStatus::Pass);
    }

    /// Snapshot the exact serde wire-format of `CheckResult`. Locks the FROZEN
    /// IPC contract: snake_case fields, lowercase status, `action` omitted/`None`.
    #[test]
    fn check_result_serialized_shape() {
        let result = CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Fail,
            detail: "No local LLM runtime detected".to_string(),
            action: Some(CheckAction::Configure),
        };
        insta::assert_json_snapshot!(result, @r#"
        {
          "id": "llm_runtime",
          "label": "LLM runtime",
          "status": "fail",
          "detail": "No local LLM runtime detected",
          "action": "configure"
        }
        "#);
    }

    /// `action: None` serializes as JSON `null` (the only way to express "no
    /// action" — there is no `CheckAction::None` variant).
    #[test]
    fn check_result_no_action_serializes_null() {
        let result = CheckResult {
            id: CheckId::VectorDatabase,
            label: "Vector database".to_string(),
            status: CheckStatus::Pending,
            detail: "Built-in".to_string(),
            action: None,
        };
        insta::assert_json_snapshot!(result, @r#"
        {
          "id": "vector_database",
          "label": "Vector database",
          "status": "pending",
          "detail": "Built-in",
          "action": null
        }
        "#);
    }

    /// Test helper: a throwaway `CheckResult` to fill the unused `result` field
    /// of an `LlmRuntimeProbe` fixture in the embedding-probe tests.
    fn probe_local_backend_placeholder() -> CheckResult {
        CheckResult {
            id: CheckId::LlmRuntime,
            label: "LLM runtime".to_string(),
            status: CheckStatus::Pass,
            detail: "fixture".to_string(),
            action: None,
        }
    }
}
