//! Model-catalog commands (Stage 1 of the LLM-interface overhaul).
//!
//! Thin typed boundary over `lens-core`'s [`lens_core::ModelCatalog`] so the
//! frontend can populate per-provider model pickers from the typed catalog
//! rather than hard-coded free strings.

use std::collections::BTreeMap;

use lens_core::{
    LensEngine, LensError, ModelInfo, ModelValidation, ProviderEntry,
    validate_model_interactive as core_validate_model_interactive,
};
use serde::Serialize;

/// Result of an interactive enrichment-model validation (issue #90).
///
/// FROZEN IPC CONTRACT: the frontend depends on this shape verbatim — `status` is
/// `"valid"` or `"invalid"`, and `reason` is present (with an actionable message)
/// only when `status == "invalid"`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelValidationResult {
    /// `"valid"` when the model is usable, `"invalid"` otherwise.
    pub status: String,
    /// Human-readable, actionable reason when `status == "invalid"`; `None` when valid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl From<ModelValidation> for ModelValidationResult {
    fn from(v: ModelValidation) -> Self {
        match v {
            ModelValidation::Pass => ModelValidationResult {
                status: "valid".to_string(),
                reason: None,
            },
            ModelValidation::Invalid(reason) => ModelValidationResult {
                status: "invalid".to_string(),
                reason: Some(reason),
            },
        }
    }
}

/// Validates a model for ANY role (enrichment or studio/chat) from RAW,
/// not-yet-persisted params (issue #90) so the onboarding `LlmConfigPanel` can
/// block a bad model BEFORE persisting config.
///
/// - Local: `provider == "ollama"` ⇒ tags-membership of `model` in the runtime at
///   `base_url` (free, deterministic).
/// - Cloud: builds a temporary provider from the raw params and runs a `max_tokens:1`
///   live probe with a 10s timeout.
///
/// The `provider` id is derived by the frontend from the active tab (local tab sends
/// `"ollama"`; cloud tab sends the selected cloud provider id). This does NOT read
/// [`lens_core::AppConfig`] — it validates exactly what the user typed.
///
/// Invoked as `invoke("validate_model_interactive", { provider, model, base_url, api_key })`.
#[tracing::instrument(skip_all, fields(provider = %provider, model = %model))]
#[tauri::command(rename_all = "snake_case")]
pub async fn validate_model_interactive(
    provider: String,
    model: String,
    base_url: String,
    api_key: String,
) -> Result<ModelValidationResult, LensError> {
    let validation = core_validate_model_interactive(&provider, &model, &base_url, &api_key).await;
    Ok(validation.into())
}

/// Returns the full typed model catalog (provider key → entry), loaded from the
/// cached `models-catalog.json` or the bundled snapshot. Never fails hard.
///
/// Invoked as `invoke("list_models")`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_models(
    engine: tauri::State<'_, LensEngine>,
) -> Result<BTreeMap<String, ProviderEntry>, LensError> {
    Ok(engine.model_catalog().await.providers.clone())
}

/// Returns the models for a single provider (model id → info), or an empty map
/// when the provider isn't in the catalog. Lets a picker load just the provider
/// it needs.
///
/// Invoked as `invoke("list_provider_models", { provider })`.
#[tracing::instrument(skip_all, fields(provider = %provider))]
#[tauri::command]
pub async fn list_provider_models(
    provider: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<BTreeMap<String, ModelInfo>, LensError> {
    let catalog = engine.model_catalog().await;
    Ok(catalog
        .provider(&provider)
        .map(|p| p.models.clone())
        .unwrap_or_default())
}

/// Lists the LOCALLY-available Ollama models at `base_url` (via the live
/// `GET /api/tags` probe), so the local-provider picker shows what the user has
/// actually pulled — models.dev only catalogs cloud providers.
///
/// Graceful by contract: when Ollama is unreachable (not running, wrong URL, a
/// non-Ollama endpoint) this returns an EMPTY list, NEVER an `Err` — so the picker
/// renders a "no local models / not reachable" state instead of an error toast.
///
/// Invoked as `invoke("list_ollama_models", { base_url })`.
#[tracing::instrument(skip_all, fields(base_url = %base_url))]
// `rename_all = "snake_case"` so the snake_case JS arg key `base_url` binds.
// Tauri v2's default convention is camelCase, so without this the `base_url`
// param silently fails to bind, the command rejects, and the JS `catch` returns
// `[]` — i.e. no Ollama models are ever detected.
#[tauri::command(rename_all = "snake_case")]
pub async fn list_ollama_models(base_url: String) -> Result<Vec<String>, LensError> {
    Ok(lens_core::list_ollama_models(&base_url).await)
}

/// Forces an on-demand catalog refresh (e.g. when a model picker opens). Still
/// gated by the staleness check, so a fresh cache is left untouched. Best-effort:
/// a fetch failure is surfaced as an `Err` the UI can ignore (the cached/bundled
/// catalog keeps serving). Returns `true` when the cache was refreshed.
///
/// Invoked as `invoke("refresh_models")`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn refresh_models(engine: tauri::State<'_, LensEngine>) -> Result<bool, LensError> {
    engine.refresh_model_catalog().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::Manager;

    #[tokio::test]
    async fn list_models_returns_bundled_catalog_offline() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        // With no cache written, the engine degrades to the bundled (full)
        // catalog, which must cover the supported CLOUD providers. (models.dev has
        // no plain `ollama` key — local models are validated live via /api/tags.)
        let providers = list_models(engine).await.unwrap();
        for key in ["anthropic", "openai", "google", "zai", "ollama-cloud"] {
            assert!(providers.contains_key(key), "missing provider {key}");
        }
    }

    #[tokio::test]
    async fn list_provider_models_returns_models_for_known_provider() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let models = list_provider_models("anthropic".to_string(), engine)
            .await
            .unwrap();
        // The FULL bundled catalog carries many Anthropic models (lower-bound
        // assertion, NOT a frozen list — model ids rotate as the catalog refreshes).
        assert!(
            models.len() >= 5,
            "anthropic should carry many models, got {}",
            models.len()
        );
        // The picker shape carries the fields it needs: each entry's `id` matches
        // its map key, a human `name`, and a numeric `context_limit` when known.
        let (key, info) = models.iter().next().expect("at least one model");
        assert_eq!(&info.id, key, "ModelInfo.id mirrors the map key");
        assert!(!info.name.is_empty(), "model carries a display name");
    }

    #[tokio::test]
    async fn list_ollama_models_empty_when_unreachable() {
        // Ollama not running (always-refused port) ⇒ Ok(empty), NEVER an Err — the
        // picker renders a not-reachable state, not an error toast.
        let models = list_ollama_models("http://127.0.0.1:1".to_string())
            .await
            .expect("graceful empty, never an error");
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn validate_model_interactive_local_unreachable_is_invalid() {
        // Ollama not running (always-refused port) ⇒ the model is not in an (empty)
        // tags list ⇒ `invalid` with an actionable `ollama pull` reason. Mirrors the
        // graceful `list_ollama_models_empty_when_unreachable` pattern (no error).
        let result = validate_model_interactive(
            "ollama".to_string(),
            "llama3.2:3b".to_string(),
            "http://127.0.0.1:1".to_string(),
            String::new(),
        )
        .await
        .expect("validation returns Ok, never an Err");
        assert_eq!(result.status, "invalid");
        let reason = result.reason.expect("invalid carries a reason");
        assert!(reason.contains("llama3.2:3b"), "got {reason}");
        assert!(reason.contains("ollama pull"), "got {reason}");
    }

    #[test]
    fn model_validation_result_serializes_frozen_ipc_shape() {
        // FROZEN IPC CONTRACT the frontend lane depends on: `{ status, reason? }`.
        // Valid ⇒ `reason` omitted (skip_serializing_if None); invalid ⇒ present.
        let valid: ModelValidationResult = ModelValidation::Pass.into();
        assert_eq!(
            serde_json::to_value(&valid).unwrap(),
            serde_json::json!({ "status": "valid" })
        );
        let invalid: ModelValidationResult =
            ModelValidation::Invalid("bad model".to_string()).into();
        assert_eq!(
            serde_json::to_value(&invalid).unwrap(),
            serde_json::json!({ "status": "invalid", "reason": "bad model" })
        );
    }

    #[tokio::test]
    async fn list_provider_models_empty_for_unknown_provider() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let models = list_provider_models("nope-not-real".to_string(), engine)
            .await
            .unwrap();
        assert!(models.is_empty());
    }
}
