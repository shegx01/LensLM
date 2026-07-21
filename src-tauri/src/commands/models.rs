//! Model-catalog commands (Stage 1 of the LLM-interface overhaul).
//!
//! Thin typed boundary over `lens-core`'s [`lens_core::ModelCatalog`] so the
//! frontend can populate per-provider model pickers from the typed catalog
//! rather than hard-coded free strings.

use std::collections::BTreeMap;

use lens_core::{
    ActiveModelCandidate, LensEngine, LensError, ModelInfo, ModelValidation, ProviderEntry,
    TaskModel, validate_model_interactive as core_validate_model_interactive,
};
use serde::Serialize;

/// FROZEN IPC CONTRACT: `{ status: "valid"|"invalid", reason? }`. The frontend
/// depends on this shape verbatim (#90).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelValidationResult {
    pub status: String,
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

/// Validates raw (not-yet-persisted) model params (#90) so the onboarding panel can
/// block a bad model before saving config. Local: Ollama tags-membership. Cloud:
/// `max_tokens:1` live probe with 10s timeout. Does not read `AppConfig`.
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

/// Returns the full typed model catalog, loaded from the cached file or the bundled
/// snapshot. Never fails hard.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_models(
    engine: tauri::State<'_, LensEngine>,
) -> Result<BTreeMap<String, ProviderEntry>, LensError> {
    Ok(engine.model_catalog().await.providers.clone())
}

/// Returns the models for a single provider, or an empty map if the provider isn't
/// in the catalog.
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

/// Lists locally-available Ollama models via `GET /api/tags`. Graceful by contract:
/// returns an empty list (never an `Err`) when Ollama is unreachable.
#[tracing::instrument(skip_all, fields(base_url = %base_url))]
// `rename_all = "snake_case"` so the snake_case JS arg key `base_url` binds.
// Tauri v2's default convention is camelCase, so without this the `base_url`
// param silently fails to bind, the command rejects, and the JS `catch` returns
// `[]` — i.e. no Ollama models are ever detected.
#[tauri::command(rename_all = "snake_case")]
pub async fn list_ollama_models(base_url: String) -> Result<Vec<String>, LensError> {
    Ok(lens_core::list_ollama_models(&base_url).await)
}

/// Forces an on-demand catalog refresh, gated by the staleness check. Returns `true`
/// when the cache was refreshed; a fetch failure surfaces as an `Err` the UI can ignore.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn refresh_models(engine: tauri::State<'_, LensEngine>) -> Result<bool, LensError> {
    engine.refresh_model_catalog().await
}

/// Active-model selector payload: the configured candidates plus the current chat-model pin
/// (`None` ⇒ routing fallback), so one call gives the picker both its options and its
/// selection.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveModelSelection {
    pub active: Option<TaskModel>,
    pub candidates: Vec<ActiveModelCandidate>,
}

/// Read-only; builds no network client. See `active_model_candidates`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_active_model_candidates(
    engine: tauri::State<'_, LensEngine>,
) -> Result<ActiveModelSelection, LensError> {
    let config = engine.config().await;
    let candidates = lens_core::active_model_candidates(&config, config.enrichment.cloud_consent);
    Ok(ActiveModelSelection {
        active: config.enrichment.chat_model.clone(),
        candidates,
    })
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
        assert!(
            models.len() >= 5,
            "anthropic should carry many models, got {}",
            models.len()
        );
        let (key, info) = models.iter().next().expect("at least one model");
        assert_eq!(&info.id, key, "ModelInfo.id mirrors the map key");
        assert!(!info.name.is_empty(), "model carries a display name");
    }

    #[tokio::test]
    async fn list_ollama_models_empty_when_unreachable() {
        let models = list_ollama_models("http://127.0.0.1:1".to_string())
            .await
            .expect("graceful empty, never an error");
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn validate_model_interactive_local_unreachable_is_invalid() {
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

    #[tokio::test]
    async fn list_active_model_candidates_reports_pin_and_availability() {
        use lens_core::config::ModelConfig;
        use lens_core::{AppConfig, EnrichmentConfig};

        let engine = LensEngine::for_test().await;
        engine
            .set_config(AppConfig {
                models: vec![ModelConfig {
                    provider: "ollama".to_string(),
                    base_url: "http://localhost:11434".to_string(),
                    model: "llama3".to_string(),
                    ..ModelConfig::default()
                }],
                enrichment: EnrichmentConfig {
                    chat_model: Some(TaskModel {
                        provider: "ollama".to_string(),
                        model: "llama3".to_string(),
                    }),
                    ..EnrichmentConfig::default()
                },
                ..AppConfig::default()
            })
            .await;

        let app = tauri::test::mock_app();
        app.manage(engine);
        let state = app.state::<LensEngine>();

        let sel = list_active_model_candidates(state).await.unwrap();
        assert_eq!(
            sel.active,
            Some(TaskModel {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
            })
        );
        assert_eq!(sel.candidates.len(), 1);
        assert!(sel.candidates[0].available);
        assert_eq!(sel.candidates[0].label, "Ollama · llama3");
    }
}
