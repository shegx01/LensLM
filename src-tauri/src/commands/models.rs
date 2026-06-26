//! Model-catalog commands (Stage 1 of the LLM-interface overhaul).
//!
//! Thin typed boundary over `lens-core`'s [`lens_core::ModelCatalog`] so the
//! frontend can populate per-provider model pickers from the typed catalog
//! rather than hard-coded free strings.

use std::collections::BTreeMap;

use lens_core::{LensEngine, LensError, ModelInfo, ProviderEntry};

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
#[tauri::command]
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
