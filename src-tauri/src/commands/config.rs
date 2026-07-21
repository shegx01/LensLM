//! Configuration commands.

use lens_core::{AppConfig, LensEngine, LensError, TaskModel};
use tauri::Manager;

/// Returns the current in-memory application configuration.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn get_config(engine: tauri::State<'_, LensEngine>) -> Result<AppConfig, LensError> {
    Ok(engine.config().await)
}

/// Whether `new` differs from `old` in any LLM-resolution-relevant field. Gates the
/// provider rebind so unrelated writes (theme/accent/etc.) don't re-enqueue enrichment.
fn llm_config_changed(old: &AppConfig, new: &AppConfig) -> bool {
    old.models != new.models
        || old.enrichment.routing != new.enrichment.routing
        || old.enrichment.coref_model != new.enrichment.coref_model
        || old.enrichment.map_model != new.enrichment.map_model
        || old.enrichment.enabled != new.enrichment.enabled
        || old.enrichment.cloud_consent != new.enrichment.cloud_consent
}

/// Swaps the in-memory config and, on an LLM delta, re-derives the cached
/// chat/enrichment provider so model changes take effect without a restart. Split
/// from [`set_config`] (which owns disk persistence) so the rebind gate is testable
/// without a concrete Tauri runtime handle.
async fn apply_config(engine: &LensEngine, config: AppConfig) -> Result<(), LensError> {
    let rebind = llm_config_changed(&engine.config().await, &config);
    engine.set_config(config).await;
    if rebind {
        engine.rescan_enrichment_on_provider_change().await?;
    }
    Ok(())
}

/// Persists `config` to `config.json` in the app data dir, then applies it in memory
/// (rebinding the provider on an LLM delta). Shared by every config-mutating command so
/// the persist-then-apply order lives in one place.
async fn persist_and_apply(
    engine: &LensEngine,
    config: AppConfig,
    app: &tauri::AppHandle,
) -> Result<(), LensError> {
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    config.save(&data_dir)?;
    apply_config(engine, config).await
}

/// Replaces the configuration in memory and persists it to `config.json` in the
/// app data directory.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn set_config(
    config: AppConfig,
    engine: tauri::State<'_, LensEngine>,
    app: tauri::AppHandle,
) -> Result<(), LensError> {
    persist_and_apply(engine.inner(), config, &app).await
}

/// Pins the active chat model to `(provider, model)` and persists it, so `chat_provider()`
/// resolves it immediately and it survives restart (fixes "no chat model configured" on
/// relaunch). Additive: routing and resolution semantics are unchanged.
#[tracing::instrument(skip_all, fields(provider = %provider, model = %model))]
#[tauri::command(rename_all = "snake_case")]
pub async fn set_active_chat_model(
    provider: String,
    model: String,
    engine: tauri::State<'_, LensEngine>,
    app: tauri::AppHandle,
) -> Result<(), LensError> {
    let mut config = engine.config().await;
    config.enrichment.chat_model = Some(TaskModel { provider, model });
    persist_and_apply(engine.inner(), config, &app).await
}

/// Clears the active chat-model pin, reverting to routing-based resolution.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn clear_active_chat_model(
    engine: tauri::State<'_, LensEngine>,
    app: tauri::AppHandle,
) -> Result<(), LensError> {
    let mut config = engine.config().await;
    config.enrichment.chat_model = None;
    persist_and_apply(engine.inner(), config, &app).await
}

/// Read-only chat-provider gate; builds no network client. (AC-11)
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn has_chat_provider(engine: tauri::State<'_, LensEngine>) -> Result<bool, LensError> {
    Ok(engine.chat_provider().await.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lens_core::EnrichmentConfig;
    use lens_core::config::ModelConfig;

    /// A usable local-Ollama config with enrichment enabled: `provider_from_config`
    /// resolves a provider from this (no network), so a rebind installs one.
    fn ollama_local_config(theme: &str) -> AppConfig {
        AppConfig {
            theme: theme.to_string(),
            models: vec![ModelConfig {
                provider: "ollama".to_string(),
                base_url: "http://localhost:11434".to_string(),
                model: "llama3".to_string(),
                context: 8192,
                temperature: 0.7,
                api_key: String::new(),
            }],
            enrichment: EnrichmentConfig {
                enabled: true,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        }
    }

    #[tokio::test]
    async fn get_config_returns_default_for_fresh_engine() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let config = get_config(engine).await.unwrap();
        assert_eq!(config, AppConfig::default());
    }

    #[test]
    fn llm_config_changed_detects_llm_deltas() {
        let base = ollama_local_config("dark");

        assert!(
            !llm_config_changed(&base, &base.clone()),
            "identical config must not be a delta"
        );

        let mut models = base.clone();
        models.models[0].model = "mistral".to_string();
        assert!(llm_config_changed(&base, &models), "models change");

        let mut routing = base.clone();
        routing.enrichment.routing = lens_core::LlmRouting::Explicit {
            provider: "ollama".to_string(),
            model: "llama3".to_string(),
        };
        assert!(llm_config_changed(&base, &routing), "routing change");

        let mut coref = base.clone();
        coref.enrichment.coref_model = Some(lens_core::TaskModel {
            provider: "ollama".to_string(),
            model: "llama3".to_string(),
        });
        assert!(llm_config_changed(&base, &coref), "coref_model change");

        let mut map = base.clone();
        map.enrichment.map_model = Some(lens_core::TaskModel {
            provider: "ollama".to_string(),
            model: "llama3".to_string(),
        });
        assert!(llm_config_changed(&base, &map), "map_model change");

        let mut enabled = base.clone();
        enabled.enrichment.enabled = !base.enrichment.enabled;
        assert!(llm_config_changed(&base, &enabled), "enabled change");

        let mut consent = base.clone();
        consent.enrichment.cloud_consent = !base.enrichment.cloud_consent;
        assert!(llm_config_changed(&base, &consent), "cloud_consent change");
    }

    #[test]
    fn llm_config_changed_ignores_unrelated_fields() {
        let base = ollama_local_config("dark");
        let mut themed = base.clone();
        themed.theme = "light".to_string();
        assert!(
            !llm_config_changed(&base, &themed),
            "a theme-only change must not be an LLM delta"
        );
    }

    #[tokio::test]
    async fn apply_config_rebinds_provider_on_llm_change() {
        let engine = LensEngine::for_test().await;
        assert!(
            engine.llm_provider().await.is_none(),
            "fresh engine has no provider installed"
        );

        apply_config(&engine, ollama_local_config("dark"))
            .await
            .unwrap();

        assert!(
            engine.llm_provider().await.is_some(),
            "an LLM-config change must rebind the cached provider without a restart"
        );
    }

    #[tokio::test]
    async fn active_chat_model_pin_round_trips_and_resolves() {
        let engine = LensEngine::for_test().await;

        let mut config = ollama_local_config("dark");
        config.enrichment.chat_model = Some(lens_core::TaskModel {
            provider: "ollama".to_string(),
            model: "llama3".to_string(),
        });
        apply_config(&engine, config).await.unwrap();

        assert_eq!(
            engine.config().await.enrichment.chat_model,
            Some(lens_core::TaskModel {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
            }),
            "the pin round-trips through get_config"
        );
        assert!(
            engine.chat_provider().await.is_some(),
            "the pinned chat model resolves a chat provider"
        );
    }

    #[tokio::test]
    async fn apply_config_does_not_rebind_on_unrelated_change() {
        let engine = LensEngine::for_test().await;

        // Seat a provider-eligible config WITHOUT rebinding (`set_config` only swaps
        // in-memory config; it does not install a provider), so the cached provider
        // stays absent and a subsequent rebind would be observable.
        engine.set_config(ollama_local_config("dark")).await;
        assert!(engine.llm_provider().await.is_none());

        // A theme-only write is not an LLM delta, so the rescan must NOT fire and no
        // provider gets installed.
        let mut themed = ollama_local_config("dark");
        themed.theme = "light".to_string();
        apply_config(&engine, themed).await.unwrap();

        assert!(
            engine.llm_provider().await.is_none(),
            "an unrelated (theme) change must not trigger a provider rebind"
        );
    }
}
