//! Configuration commands.

use lens_core::{AppConfig, LensEngine, LensError};
use tauri::Manager;

/// Returns the current in-memory application configuration.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn get_config(engine: tauri::State<'_, LensEngine>) -> Result<AppConfig, LensError> {
    Ok(engine.config().await)
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
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| LensError::Io(e.to_string()))?;
    config.save(&data_dir)?;
    engine.set_config(config).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn get_config_returns_default_for_fresh_engine() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let config = get_config(engine).await.unwrap();
        assert_eq!(config, AppConfig::default());
    }
}
