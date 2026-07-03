//! Tauri command surface, grouped by feature domain.
//!
//! Each command is a thin typed boundary over `lens-core`, returning
//! `Result<T, LensError>` so failures cross IPC as the locked `{kind, message}`
//! envelope. Milestones add files here rather than rewriting a generic handler.

pub mod config;
pub mod inspector;
pub mod models;
pub mod notebooks;
pub mod system;

use lens_core::{LensEngine, LensError};

/// Deprecated behaviorless shim retained so the existing `+page.svelte` invoke
/// and its vitest keep passing while the frontend migrates to the typed
/// commands. Removed in M1.
#[deprecated(note = "use the typed per-feature commands; removed in M1")]
#[tauri::command]
pub async fn invoke_core_action(
    _payload: String,
    _engine: tauri::State<'_, LensEngine>,
) -> Result<String, LensError> {
    Ok(String::new())
}

#[cfg(test)]
mod tests {
    #[allow(deprecated)]
    use super::invoke_core_action;
    use lens_core::LensEngine;
    use tauri::Manager;

    #[tokio::test]
    async fn invoke_core_action_returns_empty_string() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);

        let engine = app.state::<LensEngine>();
        #[allow(deprecated)]
        let result = invoke_core_action("payload".to_string(), engine).await;

        assert_eq!(result.unwrap(), "");
    }
}
