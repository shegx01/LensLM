use lens_core::{LensEngine, LensError};

/// Behaviorless structural IPC endpoint. Returns a typed [`LensError`] to the
/// frontend so command failures are programmatically distinguishable, not stringly-typed.
#[tauri::command]
pub async fn invoke_core_action(
    _payload: String,
    _engine: tauri::State<'_, LensEngine>,
) -> Result<String, LensError> {
    // Explicitly behaviorless pass-through endpoint.
    Ok(String::new())
}

#[cfg(test)]
mod tests {
    use super::invoke_core_action;
    use lens_core::LensEngine;
    use tauri::Manager;

    /// Build a headless mock app, register the managed `LensEngine`, and invoke
    /// the command with the real `State<LensEngine>` injected — exactly as the
    /// runtime does, but with no webview.
    #[tokio::test]
    async fn invoke_core_action_returns_empty_string() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);

        let engine = app.state::<LensEngine>();
        let result = invoke_core_action("payload".to_string(), engine).await;

        assert_eq!(result.unwrap(), "");
    }
}
