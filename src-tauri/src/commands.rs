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
