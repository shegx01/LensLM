// Prevents an additional console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use lens_core::LensEngine;
use tauri::Manager;

fn main() {
    // Initialize a tracing subscriber so engine/command spans are visible.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tauri::Builder::default()
        // Resolve the OS app-data dir, init the engine (open db + migrate +
        // load config) on it, and store the handle in Tauri managed state.
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            let engine = tauri::async_runtime::block_on(LensEngine::init(&data_dir))?;
            app.manage(engine);
            Ok(())
        })
        // Register the behaviorless bridge command handles.
        .invoke_handler(tauri::generate_handler![commands::invoke_core_action])
        .run(tauri::generate_context!())
        .expect("Fatal Error: Failed to launch the LensLM application context.");
}
