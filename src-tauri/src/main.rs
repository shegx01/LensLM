// Prevents an additional console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use lens_core::LensEngine;

fn main() {
    // Instantiate the headless engine container once at runtime boot.
    let lens_core_instance = LensEngine::new();

    tauri::Builder::default()
        // Cache the instance safely inside Tauri's thread-safe managed state.
        .manage(lens_core_instance)
        // Register the behaviorless bridge command handles.
        .invoke_handler(tauri::generate_handler![commands::invoke_core_action])
        .run(tauri::generate_context!())
        .expect("Fatal Error: Failed to launch the LensLM application context.");
}
