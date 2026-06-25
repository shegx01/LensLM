// Prevents an additional console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod stream;

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
        // Native file picker for the onboarding "Add sources" step.
        .plugin(tauri_plugin_dialog::init())
        // Resolve the OS app-data dir, init the engine (open db + migrate +
        // load config) on it, and store the handle in Tauri managed state.
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            let engine = tauri::async_runtime::block_on(LensEngine::init(&data_dir))?;
            app.manage(engine);
            Ok(())
        })
        // Register typed per-feature commands; the deprecated shim is kept so
        // the existing frontend invoke stays green through M0.
        .invoke_handler(tauri::generate_handler![
            #[allow(deprecated)]
            commands::invoke_core_action,
            commands::config::get_config,
            commands::config::set_config,
            commands::notebooks::list_notebooks,
            commands::notebooks::create_notebook,
            commands::notebooks::rename_notebook,
            commands::notebooks::delete_notebook,
            commands::notebooks::trash_notebook,
            commands::notebooks::restore_notebook,
            commands::notebooks::list_trashed,
            commands::notebooks::purge_notebook,
            commands::notebooks::add_source,
            commands::notebooks::list_sources,
            commands::notebooks::add_text_source,
            commands::notebooks::add_url_source,
            commands::notebooks::set_source_selected,
            commands::notebooks::trash_source,
            commands::notebooks::restore_source,
            commands::notebooks::purge_source,
            commands::notebooks::ingest_source,
            commands::system::health_check,
            commands::system::list_recent_documents,
            commands::system::run_system_check,
            commands::system::detect_llm,
            commands::system::list_tts_voices,
            commands::system::install_embedding_model,
            commands::system::download_tts_engine,
            commands::system::kokoro_downloaded,
            // Dev-only streaming demonstrator; absent from the release surface.
            #[cfg(debug_assertions)]
            commands::system::stream_demo,
        ])
        .run(tauri::generate_context!())
        .expect("Fatal Error: Failed to launch the LensLM application context.");
}
