// Prevents an additional console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// The ingest pipeline future (`LensEngine::ingest_source`) grew with the
// bounded-memory streaming PDF path (issue #71), pushing the compiler's
// auto-trait (`Send`) evaluation for the Tauri command futures past the default
// 128-frame recursion limit. Raising it to 256 is the compiler-recommended,
// zero-runtime-cost fix (it only affects type-checking depth).
#![recursion_limit = "256"]

// The Apple-native ASR bridge (issue #42): `AppleSpeechEngine` (SpeechAnalyzer via
// a Swift @_cdecl C ABI). aarch64-apple-darwin + `apple-native-asr` only; elsewhere
// it compiles out and the router picks Whisper.
#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    feature = "apple-native-asr"
))]
mod asr;

mod commands;
// The offscreen SPA-render impl (issue #78). Its `TauriJsRenderer` is injected
// into the engine in the `.setup` block below (Layer f), so its items are live.
mod render;
mod stream;

use std::sync::Arc;

use lens_core::LensEngine;
use tauri::Manager;

use crate::render::TauriJsRenderer;

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
        // Persist + restore the main window's size/position across restarts so a
        // resize survives relaunch. First launch uses the tauri.conf.json default;
        // thereafter the saved geometry supersedes it. Auto-saves on exit.
        .plugin(tauri_plugin_window_state::Builder::default().build())
        // Resolve the OS app-data dir, init the engine (open db + migrate +
        // load config) on it, and store the handle in Tauri managed state.
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            let engine = tauri::async_runtime::block_on(LensEngine::init(&data_dir))?;
            app.manage(engine);

            // Inject the offscreen-webview JS renderer (issue #78, Layer f) so
            // the URL-ingest needs_js fallback can render SPA pages. Mirrors the
            // engine init's async setup pattern (`block_on`). The renderer holds
            // an `AppHandle` to build/destroy the isolated offscreen webview; the
            // engine dispatches against the `JsRenderer` trait object without ever
            // seeing `tauri`. The `lens-render-*` label matches NO capability, and
            // `capabilities/renderer-empty.json` is auto-loaded as defense-in-depth.
            let renderer = TauriJsRenderer::new(app.handle().clone());
            let engine_state = app.state::<LensEngine>();
            tauri::async_runtime::block_on(engine_state.set_js_renderer(Some(Arc::new(renderer))));

            // Inject the Apple-native ASR engine (issue #42, Unit 7) when the
            // platform and runtime OS version are both capable. Mirrors
            // `set_js_renderer` above. The runtime macOS>=26 gate lives here (not
            // in lens-core) because OS probing is an src-tauri responsibility;
            // lens-core treats a present engine as authoritative.
            #[cfg(all(
                target_os = "macos",
                target_arch = "aarch64",
                feature = "apple-native-asr"
            ))]
            {
                let macos_ok = commands::system::macos_major_version()
                    .map(|v| v >= lens_core::MIN_MACOS_FOR_APPLE_ASR)
                    .unwrap_or(false);
                if macos_ok {
                    let apple = asr::AppleSpeechEngine::new();
                    tauri::async_runtime::block_on(
                        engine_state.set_asr_engine(Some(Arc::new(apple))),
                    );
                }
            }

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
            commands::notebooks::touch_notebook_activity,
            commands::notebooks::delete_notebook,
            commands::notebooks::trash_notebook,
            commands::notebooks::restore_notebook,
            commands::notebooks::list_trashed,
            commands::notebooks::list_trashed_sources,
            commands::notebooks::purge_notebook,
            commands::notebooks::add_source,
            commands::notebooks::list_sources,
            commands::notebooks::add_text_source,
            commands::notebooks::add_url_source,
            commands::notebooks::add_file_source,
            commands::notebooks::set_source_selected,
            commands::notebooks::trash_source,
            commands::notebooks::restore_source,
            commands::notebooks::purge_source,
            commands::notebooks::ingest_source,
            commands::notebooks::retry_ingest_source,
            commands::notebooks::retry_all_failed_sources,
            commands::notebooks::cancel_media_ingest,
            commands::notebooks::ask_notebook,
            commands::notebooks::cancel_ask,
            commands::notebooks::save_chat_user,
            commands::notebooks::save_chat_assistant,
            commands::notebooks::set_chat_feedback,
            commands::notebooks::list_chat_messages,
            commands::notebooks::set_notebook_embedding_model,
            commands::notebooks::get_notebook_embedding_model,
            commands::notebooks::set_notebook_graph_retrieval_enabled,
            commands::notebooks::get_notebook_graph_retrieval_enabled,
            commands::notebooks::latest_notebook_eval,
            commands::notebooks::run_notebook_graph_eval,
            commands::system::health_check,
            commands::system::list_recent_documents,
            commands::system::run_system_check,
            commands::system::detect_llm,
            commands::system::list_tts_voices,
            commands::system::install_embedding_model,
            commands::system::download_tts_engine,
            commands::system::kokoro_downloaded,
            commands::system::list_whisper_models,
            commands::system::download_whisper_model,
            commands::system::whisper_model_downloaded,
            commands::system::asr_apple_native_available,
            commands::system::fastembed_models_cached,
            commands::system::warm_fastembed_model,
            commands::system::gpu_accelerated_models,
            commands::models::list_models,
            commands::models::list_provider_models,
            commands::models::list_ollama_models,
            commands::models::validate_model_interactive,
            commands::models::refresh_models,
            // Dev-only streaming demonstrator; absent from the release surface.
            #[cfg(debug_assertions)]
            commands::system::stream_demo,
            // Dev/QA Embeddings Inspector; absent from the release surface.
            #[cfg(debug_assertions)]
            commands::inspector::list_source_chunks,
        ])
        .run(tauri::generate_context!())
        .expect("Fatal Error: Failed to launch the LensLM application context.");
}
