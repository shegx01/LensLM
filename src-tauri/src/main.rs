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
// The Qwen3-TTS CustomVoice sidecar host ([161e]): `QwenSidecar` is injected into
// the engine in the `.setup` block below. Apple-Silicon macOS only — MLX runs in
// an out-of-process Python sidecar (via `uv`), so the module compiles out elsewhere.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod qwen;
// The offscreen SPA-render impl (issue #78). Its `TauriJsRenderer` is injected
// into the engine in the `.setup` block below (Layer f), so its items are live.
mod render;
mod stream;

use std::sync::Arc;

use lens_core::LensEngine;
use tauri::{Manager, WebviewWindowBuilder};

use crate::render::TauriJsRenderer;

fn main_nav_allowed(url: &tauri::Url) -> bool {
    if cfg!(dev) {
        return url.scheme() == "http"
            && url.host_str() == Some("localhost")
            && url.port() == Some(1420);
    }
    match url.scheme() {
        "tauri" => url.host_str() == Some("localhost"),
        "http" if cfg!(windows) || cfg!(target_os = "android") => {
            url.host_str() == Some("tauri.localhost")
        }
        _ => false,
    }
}

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
        // Write exported notes to the path chosen via the save-file dialog (issue #25).
        .plugin(tauri_plugin_fs::init())
        // Copy exported notes text to the OS clipboard (issue #25).
        .plugin(tauri_plugin_clipboard_manager::init())
        // Persist + restore the main window's size/position across restarts so a
        // resize survives relaunch. First launch uses the tauri.conf.json default;
        // thereafter the saved geometry supersedes it. Auto-saves on exit.
        .plugin(tauri_plugin_window_state::Builder::default().build())
        // Open external provider-doc URLs in the system browser (a webview
        // `<a href>` never escapes to the OS browser). Scoped to https:// in
        // capabilities/default.json.
        .plugin(tauri_plugin_opener::init())
        // Resolve the OS app-data dir, init the engine (open db + migrate +
        // load config) on it, and store the handle in Tauri managed state.
        .setup(|app| {
            let main_cfg = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == "main")
                .ok_or("`main` window must stay declared in tauri.conf.json")?
                .clone();
            WebviewWindowBuilder::from_config(app.handle(), &main_cfg)?
                .on_navigation(main_nav_allowed)
                .build()?;

            // The OS app-data dir is a fixed anchor that never moves; it holds the
            // relocation pointer (location.json) that can re-point to a relocated
            // data dir (#238). Init on the resolved dir, then GC any superseded old
            // dir a prior relocate left behind.
            let anchor = app.path().app_data_dir()?;
            let data_dir = lens_core::relocate::resolve_data_dir(&anchor);
            let engine = tauri::async_runtime::block_on(LensEngine::init(&data_dir))?;
            lens_core::relocate::run_boot_cleanup(&anchor, &data_dir);
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

            // Inject the Qwen3-TTS CustomVoice sidecar host (161e) on Apple Silicon.
            // The resolver closure defers `uv` detection/provisioning + HF-cache
            // creation to first synth, so nothing here downloads or blocks
            // startup. `start()` is LAZY, so an unavailable runtime surfaces as a
            // generic `LensError::Tts` at synth time — never a startup panic.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                // Resolution is deferred inside the closure; see
                // `qwen::sidecar_paths` for the shared-path rationale. The venv/bin
                // follow the resolved data dir; `HF_HOME` follows the cache root (#238).
                let cfg = tauri::async_runtime::block_on(engine_state.config());
                let cache_root = cfg.cache_root(&data_dir);
                let paths = qwen::sidecar_paths(app.handle(), &data_dir, &cache_root)?;
                let resolver = qwen::spawn_resolver(&paths);
                let qwen = qwen::QwenSidecar::new(resolver);
                tauri::async_runtime::block_on(engine_state.set_tts_sidecar(Some(Arc::new(qwen))));

                // Single-flight + cancel coordinator for `--prepare` (#202); see qwen::coordinator.
                app.manage(qwen::QwenPrepareCoordinator::new());
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
            commands::config::has_chat_provider,
            commands::config::set_active_chat_model,
            commands::config::clear_active_chat_model,
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
            commands::notebooks::generate_dialogue,
            commands::notebooks::cancel_dialogue,
            commands::notebooks::synthesize_overview,
            commands::notebooks::cancel_synthesis,
            commands::notebooks::get_audio_overview_status,
            commands::notebooks::is_overview_generating,
            commands::notebooks::save_chat_user,
            commands::notebooks::save_chat_assistant,
            commands::notebooks::save_chat_marker,
            commands::notebooks::set_chat_feedback,
            commands::notebooks::list_chat_messages,
            commands::citations::resolve_citation_snippet,
            commands::citations::load_source_view,
            commands::notes::save_chat_note,
            commands::notes::save_manual_note,
            commands::notes::update_note,
            commands::notes::set_note_pinned,
            commands::notes::list_notes,
            commands::notes::delete_note,
            commands::notebooks::set_notebook_embedding_model,
            commands::notebooks::get_notebook_embedding_model,
            commands::notebooks::set_notebook_graph_retrieval_enabled,
            commands::notebooks::get_notebook_graph_retrieval_enabled,
            commands::notebooks::latest_notebook_eval,
            commands::notebooks::run_notebook_graph_eval,
            commands::system::health_check,
            commands::system::get_storage_stats,
            commands::system::clear_model_cache,
            commands::system::relocate_data_dir,
            commands::system::offload_cache,
            commands::system::reset_cache_location,
            commands::system::restart_app,
            commands::system::list_recent_documents,
            commands::system::run_system_check,
            commands::system::detect_llm,
            commands::system::list_tts_voices,
            commands::system::tts_engine_catalog,
            commands::system::install_embedding_model,
            commands::system::download_tts_model,
            commands::system::tts_model_status,
            // Qwen3-TTS explicit model download (#194); Apple-Silicon only, like
            // the sidecar it drives.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            commands::system::prepare_qwen_model,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            commands::system::cancel_prepare,
            commands::system::list_whisper_models,
            commands::system::download_whisper_model,
            commands::system::whisper_model_downloaded,
            commands::system::asr_apple_native_available,
            commands::system::fastembed_models_cached,
            commands::system::warm_fastembed_model,
            commands::system::gpu_accelerated_models,
            commands::models::list_models,
            commands::models::list_active_model_candidates,
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

#[cfg(test)]
mod tests {
    use super::main_nav_allowed;

    fn allow(s: &str) -> bool {
        main_nav_allowed(&tauri::Url::parse(s).unwrap())
    }

    #[test]
    fn main_nav_rejects_foreign_and_untrusted_origins() {
        assert!(!allow("https://evil.com/phish"));
        assert!(!allow("http://localhost:9999/x"));
        assert!(!allow("file:///etc/passwd"));
        assert!(!allow("data:text/html,<h1>x</h1>"));
        assert!(!allow("tauri://evil.example/x"));
    }

    #[test]
    fn main_nav_allows_own_origin() {
        if cfg!(dev) {
            assert!(allow("http://localhost:1420/"));
        } else {
            assert!(allow("tauri://localhost/index.html"));
        }
    }
}
