//! JS-render seam (issue #78) — the tauri-free trait boundary for the offscreen
//! webview render path.
//!
//! `lens-core` MUST NOT depend on `tauri` (crate-boundary invariant,
//! `Cargo.toml`), so it cannot own the webview. Instead it defines the
//! [`JsRenderer`] trait here; `src-tauri` implements it (`TauriJsRenderer`) and
//! injects it into [`crate::LensEngine`] via the same `Arc<RwLock<Option<Arc<dyn
//! _>>>>` DI seam used for the enrichment `LlmProvider`. The URL-ingest fallback
//! (Layer d) reads the injected renderer, renders near-empty SPA pages, and
//! feeds the rendered HTML back through the SAME static extractor.

use crate::error::LensError;
use async_trait::async_trait;

/// An async, object-safe renderer that loads a URL in an isolated offscreen
/// webview and returns its rendered DOM.
///
/// `Send + Sync` and used behind `Arc<dyn JsRenderer>` so [`crate::LensEngine`]
/// can hold it in `Arc<RwLock<Option<Arc<dyn JsRenderer>>>>` (the same
/// rebindable DI cell as `llm_provider`) and the ingest path can dispatch
/// against the trait object without `lens-core` seeing `tauri`.
#[async_trait]
pub trait JsRenderer: Send + Sync {
    /// Render `url` in an isolated offscreen webview and return the rendered DOM
    /// as `document.documentElement.outerHTML` (HTML bytes as a `String`), or
    /// `None` if rendering failed / timed out / yielded nothing.
    ///
    /// The caller feeds the returned HTML through the SAME
    /// `UrlExtractor::extract(&[u8])` the static path uses (Principle 1) — NOT a
    /// text-only path. The method is named `render_html` (not `render_text`) to
    /// make that contract unambiguous: trafilatura needs HTML bytes, so
    /// `outerHTML` (not `innerText`) is the required capture.
    async fn render_html(&self, url: &str) -> Result<Option<String>, LensError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LensEngine;
    use std::sync::Arc;

    /// A fake renderer proving the engine DI seam round-trips (set → get). The
    /// integration-level fallback behavior (Layer d) drives a fake like this
    /// through `run_ingest`; here we only assert the seam itself.
    struct FakeRenderer {
        canned: Option<String>,
    }

    #[async_trait]
    impl JsRenderer for FakeRenderer {
        async fn render_html(&self, _url: &str) -> Result<Option<String>, LensError> {
            Ok(self.canned.clone())
        }
    }

    #[tokio::test]
    async fn js_renderer_seam_set_get_round_trip() {
        let engine = LensEngine::for_test().await;
        // Starts empty.
        assert!(
            engine.js_renderer().await.is_none(),
            "renderer cell must start empty"
        );

        let fake: Arc<dyn JsRenderer> = Arc::new(FakeRenderer {
            canned: Some("<html><body>hi</body></html>".to_string()),
        });
        engine.set_js_renderer(Some(fake)).await;

        let got = engine.js_renderer().await.expect("renderer must be set");
        let html = got
            .render_html("https://example.com")
            .await
            .expect("render ok");
        assert_eq!(html.as_deref(), Some("<html><body>hi</body></html>"));

        // Clearing works (mirrors the llm_provider rebinding seam).
        engine.set_js_renderer(None).await;
        assert!(engine.js_renderer().await.is_none());
    }
}
