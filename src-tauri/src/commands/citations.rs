//! Citation read-back commands (issue #237): thin IPC boundary over
//! `lens-core`'s citation snippet/source-view engine API. Production surface
//! (not debug-gated) — the inline citation popover and "view in source"
//! viewer depend on these.

use lens_core::{LensEngine, LensError, SnippetSegments, SourceView};

/// Resolves a citation's persisted byte offsets against the retained source
/// buffer, returning bounded display segments for the inline popover.
/// `char_start`/`char_end` are byte offsets, not char indices.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn resolve_citation_snippet(
    source_id: String,
    char_start: usize,
    char_end: usize,
    engine: tauri::State<'_, LensEngine>,
) -> Result<SnippetSegments, LensError> {
    engine
        .citation_snippet(&source_id, char_start, char_end)
        .await
}

/// Loads a source for the "view in source" viewer. Both offsets must be
/// present to highlight a span; otherwise the whole text is returned
/// unhighlighted (older chat history may carry null offsets).
/// `char_start`/`char_end` are byte offsets, not char indices.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn load_source_view(
    source_id: String,
    char_start: Option<usize>,
    char_end: Option<usize>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<SourceView, LensError> {
    let span = match (char_start, char_end) {
        (Some(s), Some(e)) => Some((s, e)),
        _ => None,
    };
    engine.source_view(&source_id, span).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use lens_core::NotebookId;
    use tauri::Manager;

    /// Tempdir-backed engine so `add_text_source`'s managed files land in a
    /// scratch dir (not a cwd-relative `sources/` leak); the returned tempdir
    /// must outlive the caller's engine use.
    async fn test_engine() -> (tempfile::TempDir, tauri::App<tauri::test::MockRuntime>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = tauri::test::mock_app();
        app.manage(LensEngine::init(dir.path()).await.expect("engine init"));
        (dir, app)
    }

    async fn seed_source(engine: &LensEngine, text: &str) -> String {
        let nb = engine
            .create_notebook("Notebook", None, None)
            .await
            .unwrap();
        let notebook_id = NotebookId::from(nb.id.to_string());

        let outcome = engine
            .add_text_source(&notebook_id, "Doc", text, "text")
            .await
            .unwrap();
        outcome.source.id
    }

    #[tokio::test]
    async fn test_resolve_citation_snippet_command() {
        let (_dir, app) = test_engine().await;
        let engine = app.state::<LensEngine>();

        let text = "Alpha beta. The cited sentence here. Gamma delta.";
        let source_id = seed_source(&engine, text).await;
        let start = text.find("The cited").unwrap();
        let end = start + "The cited sentence here.".len();

        let seg = resolve_citation_snippet(source_id, start, end, engine)
            .await
            .unwrap();

        assert_eq!(seg.marked, "The cited sentence here.");
        assert_eq!(
            format!("{}{}{}", seg.before, seg.marked, seg.after),
            text,
            "small buffer reassembles losslessly"
        );
    }

    #[tokio::test]
    async fn test_load_source_view_command_with_span() {
        let (_dir, app) = test_engine().await;
        let engine = app.state::<LensEngine>();

        let text = "before-part MARKED after-part";
        let source_id = seed_source(&engine, text).await;
        let start = text.find("MARKED").unwrap();

        let view = load_source_view(source_id, Some(start), Some(start + 6), engine)
            .await
            .unwrap();

        assert_eq!(view.marked, "MARKED");
        assert_eq!(
            format!("{}{}{}", view.before, view.marked, view.after),
            text
        );
        assert_eq!(view.title, "Doc");
    }

    #[tokio::test]
    async fn test_load_source_view_command_null_span() {
        let (_dir, app) = test_engine().await;
        let engine = app.state::<LensEngine>();

        let text = "the whole document body";
        let source_id = seed_source(&engine, text).await;

        let view = load_source_view(source_id, None, None, engine)
            .await
            .unwrap();

        assert_eq!(view.before, text);
        assert!(view.marked.is_empty());
        assert!(!view.truncated);
    }
}
