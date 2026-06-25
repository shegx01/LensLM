//! Notebook commands. Thin pass-throughs to `lens-core`; full CRUD UI is M3.

use lens_core::{
    IngestProgress, LensEngine, LensError, Notebook, NotebookId, NotebookSummary, Source,
};
use tauri::ipc::Channel;

use crate::stream::{StreamEvent, send_event};

/// Lists live (non-trashed) notebooks with their source counts, newest first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_notebooks(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<NotebookSummary>, LensError> {
    engine.list_notebooks_with_counts().await
}

/// Creates a notebook with the given title and optional onboarding
/// `description`/`focus_mode`, and returns it.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn create_notebook(
    title: String,
    description: Option<String>,
    focus_mode: Option<String>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Notebook, LensError> {
    engine
        .create_notebook(&title, description.as_deref(), focus_mode.as_deref())
        .await
}

/// Inserts a file source record for a notebook (M1 onboarding "Add sources").
/// Records only — no ingestion. Returns the inserted source.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn add_source(
    notebook_id: String,
    title: String,
    locator: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Source, LensError> {
    engine
        .add_source(&NotebookId::from(notebook_id), &title, &locator)
        .await
}

/// Lists all sources for a notebook, newest first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_sources(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<Source>, LensError> {
    engine.list_sources(&NotebookId::from(notebook_id)).await
}

/// Inserts a managed text/markdown source (paste-text or `.md`/`.txt` content),
/// writing the text to a managed file and inserting a `queued` row. `kind` must
/// be `"text"` or `"markdown"`. Returns the inserted source.
#[tracing::instrument(skip(text, engine))]
#[tauri::command]
pub async fn add_text_source(
    notebook_id: String,
    title: String,
    text: String,
    kind: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Source, LensError> {
    engine
        .add_text_source(&NotebookId::from(notebook_id), &title, &text, &kind)
        .await
}

/// Inserts a URL source: inserts a `queued` row whose `locator` is the URL.
/// Returns immediately — no HTTP fetch happens here.
/// Call `ingest_source` separately to fetch + extract the page in the background.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn add_url_source(
    notebook_id: String,
    title: String,
    url: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Source, LensError> {
    engine
        .add_url_source(&NotebookId::from(notebook_id), &title, &url)
        .await
}

/// Soft-deletes a source: sets `trashed_at` to now. Keeps chunks + Lance
/// vectors so the source can be restored. Errors if missing or already trashed.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn trash_source(
    source_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.trash_source(&source_id).await
}

/// Restores a trashed source: clears `trashed_at`. Errors if live or missing.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn restore_source(
    source_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.restore_source(&source_id).await
}

/// Permanently deletes a source: drops its Lance vectors then removes the
/// `sources` row. Errors if the source does not exist.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn purge_source(
    source_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.purge_source(&source_id).await
}

/// Toggles a source's `selected` flag (persisted across reloads).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn set_source_selected(
    source_id: String,
    selected: bool,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.set_source_selected(&source_id, selected).await
}

/// Ingests a queued source end-to-end (parse → chunk → embed → index),
/// streaming progress over `on_progress` as `StreamEvent<IngestProgress>`.
///
/// Emits `Started`, then a `Progress { done, total }` plus a `Chunk` carrying
/// the per-phase [`IngestProgress`] for each pipeline phase, then `Done` on
/// success or `Failed(LensError)` on failure (the source is left in `error`).
///
/// Invoked as `invoke("ingest_source", { sourceId, onProgress })` where
/// `onProgress` is a `Channel<StreamEvent<IngestProgress>>`.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn ingest_source(
    source_id: String,
    on_progress: Channel<StreamEvent<IngestProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    // A send failure means the frontend dropped the channel; log and keep going
    // (the ingest itself is unaffected and will still complete).
    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("ingest_source: started event send failed: {e}");
    }

    let result = engine
        .ingest_source(&source_id, |progress| {
            let done = progress.done;
            let total = progress.total;
            if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                tracing::warn!("ingest_source: progress chunk send failed: {e}");
            }
            if let Err(e) = send_event(&on_progress, StreamEvent::Progress { done, total }) {
                tracing::warn!("ingest_source: progress event send failed: {e}");
            }
        })
        .await;

    match &result {
        Ok(()) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("ingest_source: done event send failed: {e}");
            }
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("ingest_source: failed event send failed: {e}");
            }
        }
    }

    result
}

/// Renames a notebook.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn rename_notebook(
    id: String,
    title: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.rename_notebook(&NotebookId::from(id), &title).await
}

/// Soft-deletes a notebook (backward-compat alias for `trash_notebook`).
///
/// Sets `trashed_at` rather than hard-deleting; the notebook is recoverable from
/// Trash. `purge_notebook` is the only permanent delete.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn delete_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.trash_notebook(&NotebookId::from(id)).await
}

/// Soft-deletes a notebook: sets `trashed_at`, bumps `updated_at`. Recoverable.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn trash_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.trash_notebook(&NotebookId::from(id)).await
}

/// Restores a trashed notebook: clears `trashed_at`, bumps `updated_at`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn restore_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.restore_notebook(&NotebookId::from(id)).await
}

/// Lists trashed notebooks with their source counts, newest-trashed first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_trashed(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<NotebookSummary>, LensError> {
    engine.list_trashed_with_counts().await
}

/// Permanently deletes a notebook (child rows cascade). Used by "Delete forever".
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn purge_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.purge_notebook(&NotebookId::from(id)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::Manager;

    #[tokio::test]
    async fn list_is_empty_then_reflects_create() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());

        let created = create_notebook("My Notebook".into(), None, None, engine.clone())
            .await
            .unwrap();
        assert_eq!(created.title, "My Notebook");

        let listed = list_notebooks(engine).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].notebook.id, created.id);
    }

    #[tokio::test]
    async fn create_notebook_persists_description_and_focus_mode() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let created = create_notebook(
            "Research".into(),
            Some("My notes".into()),
            Some("research".into()),
            engine.clone(),
        )
        .await
        .unwrap();
        assert_eq!(created.description.as_deref(), Some("My notes"));
        assert_eq!(created.focus_mode.as_deref(), Some("research"));

        let listed = list_notebooks(engine).await.unwrap();
        assert_eq!(listed[0].notebook.description.as_deref(), Some("My notes"));
        assert_eq!(listed[0].notebook.focus_mode.as_deref(), Some("research"));
    }

    #[tokio::test]
    async fn add_source_then_list_sources_scoped_by_notebook() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();
        let other = create_notebook("Other".into(), None, None, engine.clone())
            .await
            .unwrap();

        let src = add_source(
            nb.id.to_string(),
            "report.pdf".into(),
            "/abs/path/report.pdf".into(),
            engine.clone(),
        )
        .await
        .unwrap();
        assert_eq!(src.kind, "file");
        assert_eq!(src.status, "pending");
        assert_eq!(src.locator, "/abs/path/report.pdf");
        assert_eq!(src.selected, 1);

        let sources = list_sources(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id, src.id);

        // Sources are scoped to their notebook.
        assert!(
            list_sources(other.id.to_string(), engine)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn rename_then_delete_is_soft() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("Original".into(), None, None, engine.clone())
            .await
            .unwrap();
        rename_notebook(nb.id.to_string(), "Renamed".into(), engine.clone())
            .await
            .unwrap();
        let listed = list_notebooks(engine.clone()).await.unwrap();
        assert_eq!(listed[0].notebook.title, "Renamed");

        // `delete_notebook` is now a soft-delete: the notebook leaves the live
        // list but appears in the trashed list (recoverable).
        delete_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        let trashed = list_trashed(engine).await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].notebook.id, nb.id);
        assert!(trashed[0].notebook.trashed_at.is_some());
    }

    #[tokio::test]
    async fn list_notebooks_includes_source_count() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();

        // Zero sources -> count is 0.
        let listed = list_notebooks(engine.clone()).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].source_count, 0);

        // Add two sources -> count is 2.
        for i in 0..2 {
            add_source(
                nb.id.to_string(),
                format!("file{i}.pdf"),
                format!("/abs/file{i}.pdf"),
                engine.clone(),
            )
            .await
            .unwrap();
        }
        let listed = list_notebooks(engine).await.unwrap();
        assert_eq!(listed[0].source_count, 2);
    }

    #[tokio::test]
    async fn trash_restore_purge_lifecycle() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();

        // Trash: leaves live list, enters trashed list.
        trash_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        assert_eq!(list_trashed(engine.clone()).await.unwrap().len(), 1);

        // Restore: returns to live list, leaves trashed list.
        restore_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(list_notebooks(engine.clone()).await.unwrap().len(), 1);
        assert!(list_trashed(engine.clone()).await.unwrap().is_empty());

        // Trash again, then purge: gone from both lists.
        trash_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        purge_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        assert!(list_trashed(engine).await.unwrap().is_empty());
    }
}
