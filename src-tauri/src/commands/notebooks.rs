//! Notebook commands. Thin pass-throughs to `lens-core`; full CRUD UI is M3.

use lens_core::{
    AddSourceOutcome, IngestProgress, LensEngine, LensError, Notebook, NotebookId, NotebookSummary,
    Source, TrashedSource,
};
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

use crate::stream::{StreamEvent, send_event};

/// Progress event streamed by [`set_notebook_embedding_model`] while re-embedding
/// chunks under the new coordinate. Mirrors [`IngestProgress`]'s shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReembedProgress {
    /// Chunks processed so far.
    pub done: usize,
    /// Total chunks to process.
    pub total: usize,
}

/// IPC return type for [`get_notebook_embedding_model`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    /// Canonical model id (e.g. `"nomic-embed-text-v1.5"`).
    pub model_id: String,
    /// Output vector dimension (e.g. `768`).
    pub dim: usize,
    /// Embedding backend serving this coordinate (`"fastembed"` | `"ollama"`).
    pub backend: String,
    /// Whether an active embedding_index row exists for this
    /// (notebook, backend, model, dim) coordinate: `"active"` when the coordinate
    /// is live, `"none"` when no index exists yet.
    pub status: String,
}

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

/// Inserts a file source record (no ingestion). On a path-based dedup hit (#100)
/// returns the existing live source with `wasExisting = true`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn add_source(
    notebook_id: String,
    title: String,
    locator: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
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

/// Inserts a managed text/markdown source. `kind` must be `"text"` or `"markdown"`.
/// On a content-dedup hit (#100) returns the existing live source with `wasExisting = true`.
#[tracing::instrument(skip(text, engine))]
#[tauri::command]
pub async fn add_text_source(
    notebook_id: String,
    title: String,
    text: String,
    kind: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
    engine
        .add_text_source(&NotebookId::from(notebook_id), &title, &text, &kind)
        .await
}

/// Inserts a URL source (`queued` row). Returns immediately — no HTTP fetch.
/// On a URL-based dedup hit (#100) returns the existing source with `wasExisting = true`.
/// `force_js_render` (#78) is `Option<bool>`: omitted by non-SPA callers → treated
/// as `false` (Tauri params cannot carry `#[serde(default)]`).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn add_url_source(
    notebook_id: String,
    title: String,
    url: String,
    force_js_render: Option<bool>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
    engine
        .add_url_source(
            &NotebookId::from(notebook_id),
            &title,
            &url,
            force_js_render.unwrap_or(false),
        )
        .await
}

/// Inserts a managed local-file source (PDF/DOCX/text/markdown). Kind is detected from
/// extension; unsupported extensions are rejected. On a content-dedup hit (#96) returns
/// the existing source with `wasExisting = true`. Call `ingest_source` to index.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn add_file_source(
    notebook_id: String,
    path: String,
    title: Option<String>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
    engine
        .add_file_source(
            &NotebookId::from(notebook_id),
            std::path::Path::new(&path),
            title.as_deref(),
        )
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

/// Ingests a queued source end-to-end (parse → chunk → embed → index), streaming
/// `Started` → `Chunk`/`Progress` per phase → `Done` or `Failed` over `on_progress`.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn ingest_source(
    source_id: String,
    on_progress: Channel<StreamEvent<IngestProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
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

/// Retries a failed source in place (#73), re-running the ingest pipeline on the same
/// row. Rejects non-`error` and trashed sources. Mirrors [`ingest_source`] progress stream.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn retry_ingest_source(
    source_id: String,
    on_progress: Channel<StreamEvent<IngestProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("retry_ingest_source: started event send failed: {e}");
    }

    let result = engine
        .retry_source(&source_id, |progress| {
            let done = progress.done;
            let total = progress.total;
            if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                tracing::warn!("retry_ingest_source: progress chunk send failed: {e}");
            }
            if let Err(e) = send_event(&on_progress, StreamEvent::Progress { done, total }) {
                tracing::warn!("retry_ingest_source: progress event send failed: {e}");
            }
        })
        .await;

    match &result {
        Ok(()) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("retry_ingest_source: done event send failed: {e}");
            }
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("retry_ingest_source: failed event send failed: {e}");
            }
        }
    }

    result
}

/// Retries every failed source in a notebook (#73). Sequential with continue-on-failure:
/// a source that fails again updates only its own `error_meta` and does not abort the
/// batch. All per-source progress streams over the shared `on_progress` channel.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn retry_all_failed_sources(
    notebook_id: String,
    on_progress: Channel<StreamEvent<IngestProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    let failed: Vec<String> = engine
        .list_sources(&NotebookId::from(notebook_id))
        .await?
        .into_iter()
        .filter(|s| s.status == lens_core::notebooks::SourceStatus::Error.as_str())
        .map(|s| s.id)
        .collect();

    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("retry_all_failed_sources: started event send failed: {e}");
    }

    for source_id in &failed {
        let result = engine
            .retry_source(source_id, |progress| {
                let done = progress.done;
                let total = progress.total;
                if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                    tracing::warn!("retry_all_failed_sources: progress chunk send failed: {e}");
                }
                if let Err(e) = send_event(&on_progress, StreamEvent::Progress { done, total }) {
                    tracing::warn!("retry_all_failed_sources: progress event send failed: {e}");
                }
            })
            .await;
        if let Err(err) = &result {
            tracing::warn!(
                source_id,
                "retry_all_failed_sources: source retry failed: {err}"
            );
        }
    }

    if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
        tracing::warn!("retry_all_failed_sources: done event send failed: {e}");
    }

    Ok(())
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

/// Bumps `last_activity_at` (records an "open" for cold-launch MRU auto-open).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn touch_notebook_activity(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine
        .touch_notebook_activity(&NotebookId::from(notebook_id))
        .await
}

/// Soft-deletes a notebook (backward-compat alias for `trash_notebook`). Recoverable.
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

/// Lists individually-trashed sources whose parent notebook is still live,
/// newest-trashed first. Used by the Trash modal Sources section (issue #94).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_trashed_sources(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<TrashedSource>, LensError> {
    engine.list_trashed_sources().await
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

/// Sets a notebook's embedding model and re-embeds all chunks, streaming
/// `Started` → `Chunk`/`Progress` per batch → `Done` or `Failed`. Unknown model
/// ids are rejected against the registry before any work begins.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn set_notebook_embedding_model(
    notebook_id: String,
    model_id: String,
    backend: String,
    on_progress: Channel<StreamEvent<ReembedProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    let nb_id = NotebookId::from(notebook_id);
    let backend = lens_core::EmbeddingBackend::from_opt_str(Some(&backend));

    engine
        .set_notebook_embedding_model(&nb_id, &model_id, backend)
        .await?;

    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("set_notebook_embedding_model: started event send failed: {e}");
    }

    let result = engine
        .reembed_notebook(&nb_id, |done, total| {
            let progress = ReembedProgress { done, total };
            if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                tracing::warn!("set_notebook_embedding_model: progress chunk send failed: {e}");
            }
            if let Err(e) = send_event(
                &on_progress,
                StreamEvent::Progress {
                    done: done as u64,
                    total: Some(total as u64),
                },
            ) {
                tracing::warn!("set_notebook_embedding_model: progress event send failed: {e}");
            }
        })
        .await;

    match &result {
        Ok(_outcome) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("set_notebook_embedding_model: done event send failed: {e}");
            }
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("set_notebook_embedding_model: failed event send failed: {e}");
            }
        }
    }

    result.map(|_| ())
}

/// Returns the notebook's current embedding model id, dimension, backend, and index
/// status (`"active"` when a live row exists for the full coordinate, `"none"` otherwise).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn get_notebook_embedding_model(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<EmbeddingModelInfo, LensError> {
    let nb_id = NotebookId::from(notebook_id);
    let (model_id, dim, backend, status) = engine.get_notebook_embedding_info(&nb_id).await?;
    Ok(EmbeddingModelInfo {
        model_id,
        dim,
        backend: backend.as_str().to_string(),
        status,
    })
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
        .unwrap()
        .source;
        assert_eq!(src.kind, "file");
        assert_eq!(src.status, "pending");
        assert_eq!(src.locator, "/abs/path/report.pdf");
        assert_eq!(src.selected, 1);

        let sources = list_sources(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id, src.id);

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

        let listed = list_notebooks(engine.clone()).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].source_count, 0);

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

        trash_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        assert_eq!(list_trashed(engine.clone()).await.unwrap().len(), 1);

        restore_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(list_notebooks(engine.clone()).await.unwrap().len(), 1);
        assert!(list_trashed(engine.clone()).await.unwrap().is_empty());

        trash_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        purge_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        assert!(list_trashed(engine).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn set_embedding_model_persists_and_get_returns_it() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("Embed Test".into(), None, None, engine.clone())
            .await
            .unwrap();

        let info = get_notebook_embedding_model(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(info.model_id, "nomic-embed-text-v1.5");
        assert_eq!(info.dim, 768);
        assert_eq!(info.backend, "fastembed");
        assert_eq!(info.status, "none");

        engine
            .set_notebook_embedding_model(
                &lens_core::NotebookId::from(nb.id.to_string()),
                "mxbai-embed-large",
                lens_core::EmbeddingBackend::Fastembed,
            )
            .await
            .unwrap();

        let info2 = get_notebook_embedding_model(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(info2.model_id, "mxbai-embed-large");
        assert_eq!(info2.dim, 1024);
        assert_eq!(info2.backend, "fastembed");
        assert_eq!(info2.status, "none");
    }

    #[tokio::test]
    async fn set_embedding_model_rejects_unknown_id() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("Reject Test".into(), None, None, engine.clone())
            .await
            .unwrap();

        let err = engine
            .set_notebook_embedding_model(
                &lens_core::NotebookId::from(nb.id.to_string()),
                "totally-unknown-model",
                lens_core::EmbeddingBackend::Fastembed,
            )
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown embedding model id"),
            "expected validation error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn trash_restore_purge_source_lifecycle() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();
        let src = add_source(
            nb.id.to_string(),
            "report.pdf".into(),
            "/abs/report.pdf".into(),
            engine.clone(),
        )
        .await
        .unwrap();

        assert_eq!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(
            list_trashed_sources(engine.clone())
                .await
                .unwrap()
                .is_empty()
        );

        trash_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        assert!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .is_empty()
        );
        let trashed = list_trashed_sources(engine.clone()).await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].source.id, src.source.id);
        assert_eq!(trashed[0].notebook_title, "NB");

        restore_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        assert_eq!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(
            list_trashed_sources(engine.clone())
                .await
                .unwrap()
                .is_empty()
        );

        trash_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        purge_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        assert!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .is_empty()
        );
        assert!(list_trashed_sources(engine).await.unwrap().is_empty());
    }
}
