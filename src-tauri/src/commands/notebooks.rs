//! Notebook commands. Thin pass-throughs to `lens-core`; full CRUD UI is M3.

use lens_core::{LensEngine, LensError, Notebook, NotebookId, Source};

/// Lists live (non-trashed) notebooks, newest first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_notebooks(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<Notebook>, LensError> {
    engine.list_notebooks().await
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

/// Deletes a notebook (child rows cascade).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn delete_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.delete_notebook(&NotebookId::from(id)).await
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
        assert_eq!(listed[0].id, created.id);
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
        assert_eq!(listed[0].description.as_deref(), Some("My notes"));
        assert_eq!(listed[0].focus_mode.as_deref(), Some("research"));
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
    async fn rename_then_delete() {
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
        assert_eq!(listed[0].title, "Renamed");

        delete_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine).await.unwrap().is_empty());
    }
}
