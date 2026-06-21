//! Notebook commands. Thin pass-throughs to `lens-core`; full CRUD UI is M3.

use lens_core::{LensEngine, LensError, Notebook, NotebookId};

/// Lists live (non-trashed) notebooks, newest first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_notebooks(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<Notebook>, LensError> {
    engine.list_notebooks().await
}

/// Creates a notebook with the given title and returns it.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn create_notebook(
    title: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Notebook, LensError> {
    engine.create_notebook(&title).await
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

        let created = create_notebook("My Notebook".into(), engine.clone())
            .await
            .unwrap();
        assert_eq!(created.title, "My Notebook");

        let listed = list_notebooks(engine).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, created.id);
    }

    #[tokio::test]
    async fn rename_then_delete() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("Original".into(), engine.clone())
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
