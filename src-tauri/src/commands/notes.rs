//! Notes IPC commands (#24): save a grounded chat answer as a durable note,
//! and list/delete a notebook's notes. Thin typed boundary over `lens-core`.

use lens_core::{Citation, LensEngine, LensError, Note, NotebookId};

/// Saves a completed assistant answer as an `origin=chat` note snapshot.
#[tauri::command]
pub async fn save_chat_note(
    notebook_id: String,
    content: String,
    citations: Option<Vec<Citation>>,
    source_message_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Note, LensError> {
    engine
        .save_chat_note(
            &NotebookId::from(notebook_id),
            &content,
            citations.as_deref(),
            &source_message_id,
        )
        .await
}

/// Saves a user-authored manual note (#25).
#[tauri::command]
pub async fn save_manual_note(
    notebook_id: String,
    content: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Note, LensError> {
    engine
        .save_manual_note(&NotebookId::from(notebook_id), &content)
        .await
}

/// Lists a notebook's notes, newest first.
#[tauri::command]
pub async fn list_notes(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<Note>, LensError> {
    engine.list_notes(&NotebookId::from(notebook_id)).await
}

/// Updates a note's content (#25). Routes through the engine wrapper, which
/// enforces the empty/whitespace guard — do NOT call `NotesRepo::update_note`
/// directly here (that would bypass the guard).
#[tauri::command]
pub async fn update_note(
    note_id: String,
    content: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Note, LensError> {
    engine.update_note(&note_id, &content).await
}

/// Sets a note's pinned flag (#25, pin-to-top).
#[tauri::command]
pub async fn set_note_pinned(
    note_id: String,
    pinned: bool,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Note, LensError> {
    engine.set_note_pinned(&note_id, pinned).await
}

/// Deletes a note by id (drives chat toggle-unsave).
#[tauri::command]
pub async fn delete_note(
    note_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.delete_note(&note_id).await
}
