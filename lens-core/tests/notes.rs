//! Integration tests for notes persistence (#24): the `notes` table (extended by
//! migration 0019), `NotesRepo`, and the `LensEngine` notes methods. Offline
//! (tempfile scratch DB via `LensEngine::for_test`).

use lens_core::{Citation, LensEngine, Locator, NoteOrigin};
use uuid::Uuid;

fn citation(source_id: &str, ordinal: u32) -> Citation {
    Citation {
        source_id: source_id.into(),
        ordinal,
        locators: vec![Locator {
            chunk_id: format!("chunk-{source_id}"),
            anchor: Some("s1".into()),
            section_path: Some("Intro".into()),
            page: Some(1),
            char_start: Some(0),
            char_end: Some(10),
        }],
    }
}

/// A chat note round-trips through the engine, lists newest-first, and deletes.
#[tokio::test]
async fn save_list_newest_first_and_delete() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("notes", None, None).await.unwrap();

    let cites = vec![citation("src-1", 1)];
    let first = engine
        .save_chat_note(&nb.id, "first answer [1].", Some(&cites), "msg-1")
        .await
        .unwrap();
    assert_eq!(first.origin, NoteOrigin::Chat);
    assert_eq!(first.source_message_id.as_deref(), Some("msg-1"));

    // Distinct created_at so ordering is deterministic (rfc3339 sorts lexically).
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    let second = engine
        .save_chat_note(&nb.id, "second answer [1].", Some(&cites), "msg-2")
        .await
        .unwrap();

    let listed = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].id, second.id, "newest note first");
    assert_eq!(listed[1].id, first.id);

    let parsed = listed[0].citations_parsed().unwrap();
    assert_eq!(parsed, Some(cites));

    engine.delete_note(&second.id).await.unwrap();
    let after = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].id, first.id);
}

/// A saved note's frozen `source_title` survives deletion of the originating
/// source (notes have no FK to `sources`) — the real durability threat (AC9).
#[tokio::test]
async fn source_deletion_leaves_note_and_frozen_title_intact() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("notes", None, None).await.unwrap();
    let pool = engine.pool().await;

    let source_id = Uuid::now_v7().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, created_at) \
         VALUES (?, ?, 'text', 'The Source Title', 'indexed', 'loc', ?)",
    )
    .bind(&source_id)
    .bind(&nb.id)
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let cites = vec![citation(&source_id, 1)];
    let note = engine
        .save_chat_note(&nb.id, "grounded [1].", Some(&cites), "msg-1")
        .await
        .unwrap();
    assert_eq!(note.source_title.as_deref(), Some("The Source Title"));

    sqlx::query("DELETE FROM sources WHERE id = ?")
        .bind(&source_id)
        .execute(&pool)
        .await
        .unwrap();

    let listed = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 1, "note survives source deletion");
    assert_eq!(
        listed[0].source_title.as_deref(),
        Some("The Source Title"),
        "frozen title unchanged after source deletion"
    );
}

/// A note saved with no citations has a NULL `source_title`, lists fine, and
/// `citations_parsed()` yields `None` (toggle keys on `source_message_id`).
#[tokio::test]
async fn null_citations_note_has_no_title() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("notes", None, None).await.unwrap();

    let note = engine
        .save_chat_note(&nb.id, "an uncited answer.", None, "msg-1")
        .await
        .unwrap();
    assert_eq!(note.source_title, None);
    assert_eq!(note.citations, None);

    let listed = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].source_message_id.as_deref(), Some("msg-1"));
    assert_eq!(listed[0].source_title, None);
    assert_eq!(listed[0].citations_parsed().unwrap(), None);
}

/// `NoteOrigin` rejects unknown persisted strings rather than mis-decoding.
#[test]
fn note_origin_enum_wire_values() {
    assert_eq!("chat".parse::<NoteOrigin>().unwrap(), NoteOrigin::Chat);
    assert_eq!("manual".parse::<NoteOrigin>().unwrap(), NoteOrigin::Manual);
    assert!("user".parse::<NoteOrigin>().is_err());
    assert!("".parse::<NoteOrigin>().is_err());
}
