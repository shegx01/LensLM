//! Integration tests for notes persistence (#24): the `notes` table (extended by
//! migration 0019), `NotesRepo`, and the `LensEngine` notes methods. Offline
//! (tempfile scratch DB via `LensEngine::for_test`).

use lens_core::{Citation, LensEngine, LensError, Locator, NoteOrigin};
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

/// A manual note persists `origin=manual` with NULL citations/source, lists
/// newest-first alongside chat notes, and empty content is rejected (#25).
#[tokio::test]
async fn manual_note_persists_and_lists_alongside_chat() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("notes", None, None).await.unwrap();

    let chat = engine
        .save_chat_note(
            &nb.id,
            "grounded [1].",
            Some(&[citation("src-1", 1)]),
            "msg-1",
        )
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    let manual = engine
        .save_manual_note(&nb.id, "a personal thought")
        .await
        .unwrap();
    assert_eq!(manual.origin, NoteOrigin::Manual);
    assert_eq!(manual.content, "a personal thought");
    assert_eq!(manual.citations, None);
    assert_eq!(manual.source_title, None);
    assert_eq!(manual.source_message_id, None);

    let listed = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].id, manual.id, "newest (manual) note first");
    assert_eq!(listed[1].id, chat.id);

    assert!(
        engine.save_manual_note(&nb.id, "   ").await.is_err(),
        "empty/whitespace content rejected"
    );
}

/// Editing a chat note bumps `updated_at`, keeps `[n]` markers in content that
/// the edit leaves intact, and preserves grounding columns (AC4).
#[tokio::test]
async fn update_note_bumps_updated_at_and_preserves_grounding() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("notes", None, None).await.unwrap();

    let cites = vec![citation("src-1", 1)];
    let note = engine
        .save_chat_note(&nb.id, "grounded answer [1].", Some(&cites), "msg-1")
        .await
        .unwrap();
    assert_eq!(
        note.created_at, note.updated_at,
        "fresh note not yet edited"
    );

    // Ensure the bumped timestamp is strictly later (rfc3339 sorts lexically).
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    let edited = engine
        .update_note(&note.id, "grounded answer [1] with more detail.")
        .await
        .unwrap();

    assert_eq!(edited.content, "grounded answer [1] with more detail.");
    assert!(
        edited.content.contains("[1]"),
        "citation marker survives an edit that leaves it intact"
    );
    assert_ne!(
        edited.updated_at, edited.created_at,
        "updated_at bumped past created_at"
    );
    assert_eq!(edited.created_at, note.created_at, "created_at unchanged");
    assert_eq!(edited.origin, NoteOrigin::Chat);
    assert_eq!(edited.source_message_id.as_deref(), Some("msg-1"));
    assert_eq!(edited.citations_parsed().unwrap(), Some(cites));

    // Rehydrate from the DB to confirm the persisted row matches.
    let listed = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].content, "grounded answer [1] with more detail.");
    assert_eq!(listed[0].source_message_id.as_deref(), Some("msg-1"));
    assert_ne!(listed[0].updated_at, listed[0].created_at);
}

/// Empty/whitespace edits are rejected by the engine wrapper (AC5) — the single
/// enforcement point, mirroring `save_manual_note`.
#[tokio::test]
async fn update_note_rejects_empty_content() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("notes", None, None).await.unwrap();

    let note = engine
        .save_manual_note(&nb.id, "original content")
        .await
        .unwrap();

    let err = engine.update_note(&note.id, "   ").await.unwrap_err();
    assert!(
        matches!(err, LensError::Validation(_)),
        "whitespace-only edit is a Validation error, got {err:?}"
    );

    // The original content is untouched (guard runs before the repo write).
    let listed = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].content, "original content");
}

/// Pinning the older note floats it to the top of `list_notes` (AC8).
#[tokio::test]
async fn set_pinned_floats_note_to_top() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("notes", None, None).await.unwrap();

    let older = engine.save_manual_note(&nb.id, "older note").await.unwrap();
    assert!(!older.pinned, "notes start unpinned");

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    let newer = engine.save_manual_note(&nb.id, "newer note").await.unwrap();

    // Unpinned: newest first.
    let before = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(before[0].id, newer.id);
    assert_eq!(before[1].id, older.id);

    let pinned = engine.set_note_pinned(&older.id, true).await.unwrap();
    assert!(pinned.pinned);

    // Pinned older note now floats above the newer unpinned one.
    let after = engine.list_notes(&nb.id).await.unwrap();
    assert_eq!(after[0].id, older.id, "pinned note floats to top");
    assert!(after[0].pinned);
    assert_eq!(after[1].id, newer.id);
    assert!(!after[1].pinned);
}

/// `NoteOrigin` rejects unknown persisted strings rather than mis-decoding.
#[test]
fn note_origin_enum_wire_values() {
    assert_eq!("chat".parse::<NoteOrigin>().unwrap(), NoteOrigin::Chat);
    assert_eq!("manual".parse::<NoteOrigin>().unwrap(), NoteOrigin::Manual);
    assert!("user".parse::<NoteOrigin>().is_err());
    assert!("".parse::<NoteOrigin>().is_err());
}
