//! Integration tests for chat persistence (#22): the `chat_messages` table,
//! `ChatRepo`, and the `LensEngine` chat methods. Offline (tempfile scratch DB).

use lens_core::{ChatFeedback, ChatRole, ChatState, Citation, LensEngine, Locator};

/// A user then an assistant message round-trip through the engine in order, and
/// the citation payload survives DB → hydrate intact (AC16, AC17).
#[tokio::test]
async fn user_and_assistant_round_trip_with_citations() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();
    let turn = "turn-1";

    let user = engine
        .save_chat_user(&nb.id, turn, "what is it?")
        .await
        .unwrap();
    assert_eq!(user.role, ChatRole::User);
    assert_eq!(user.citations, None);

    let citations = vec![Citation {
        source_id: "src-1".into(),
        ordinal: 1,
        locators: vec![Locator {
            chunk_id: "chunk-1".into(),
            anchor: Some("s1".into()),
            section_path: Some("Intro".into()),
            page: Some(3),
            char_start: Some(0),
            char_end: Some(42),
        }],
    }];
    let assistant = engine
        .save_chat_assistant(&nb.id, turn, "it is [1].", Some(&citations), 128)
        .await
        .unwrap();
    assert_eq!(assistant.role, ChatRole::Assistant);
    assert_eq!(assistant.tokens_used, Some(128));

    let listed = engine.list_chat_messages(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].role, ChatRole::User);
    assert_eq!(listed[1].role, ChatRole::Assistant);
    assert_eq!(listed[0].turn_id, listed[1].turn_id);

    let parsed = listed[1].citations_parsed().unwrap();
    assert_eq!(parsed, Some(citations));
}

/// `save_chat_assistant(None, ..)` stores NULL citations; hydrate yields `None`.
#[tokio::test]
async fn assistant_without_citations_stores_null() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();
    engine.save_chat_user(&nb.id, "t", "q").await.unwrap();
    let a = engine
        .save_chat_assistant(&nb.id, "t", "a", None, 0)
        .await
        .unwrap();
    assert_eq!(a.citations, None);
    assert_eq!(a.citations_parsed().unwrap(), None);
}

/// An assistant insert for a turn with no user row is rejected (turn-integrity
/// guard), never silently written.
#[tokio::test]
async fn assistant_without_user_row_is_rejected() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();
    let err = engine
        .save_chat_assistant(&nb.id, "orphan-turn", "a", None, 0)
        .await
        .unwrap_err();
    assert!(matches!(err, lens_core::LensError::Validation(_)));
    assert!(engine.list_chat_messages(&nb.id).await.unwrap().is_empty());
}

/// Feedback is settable, switchable, and clearable back to NULL (AC14, AC22).
#[tokio::test]
async fn feedback_toggles_up_down_and_clears() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();
    engine.save_chat_user(&nb.id, "t", "q").await.unwrap();
    let a = engine
        .save_chat_assistant(&nb.id, "t", "a", None, 1)
        .await
        .unwrap();

    engine
        .set_chat_feedback(&a.id, Some(ChatFeedback::Down))
        .await
        .unwrap();
    assert_eq!(
        feedback_of(&engine, &nb.id, &a.id).await,
        Some(ChatFeedback::Down)
    );

    engine
        .set_chat_feedback(&a.id, Some(ChatFeedback::Up))
        .await
        .unwrap();
    assert_eq!(
        feedback_of(&engine, &nb.id, &a.id).await,
        Some(ChatFeedback::Up)
    );

    engine.set_chat_feedback(&a.id, None).await.unwrap();
    assert_eq!(feedback_of(&engine, &nb.id, &a.id).await, None);
}

async fn feedback_of(
    engine: &LensEngine,
    nb: &lens_core::NotebookId,
    id: &str,
) -> Option<ChatFeedback> {
    engine
        .list_chat_messages(nb)
        .await
        .unwrap()
        .into_iter()
        .find(|m| m.id == id)
        .unwrap()
        .feedback
}

/// Purging a notebook cascades: zero orphan chat rows remain (AC18).
#[tokio::test]
async fn purging_notebook_cascades_chat_messages() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();
    engine.save_chat_user(&nb.id, "t", "q").await.unwrap();
    engine
        .save_chat_assistant(&nb.id, "t", "a", None, 5)
        .await
        .unwrap();

    let pool = engine.pool().await;
    let before: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chat_messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(before, 2);

    // `purge_notebook` is the hard-delete path (`delete_notebook` soft-deletes),
    // so it is what exercises the SQLite ON DELETE CASCADE.
    engine.trash_notebook(&nb.id).await.unwrap();
    engine.purge_notebook(&nb.id).await.unwrap();

    let after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chat_messages")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        after, 0,
        "ON DELETE CASCADE must leave zero orphan chat rows"
    );
}

/// `insert_terminal_marker` (Plan 2 / PC-1) after a user row persists a marker
/// that reloads with the given state and partial content; a normal user row in
/// the same turn still reloads with `state = None`.
#[tokio::test]
async fn terminal_marker_persists_state_and_reloads() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();
    let turn = "turn-1";

    engine
        .save_chat_user(&nb.id, turn, "what is it?")
        .await
        .unwrap();
    let marker = engine
        .save_chat_marker(
            &nb.id,
            turn,
            "partial answer so f",
            ChatState::Cancelled,
            None,
        )
        .await
        .unwrap();
    assert_eq!(marker.state, Some(ChatState::Cancelled));
    assert_eq!(marker.content, "partial answer so f");

    let listed = engine.list_chat_messages(&nb.id).await.unwrap();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].role, ChatRole::User);
    assert_eq!(
        listed[0].state, None,
        "a normal user row must reload with no state"
    );
    assert_eq!(listed[1].id, marker.id);
    assert_eq!(listed[1].state, Some(ChatState::Cancelled));
    assert_eq!(listed[1].content, "partial answer so f");
}

/// An `Errored` marker persists `error_kind` alongside `state` and both reload
/// intact.
#[tokio::test]
async fn errored_marker_persists_error_kind() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();
    let turn = "turn-1";

    engine.save_chat_user(&nb.id, turn, "q").await.unwrap();
    let marker = engine
        .save_chat_marker(&nb.id, turn, "", ChatState::Errored, Some("Network"))
        .await
        .unwrap();
    assert_eq!(marker.state, Some(ChatState::Errored));
    assert_eq!(marker.error_kind, Some("Network".to_string()));

    let listed = engine.list_chat_messages(&nb.id).await.unwrap();
    let reloaded = listed.iter().find(|m| m.id == marker.id).unwrap();
    assert_eq!(reloaded.state, Some(ChatState::Errored));
    assert_eq!(reloaded.error_kind, Some("Network".to_string()));
}

/// `insert_terminal_marker` for a turn with no user row is rejected, mirroring
/// `insert_assistant`'s turn-integrity guard.
#[tokio::test]
async fn terminal_marker_without_user_row_is_rejected() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();

    let err = engine
        .save_chat_marker(&nb.id, "orphan-turn", "", ChatState::Cancelled, None)
        .await
        .unwrap_err();
    assert!(matches!(err, lens_core::LensError::Validation(_)));
    assert!(engine.list_chat_messages(&nb.id).await.unwrap().is_empty());
}

/// `ChatRepo::history` excludes the current turn's rows and skips marker rows:
/// a cancelled marker from an earlier turn must not surface as prior context.
#[tokio::test]
async fn history_excludes_current_turn_and_skips_marker_rows() {
    let engine = LensEngine::for_test().await;
    let nb = engine.create_notebook("chat", None, None).await.unwrap();

    engine.save_chat_user(&nb.id, "t1", "q1").await.unwrap();
    engine
        .save_chat_marker(&nb.id, "t1", "partial", ChatState::Cancelled, None)
        .await
        .unwrap();

    engine.save_chat_user(&nb.id, "t2", "q2").await.unwrap();
    engine
        .save_chat_assistant(&nb.id, "t2", "a2", None, 0)
        .await
        .unwrap();

    engine.save_chat_user(&nb.id, "t3", "q3").await.unwrap();

    let pool = engine.pool().await;
    let history = lens_core::chat::ChatRepo::new(&pool)
        .history(nb.id.as_str(), "t3", 10)
        .await
        .unwrap();

    let contents: Vec<&str> = history.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(
        contents,
        vec!["q2", "a2"],
        "t1 is incomplete (only a cancelled marker, no real answer) so its user row is \
         dropped WHOLE — no dangling question; t3 (current turn) excluded"
    );
}

/// Both chat enums reject an unknown persisted string rather than silently
/// mis-decoding (enum discipline).
#[test]
fn chat_enums_reject_unknown_strings() {
    assert!("maybe".parse::<ChatFeedback>().is_err());
    assert!("".parse::<ChatFeedback>().is_err());
    assert!("system".parse::<ChatRole>().is_err());

    assert_eq!("up".parse::<ChatFeedback>().unwrap(), ChatFeedback::Up);
    assert_eq!("down".parse::<ChatFeedback>().unwrap(), ChatFeedback::Down);
    assert_eq!("user".parse::<ChatRole>().unwrap(), ChatRole::User);
    assert_eq!(
        "assistant".parse::<ChatRole>().unwrap(),
        ChatRole::Assistant
    );
}
