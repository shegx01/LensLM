//! Integration tests for the audio ingest branch (issue #43): end-to-end
//! decode → transcribe (MockAsrEngine) → chunk/embed/index, deterministic
//! cancellation, and the all-silent EmptyAudio guard.
//!
//! Offline: uses the model-free `MockAsrEngine` (routed via `apple_native`) and
//! the `CountingEmbedder`; no `LENS_RUN_MODEL_TESTS`, no downloads. Chunk-bearing
//! assertions still need the nomic tokenizer, so they skip cleanly when offline.
//!
//! `ingest_phase` is `pub(crate)`, so these tests assert on the string literals
//! `"decoding"`/`"transcribing"` rather than importing the constants.

#![recursion_limit = "256"]

use std::path::Path;
use std::sync::Arc;

use lens_core::TranscriptSegment;
use sqlx::Row;

mod support;
use support::{inject_counting_engine, tokenizer_available, use_mock_asr, write_tone_wav};

fn canned_segments() -> Vec<TranscriptSegment> {
    vec![
        TranscriptSegment {
            text: "hello world".to_string(),
            start_second: 0.0,
            end_second: 1.0,
        },
        TranscriptSegment {
            text: "foo bar".to_string(),
            start_second: 1.0,
            end_second: 2.0,
        },
    ]
}

/// AC3/AC4: an audio source ingests end-to-end — status `indexed`, chunks
/// persisted with the concatenated transcript, and the stream emits both the
/// `decoding` and `transcribing` phases plus `done`.
#[tokio::test]
async fn audio_ingest_end_to_end_indexed_and_searchable() {
    if !tokenizer_available().await {
        eprintln!("skipping audio_ingest_end_to_end: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    use_mock_asr(&engine, canned_segments()).await;

    let wav = dir.path().join("clip.wav");
    write_tone_wav(&wav, 2);

    let nb = engine
        .create_notebook("audio-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_file_source(&nb.id, &wav, None)
        .await
        .unwrap()
        .source;
    assert_eq!(src.kind, lens_core::parse::SourceKind::Audio, "wav maps to the audio kind");

    let mut phases: Vec<String> = Vec::new();
    engine
        .ingest_source(&src.id, |p| phases.push(p.phase))
        .await
        .expect("audio ingest should succeed");

    assert!(
        phases.iter().any(|p| p == "decoding"),
        "expected a decoding phase, got {phases:?}"
    );
    assert!(
        phases.iter().any(|p| p == "transcribing"),
        "expected a transcribing phase, got {phases:?}"
    );
    assert_eq!(phases.last().map(String::as_str), Some("done"));

    let pool = engine.pool().await;
    let status: String = sqlx::query("SELECT status FROM sources WHERE id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .get("status");
    assert_eq!(status, "indexed");

    let chunk_text: String =
        sqlx::query("SELECT group_concat(text, '\n') AS t FROM chunks WHERE source_id = ?")
            .bind(&src.id)
            .fetch_one(&pool)
            .await
            .unwrap()
            .get("t");
    assert!(
        chunk_text.contains("hello world"),
        "chunk text missing first segment: {chunk_text:?}"
    );
    assert!(
        chunk_text.contains("foo bar"),
        "chunk text missing second segment: {chunk_text:?}"
    );
}

#[tokio::test]
async fn audio_ingest_cancel_is_deterministic() {
    let (dir, engine) = inject_counting_engine().await;
    let engine = Arc::new(engine);
    use_mock_asr(&engine, canned_segments()).await;

    let wav = dir.path().join("long.wav");
    write_tone_wav(&wav, 61);

    let nb = engine
        .create_notebook("audio-cancel-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_file_source(&nb.id, &wav, None)
        .await
        .unwrap()
        .source;
    let source_id = src.id.clone();

    {
        let pool = engine.pool().await;
        sqlx::query("UPDATE sources SET error_meta = ? WHERE id = ?")
            .bind(r#"{"kind":"Network","message":"prior failure","attempt_count":1}"#)
            .bind(&source_id)
            .execute(&pool)
            .await
            .unwrap();
    }

    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let mut started_tx = Some(started_tx);

    let ingest_engine = engine.clone();
    let ingest_id = source_id.clone();
    let handle = tokio::spawn(async move {
        ingest_engine
            .ingest_source(&ingest_id, move |p| {
                if p.phase == "decoding"
                    && let Some(tx) = started_tx.take()
                {
                    let _ = tx.send(());
                }
            })
            .await
    });

    started_rx.await.expect("decode should emit a first window");
    assert!(
        engine.cancel_media_ingest(&source_id),
        "a token must be registered while the ingest is in flight"
    );

    let result = handle.await.expect("ingest task join");
    let err = result.expect_err("cancelled ingest must return Err");
    assert_eq!(err.kind(), "Cancelled", "error kind must be Cancelled");

    let pool = engine.pool().await;
    let row = sqlx::query("SELECT status, error_meta FROM sources WHERE id = ?")
        .bind(&source_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        row.get::<String, _>("status"),
        "queued",
        "a cancelled ingest must reset status to queued, not error"
    );
    assert!(
        row.get::<Option<String>, _>("error_meta").is_none(),
        "a cancel must clear error_meta, even one left by a prior failed attempt"
    );

    let chunk_count: i64 = sqlx::query("SELECT COUNT(*) AS c FROM chunks WHERE source_id = ?")
        .bind(&source_id)
        .fetch_one(&pool)
        .await
        .unwrap()
        .get("c");
    assert_eq!(
        chunk_count, 0,
        "no chunks may persist for a cancelled ingest"
    );

    assert!(
        !engine.cancel_media_ingest(&source_id),
        "the registry entry must be removed after the ingest exits"
    );
}

/// AC8/silence: an all-silent audio source fails with `EmptyAudio` (the guard
/// replicated from `decode_and_resample_audio`), and never reaches transcription.
#[tokio::test]
async fn audio_ingest_all_silent_is_empty_audio() {
    let (dir, engine) = inject_counting_engine().await;
    use_mock_asr(&engine, canned_segments()).await;

    // The #41 fixture is a 1 s all-zero 16 kHz mono wav.
    let silent =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio/silence_16000_mono.wav");
    let dest = dir.path().join("silent.wav");
    std::fs::copy(&silent, &dest).expect("copy silence fixture");

    let nb = engine
        .create_notebook("audio-silent-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_file_source(&nb.id, &dest, None)
        .await
        .unwrap()
        .source;

    let err = engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect_err("all-silent audio must error");
    assert_eq!(err.kind(), "EmptyAudio", "expected EmptyAudio, got {err:?}");
}
