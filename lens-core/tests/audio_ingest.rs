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

use lens_core::{LensEngine, MockAsrEngine, TranscriptSegment};
use sqlx::Row;

mod support;
use support::{inject_counting_engine, tokenizer_available};

/// Writes a mono 16 kHz PCM16 WAV of `seconds` seconds carrying a 440 Hz tone
/// (nonzero, so it survives the all-silent guard) to `path`. At the default
/// ~30 s window this yields `ceil(seconds / 30)` decode windows — pass ≥ 61 s
/// for the ≥ 3 windows the deterministic cancel test needs.
fn write_tone_wav(path: &Path, seconds: u32) {
    const SAMPLE_RATE: u32 = 16_000;
    let n_samples = SAMPLE_RATE * seconds;
    let data_len = n_samples * 2; // 16-bit mono
    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_len as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());

    for i in 0..n_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        let s = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5;
        let v = (s * i16::MAX as f32) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }

    std::fs::write(path, &buf).expect("write tone wav");
}

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

/// Routes `LensEngine::transcribe` to the injected mock: the `apple_native`
/// backend with an injected engine hits the `(AppleNative, Some)` arm (the mock
/// is the Apple-native seam in tests).
async fn use_mock_asr(engine: &LensEngine, segments: Vec<TranscriptSegment>) {
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    engine.set_config(config).await;
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(segments))))
        .await;
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
    assert_eq!(src.kind, "audio", "wav maps to the audio kind");

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

/// AC6: cancelling mid-decode returns `Err(Cancelled)`, flowing through the
/// existing error handler — status `error` with `error_meta.kind = "Cancelled"`,
/// no chunks persisted, and the registry entry removed.
///
/// Determinism: the bounded decode-progress channel (capacity 1) pauses the
/// decode thread on its second `blocking_send`. The test awaits the first
/// decoding event (proving decode started and is now blocked), cancels, then
/// resumes draining — the closure observes the cancellation on its next window.
/// A ≥ 61 s fixture guarantees ≥ 3 decode windows so this branch is reached.
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

    // Wait until decode has emitted its first window (closure now blocked on the
    // next bounded `blocking_send`), then flip the cancellation token.
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
    assert_eq!(row.get::<String, _>("status"), "error");
    let meta_json: String = row
        .get::<Option<String>, _>("error_meta")
        .expect("error_meta set");
    assert!(
        meta_json.contains("\"kind\":\"Cancelled\""),
        "error_meta.kind must be Cancelled: {meta_json}"
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
