//! Gated real-model Whisper test (#42 Unit 5). Downloads the base ggml model and
//! a public-domain spoken-word sample, decodes it via #41's
//! [`decode_and_resample_audio`], transcribes through the real `WhisperEngine`,
//! and asserts non-empty, time-ordered segments.
//!
//! Gated twice so a plain `cargo test` never runs it:
//!
//! * `#[ignore]` — excluded unless `--ignored`/`--include-ignored` is passed.
//! * `LENS_RUN_MODEL_TESTS=1` — the repo's opt-in for model/network tests.
//!
//! Requires the `local-whisper` feature (the whole file is cfg-gated on it).

#![cfg(feature = "local-whisper")]

use std::path::Path;

use lens_core::{
    AsrEngine, DEFAULT_WHISPER_MODEL_ID, TranscribeConfig, WhisperEngine,
    decode_and_resample_audio, download_whisper_model, whisper_model_path,
};

/// Canonical public-domain JFK sample shipped with whisper.cpp (16 kHz mono WAV).
const JFK_SAMPLE_URL: &str =
    "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav";

fn model_tests_enabled() -> bool {
    std::env::var("LENS_RUN_MODEL_TESTS").is_ok()
}

async fn fetch_to(path: &Path, url: &str) {
    let bytes = reqwest::get(url)
        .await
        .expect("fetch sample")
        .error_for_status()
        .expect("sample http status")
        .bytes()
        .await
        .expect("read sample bytes");
    std::fs::write(path, &bytes).expect("write sample");
}

#[tokio::test]
#[ignore = "downloads the base ggml model + a spoken sample; run with LENS_RUN_MODEL_TESTS=1 --ignored"]
async fn whisper_transcribes_fixture() {
    if !model_tests_enabled() {
        eprintln!("skipping whisper_transcribes_fixture (set LENS_RUN_MODEL_TESTS=1)");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");

    // Fetch the base model into the standard cache layout, then load it offline.
    download_whisper_model(dir.path(), DEFAULT_WHISPER_MODEL_ID, |_| {})
        .await
        .expect("download base whisper model");
    let model_path = whisper_model_path(dir.path(), DEFAULT_WHISPER_MODEL_ID);
    let engine = WhisperEngine::load(&model_path).expect("load whisper engine");

    // Fetch a spoken-word sample and decode it to 16 kHz mono f32 PCM (#41).
    let sample_path = dir.path().join("jfk.wav");
    fetch_to(&sample_path, JFK_SAMPLE_URL).await;
    let pcm = decode_and_resample_audio(&sample_path).expect("decode sample");
    assert!(!pcm.is_empty(), "decoded PCM must be non-empty");

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let segments = engine
        .transcribe_pcm(&pcm, &TranscribeConfig::default(), Some(tx))
        .await
        .expect("transcribe");

    assert!(!segments.is_empty(), "expected non-empty segments");
    for seg in &segments {
        assert!(
            seg.start_second < seg.end_second,
            "segment must be time-ordered: {seg:?}"
        );
        assert!(
            !seg.text.trim().is_empty(),
            "segment text must be non-empty"
        );
    }

    // Progress must have been forwarded in the 0.0..=1.0 range.
    let mut got_progress = false;
    while let Ok(p) = rx.try_recv() {
        assert!((0.0..=1.0).contains(&p), "progress {p} out of range");
        got_progress = true;
    }
    assert!(got_progress, "expected forwarded progress values");
}
