//! Engine-level ASR dispatch (#42, Units 2+3): `LensEngine::transcribe` routes
//! through `select_asr_backend` and delegates to the injected `AsrEngine` seam.
//!
//! Offline: uses the model-free `MockAsrEngine`, no downloads.

use std::sync::Arc;

use lens_core::{LensEngine, MockAsrEngine, TranscribeConfig, TranscriptSegment};

fn canned() -> Vec<TranscriptSegment> {
    vec![
        TranscriptSegment {
            text: "hello".to_string(),
            start_second: 0.0,
            end_second: 1.0,
        },
        TranscriptSegment {
            text: "world".to_string(),
            start_second: 1.0,
            end_second: 2.0,
        },
    ]
}

/// An injected engine (mock) receives the call — even when the config explicitly
/// forces `local_whisper`, the override path still dispatches to the injected seam.
#[tokio::test]
async fn transcribe_dispatches_to_injected_engine() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "local_whisper".to_string();
    engine.set_config(config).await;

    let expected = canned();
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(expected.clone()))))
        .await;

    let out = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None)
        .await
        .expect("mock transcribe should succeed");

    assert_eq!(out, expected);
}

/// With no injected engine and LocalWhisper selected, the internal Whisper path
/// is not yet wired → a typed `Transcription` error.
///
/// TODO(#42 Unit 5): REPLACE this test once `WhisperEngine` lands — it should
/// then transcribe rather than error.
#[tokio::test]
async fn transcribe_local_whisper_not_yet_wired_errors() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "local_whisper".to_string();
    engine.set_config(config).await;

    let err = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None)
        .await
        .expect_err("local whisper is not wired until Unit 5");

    assert_eq!(err.kind(), "Transcription");
    assert!(
        err.message().contains("Unit 5"),
        "message: {}",
        err.message()
    );
}
