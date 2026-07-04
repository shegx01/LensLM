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
/// resolves the configured model but errors clearly when it is not downloaded —
/// `transcribe` never auto-downloads (that is the onboarding step's job).
#[tokio::test]
async fn transcribe_local_whisper_missing_model_errors() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "local_whisper".to_string();
    engine.set_config(config).await;

    let err = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None)
        .await
        .expect_err("no downloaded whisper model → typed error");

    assert_eq!(err.kind(), "Transcription");
    // Feature-on: "not downloaded"; feature-off: "feature not built". Both are a
    // clear typed Transcription error the caller can surface.
    let msg = err.message();
    assert!(
        msg.contains("not downloaded") || msg.contains("feature not built"),
        "message: {msg}"
    );
}
