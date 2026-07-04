//! Engine-level ASR dispatch (#42, Units 2+3): `LensEngine::transcribe` routes
//! through `select_asr_backend`. The injected `AsrEngine` is the Apple-native seam
//! (Apple in prod, a mock in tests) — used ONLY when the router selects AppleNative;
//! LocalWhisper always uses the internal WhisperEngine, never the injected engine.
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

/// The injected engine is the Apple-native seam (Apple in prod, a mock in tests).
/// Forcing `apple_native` routes to it, so its canned segments come back.
#[tokio::test]
async fn transcribe_dispatches_to_injected_apple_engine() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    engine.set_config(config).await;

    let expected = canned();
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(expected.clone()))))
        .await;

    let out = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None)
        .await
        .expect("apple-native seam transcribe should succeed");

    assert_eq!(out, expected);
}

/// The injected engine is Apple-only: even with a mock injected, forcing
/// `local_whisper` must use the internal WhisperEngine (never the mock's canned
/// output), erroring clearly when the model is not downloaded.
#[tokio::test]
async fn transcribe_local_whisper_ignores_injected_engine() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "local_whisper".to_string();
    engine.set_config(config).await;

    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned()))))
        .await;

    let err = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None)
        .await
        .expect_err("local_whisper must not fall through to the injected Apple mock");

    assert_eq!(err.kind(), "Transcription");
    let msg = err.message();
    assert!(
        msg.contains("not downloaded") || msg.contains("feature not built"),
        "message: {msg}"
    );
}

/// Forcing `apple_native` with no injected engine hits the typed `(AppleNative, None)`
/// arm: a clear Transcription error rather than a silent LocalWhisper fallback.
#[tokio::test]
async fn transcribe_apple_forced_without_engine_errors() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    engine.set_config(config).await;

    let err = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None)
        .await
        .expect_err("apple_native with no injected engine → typed error");

    assert_eq!(err.kind(), "Transcription");
    assert!(
        err.message().contains("no engine is injected"),
        "message: {}",
        err.message()
    );
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
