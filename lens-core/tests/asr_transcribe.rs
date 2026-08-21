//! Engine-level ASR dispatch (#42, Units 2+3): `LensEngine::transcribe` routes
//! through `select_asr_backend`. The injected `AsrEngine` is the Apple-native seam
//! (Apple in prod, a mock in tests) — used ONLY when the router selects AppleNative;
//! LocalWhisper always uses the internal WhisperEngine, never the injected engine.
//!
//! Offline: uses the model-free `MockAsrEngine`, no downloads.

use std::sync::Arc;

use lens_core::{AsrBackend, LensEngine, MockAsrEngine, TranscribeConfig, TranscriptSegment};
#[cfg(feature = "local-whisper")]
use lens_core::{
    DEFAULT_WHISPER_MODEL_ID, download_whisper_model, resolve_whisper, whisper_model_path,
};

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

    let (out, backend) = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None, None)
        .await
        .expect("apple-native seam transcribe should succeed");

    assert_eq!(out, expected);
    assert_eq!(backend, "apple_native");
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
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None, None)
        .await
        .expect_err("local_whisper must not fall through to the injected Apple mock");

    assert_eq!(err.kind(), "Transcription");
    let msg = err.message();
    assert!(
        msg.contains("not downloaded") || msg.contains("feature not built"),
        "message: {msg}"
    );
}

/// Forcing `apple_native` with no injected engine and no Whisper model on disk has
/// nothing to fall back to, so the original cause is preserved.
#[tokio::test]
async fn transcribe_apple_forced_without_engine_or_whisper_errors() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    engine.set_config(config).await;

    let err = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None, None)
        .await
        .expect_err("apple_native with no injected engine and no whisper → typed error");

    assert_eq!(err.kind(), "Transcription");
    assert!(
        err.message().contains("no engine is injected"),
        "message: {}",
        err.message()
    );
}

/// With a Whisper model present, that same forced `apple_native` routes into
/// the LocalWhisper fallback — the loader error proves it, since a preserved-original
/// run would still say "no engine is injected". The fixture MUST stay 0 bytes: a file
/// whose first four bytes match the ggml magic makes whisper.cpp SIGSEGV on garbage
/// header fields.
#[cfg(feature = "local-whisper")]
#[tokio::test]
async fn transcribe_apple_forced_without_engine_falls_back_to_whisper() {
    let cache = tempfile::tempdir().expect("tempdir");
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    config.paths.cache_dir = Some(cache.path().to_string_lossy().into_owned());
    let spec = resolve_whisper(&config.asr.whisper_model)
        .or_else(|| resolve_whisper(DEFAULT_WHISPER_MODEL_ID))
        .expect("configured or default whisper spec resolves");
    engine.set_config(config).await;

    let model_path = whisper_model_path(cache.path(), spec.id);
    std::fs::create_dir_all(model_path.parent().expect("model path has a parent"))
        .expect("create whisper model dir");
    std::fs::write(&model_path, b"").expect("write 0-byte ggml fixture");

    let err = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None, None)
        .await
        .expect_err("a 0-byte ggml cannot load");

    assert_eq!(err.kind(), "Transcription");
    let msg = err.message();
    assert!(
        !msg.contains("no engine is injected"),
        "the Whisper fallback was never entered: {msg}"
    );
    assert!(
        msg.contains("whisper"),
        "expected a whisper load failure, got: {msg}"
    );
}

/// The positive leg of the fallback: a real ggml yields segments tagged
/// `local_whisper (fallback)`. Needs a downloaded model, so it is gated.
#[cfg(feature = "local-whisper")]
#[tokio::test]
#[ignore = "downloads the default ggml model; run with LENS_RUN_MODEL_TESTS=1 --ignored"]
async fn transcribe_apple_fallback_reports_local_whisper_fallback() {
    if std::env::var("LENS_RUN_MODEL_TESTS").is_err() {
        eprintln!(
            "skipping transcribe_apple_fallback_reports_local_whisper_fallback (set LENS_RUN_MODEL_TESTS=1)"
        );
        return;
    }

    let cache = tempfile::tempdir().expect("tempdir");
    download_whisper_model(
        cache.path(),
        DEFAULT_WHISPER_MODEL_ID,
        &tokio_util::sync::CancellationToken::new(),
        |_| {},
    )
    .await
    .expect("download the default whisper model");

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    config.asr.whisper_model = DEFAULT_WHISPER_MODEL_ID.to_string();
    config.paths.cache_dir = Some(cache.path().to_string_lossy().into_owned());
    engine.set_config(config).await;

    let (_segments, backend) = engine
        .transcribe(&[0.0_f32; 16000], &TranscribeConfig::default(), None, None)
        .await
        .expect("the whisper fallback transcribes");

    assert_eq!(backend, "local_whisper (fallback)");
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
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None, None)
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

/// The resolved-backend seam the UI consumes, end-to-end through the real engine,
/// returning the typed enum rather than a re-derived string. A stored preference
/// that cannot run is reported as the backend that WILL run, not as itself.
#[tokio::test]
async fn resolve_asr_backend_reports_the_engine_decision() {
    let engine = LensEngine::for_test().await;

    assert_eq!(
        engine
            .resolve_asr_backend(None, false)
            .await
            .expect("resolve"),
        AsrBackend::LocalWhisper,
        "no injected engine → Whisper"
    );

    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned()))))
        .await;
    assert_eq!(
        engine
            .resolve_asr_backend(None, false)
            .await
            .expect("resolve"),
        AsrBackend::AppleNative,
        "an injected engine with no override → Apple"
    );

    // `cloud` is stored but unconfigured, so it resolves to the local cascade the
    // run would actually take — the Apple engine injected above.
    for (token, expected) in [
        ("local_whisper", AsrBackend::LocalWhisper),
        ("cloud", AsrBackend::AppleNative),
        ("apple_native", AsrBackend::AppleNative),
    ] {
        let mut config = engine.config().await;
        config.asr.backend = token.to_string();
        engine.set_config(config).await;
        let resolved = engine
            .resolve_asr_backend(None, false)
            .await
            .expect("resolve");
        assert_eq!(resolved, expected, "stored backend {token}");
    }
}

/// Settings said "Cloud" while every ingest ran Whisper: consent revoked on an
/// otherwise complete cloud config, with no Apple engine to catch it.
#[tokio::test]
async fn resolve_asr_backend_demotes_a_cloud_backend_left_without_consent() {
    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_provider = Some(lens_core::config::CloudAsrProvider::OpenAiCompatible);
    config.asr.cloud_base_url = "https://api.openai.com".to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-test".to_string();
    config.audio_cloud_consent = true;
    engine.set_config(config.clone()).await;
    assert_eq!(
        engine
            .resolve_asr_backend(None, false)
            .await
            .expect("resolve"),
        AsrBackend::Cloud,
        "a complete, consented cloud config resolves to Cloud"
    );

    config.audio_cloud_consent = false;
    engine.set_config(config).await;
    assert_eq!(
        engine
            .resolve_asr_backend(None, false)
            .await
            .expect("resolve"),
        AsrBackend::LocalWhisper,
        "revoking consent must change what the UI is told"
    );
}

/// Persisting the same revocation also repairs the stored value, so the config on
/// disk cannot keep naming a backend the engine will never run.
#[test]
fn saving_a_config_without_audio_consent_demotes_a_cloud_backend() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = lens_core::AppConfig {
        audio_cloud_consent: false,
        ..lens_core::AppConfig::default()
    };
    // Fully configured, so the withheld consent is the ONLY reason to demote.
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_provider = Some(lens_core::config::CloudAsrProvider::OpenAiCompatible);
    config.asr.cloud_base_url = "https://api.openai.com".to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-test".to_string();
    config.save(dir.path()).expect("save");

    assert_eq!(config.asr.backend, "");
    assert_eq!(
        lens_core::AppConfig::load(dir.path())
            .expect("load")
            .asr
            .backend,
        ""
    );
}

/// Equal answers alone would not catch a re-read across an await racing a concurrent
/// `set_asr_engine`, so the single-observation property is asserted too.
#[tokio::test]
async fn resolve_asr_backend_agrees_with_transcribe_and_reads_the_engine_once() {
    let engine = LensEngine::for_test().await;
    let expected = canned();
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(expected.clone()))))
        .await;

    let before = engine.asr_engine_reads();
    let resolved = engine
        .resolve_asr_backend(None, false)
        .await
        .expect("resolve");
    assert_eq!(
        engine.asr_engine_reads() - before,
        1,
        "the resolution path must read the engine slot exactly once"
    );

    let before = engine.asr_engine_reads();
    let (out, label) = engine
        .transcribe(&[0.0_f32; 16], &TranscribeConfig::default(), None, None)
        .await
        .expect("transcribe");
    assert_eq!(
        engine.asr_engine_reads() - before,
        1,
        "transcribe must reach the same decision from a single engine read"
    );

    assert_eq!(out, expected);
    assert_eq!(label, resolved.as_str());
}
