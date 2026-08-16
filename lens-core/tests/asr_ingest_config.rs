//! Pins that audio ingest forwards the persisted `AsrConfig` language/translate
//! settings into the per-call `TranscribeConfig` (#136) — a hard-coded default
//! here would silently discard whatever the settings panel persisted.

#![recursion_limit = "256"]

use std::sync::{Arc, Mutex};

use lens_core::{Lang, MockAsrEngine, TranscribeConfig, TranscriptSegment};

mod support;
use support::{inject_counting_engine, tokenizer_available, write_tone_wav};

fn canned() -> Vec<TranscriptSegment> {
    vec![TranscriptSegment {
        text: "hola mundo".to_string(),
        start_second: 0.0,
        end_second: 1.0,
    }]
}

/// Ingests one clip with `language`/`translate` set and returns the
/// `TranscribeConfig` the engine actually received.
async fn ingest_and_capture(language: Option<Lang>, translate: bool) -> TranscribeConfig {
    ingest_with(language, translate)
        .await
        .expect("ingest failed")
        .expect("the ASR engine was never invoked")
}

/// `Ok(Some(cfg))` = the injected engine ran and recorded `cfg`; `Err` = ingest itself failed.
async fn ingest_with(
    language: Option<Lang>,
    translate: bool,
) -> Result<Option<TranscribeConfig>, lens_core::LensError> {
    let (dir, engine) = inject_counting_engine().await;

    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    config.asr.language = language;
    config.asr.translate = translate;
    engine.set_config(config).await;

    let seen = Arc::new(Mutex::new(None));
    engine
        .set_asr_engine(Some(Arc::new(
            MockAsrEngine::new(canned()).recording_config(Arc::clone(&seen)),
        )))
        .await;

    let wav = dir.path().join("clip.wav");
    write_tone_wav(&wav, 2);

    let nb = engine
        .create_notebook("asr-config-nb", None, None)
        .await
        .unwrap();
    let src = engine.add_file_source(&nb.id, &wav, None).await.unwrap();
    engine.ingest_source(&src.source.id, |_| {}).await?;

    let captured = seen.lock().unwrap().clone();
    Ok(captured)
}

#[tokio::test]
async fn ingest_forwards_persisted_language_to_the_engine() {
    if !tokenizer_available().await {
        eprintln!(
            "skipping ingest_forwards_persisted_language_to_the_engine: no tokenizer (offline)"
        );
        return;
    }
    let seen = ingest_and_capture(Some(Lang::Es), false).await;
    assert_eq!(seen.language, Some(Lang::Es));
}

/// `translate` reaching the engine is observable through its documented side effect:
/// Apple reroutes to Whisper before dispatch (`lib.rs` translate arm), so with Apple
/// selected and no Whisper model on disk, ingest fails on the missing model. While
/// `translate` was dropped at this call site that reroute could never fire.
#[tokio::test]
async fn translate_reaches_the_engine_and_reroutes_apple_to_whisper() {
    if !tokenizer_available().await {
        eprintln!(
            "skipping translate_reaches_the_engine_and_reroutes_apple_to_whisper: no tokenizer (offline)"
        );
        return;
    }
    let err = ingest_with(None, true)
        .await
        .expect_err("translate under Apple must reroute to Whisper");
    let msg = err.to_string();
    assert!(
        msg.contains("whisper model"),
        "expected the Whisper-model error proving the reroute fired, got: {msg}"
    );

    // Same config without `translate` transcribes on Apple, isolating translate as the cause.
    let seen = ingest_and_capture(None, false).await;
    assert!(!seen.translate);
}

#[tokio::test]
async fn ingest_forwards_the_other_language_hatch() {
    if !tokenizer_available().await {
        eprintln!("skipping ingest_forwards_the_other_language_hatch: no tokenizer (offline)");
        return;
    }
    let seen = ingest_and_capture(Some(Lang::Other("ar".to_string())), false).await;
    assert_eq!(seen.language, Some(Lang::Other("ar".to_string())));
}
