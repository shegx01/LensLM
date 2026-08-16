// issue #71: deep `Send` auto-trait evaluation can overflow the default 128-frame
// limit under stricter toolchains.
#![recursion_limit = "256"]
//! Offline tests for the cloud TTS adapter (#195).
//!
//! Coverage (wiremock, no live network): OpenAI-compatible happy path → 24 kHz
//! mono, 48 kHz stereo → resampled/downmixed 24 kHz mono, bearer auth, voice
//! pass-through, HTTP error matrix (401/429/5xx), no-key-leak, undecodable/empty
//! body, `VoiceRef::Reference` rejection, empty-key `Validation`, and the
//! not-yet-supported Deepgram/ElevenLabs kinds. All offline.

use base64::Engine;
use lens_core::config::{VoiceConfig, VoiceRef};
use lens_core::dialogue::{DialogueScript, Emotion, Speaker, Turn};
use lens_core::error::LensError;
use lens_core::tts::TtsProvider;
use lens_core::tts::audio::TARGET_RATE;
use lens_core::tts::cloud::CloudTtsAdapter;
use lens_core::tts::{CloudTtsConsent, cloud_tts_usable};
use lens_core::{CloudTtsKind, TtsPhase};
use rstest::rstest;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn raw_pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&s| ((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes())
        .collect()
}

fn dialogue_adapter(kind: CloudTtsKind, uri: &str, key: &str, model: &str) -> CloudTtsAdapter {
    CloudTtsAdapter::with_client(kind, uri, key, model, reqwest::Client::new())
}

fn hg(host: &str, guest: &str) -> DialogueScript {
    DialogueScript {
        turns: vec![
            Turn {
                speaker: Speaker::Host,
                text: host.to_string(),
                emotion: Some(Emotion::Laugh),
                source_ids: Vec::new(),
            },
            Turn {
                speaker: Speaker::Guest,
                text: guest.to_string(),
                emotion: None,
                source_ids: Vec::new(),
            },
        ],
    }
}

fn wav_bytes(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for &s in samples {
            let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(clamped).unwrap();
        }
        writer.finalize().unwrap();
    }
    buf.into_inner()
}

fn turn(text: &str) -> Turn {
    Turn {
        speaker: Speaker::Host,
        text: text.to_string(),
        emotion: None,
        source_ids: Vec::new(),
    }
}

fn adapter(kind: CloudTtsKind, uri: &str, key: &str) -> CloudTtsAdapter {
    CloudTtsAdapter::with_client(kind, uri, key, "gpt-4o-mini-tts", reqwest::Client::new())
}

async fn mount_wav(server: &MockServer, bytes: Vec<u8>) {
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
        .mount(server)
        .await;
}

#[tokio::test]
async fn happy_path_decodes_24k_mono() {
    let server = MockServer::start().await;
    mount_wav(&server, wav_bytes(&[0.2f32; 240], TARGET_RATE, 1)).await;

    let out = adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "sk-test")
        .synthesize_turn(
            &turn("hello"),
            &VoiceConfig::default(),
            &CancellationToken::new(),
        )
        .await
        .expect("happy path");

    assert_eq!(out.sample_rate, TARGET_RATE);
    assert_eq!(out.channels, 1);
    assert_eq!(out.samples.len(), 240);
}

#[tokio::test]
async fn stereo_48k_synthesize_turn_downmixes_at_native_rate() {
    let server = MockServer::start().await;
    // 4 interleaved stereo frames at 48k -> 4 mono samples at 48k (no resample here).
    let interleaved = vec![0.4, 0.6, -0.2, 0.2, 0.1, 0.1, 0.0, 0.0];
    mount_wav(&server, wav_bytes(&interleaved, 48_000, 2)).await;

    let out = adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "k")
        .synthesize_turn(
            &turn("hi"),
            &VoiceConfig::default(),
            &CancellationToken::new(),
        )
        .await
        .expect("stereo decode");

    assert_eq!(out.sample_rate, 48_000);
    assert_eq!(out.channels, 1);
    assert_eq!(out.samples.len(), 4);
    assert!((out.samples[0] - 0.5).abs() < 1e-3);
}

#[tokio::test]
async fn synthesize_script_resamples_and_downmixes_to_24k() {
    let server = MockServer::start().await;
    mount_wav(&server, wav_bytes(&[0.3f32; 9_600], 48_000, 2)).await;

    let script = DialogueScript {
        turns: vec![turn("hi")],
    };
    let cancel = CancellationToken::new();
    let noop = |_p: TtsPhase| {};
    let out = adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "k")
        .synthesize_script(&script, &VoiceConfig::default(), &noop, &cancel)
        .await
        .expect("script synth");

    assert_eq!(out.sample_rate, TARGET_RATE);
    assert_eq!(out.channels, 1);
    // 4800 mono frames at 48k -> ~2400 at 24k.
    assert!(
        (out.samples.len() as i64 - 2_400).abs() <= 4,
        "len {}",
        out.samples.len()
    );
}

#[tokio::test]
async fn synthesize_script_stitches_multiple_turns() {
    let server = MockServer::start().await;
    mount_wav(&server, wav_bytes(&[0.2f32; 2_400], TARGET_RATE, 1)).await;

    let script = DialogueScript {
        turns: vec![
            Turn {
                speaker: Speaker::Host,
                text: "hello".to_string(),
                emotion: None,
                source_ids: Vec::new(),
            },
            Turn {
                speaker: Speaker::Guest,
                text: "hi there".to_string(),
                emotion: None,
                source_ids: Vec::new(),
            },
        ],
    };
    let cancel = CancellationToken::new();
    let noop = |_p: TtsPhase| {};
    let out = adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "k")
        .synthesize_script(&script, &VoiceConfig::default(), &noop, &cancel)
        .await
        .expect("multi-turn script synth");

    assert_eq!(out.sample_rate, TARGET_RATE);
    assert_eq!(out.channels, 1);
    // A Host->Guest turn boundary inserts a silence gap, so the stitched result
    // must be longer than a single turn's raw sample count.
    assert!(
        out.samples.len() > 2_400,
        "expected stitched length > single turn, got {}",
        out.samples.len()
    );
}

#[tokio::test]
async fn request_carries_bearer_auth_voice_and_wav_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header("Authorization", "Bearer sk-bearer"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(wav_bytes(
            &[0.1f32; 24],
            TARGET_RATE,
            1,
        )))
        .mount(&server)
        .await;

    let voices = VoiceConfig {
        host: VoiceRef::Named("nova".into()),
        guest: VoiceRef::default(),
    };
    adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "sk-bearer")
        .synthesize_turn(&turn("hello world"), &voices, &CancellationToken::new())
        .await
        .expect("bearer + body");

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1);
    let body = String::from_utf8_lossy(&calls[0].body);
    assert!(
        body.contains("\"voice\":\"nova\""),
        "voice pass-through: {body}"
    );
    assert!(
        body.contains("\"response_format\":\"wav\""),
        "wav format: {body}"
    );
    assert!(
        body.contains("\"input\":\"hello world\""),
        "input text: {body}"
    );
}

#[tokio::test]
async fn status_401_maps_to_validation_and_leaks_nothing() {
    let server = MockServer::start().await;
    // A hostile body echoing secrets; it must never surface in the error message.
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string("bad key sk-super-secret at http://leak.internal"),
        )
        .mount(&server)
        .await;

    let base = server.uri();
    let err = adapter(CloudTtsKind::OpenAiCompatible, &base, "sk-super-secret")
        .synthesize_turn(
            &turn("x"),
            &VoiceConfig::default(),
            &CancellationToken::new(),
        )
        .await
        .expect_err("401");

    assert!(matches!(err, LensError::Validation(_)));
    let msg = err.to_string();
    assert!(!msg.contains("sk-super-secret"), "api key leaked: {msg}");
    assert!(!msg.contains(&base), "base url leaked: {msg}");
    assert!(
        !msg.contains("leak.internal"),
        "upstream body leaked: {msg}"
    );
}

#[tokio::test]
async fn status_429_and_5xx_map_to_network() {
    for status in [429u16, 500, 503] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;

        let err = adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "k")
            .synthesize_turn(
                &turn("x"),
                &VoiceConfig::default(),
                &CancellationToken::new(),
            )
            .await
            .expect_err("error status");
        assert!(matches!(err, LensError::Network(_)), "status {status}");
    }
}

#[tokio::test]
async fn undecodable_and_empty_body_map_to_tts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(200).set_body_bytes(b"ID3\x03fake mp3 payload".to_vec()),
        )
        .mount(&server)
        .await;
    let err = adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "k")
        .synthesize_turn(
            &turn("x"),
            &VoiceConfig::default(),
            &CancellationToken::new(),
        )
        .await
        .expect_err("mp3 body");
    assert!(matches!(err, LensError::Tts(_)));

    let empty = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(Vec::<u8>::new()))
        .mount(&empty)
        .await;
    let err = adapter(CloudTtsKind::OpenAiCompatible, &empty.uri(), "k")
        .synthesize_turn(
            &turn("x"),
            &VoiceConfig::default(),
            &CancellationToken::new(),
        )
        .await
        .expect_err("empty body");
    assert!(matches!(err, LensError::Tts(_)));
}

#[tokio::test]
async fn voice_reference_clone_is_rejected_before_network() {
    let voices = VoiceConfig {
        host: VoiceRef::Reference {
            clip_path: "/x.wav".into(),
            transcript: "hi".into(),
        },
        guest: VoiceRef::default(),
    };
    // Base URL is unreachable on purpose: the reference must be rejected first.
    let err = adapter(CloudTtsKind::OpenAiCompatible, "http://127.0.0.1:1", "k")
        .synthesize_turn(&turn("x"), &voices, &CancellationToken::new())
        .await
        .expect_err("reference rejected");
    assert!(matches!(err, LensError::Tts(_)));
}

#[tokio::test]
async fn empty_api_key_yields_validation() {
    let err = adapter(CloudTtsKind::OpenAiCompatible, "http://127.0.0.1:1", "")
        .synthesize_turn(
            &turn("x"),
            &VoiceConfig::default(),
            &CancellationToken::new(),
        )
        .await
        .expect_err("empty key");
    assert!(matches!(err, LensError::Validation(_)));
}

#[tokio::test]
async fn deepgram_turn_unsupported_dialogue_kinds_reject_single_turn() {
    // Deepgram's per-turn path is still unimplemented -> Tts.
    let err = adapter(CloudTtsKind::Deepgram, "http://127.0.0.1:1", "k")
        .synthesize_turn(
            &turn("x"),
            &VoiceConfig::default(),
            &CancellationToken::new(),
        )
        .await
        .expect_err("deepgram unsupported");
    assert!(matches!(err, LensError::Tts(_)));

    // Dialogue engines have no single-turn contract; a direct synthesize_turn is a
    // Validation error (unreachable via synthesize_overview, which calls the script path).
    for kind in [CloudTtsKind::ElevenLabs, CloudTtsKind::GoogleCloud] {
        let err = dialogue_adapter(kind, "http://127.0.0.1:1", "k", "m")
            .synthesize_turn(
                &turn("x"),
                &VoiceConfig::default(),
                &CancellationToken::new(),
            )
            .await
            .expect_err("dialogue kind rejects single turn");
        assert!(matches!(err, LensError::Validation(_)), "kind {kind:?}");
    }
}

#[tokio::test]
async fn openai_emotive_turn_emits_instructions_field() {
    // AC5 on the re-homed OpenAI path: a per-turn emotion still drives the
    // `instructions` hint through the whole-exchange contract (no behavior change).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(wav_bytes(
            &[0.1f32; 24],
            TARGET_RATE,
            1,
        )))
        .mount(&server)
        .await;

    let script = DialogueScript {
        turns: vec![Turn {
            speaker: Speaker::Host,
            text: "hooray".into(),
            emotion: Some(Emotion::Excited),
            source_ids: Vec::new(),
        }],
    };
    let noop = |_p: TtsPhase| {};
    adapter(CloudTtsKind::OpenAiCompatible, &server.uri(), "k")
        .synthesize_script(
            &script,
            &VoiceConfig::default(),
            &noop,
            &CancellationToken::new(),
        )
        .await
        .expect("openai script synth");
    let calls = server.received_requests().await.unwrap();
    let body = String::from_utf8_lossy(&calls[0].body);
    assert!(
        body.contains("\"instructions\""),
        "emotion drives instructions (AC5): {body}"
    );
}

// ---- ElevenLabs Text-to-Dialogue (#40) ----

#[tokio::test]
async fn elevenlabs_dialogue_request_shape_auth_order_emotion_and_pcm_decode() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/text-to-dialogue"))
        .and(query_param("output_format", "pcm_24000"))
        .and(header("xi-api-key", "el-key"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(raw_pcm16(&[0.1f32; 240])))
        .mount(&server)
        .await;

    let noop = |_p: TtsPhase| {};
    let out = dialogue_adapter(
        CloudTtsKind::ElevenLabs,
        &server.uri(),
        "el-key",
        "eleven_v3",
    )
    .synthesize_script(
        &hg("Hello there", "Hi back"),
        &VoiceConfig::default(),
        &noop,
        &CancellationToken::new(),
    )
    .await
    .expect("dialogue synth");
    assert_eq!(out.sample_rate, TARGET_RATE);
    // One chunk -> raw 240 samples (no inter-chunk gap).
    assert_eq!(out.samples.len(), 240);

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1);
    let body = String::from_utf8_lossy(&calls[0].body);
    assert!(body.contains("\"model_id\":\"eleven_v3\""), "model: {body}");
    assert!(body.contains("\"inputs\""), "inputs list: {body}");
    // Per-line emotion audio tag prepended to the host line (AC5).
    assert!(body.contains("[laughs] Hello there"), "emotion tag: {body}");
    // Turn order preserved (host line before guest line).
    let host_idx = body.find("Hello there").expect("host line");
    let guest_idx = body.find("Hi back").expect("guest line");
    assert!(host_idx < guest_idx, "turn order preserved");
}

#[tokio::test]
async fn elevenlabs_over_limit_chunks_and_never_calls_per_turn() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/text-to-dialogue"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(raw_pcm16(&[0.1f32; 240])))
        .mount(&server)
        .await;

    // 3 turns of 800 chars, ElevenLabs 2000-char cap -> 2 scene chunks.
    let big = "a".repeat(800);
    let script = DialogueScript {
        turns: vec![
            Turn {
                speaker: Speaker::Host,
                text: big.clone(),
                emotion: None,
                source_ids: Vec::new(),
            },
            Turn {
                speaker: Speaker::Guest,
                text: big.clone(),
                emotion: None,
                source_ids: Vec::new(),
            },
            Turn {
                speaker: Speaker::Host,
                text: big.clone(),
                emotion: None,
                source_ids: Vec::new(),
            },
        ],
    };
    let phases = std::sync::Mutex::new(Vec::new());
    let on_phase = |p: TtsPhase| phases.lock().unwrap().push(p);
    let out = dialogue_adapter(CloudTtsKind::ElevenLabs, &server.uri(), "k", "eleven_v3")
        .synthesize_script(
            &script,
            &VoiceConfig::default(),
            &on_phase,
            &CancellationToken::new(),
        )
        .await
        .expect("chunked synth");

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 2, "two scene chunks -> two dialogue requests");
    assert!(
        calls.iter().all(|c| c.url.path() == "/v1/text-to-dialogue"),
        "never falls back to a per-turn /v1/audio/speech call"
    );
    // Two chunks stitched with one scene gap between them.
    assert!(
        out.samples.len() > 480,
        "stitched len {}",
        out.samples.len()
    );
    assert!(
        phases
            .lock()
            .unwrap()
            .iter()
            .any(|p| matches!(p, TtsPhase::Stitching))
    );
}

#[tokio::test]
async fn elevenlabs_401_maps_validation_and_leaks_no_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/text-to-dialogue"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key el-super-secret"))
        .mount(&server)
        .await;

    let base = server.uri();
    let noop = |_p: TtsPhase| {};
    let err = dialogue_adapter(
        CloudTtsKind::ElevenLabs,
        &base,
        "el-super-secret",
        "eleven_v3",
    )
    .synthesize_script(
        &hg("a", "b"),
        &VoiceConfig::default(),
        &noop,
        &CancellationToken::new(),
    )
    .await
    .expect_err("401");
    assert!(matches!(err, LensError::Validation(_)));
    let msg = err.to_string();
    assert!(!msg.contains("el-super-secret"), "xi-api-key leaked: {msg}");
    assert!(!msg.contains(&base), "base url leaked: {msg}");
}

// ---- Google Gemini multi-speaker (#40) ----

fn gemini_audio_response(samples: &[f32]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(raw_pcm16(samples));
    format!(
        r#"{{"candidates":[{{"content":{{"parts":[{{"inlineData":{{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"{b64}"}}}}]}}}}]}}"#
    )
}

#[tokio::test]
async fn google_dialogue_request_shape_auth_and_base64_l16_decode() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.5-flash-preview-tts:generateContent",
        ))
        .and(header("x-goog-api-key", "g-key"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(gemini_audio_response(&[0.1f32; 240])),
        )
        .mount(&server)
        .await;

    let script = DialogueScript {
        turns: vec![
            Turn {
                speaker: Speaker::Host,
                text: "Hello".into(),
                emotion: Some(Emotion::Excited),
                source_ids: Vec::new(),
            },
            Turn {
                speaker: Speaker::Guest,
                text: "Hi".into(),
                emotion: None,
                source_ids: Vec::new(),
            },
        ],
    };
    let noop = |_p: TtsPhase| {};
    let out = dialogue_adapter(
        CloudTtsKind::GoogleCloud,
        &server.uri(),
        "g-key",
        "gemini-2.5-flash-preview-tts",
    )
    .synthesize_script(
        &script,
        &VoiceConfig::default(),
        &noop,
        &CancellationToken::new(),
    )
    .await
    .expect("gemini synth");
    assert_eq!(out.sample_rate, TARGET_RATE);
    assert_eq!(out.samples.len(), 240);

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1);
    // Google uses x-goog-api-key, NEVER bearer Authorization.
    assert!(
        calls[0].headers.get("authorization").is_none(),
        "must not send bearer auth"
    );
    let body = String::from_utf8_lossy(&calls[0].body);
    assert!(
        body.contains("Host: (bright, energetic excitement) Hello"),
        "labelled natural-language style line: {body}"
    );
    assert!(body.contains("Guest: Hi"), "guest line: {body}");
    assert!(
        body.contains("multiSpeakerVoiceConfig"),
        "multi-speaker config: {body}"
    );
    assert!(
        body.contains("\"speaker\":\"Host\"") && body.contains("\"speaker\":\"Guest\""),
        "both speaker configs: {body}"
    );
}

#[tokio::test]
async fn google_403_maps_validation_and_leaks_no_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403).set_body_string("denied for g-super-secret"))
        .mount(&server)
        .await;

    let base = server.uri();
    let noop = |_p: TtsPhase| {};
    let err = dialogue_adapter(
        CloudTtsKind::GoogleCloud,
        &base,
        "g-super-secret",
        "gemini-2.5-flash-preview-tts",
    )
    .synthesize_script(
        &hg("a", "b"),
        &VoiceConfig::default(),
        &noop,
        &CancellationToken::new(),
    )
    .await
    .expect_err("403");
    assert!(matches!(err, LensError::Validation(_)));
    let msg = err.to_string();
    assert!(
        !msg.contains("g-super-secret"),
        "x-goog-api-key leaked: {msg}"
    );
    assert!(!msg.contains(&base), "base url leaked: {msg}");
}

#[tokio::test]
async fn google_safety_blocked_no_audio_maps_to_tts_not_panic() {
    let server = MockServer::start().await;
    // 200 OK but no audio part (finishReason SAFETY) — must map to Tts, not panic.
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(r#"{"candidates":[{"finishReason":"SAFETY"}]}"#),
        )
        .mount(&server)
        .await;

    let noop = |_p: TtsPhase| {};
    let err = dialogue_adapter(
        CloudTtsKind::GoogleCloud,
        &server.uri(),
        "k",
        "gemini-2.5-flash-preview-tts",
    )
    .synthesize_script(
        &hg("a", "b"),
        &VoiceConfig::default(),
        &noop,
        &CancellationToken::new(),
    )
    .await
    .expect_err("no audio");
    assert!(matches!(err, LensError::Tts(_)));
}

// ===========================================================================
// Consent gate: `tts_backend_available` (the notebook-side availability gate)
// ===========================================================================

/// Fakes both Orpheus weights at their exact pinned sizes so `orpheus_ready` is true
/// without a real multi-GB download.
fn fake_orpheus_on_disk(cache_root: &std::path::Path) {
    for id in ["orpheus", "snac"] {
        let spec = lens_core::resolve_tts(id).expect("registry spec");
        let path = lens_core::tts_model_path(cache_root, id).expect("model path");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::File::create(&path)
            .expect("create")
            .set_len(spec.size_bytes)
            .expect("set_len");
    }
}

async fn tts_available(consent: bool, api_key: &str, orpheus_on_disk: bool) -> bool {
    let dir = tempfile::tempdir().expect("tempdir");
    if orpheus_on_disk {
        fake_orpheus_on_disk(dir.path());
    }
    let engine = lens_core::LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.paths.data_dir = dir.path().display().to_string();
    config.tts_cloud_consent = consent;
    config.tts = lens_core::config::TtsConfig {
        version: 1,
        backend: lens_core::TtsBackend::Cloud(CloudTtsKind::OpenAiCompatible),
        model: String::new(),
        clouds: std::collections::BTreeMap::from([(
            CloudTtsKind::OpenAiCompatible,
            lens_core::config::CloudTtsCreds {
                api_key: api_key.to_string(),
                base_url: String::new(),
            },
        )]),
    };
    let cfg = config.tts.clone();
    engine.set_config(config).await;
    engine.tts_backend_available(&cfg).await
}

/// The gate must equal `cloud_tts_usable || orpheus_ready`: a keyed cloud backend with
/// consent withheld is available only because the offline voice can stand in for it.
#[tokio::test]
async fn tts_backend_available_tracks_consent_and_the_orpheus_fallback() {
    assert!(tts_available(true, "sk-key", false).await);
    assert!(!tts_available(false, "sk-key", false).await);
    assert!(!tts_available(true, "", false).await);
    assert!(tts_available(false, "sk-key", true).await);
}

fn tts_cfg_with_base_url(base_url: &str) -> lens_core::config::TtsConfig {
    lens_core::config::TtsConfig {
        version: 1,
        backend: lens_core::TtsBackend::Cloud(CloudTtsKind::OpenAiCompatible),
        model: String::new(),
        clouds: std::collections::BTreeMap::from([(
            CloudTtsKind::OpenAiCompatible,
            lens_core::config::CloudTtsCreds {
                api_key: "sk-key".to_string(),
                base_url: base_url.to_string(),
            },
        )]),
    }
}

// SYNC-CHECK: the cloud-ASR tables in `cloud_asr.rs` cover the same predicate; both
// subsystems must reject the same endpoints.
#[rstest]
#[case::plain_http("http://tts.vendor.example")]
#[case::http_lookalike_host("http://localhost.evil.example")]
#[case::public_ipv4("http://93.184.216.34")]
#[case::ftp("ftp://tts.vendor.example")]
#[case::relative("tts.vendor.example")]
#[case::file("file:///etc/passwd")]
fn cloud_tts_is_unusable_over_a_cleartext_endpoint(#[case] base_url: &str) {
    assert!(
        !cloud_tts_usable(
            &tts_cfg_with_base_url(base_url),
            CloudTtsKind::OpenAiCompatible,
            CloudTtsConsent::Granted,
        ),
        "{base_url} must not carry the API key and dialogue text"
    );
}

#[rstest]
#[case::blank_takes_the_https_vendor_default("")]
#[case::https("https://api.openai.com")]
#[case::loopback_name("http://localhost:9000")]
#[case::loopback_v4("http://127.0.0.1:9000")]
#[case::rfc1918("http://192.168.1.5:9000")]
fn cloud_tts_stays_usable_over_a_transport_safe_endpoint(#[case] base_url: &str) {
    assert!(
        cloud_tts_usable(
            &tts_cfg_with_base_url(base_url),
            CloudTtsKind::OpenAiCompatible,
            CloudTtsConsent::Granted,
        ),
        "{base_url} must remain usable"
    );
}

/// The gate is load-bearing at the resolution seam too: a cleartext endpoint must
/// not produce a cloud adapter, even with consent and a key.
#[test]
fn resolve_refuses_a_cloud_adapter_for_a_cleartext_endpoint() {
    assert!(
        lens_core::tts::resolve_tts_provider(
            lens_core::TtsBackend::Cloud(CloudTtsKind::OpenAiCompatible),
            &tts_cfg_with_base_url("http://tts.vendor.example"),
            CloudTtsConsent::Granted,
            std::path::Path::new("/data"),
        )
        .is_none()
    );
}
