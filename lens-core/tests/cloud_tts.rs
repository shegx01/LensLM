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

use lens_core::config::{VoiceConfig, VoiceRef};
use lens_core::dialogue::{DialogueScript, Speaker, Turn};
use lens_core::error::LensError;
use lens_core::tts::TtsProvider;
use lens_core::tts::audio::TARGET_RATE;
use lens_core::tts::cloud::CloudTtsAdapter;
use lens_core::{CloudTtsKind, TtsPhase};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
async fn deepgram_and_elevenlabs_kinds_are_not_yet_supported() {
    for kind in [CloudTtsKind::Deepgram, CloudTtsKind::ElevenLabs] {
        let err = adapter(kind, "http://127.0.0.1:1", "k")
            .synthesize_turn(
                &turn("x"),
                &VoiceConfig::default(),
                &CancellationToken::new(),
            )
            .await
            .expect_err("unsupported kind");
        assert!(matches!(err, LensError::Tts(_)), "kind {kind:?}");
    }
}
