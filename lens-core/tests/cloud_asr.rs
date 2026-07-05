// issue #71: deep `Send` auto-trait evaluation can overflow the default 128-frame
// limit under stricter toolchains.
#![recursion_limit = "256"]
//! Offline tests for the cloud ASR fallback tier (#45).
//!
//! # Coverage
//!
//! Unit: WAV header, chunk split/stitch, config backward-compat, api_key
//! redaction, consent isolation, preflight gates.
//!
//! Integration (wiremock): OpenAI + Deepgram happy paths, HTTP error matrix
//! (401/413/429/500/malformed-200), pre-flight zero-call assertions,
//! chunked multi-request transcription, effective-backend transparency.
//!
//! All offline — no live network, no `LENS_RUN_MODEL_TESTS` gate.

use std::sync::Arc;

use lens_core::LensEngine;
use lens_core::asr::cloud::chunk::{split_if_needed, stitch_segments};
use lens_core::asr::cloud::wav::{WAV_HEADER_BYTES, pcm_to_wav};
use lens_core::asr::cloud::{CloudAsrEngine, preflight_check};
use lens_core::asr::{AsrEngine, MockAsrEngine, TranscribeConfig, TranscriptSegment};
use lens_core::config::{AppConfig, AsrConfig, CloudAsrProvider};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ===========================================================================
// Unit: WAV header validity
// ===========================================================================

#[test]
fn wav_header_is_44_bytes() {
    let wav = pcm_to_wav(&[0.0_f32; 0], 16_000).expect("wav encode");
    assert_eq!(
        wav.len(),
        WAV_HEADER_BYTES,
        "empty PCM must produce exactly a 44-byte header"
    );
}

#[test]
fn wav_header_riff_magic() {
    let wav = pcm_to_wav(&[0.0_f32; 4], 16_000).expect("wav encode");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
}

#[test]
fn wav_total_length_matches_samples() {
    let n_samples: usize = 1024;
    let wav = pcm_to_wav(&vec![0.25_f32; n_samples], 16_000).expect("wav encode");
    // 44-byte header + n_samples * 2 bytes (16-bit)
    assert_eq!(wav.len(), WAV_HEADER_BYTES + n_samples * 2);
}

#[test]
fn wav_header_fields_16khz_mono_16bit() {
    let wav = pcm_to_wav(&[0.0_f32; 0], 16_000).expect("wav encode");

    // fmt chunk size: bytes 16..20 = 16u32 LE
    let fmt_chunk_size = u32::from_le_bytes(wav[16..20].try_into().unwrap());
    assert_eq!(fmt_chunk_size, 16, "PCM fmt chunk is 16 bytes");

    // audio format: bytes 20..22 = 1u16 (PCM)
    let audio_fmt = u16::from_le_bytes(wav[20..22].try_into().unwrap());
    assert_eq!(audio_fmt, 1, "format must be PCM (1)");

    // channels: bytes 22..24 = 1u16 (mono)
    let channels = u16::from_le_bytes(wav[22..24].try_into().unwrap());
    assert_eq!(channels, 1, "must be mono");

    // sample rate: bytes 24..28 = 16000u32 LE
    let sample_rate = u32::from_le_bytes(wav[24..28].try_into().unwrap());
    assert_eq!(sample_rate, 16_000);

    // bits per sample: bytes 34..36 = 16u16
    let bits = u16::from_le_bytes(wav[34..36].try_into().unwrap());
    assert_eq!(bits, 16);
}

#[test]
fn wav_riff_chunk_size_correct() {
    let n_samples: usize = 100;
    let wav = pcm_to_wav(&vec![0.0_f32; n_samples], 16_000).expect("wav encode");
    // RIFF chunk size = 36 + data_len; data_len = n_samples * 2
    let data_len = (n_samples * 2) as u32;
    let expected_riff = 36 + data_len;
    let actual_riff = u32::from_le_bytes(wav[4..8].try_into().unwrap());
    assert_eq!(actual_riff, expected_riff);
}

#[test]
fn wav_data_chunk_size_correct() {
    let n_samples: usize = 50;
    let wav = pcm_to_wav(&vec![0.5_f32; n_samples], 16_000).expect("wav encode");
    let data_len = u32::from_le_bytes(wav[40..44].try_into().unwrap());
    assert_eq!(data_len, (n_samples * 2) as u32);
}

#[test]
fn wav_samples_are_clamped_and_scaled() {
    // 1.0 → i16::MAX, -1.0 → -i16::MAX, 0.0 → 0
    let wav = pcm_to_wav(&[1.0_f32, -1.0_f32, 0.0_f32], 16_000).expect("wav encode");
    let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
    let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
    let s2 = i16::from_le_bytes(wav[48..50].try_into().unwrap());
    assert_eq!(s0, i16::MAX);
    assert_eq!(s1, -i16::MAX);
    assert_eq!(s2, 0);
}

// ===========================================================================
// Unit: chunk::split_if_needed / stitch_segments
// ===========================================================================

#[test]
fn split_if_needed_under_limit_returns_single_chunk() {
    // 1 second at 16kHz = 16000 f32 samples, WAV ~= 44 + 32000 bytes ≈ 32 KB → way under 25 MB
    let pcm: Vec<f32> = vec![0.1_f32; 16_000];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_second, 0.0);
    assert_eq!(chunks[0].data.len(), pcm.len());
}

#[test]
fn split_if_needed_deepgram_short_audio_single_chunk() {
    let pcm: Vec<f32> = vec![0.1_f32; 16_000 * 60];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::Deepgram, 16_000);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn split_if_needed_deepgram_over_duration_cap_produces_multiple_chunks() {
    let sample_rate = 16_000usize;
    let n = sample_rate * 60 * 20;
    let pcm: Vec<f32> = vec![0.1_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::Deepgram, sample_rate as u32);
    assert!(
        chunks.len() >= 2,
        "20 minutes of Deepgram audio must split on the duration cap, got {}",
        chunks.len()
    );
    let max_chunk_samples = 480 * sample_rate;
    for (i, c) in chunks.iter().enumerate() {
        assert!(
            c.data.len() <= max_chunk_samples,
            "chunk {i} of {} samples exceeds the ~8-minute duration cap ({max_chunk_samples})",
            c.data.len()
        );
    }
    let total: usize = chunks.iter().map(|c| c.data.len()).sum();
    assert_eq!(total, n, "all samples must be covered across chunks");
    let mut prev = -1.0f32;
    for c in &chunks {
        assert!(
            c.start_second > prev,
            "chunk starts must be strictly increasing"
        );
        prev = c.start_second;
    }
}

#[test]
fn split_if_needed_zero_sample_rate_terminates() {
    let pcm: Vec<f32> = vec![0.1_f32; 10];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::Deepgram, 0);
    let total: usize = chunks.iter().map(|c| c.data.len()).sum();
    assert_eq!(
        total,
        pcm.len(),
        "a zero sample rate must not lose or loop over samples"
    );
}

#[test]
fn split_if_needed_over_openai_25mb_cap_produces_multiple_chunks() {
    // OpenAI WAV cap: 25 MB. 16-bit WAV: each sample = 2 bytes.
    // 25 MB = 26_214_400 bytes. Data portion = 25 MB - 44 header = 26_214_356 bytes
    // → max ~13_107_178 samples. Use 14 million to be safely over.
    let n = 14_000_000usize;
    let pcm: Vec<f32> = vec![0.1_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    assert!(
        chunks.len() >= 2,
        "over-limit PCM must split into at least 2 chunks, got {}",
        chunks.len()
    );
    // Every chunk's encoded WAV size must be ≤ 25 MB
    let cap = 25 * 1024 * 1024usize;
    for (i, c) in chunks.iter().enumerate() {
        let encoded = WAV_HEADER_BYTES + c.data.len() * 2;
        assert!(
            encoded <= cap,
            "chunk {i} encoded size {encoded} exceeds 25 MB cap"
        );
    }
}

#[test]
fn split_if_needed_chunk_start_seconds_are_monotonic() {
    let n = 14_000_000usize;
    let pcm: Vec<f32> = vec![0.0_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    let mut prev = -1.0f32;
    for c in &chunks {
        assert!(
            c.start_second > prev,
            "chunk start_second must be strictly increasing"
        );
        prev = c.start_second;
    }
    // First chunk always starts at 0.0
    assert_eq!(chunks[0].start_second, 0.0);
}

#[test]
fn split_if_needed_chunks_cover_all_samples() {
    let n = 14_000_000usize;
    let pcm: Vec<f32> = vec![0.1_f32; n];
    let chunks = split_if_needed(&pcm, CloudAsrProvider::OpenAiCompatible, 16_000);
    let total: usize = chunks.iter().map(|c| c.data.len()).sum();
    assert_eq!(total, n, "all samples must be covered across chunks");
}

#[test]
fn stitch_segments_empty_input() {
    let out = stitch_segments(&[]);
    assert!(out.is_empty());
}

#[test]
fn stitch_segments_single_chunk_passthrough() {
    let segs = vec![
        TranscriptSegment {
            text: "hello".into(),
            start_second: 0.0,
            end_second: 1.0,
        },
        TranscriptSegment {
            text: "world".into(),
            start_second: 1.0,
            end_second: 2.5,
        },
    ];
    let out = stitch_segments(&[(0.0, segs.clone())]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].text, "hello");
    assert!((out[0].start_second - 0.0).abs() < 1e-4);
    assert!((out[1].start_second - 1.0).abs() < 1e-4);
}

#[test]
fn stitch_segments_reoffsets_second_chunk() {
    // Chunk 1: 10s long, segments at 0–5 and 5–10
    // Chunk 2: starts at t=10s (start_second=10.0), segments at 0–3 and 3–6
    let chunk1_segs = vec![
        TranscriptSegment {
            text: "a".into(),
            start_second: 0.0,
            end_second: 5.0,
        },
        TranscriptSegment {
            text: "b".into(),
            start_second: 5.0,
            end_second: 10.0,
        },
    ];
    let chunk2_segs = vec![
        TranscriptSegment {
            text: "c".into(),
            start_second: 0.0,
            end_second: 3.0,
        },
        TranscriptSegment {
            text: "d".into(),
            start_second: 3.0,
            end_second: 6.0,
        },
    ];
    let out = stitch_segments(&[(0.0, chunk1_segs), (10.0, chunk2_segs)]);
    assert_eq!(out.len(), 4);
    // "a": unchanged
    assert!(
        (out[0].start_second - 0.0).abs() < 1e-3,
        "a.start={}",
        out[0].start_second
    );
    assert!((out[0].end_second - 5.0).abs() < 1e-3);
    // "b": unchanged
    assert!((out[1].start_second - 5.0).abs() < 1e-3);
    assert!((out[1].end_second - 10.0).abs() < 1e-3);
    // "c": offset by 10.0
    assert!(
        (out[2].start_second - 10.0).abs() < 1e-3,
        "c.start={}",
        out[2].start_second
    );
    assert!((out[2].end_second - 13.0).abs() < 1e-3);
    // "d": offset by 10.0
    assert!(
        (out[3].start_second - 13.0).abs() < 1e-3,
        "d.start={}",
        out[3].start_second
    );
    assert!((out[3].end_second - 16.0).abs() < 1e-3);
}

#[test]
fn stitch_segments_global_monotonic_non_decreasing() {
    // Three chunks whose individual segments are all relative to chunk start.
    let c1 = vec![TranscriptSegment {
        text: "x".into(),
        start_second: 0.0,
        end_second: 4.0,
    }];
    let c2 = vec![TranscriptSegment {
        text: "y".into(),
        start_second: 0.0,
        end_second: 3.0,
    }];
    let c3 = vec![TranscriptSegment {
        text: "z".into(),
        start_second: 0.0,
        end_second: 5.0,
    }];
    let out = stitch_segments(&[(0.0, c1), (4.0, c2), (7.0, c3)]);
    for w in out.windows(2) {
        assert!(
            w[1].start_second >= w[0].end_second - 1e-4,
            "segments not monotonic: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn stitch_segments_total_coverage_preserved() {
    // Sum of each chunk's max end_second should ≈ total duration
    let c1 = vec![TranscriptSegment {
        text: "a".into(),
        start_second: 0.0,
        end_second: 8.0,
    }];
    let c2 = vec![TranscriptSegment {
        text: "b".into(),
        start_second: 0.0,
        end_second: 7.0,
    }];
    let out = stitch_segments(&[(0.0, c1), (8.0, c2)]);
    let last_end = out.last().unwrap().end_second;
    // Expected ~15.0 (8 + 7)
    assert!((last_end - 15.0).abs() < 0.1, "last_end={last_end}");
}

// ===========================================================================
// Unit: config backward-compat, serde, and api_key redaction
// ===========================================================================

#[test]
fn asr_config_cloud_fields_default_to_empty() {
    let cfg = AsrConfig::default();
    assert!(cfg.cloud_provider.is_none());
    assert!(cfg.cloud_base_url.is_empty());
    assert!(cfg.cloud_model.is_empty());
    assert!(cfg.cloud_api_key.is_empty());
}

#[test]
fn app_config_audio_cloud_consent_defaults_false() {
    // AppConfig::default() is the authoritative default; audio_cloud_consent must be false.
    let cfg = AppConfig::default();
    assert!(
        !cfg.audio_cloud_consent,
        "audio_cloud_consent must default to false"
    );

    // Old on-disk JSON that predates #45 also gets false via #[serde(default)].
    let json = r#"{"theme":"dark","user_name":"","embedding_model":"","embedding_backend":"","max_source_mb":"","models":[],"endpoints":{},"voices":{"host":"","guest":""},"paths":{"data_dir":""},"tier_thresholds":{"tier1_token_cap":4000,"tier2_token_cap":16000},"onboarding_complete":false}"#;
    let cfg2: AppConfig = serde_json::from_str(json).expect("old config must parse");
    assert!(
        !cfg2.audio_cloud_consent,
        "audio_cloud_consent must default to false when absent from old JSON"
    );
}

#[test]
fn asr_config_backward_compat_old_json_no_cloud_fields() {
    // Old config without cloud keys must parse fine, cloud fields get defaults
    let json = r#"{"backend":"local_whisper","whisper_model":"base"}"#;
    let cfg: AsrConfig = serde_json::from_str(json).expect("old asr config must parse");
    assert!(cfg.cloud_provider.is_none());
    assert!(cfg.cloud_api_key.is_empty());
}

#[test]
fn cloud_asr_provider_serde_snake_case_roundtrip() {
    let oai = CloudAsrProvider::OpenAiCompatible;
    let dg = CloudAsrProvider::Deepgram;

    let oai_json = serde_json::to_string(&oai).unwrap();
    let dg_json = serde_json::to_string(&dg).unwrap();

    assert_eq!(oai_json, r#""open_ai_compatible""#);
    assert_eq!(dg_json, r#""deepgram""#);

    let back_oai: CloudAsrProvider = serde_json::from_str(&oai_json).unwrap();
    let back_dg: CloudAsrProvider = serde_json::from_str(&dg_json).unwrap();
    assert_eq!(back_oai, CloudAsrProvider::OpenAiCompatible);
    assert_eq!(back_dg, CloudAsrProvider::Deepgram);
}

#[test]
fn asr_config_debug_redacts_api_key() {
    let cfg = AsrConfig {
        cloud_api_key: "super-secret-key".to_string(),
        cloud_provider: Some(CloudAsrProvider::OpenAiCompatible),
        ..AsrConfig::default()
    };
    let debug_str = format!("{cfg:?}");
    assert!(
        !debug_str.contains("super-secret-key"),
        "api_key must not appear in Debug output: {debug_str}"
    );
    assert!(
        debug_str.contains("***"),
        "Debug output must show *** for non-empty key: {debug_str}"
    );
}

#[test]
fn asr_config_debug_shows_empty_for_absent_key() {
    let cfg = AsrConfig {
        cloud_api_key: String::new(),
        ..AsrConfig::default()
    };
    let debug_str = format!("{cfg:?}");
    // Empty key shows as "" not ***
    assert!(
        !debug_str.contains("***"),
        "empty key must not show ***: {debug_str}"
    );
}

#[test]
fn audio_cloud_consent_independent_from_enrichment_cloud_consent() {
    use lens_core::config::EnrichmentConfig;

    // Case 1: audio consent ON, enrichment consent OFF — must be independent
    let cfg = AppConfig {
        audio_cloud_consent: true,
        enrichment: EnrichmentConfig {
            cloud_consent: false,
            ..EnrichmentConfig::default()
        },
        ..AppConfig::default()
    };
    assert!(cfg.audio_cloud_consent, "audio_cloud_consent must be true");
    assert!(
        !cfg.enrichment.cloud_consent,
        "enrichment.cloud_consent must remain false"
    );

    // Case 2: audio consent OFF, enrichment consent ON — must be independent
    let cfg2 = AppConfig {
        audio_cloud_consent: false,
        enrichment: EnrichmentConfig {
            cloud_consent: true,
            ..EnrichmentConfig::default()
        },
        ..AppConfig::default()
    };
    assert!(
        !cfg2.audio_cloud_consent,
        "audio_cloud_consent must be false"
    );
    assert!(
        cfg2.enrichment.cloud_consent,
        "enrichment.cloud_consent must be true"
    );

    // Case 3: serde round-trip preserves each flag independently
    let src = AppConfig {
        audio_cloud_consent: true,
        enrichment: EnrichmentConfig {
            cloud_consent: false,
            ..EnrichmentConfig::default()
        },
        ..AppConfig::default()
    };
    let json = serde_json::to_string(&src).unwrap();
    let back: AppConfig = serde_json::from_str(&json).unwrap();
    assert!(back.audio_cloud_consent);
    assert!(!back.enrichment.cloud_consent);
}

// ===========================================================================
// Unit: preflight_check gates (no network)
// ===========================================================================

fn app_config_with_cloud(
    consent: bool,
    api_key: &str,
    provider: Option<CloudAsrProvider>,
) -> AppConfig {
    AppConfig {
        audio_cloud_consent: consent,
        asr: AsrConfig {
            cloud_provider: provider,
            cloud_api_key: api_key.to_string(),
            cloud_base_url: "https://api.openai.com".to_string(),
            cloud_model: "whisper-1".to_string(),
            ..AsrConfig::default()
        },
        ..AppConfig::default()
    }
}

#[test]
fn preflight_consent_false_returns_validation_error() {
    let cfg = app_config_with_cloud(false, "sk-test", Some(CloudAsrProvider::OpenAiCompatible));
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(
        err.kind(),
        "Validation",
        "no consent → Validation, got {err:?}"
    );
    assert!(
        err.message().contains("consent"),
        "error must mention consent: {}",
        err.message()
    );
}

#[test]
fn preflight_empty_key_returns_validation_error() {
    let cfg = app_config_with_cloud(true, "", Some(CloudAsrProvider::OpenAiCompatible));
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(err.message().contains("key") || err.message().contains("API"));
}

#[test]
fn preflight_no_provider_returns_validation_error() {
    let cfg = app_config_with_cloud(true, "sk-test", None);
    let err = preflight_check(&cfg).unwrap_err();
    assert_eq!(err.kind(), "Validation");
    assert!(
        err.message().contains("provider"),
        "error must mention provider: {}",
        err.message()
    );
}

#[test]
fn preflight_all_present_returns_ok() {
    let cfg = app_config_with_cloud(true, "sk-test", Some(CloudAsrProvider::OpenAiCompatible));
    assert!(preflight_check(&cfg).is_ok());
}

// ===========================================================================
// Integration helpers
// ===========================================================================

/// Minimal flat PCM — just enough samples to produce a non-empty WAV body.
fn tiny_pcm() -> Vec<f32> {
    vec![0.1_f32; 160] // 10 ms at 16 kHz
}

fn openai_segments_response() -> serde_json::Value {
    serde_json::json!({
        "segments": [
            { "start": 0.0, "end": 1.5, "text": " hello from openai" },
            { "start": 1.5, "end": 3.0, "text": " goodbye from openai" }
        ]
    })
}

fn deepgram_utterances_response() -> serde_json::Value {
    serde_json::json!({
        "results": {
            "utterances": [
                { "start": 0.0, "end": 1.2, "transcript": "hello from deepgram" },
                { "start": 1.2, "end": 2.8, "transcript": "goodbye from deepgram" }
            ]
        }
    })
}

// ===========================================================================
// Integration: OpenAI happy path
// ===========================================================================

#[tokio::test]
async fn openai_happy_path_maps_segments_correctly() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    );

    let out = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("happy-path openai");

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].text, "hello from openai");
    assert!((out[0].start_second - 0.0).abs() < 1e-4);
    assert!((out[0].end_second - 1.5).abs() < 1e-4);
    assert_eq!(out[1].text, "goodbye from openai");
    assert!((out[1].start_second - 1.5).abs() < 1e-4);
    assert!((out[1].end_second - 3.0).abs() < 1e-4);
}

#[tokio::test]
async fn openai_request_carries_bearer_auth_and_multipart_fields() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("Authorization", "Bearer sk-bearer-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-bearer-test",
        reqwest::Client::new(),
    );

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("bearer auth test");

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1);
    let body_str = String::from_utf8_lossy(&calls[0].body);
    // multipart should contain model and response_format
    assert!(
        body_str.contains("whisper-1"),
        "body must contain model name: {body_str}"
    );
    assert!(
        body_str.contains("verbose_json"),
        "body must contain response_format: {body_str}"
    );
}

#[tokio::test]
async fn cloud_chunk_retries_then_succeeds_on_transient_5xx() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(2)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .with_priority(2)
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    )
    .with_retry_policy(3, std::time::Duration::from_millis(1));

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("two transient 5xx then success must recover via retry");

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 3, "expected 2 failed attempts + 1 success");
}

#[tokio::test]
async fn cloud_chunk_gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    )
    .with_retry_policy(2, std::time::Duration::from_millis(1));

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err("persistent 5xx must propagate after retries are exhausted");
    assert!(
        matches!(err, lens_core::LensError::Network(_)),
        "a retried-out 5xx surfaces as Network, got {err:?}"
    );

    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 3, "expected 1 initial attempt + 2 retries");
}

// ===========================================================================
// Integration: Deepgram happy path
// ===========================================================================

#[tokio::test]
async fn deepgram_happy_path_maps_utterances_correctly() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .respond_with(ResponseTemplate::new(200).set_body_json(deepgram_utterances_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-key",
        reqwest::Client::new(),
    );

    let out = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("happy-path deepgram");

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].text, "hello from deepgram");
    assert!((out[0].start_second - 0.0).abs() < 1e-4);
    assert!((out[0].end_second - 1.2).abs() < 1e-4);
    assert_eq!(out[1].text, "goodbye from deepgram");
}

#[tokio::test]
async fn deepgram_request_carries_token_auth_and_query_params() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(header("Authorization", "Token dg-token-test"))
        .and(query_param("encoding", "linear32"))
        .and(query_param("utterances", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(deepgram_utterances_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-token-test",
        reqwest::Client::new(),
    );

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("deepgram token auth + query params");
}

#[tokio::test]
async fn deepgram_request_content_type_is_audio_raw() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(header("Content-Type", "audio/raw"))
        .respond_with(ResponseTemplate::new(200).set_body_json(deepgram_utterances_response()))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-key",
        reqwest::Client::new(),
    );

    engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect("deepgram content-type audio/raw");
}

// ===========================================================================
// Integration: error mapping for both providers
// ===========================================================================

/// Helper: assert a given HTTP status maps to the expected LensError kind.
async fn assert_status_maps_to(
    provider: CloudAsrProvider,
    path_str: &str,
    status: u16,
    expected_kind: &str,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(path_str))
        .respond_with(ResponseTemplate::new(status))
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        provider,
        server.uri(),
        "model",
        "key",
        reqwest::Client::new(),
    )
    .with_retry_policy(0, std::time::Duration::ZERO);

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err(&format!("status {status} must error"));
    assert_eq!(
        err.kind(),
        expected_kind,
        "status {status}: expected {expected_kind}, got {} ({})",
        err.kind(),
        err.message()
    );
}

#[tokio::test]
async fn openai_401_maps_to_validation() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        401,
        "Validation",
    )
    .await;
}

#[tokio::test]
async fn openai_413_maps_to_validation() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        413,
        "Validation",
    )
    .await;
}

#[tokio::test]
async fn openai_429_maps_to_network() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        429,
        "Network",
    )
    .await;
}

#[tokio::test]
async fn openai_500_maps_to_network() {
    assert_status_maps_to(
        CloudAsrProvider::OpenAiCompatible,
        "/v1/audio/transcriptions",
        500,
        "Network",
    )
    .await;
}

#[tokio::test]
async fn deepgram_401_maps_to_validation() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 401, "Validation").await;
}

#[tokio::test]
async fn deepgram_413_maps_to_validation() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 413, "Validation").await;
}

#[tokio::test]
async fn deepgram_429_maps_to_network() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 429, "Network").await;
}

#[tokio::test]
async fn deepgram_500_maps_to_network() {
    assert_status_maps_to(CloudAsrProvider::Deepgram, "/v1/listen", 500, "Network").await;
}

#[tokio::test]
async fn openai_malformed_json_200_maps_to_transcription_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("not valid json {{{")
                .insert_header("content-type", "application/json"),
        )
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-test",
        reqwest::Client::new(),
    );

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err("malformed JSON must error");
    assert_eq!(err.kind(), "Transcription", "malformed 200: got {err:?}");
}

#[tokio::test]
async fn deepgram_malformed_json_200_maps_to_transcription_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("{bad json")
                .insert_header("content-type", "application/json"),
        )
        .mount(&server)
        .await;

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "dg-key",
        reqwest::Client::new(),
    );

    let err = engine
        .transcribe_pcm(&tiny_pcm(), &TranscribeConfig::default(), None)
        .await
        .expect_err("malformed JSON must error");
    assert_eq!(err.kind(), "Transcription", "malformed 200: got {err:?}");
}

// ===========================================================================
// Integration: pre-flight gates → zero wiremock requests
// ===========================================================================

/// Runs `LensEngine::transcribe` with the cloud backend configured. Asserts the
/// wiremock received zero requests (pre-flight blocked), and that the fallback
/// produced the mock's canned segments.
async fn assert_preflight_blocks_with_zero_requests(
    server: &MockServer,
    cfg_override: impl FnOnce(&mut AppConfig),
) {
    let engine = LensEngine::for_test().await;

    // Set backend=cloud but override via the caller's closure
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    cfg_override(&mut config);
    engine.set_config(config).await;

    // Inject mock Apple engine as fallback (uses apple_native seam)
    let canned = vec![TranscriptSegment {
        text: "local fallback".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let pcm = tiny_pcm();
    let (out, _backend) = engine
        .transcribe(&pcm, &TranscribeConfig::default(), None, None)
        .await
        .expect("pre-flight blocked → local fallback should succeed");

    assert_eq!(out, canned, "fallback must return mock segments");

    let received = server.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        0,
        "pre-flight must issue ZERO cloud requests, got {}",
        received.len()
    );
}

#[tokio::test]
async fn preflight_no_consent_zero_cloud_requests() {
    let server = MockServer::start().await;
    // Mount a catch-all so any request would be recorded
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    assert_preflight_blocks_with_zero_requests(&server, |cfg| {
        cfg.audio_cloud_consent = false;
        cfg.asr.cloud_api_key = "sk-test".to_string();
        cfg.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    })
    .await;
}

#[tokio::test]
async fn preflight_no_key_zero_cloud_requests() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    assert_preflight_blocks_with_zero_requests(&server, |cfg| {
        cfg.audio_cloud_consent = true;
        cfg.asr.cloud_api_key = "".to_string(); // empty key
        cfg.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    })
    .await;
}

#[tokio::test]
async fn preflight_no_provider_zero_cloud_requests() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    assert_preflight_blocks_with_zero_requests(&server, |cfg| {
        cfg.audio_cloud_consent = true;
        cfg.asr.cloud_api_key = "sk-test".to_string();
        cfg.asr.cloud_provider = None; // no provider
    })
    .await;
}

// ===========================================================================
// Integration: LensEngine::transcribe cloud → fallback on HTTP errors
// ===========================================================================

/// Configures `LensEngine` for cloud with a wiremock that returns `status`, then
/// asserts the error triggers fallback to the injected MockAsrEngine.
async fn assert_cloud_error_falls_back_to_mock(
    server: &MockServer,
    _route_path: &str,
    status: u16,
) {
    let engine = LensEngine::for_test().await;

    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "fallback segment".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let (out, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .unwrap_or_else(|e| panic!("status {status} must fallback, not hard-fail: {e:?}"));

    assert_eq!(out, canned, "fallback segments must match mock");
    assert!(
        label.contains("fallback"),
        "backend label must indicate fallback: {label}"
    );
}

#[tokio::test]
async fn cloud_401_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(&server, "/v1/audio/transcriptions", 401).await;
}

#[tokio::test]
async fn cloud_413_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(413))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(&server, "/v1/audio/transcriptions", 413).await;
}

#[tokio::test]
async fn cloud_429_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(&server, "/v1/audio/transcriptions", 429).await;
}

#[tokio::test]
async fn cloud_500_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(&server, "/v1/audio/transcriptions", 500).await;
}

#[tokio::test]
async fn cloud_malformed_200_triggers_fallback_to_local() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("{bad}")
                .insert_header("content-type", "application/json"),
        )
        .mount(&server)
        .await;
    assert_cloud_error_falls_back_to_mock(&server, "/v1/audio/transcriptions", 200).await;
}

// ===========================================================================
// Integration: effective-backend transparency via LensEngine::transcribe
// ===========================================================================

#[tokio::test]
async fn transcribe_cloud_success_returns_cloud_label() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-label-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let (_segs, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("cloud success");
    assert_eq!(label, "cloud");
}

#[tokio::test]
async fn transcribe_cloud_fallback_returns_fallback_label() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-fallback-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "fallback".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned))))
        .await;

    let (_segs, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("500 → fallback");
    assert!(
        label.contains("fallback"),
        "label must contain fallback: {label}"
    );
}

// ===========================================================================
// Integration: chunked transcription (multiple wiremock calls, stitched timestamps)
// ===========================================================================

#[tokio::test]
async fn chunked_transcription_calls_server_multiple_times_and_stitches() {
    let server = MockServer::start().await;

    // The mock returns different responses each call:
    // call 1 → chunk 1 segments; call 2 → chunk 2 segments
    // wiremock 0.6 doesn't support per-call ordering easily, so return the same body
    // but we verify 2+ calls and that the stitched output has re-offset timestamps.
    let chunk_response = serde_json::json!({
        "segments": [
            { "start": 0.0, "end": 2.0, "text": "chunk segment" }
        ]
    });
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chunk_response))
        .mount(&server)
        .await;

    // Build a PCM buffer large enough to split into 2 chunks (> 25 MB WAV)
    // 14 million f32 samples → WAV ~= 44 + 28_000_000 bytes ≈ 26.7 MB > 25 MB cap
    let n = 14_000_000usize;
    let pcm = vec![0.1_f32; n];

    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "sk-chunk-test",
        reqwest::Client::new(),
    );

    let out = engine
        .transcribe_pcm(&pcm, &TranscribeConfig::default(), None)
        .await
        .expect("chunked transcription");

    let received = server.received_requests().await.unwrap();
    assert!(
        received.len() >= 2,
        "over-limit PCM must produce 2+ requests, got {}",
        received.len()
    );

    // Stitched output should have segments from all chunks; timestamps must be non-decreasing
    assert!(out.len() >= 2, "stitched output must have 2+ segments");
    for w in out.windows(2) {
        assert!(
            w[1].start_second >= w[0].start_second - 1e-4,
            "stitched segments must be monotonic: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
    // Second chunk's segment must be offset from 0 (the chunk starts after the first chunk)
    assert!(
        out[1].start_second > 0.0,
        "second chunk segment must be offset from zero, got {}",
        out[1].start_second
    );
}

// ===========================================================================
// Integration: consent isolation — audio_cloud_consent independent from enrichment.cloud_consent
// ===========================================================================

#[tokio::test]
async fn consent_isolation_audio_consent_true_enrichment_false_allows_cloud() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-isolation-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    // audio consent ON, enrichment consent OFF
    config.audio_cloud_consent = true;
    config.enrichment.cloud_consent = false;
    engine.set_config(config).await;

    let (_segs, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect(
            "audio_cloud_consent=true must allow cloud even when enrichment.cloud_consent=false",
        );
    assert_eq!(label, "cloud", "must use cloud: {label}");

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "must have made exactly 1 cloud request");
}

#[tokio::test]
async fn consent_isolation_audio_consent_false_enrichment_true_blocks_cloud() {
    let server = MockServer::start().await;
    // Mount a catch-all to detect any unauthorized request
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_segments_response()))
        .mount(&server)
        .await;

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = server.uri();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-isolation-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    // audio consent OFF, enrichment consent ON
    config.audio_cloud_consent = false;
    config.enrichment.cloud_consent = true;
    engine.set_config(config).await;

    // Inject mock for fallback
    let canned = vec![TranscriptSegment {
        text: "local only".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let (out, _label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("audio_cloud_consent=false must fallback, not hard-fail");
    assert_eq!(out, canned, "must use local fallback");

    let received = server.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        0,
        "enrichment.cloud_consent=true must NOT unblock audio cloud requests"
    );
}

// ===========================================================================
// Integration: dead-port (unreachable) triggers fallback (1 failed request expected)
// ===========================================================================

#[tokio::test]
async fn unreachable_cloud_triggers_local_fallback() {
    // Use a port that is almost certainly not listening (high ephemeral port)
    let dead_url = "http://127.0.0.1:19823";

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "cloud".to_string();
    config.asr.cloud_base_url = dead_url.to_string();
    config.asr.cloud_model = "whisper-1".to_string();
    config.asr.cloud_api_key = "sk-dead-test".to_string();
    config.asr.cloud_provider = Some(CloudAsrProvider::OpenAiCompatible);
    config.audio_cloud_consent = true;
    engine.set_config(config).await;

    let canned = vec![TranscriptSegment {
        text: "local result".into(),
        start_second: 0.0,
        end_second: 1.0,
    }];
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(canned.clone()))))
        .await;

    let (out, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("dead port must degrade gracefully to local fallback");

    assert_eq!(out, canned);
    assert!(
        label.contains("fallback"),
        "label must indicate fallback: {label}"
    );
}

// ===========================================================================
// Unit: select_asr_backend Cloud gate-1 passthrough
// ===========================================================================

#[test]
fn select_asr_backend_cloud_config_override_is_passthrough() {
    use lens_core::asr::{Platform, select_asr_backend};

    // Gate-1: explicit Cloud override wins regardless of platform/availability.
    let non_apple = Platform {
        is_apple_silicon_macos: false,
        macos_major: None,
    };
    let result = select_asr_backend(Some(lens_core::AsrBackend::Cloud), non_apple, false, false);
    assert_eq!(
        result,
        lens_core::AsrBackend::Cloud,
        "explicit Cloud config override must pass through unconditionally"
    );
}

// ===========================================================================
// Unit: IngestProgress effective_backend serde
// ===========================================================================

#[test]
fn ingest_progress_effective_backend_roundtrip_and_omit() {
    // With effective_backend set: must survive a JSON round-trip.
    let with_backend = lens_core::IngestProgress {
        phase: "transcribing".to_string(),
        done: 1,
        total: Some(1),
        effective_backend: Some("cloud".to_string()),
    };
    let json = serde_json::to_string(&with_backend).expect("serialize with backend");
    let back: lens_core::IngestProgress = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.effective_backend, Some("cloud".to_string()));

    // Without effective_backend: the key must be absent from the JSON wire format.
    let without_backend = lens_core::IngestProgress {
        phase: "transcribing".to_string(),
        done: 1,
        total: Some(1),
        effective_backend: None,
    };
    let json_none = serde_json::to_string(&without_backend).expect("serialize without backend");
    assert!(
        !json_none.contains("effective_backend"),
        "None effective_backend must be omitted from JSON: {json_none}"
    );
}
