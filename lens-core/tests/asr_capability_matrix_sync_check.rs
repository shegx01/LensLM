//! SYNC-CHECK (issue #136): behaviourally pins the TS `ASR_CAPABILITY_MATRIX`
//! (`src/lib/asr/catalog.ts`) against the engines that implement it. Each cell
//! is parsed from the live TS literal (an inverted cell fails) and, where
//! reachable offline, proven against the real engine — not grepped source
//! text, which can't tell live code from a comment or a dead branch.
//!
//! Exception: `apple_native.language` lives in `src-tauri/src/asr/mod.rs:278`
//! (Apple's `lang_to_bcp47`), outside lens-core's reach — pinned by literal
//! value only, reviewed against that source.

use std::sync::Arc;

use lens_core::LensEngine;
use lens_core::asr::cloud::CloudAsrEngine;
use lens_core::asr::{AsrEngine, Lang, MockAsrEngine, TranscribeConfig};
use lens_core::config::CloudAsrProvider;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn read_catalog_ts() -> String {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src/lib/asr/catalog.ts");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// Extracts the `ASR_CAPABILITY_MATRIX = { ... }` object literal body via brace
/// matching, tolerant of comments/whitespace (unlike a plain grep).
fn matrix_body(src: &str) -> &str {
    let decl = src
        .find("ASR_CAPABILITY_MATRIX")
        .expect("ASR_CAPABILITY_MATRIX not found in catalog.ts");
    let eq = src[decl..]
        .find('=')
        .map(|j| decl + j)
        .expect("no `=` after ASR_CAPABILITY_MATRIX");
    let brace_start = src[eq..]
        .find('{')
        .map(|j| eq + j)
        .expect("no `{` after ASR_CAPABILITY_MATRIX =");
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut i = brace_start;
    loop {
        match bytes[i] as char {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    &src[brace_start..=i]
}

/// Reads the quoted value of `engine: { ..., field: '<value>', ... }`.
fn cell(body: &str, engine: &str, field: &str) -> String {
    let ekey = format!("{engine}:");
    let estart = body
        .find(&ekey)
        .unwrap_or_else(|| panic!("{engine} entry missing from ASR_CAPABILITY_MATRIX"))
        + ekey.len();
    let erest = &body[estart..];
    let obj_start = erest.find('{').expect("engine entry has no object body") + 1;
    let obj_rest = &erest[obj_start..];
    let obj_end = obj_rest
        .find('}')
        .expect("engine entry object never closes");
    let obj_body = &obj_rest[..obj_end];
    let fkey = format!("{field}:");
    let fstart = obj_body
        .find(&fkey)
        .unwrap_or_else(|| panic!("{engine}.{field} missing from ASR_CAPABILITY_MATRIX"))
        + fkey.len();
    let after = &obj_body[fstart..];
    let q = after.find(['\'', '"']).expect("quoted value expected");
    let quote = after.as_bytes()[q] as char;
    let after2 = &after[q + 1..];
    let end = after2.find(quote).expect("closing quote expected");
    after2[..end].to_string()
}

fn tiny_pcm() -> Vec<f32> {
    vec![0.1_f32; 160]
}

// ---------------------------------------------------------------------------
// apple_native.language — reviewed against src-tauri, not automated (see
// module doc for why lens-core cannot reach the real Apple engine).
// ---------------------------------------------------------------------------

#[test]
fn apple_native_language_cell_matches_the_reviewed_src_tauri_mapping() {
    let src = read_catalog_ts();
    let body = matrix_body(&src);
    assert_eq!(
        cell(body, "apple_native", "language"),
        "honoured",
        "apple_native.language changed — re-review src-tauri/src/asr/mod.rs:278 (lang_to_bcp47) before updating this pin"
    );
}

// ---------------------------------------------------------------------------
// apple_native.translate — reroutes to LocalWhisper (lib.rs:2099-2104).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn apple_native_translate_cell_matches_the_reroute_behaviour() {
    let src = read_catalog_ts();
    let body = matrix_body(&src);
    assert_eq!(
        cell(body, "apple_native", "translate"),
        "reroutes",
        "apple_native.translate changed — this check only verifies 'reroutes'"
    );

    let engine = LensEngine::for_test().await;
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    engine.set_config(config).await;
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(vec![]))))
        .await;

    let (_, label) = engine
        .transcribe(&tiny_pcm(), &TranscribeConfig::default(), None, None)
        .await
        .expect("translate=false must transcribe via apple_native, not reroute");
    assert_eq!(label, "apple_native");

    // translate=true reroutes to LocalWhisper, which errors here because no model
    // is downloaded in this test engine — that error IS the proof the reroute fired.
    let translate_cfg = TranscribeConfig {
        language: None,
        translate: true,
    };
    let err = engine
        .transcribe(&tiny_pcm(), &translate_cfg, None, None)
        .await
        .expect_err("translate=true under apple_native must reroute to LocalWhisper");
    assert!(
        err.to_string().contains("whisper model"),
        "expected the LocalWhisper missing-model error proving the reroute fired, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// local_whisper.language / local_whisper.translate — honoured
// (whisper.rs::resolve_transcribe_params, unit-tested in whisper.rs itself).
// ---------------------------------------------------------------------------

#[test]
fn local_whisper_cells_match_the_honoured_pin() {
    let src = read_catalog_ts();
    let body = matrix_body(&src);
    assert_eq!(
        cell(body, "local_whisper", "language"),
        "honoured",
        "local_whisper.language changed — see whisper.rs::resolve_transcribe_params unit tests"
    );
    assert_eq!(
        cell(body, "local_whisper", "translate"),
        "honoured",
        "local_whisper.translate changed — see whisper.rs::resolve_transcribe_params unit tests"
    );
}

// ---------------------------------------------------------------------------
// cloud.language — honoured (deepgram.rs:52, openai_compat.rs:53-54).
// cloud.translate — ignored (neither adapter reads `config.translate`).
// ---------------------------------------------------------------------------

async fn deepgram_query(config: TranscribeConfig) -> String {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"results": {"utterances": []}})),
        )
        .mount(&server)
        .await;
    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::Deepgram,
        server.uri(),
        "nova-3",
        "k",
        reqwest::Client::new(),
    );
    engine
        .transcribe_pcm(&tiny_pcm(), &config, None)
        .await
        .expect("deepgram call");
    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1);
    calls[0].url.query().unwrap_or("").to_string()
}

async fn openai_body(config: TranscribeConfig) -> String {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"segments": []})))
        .mount(&server)
        .await;
    let engine = CloudAsrEngine::with_client(
        CloudAsrProvider::OpenAiCompatible,
        server.uri(),
        "whisper-1",
        "k",
        reqwest::Client::new(),
    );
    engine
        .transcribe_pcm(&tiny_pcm(), &config, None)
        .await
        .expect("openai-compat call");
    let calls = server.received_requests().await.unwrap();
    assert_eq!(calls.len(), 1);
    String::from_utf8_lossy(&calls[0].body).to_string()
}

/// Multipart bodies embed a random per-request boundary; replace it with a
/// fixed placeholder so two otherwise-identical requests compare equal.
fn normalized_multipart(raw: &str) -> String {
    let boundary = raw
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("--"))
        .unwrap_or("")
        .to_string();
    if boundary.is_empty() {
        raw.to_string()
    } else {
        raw.replace(&boundary, "BOUNDARY")
    }
}

#[tokio::test]
async fn cloud_language_cell_matches_the_wire_request() {
    let src = read_catalog_ts();
    let body = matrix_body(&src);
    assert_eq!(
        cell(body, "cloud", "language"),
        "honoured",
        "cloud.language changed — this check only verifies 'honoured'"
    );

    let query_es = deepgram_query(TranscribeConfig {
        language: Some(Lang::Es),
        translate: false,
    })
    .await;
    assert!(
        query_es.contains("language=es"),
        "deepgram must forward the pinned language, got query: {query_es}"
    );
    let query_auto = deepgram_query(TranscribeConfig::default()).await;
    assert!(
        query_auto.contains("language=multi"),
        "deepgram auto-detect must send language=multi, got: {query_auto}"
    );

    let body_es = openai_body(TranscribeConfig {
        language: Some(Lang::Es),
        translate: false,
    })
    .await;
    assert!(
        body_es.contains("name=\"language\"") && body_es.contains("\r\n\r\nes"),
        "openai-compat must forward the pinned language field, got: {body_es}"
    );
    let body_auto = openai_body(TranscribeConfig::default()).await;
    assert!(
        !body_auto.contains("name=\"language\""),
        "openai-compat must omit the language field on auto-detect, got: {body_auto}"
    );
}

#[tokio::test]
async fn cloud_translate_cell_has_zero_effect_on_the_wire_request() {
    let src = read_catalog_ts();
    let body = matrix_body(&src);
    assert_eq!(
        cell(body, "cloud", "translate"),
        "ignored",
        "cloud.translate changed — this check only verifies 'ignored'"
    );

    let dg_off = deepgram_query(TranscribeConfig {
        language: Some(Lang::En),
        translate: false,
    })
    .await;
    let dg_on = deepgram_query(TranscribeConfig {
        language: Some(Lang::En),
        translate: true,
    })
    .await;
    assert_eq!(
        dg_off, dg_on,
        "deepgram request must be identical regardless of translate"
    );

    let oai_off = normalized_multipart(
        &openai_body(TranscribeConfig {
            language: Some(Lang::En),
            translate: false,
        })
        .await,
    );
    let oai_on = normalized_multipart(
        &openai_body(TranscribeConfig {
            language: Some(Lang::En),
            translate: true,
        })
        .await,
    );
    assert_eq!(
        oai_off, oai_on,
        "openai-compat request must be identical regardless of translate"
    );
}
