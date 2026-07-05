//! Integration tests for timestamped transcript anchors (issue #44): every
//! persisted transcript chunk carries a `SourceAnchor::Audio { start, end }`
//! equal to `[min,max]` over the segments it covers, survives enrichment
//! byte-identically, and hydrates through the same chunk-row path a PDF anchor
//! would (proving #23 needs no new API).
//!
//! Offline: model-free `MockAsrEngine` (routed via `apple_native`) + the nomic
//! tokenizer (chunk-bearing assertions skip cleanly when it is unreachable). No
//! `LENS_RUN_MODEL_TESTS`, no downloads beyond the tokenizer.

#![recursion_limit = "256"]

use std::path::Path;
use std::sync::Arc;

use lens_core::{LensEngine, MockAsrEngine, SourceAnchor, TranscriptSegment};

mod support;
use support::{inject_counting_engine, tokenizer_available};

/// Mono 16 kHz PCM16 WAV carrying a 440 Hz tone (nonzero, survives the all-silent
/// guard). The mock ignores the PCM, so the exact duration is irrelevant to the
/// canned segments — only that decode yields nonzero samples.
fn write_tone_wav(path: &Path, seconds: u32) {
    const SAMPLE_RATE: u32 = 16_000;
    let n_samples = SAMPLE_RATE * seconds;
    let data_len = n_samples * 2;
    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..n_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        let s = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5;
        let v = (s * i16::MAX as f32) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &buf).expect("write tone wav");
}

fn seg(text: &str, start: f32, end: f32) -> TranscriptSegment {
    TranscriptSegment {
        text: text.to_string(),
        start_second: start,
        end_second: end,
    }
}

/// Many short segments so the child (128-token) chunker packs several segments
/// per chunk — exercising the `[min,max]` aggregation, not a 1:1 mapping.
fn many_segments() -> Vec<TranscriptSegment> {
    (0..24)
        .map(|i| {
            let start = i as f32;
            seg(
                &format!(
                    "Segment number {i} discusses topic {i} with several extra words \
                     to lengthen the transcript enough that chunking packs multiple \
                     segments into a single child chunk for aggregation coverage."
                ),
                start,
                start + 1.0,
            )
        })
        .collect()
}

async fn use_mock_asr(engine: &LensEngine, segments: Vec<TranscriptSegment>) {
    let mut config = engine.config().await;
    config.asr.backend = "apple_native".to_string();
    engine.set_config(config).await;
    engine
        .set_asr_engine(Some(Arc::new(MockAsrEngine::new(segments))))
        .await;
}

/// Ingests a fresh audio source with the given canned segments; returns the
/// engine, temp dir (kept alive), and source id. Skips (returns None) offline.
async fn ingest_audio(
    segments: Vec<TranscriptSegment>,
) -> Option<(tempfile::TempDir, LensEngine, String)> {
    if !tokenizer_available().await {
        eprintln!("skipping: no tokenizer (offline)");
        return None;
    }
    let (dir, engine) = inject_counting_engine().await;
    use_mock_asr(&engine, segments).await;

    let wav = dir.path().join("clip.wav");
    write_tone_wav(&wav, 2);

    let nb = engine
        .create_notebook("audio-anchors-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_file_source(&nb.id, &wav, None)
        .await
        .unwrap()
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("audio ingest should succeed");
    Some((dir, engine, src.id))
}

fn parse_audio(json: &str) -> (f32, f32) {
    match serde_json::from_str::<SourceAnchor>(json).expect("deserialize anchor") {
        SourceAnchor::Audio {
            start_second,
            end_second,
        } => (start_second, end_second),
        other => panic!("expected SourceAnchor::Audio, got {other:?}"),
    }
}

/// AC (a)(b)(d): every persisted chunk has an Audio anchor with `start ≤ end`,
/// both within the transcript's `[0, duration]`, and equal to `[min,max]` over
/// the segments the chunk textually covers.
#[tokio::test]
async fn every_chunk_has_audio_anchor_covering_its_segments() {
    let segments = many_segments();
    let Some((_dir, engine, src_id)) = ingest_audio(segments.clone()).await else {
        return;
    };

    // Canonical buffer mirrors ingest: segment texts joined with "\n\n".
    let buffer = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    // Byte range of each segment in the buffer (matches transcript_extract_output).
    let mut ranges: Vec<(usize, usize, f32, f32)> = Vec::new();
    let mut cursor = 0usize;
    for (i, s) in segments.iter().enumerate() {
        if i > 0 {
            cursor += 2; // "\n\n"
        }
        let start = cursor;
        cursor += s.text.len();
        ranges.push((start, cursor, s.start_second, s.end_second));
    }
    let duration = segments
        .iter()
        .map(|s| s.end_second)
        .fold(0.0_f32, f32::max);

    let chunks = engine.list_source_chunks(&src_id).await.unwrap();
    assert!(!chunks.is_empty(), "audio source must produce chunks");

    for c in &chunks {
        let json = c
            .source_anchor
            .as_deref()
            .unwrap_or_else(|| panic!("chunk {} missing source_anchor", c.id));
        let (start, end) = parse_audio(json);

        assert!(start.is_finite() && end.is_finite(), "finite timestamps");
        assert!(
            start <= end,
            "start {start} <= end {end} for chunk {}",
            c.id
        );
        assert!(
            (0.0..=duration).contains(&start) && (0.0..=duration).contains(&end),
            "anchor [{start},{end}] within [0,{duration}] for chunk {}",
            c.id
        );

        // Expected [min,max] over segments whose char_start ∈ [char_start, char_end).
        let cs = c.char_start.expect("char_start") as usize;
        let ce = c.char_end.expect("char_end") as usize;
        let covered: Vec<&(usize, usize, f32, f32)> = ranges
            .iter()
            .filter(|(seg_start, _, _, _)| *seg_start >= cs && *seg_start < ce)
            .collect();
        assert!(
            !covered.is_empty(),
            "chunk {} [{cs},{ce}) covers no segment (buffer len {})",
            c.id,
            buffer.len()
        );
        let exp_min = covered.iter().map(|r| r.2).fold(f32::INFINITY, f32::min);
        let exp_max = covered
            .iter()
            .map(|r| r.3)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(start, exp_min, "chunk {} start = min(covered)", c.id);
        assert_eq!(end, exp_max, "chunk {} end = max(covered)", c.id);
    }
}

/// AC (c): `start_second` is monotonic non-decreasing across sequential chunks
/// evaluated PER LEVEL (the persisted output interleaves parents and children).
#[tokio::test]
async fn anchor_start_is_monotonic_per_level() {
    let Some((_dir, engine, src_id)) = ingest_audio(many_segments()).await else {
        return;
    };
    let chunks = engine.list_source_chunks(&src_id).await.unwrap();

    let mut levels: std::collections::BTreeMap<i32, Vec<f32>> = std::collections::BTreeMap::new();
    for c in &chunks {
        let (start, _) = parse_audio(c.source_anchor.as_deref().expect("anchor"));
        levels.entry(c.level).or_default().push(start);
    }
    assert!(levels.len() >= 2, "expected parent + child levels");
    for (level, starts) in &levels {
        for w in starts.windows(2) {
            assert!(
                w[0] <= w[1],
                "level {level} starts must be non-decreasing: {starts:?}"
            );
        }
    }
}

/// AC: a chunk_id resolves to a row whose `source_anchor` deserializes to Audio —
/// the exact chunk-row path #23 reads, with both parent and child levels present.
#[tokio::test]
async fn chunks_hydrate_to_audio_anchor_rows_across_levels() {
    let Some((_dir, engine, src_id)) = ingest_audio(many_segments()).await else {
        return;
    };
    let chunks = engine.list_source_chunks(&src_id).await.unwrap();

    let levels: std::collections::BTreeSet<i32> = chunks.iter().map(|c| c.level).collect();
    assert!(
        levels.contains(&0) && levels.len() >= 2,
        "parent (level 0) and child levels must both persist, got {levels:?}"
    );
    for c in &chunks {
        let anchor: SourceAnchor =
            serde_json::from_str(c.source_anchor.as_deref().expect("anchor json")).unwrap();
        assert!(
            matches!(anchor, SourceAnchor::Audio { .. }),
            "chunk {} anchor must be Audio",
            c.id
        );
    }
}

/// Regression: enrichment must leave `source_anchor` byte-identical. Runs the
/// worker to a terminal enrichment status, comparing the anchor JSON before/after.
#[tokio::test]
async fn enrichment_preserves_source_anchor() {
    use std::time::Duration;

    if !tokenizer_available().await {
        eprintln!("skipping enrichment_preserves_source_anchor: no tokenizer (offline)");
        return;
    }
    // Enable enrichment ON DISK before init so the worker's config-gated write
    // path actually runs (matches the enrichment integration harness).
    let dir = tempfile::tempdir().expect("tempdir");
    lens_core::LensEngine::init(dir.path())
        .await
        .expect("engine init");
    {
        let mut cfg = lens_core::config::AppConfig::load(dir.path()).expect("load config");
        cfg.enrichment.enabled = true;
        cfg.save(dir.path()).expect("save config");
    }
    let engine = lens_core::LensEngine::init(dir.path())
        .await
        .expect("engine re-init");
    support::inject_fake_embedder(&engine);
    use_mock_asr(&engine, many_segments()).await;

    let wav = dir.path().join("clip.wav");
    write_tone_wav(&wav, 2);
    let nb = engine
        .create_notebook("audio-enrich-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_file_source(&nb.id, &wav, None)
        .await
        .unwrap()
        .source;
    engine.ingest_source(&src.id, |_p| {}).await.unwrap();

    let before: Vec<(String, Option<String>)> = engine
        .list_source_chunks(&src.id)
        .await
        .unwrap()
        .into_iter()
        .map(|c| (c.id, c.source_anchor))
        .collect();
    assert!(
        before.iter().all(|(_, a)| a.is_some()),
        "all chunks anchored pre-enrichment"
    );

    engine.enqueue_enrichment_for_test(&src.id);
    let pool = engine.pool().await;
    let mut settled = false;
    for _ in 0..100 {
        let status: Option<String> =
            sqlx::query_scalar("SELECT enrichment_status FROM sources WHERE id = ?")
                .bind(&src.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        if matches!(status.as_deref(), Some("done") | Some("pending")) {
            settled = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(settled, "enrichment worker did not reach a terminal status");

    let after: Vec<(String, Option<String>)> = engine
        .list_source_chunks(&src.id)
        .await
        .unwrap()
        .into_iter()
        .map(|c| (c.id, c.source_anchor))
        .collect();
    assert_eq!(
        before, after,
        "source_anchor must be byte-identical before/after enrichment"
    );
}

/// Edge case: an empty transcript (zero segments) produces zero chunks and a
/// coherent terminal status without panicking.
#[tokio::test]
async fn empty_transcript_yields_zero_chunks() {
    if !tokenizer_available().await {
        eprintln!("skipping empty_transcript_yields_zero_chunks: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    use_mock_asr(&engine, Vec::new()).await;

    let wav = dir.path().join("clip.wav");
    write_tone_wav(&wav, 2);
    let nb = engine
        .create_notebook("audio-empty-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_file_source(&nb.id, &wav, None)
        .await
        .unwrap()
        .source;
    engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect("empty transcript must not panic");

    let chunks = engine.list_source_chunks(&src.id).await.unwrap();
    assert!(chunks.is_empty(), "empty transcript -> zero chunks");

    let pool = engine.pool().await;
    let status: String = sqlx::query_scalar("SELECT status FROM sources WHERE id = ?")
        .bind(&src.id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(
        status == "indexed" || status == "error",
        "source must reach a coherent terminal status, got {status}"
    );
}

/// A non-finite (NaN) segment timestamp is rejected before it can poison the
/// min/max aggregation.
#[tokio::test]
async fn non_finite_timestamp_is_rejected() {
    if !tokenizer_available().await {
        eprintln!("skipping non_finite_timestamp_is_rejected: no tokenizer (offline)");
        return;
    }
    let (dir, engine) = inject_counting_engine().await;
    use_mock_asr(
        &engine,
        vec![seg("first", 0.0, 1.0), seg("nan", f32::NAN, 2.0)],
    )
    .await;

    let wav = dir.path().join("clip.wav");
    write_tone_wav(&wav, 2);
    let nb = engine
        .create_notebook("audio-nan-nb", None, None)
        .await
        .unwrap();
    let src = engine
        .add_file_source(&nb.id, &wav, None)
        .await
        .unwrap()
        .source;
    let err = engine
        .ingest_source(&src.id, |_p| {})
        .await
        .expect_err("non-finite timestamp must error");
    assert_eq!(err.kind(), "Parse", "expected a Parse error, got {err:?}");
}
