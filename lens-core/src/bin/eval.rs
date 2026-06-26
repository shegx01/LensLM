//! Retrieval-quality eval harness (M4 Phase 3, AC15 — A/B enrichment measurement).
//!
//! Ingests fixture corpora under `tests/fixtures/eval/*.md` into a temporary
//! LanceDB, runs canned queries, prints the top-5 chunk ids + a text snippet per
//! query, and computes **recall@5**. Two embedding paths are measured on the SAME
//! corpus so the enrichment delta is visible in one run:
//!
//! - **raw**: embeds each chunk's canonical `chunk.text` (the Phase-1/2 floor).
//! - **enriched**: embeds `COALESCE(embedding_text, text)` where `embedding_text`
//!   is composed EXACTLY as the production worker composes it
//!   ([`lens_core::enrichment::compose_prefix`] + [`compose_embedding_text`]) plus a
//!   synthesized doc-summary RAPTOR node — i.e. the eval embeds what the production
//!   re-embed flip embeds. This is a DETERMINISTIC stand-in: NO live LLM, NO mpsc,
//!   NO table flip. The per-doc "summary" the worker would get from the LLM
//!   structural map is derived deterministically from the document's own title +
//!   lead sentence (the same contextual signal the structural map carries:
//!   the document's named entity/topic). The summary node id is a content-derived
//!   hash so its gold membership is stable run-to-run.
//!
//! # Why two corpora
//!
//! The original 4-doc corpus (`queries.json`) is already SATURATED at raw
//! recall@5 = 1.00, so it can only prove *no regression* (`enriched >= raw`). To
//! prove enrichment *improves* retrieval we add a pronoun/coref fixture
//! (`pronoun_context.md` + `pronoun_queries.json`) authored so the gold chunks name
//! their subject only in the document title — the bodies use bare pronouns ("It
//! used …"). RAW embedding of the body alone genuinely MISSES at least one gold
//! chunk (recall < 1.00, recorded as [`RAW_RECALL_PRONOUN_FIXTURE`]); the enriched
//! `embedding_text`, which prepends the `[Document: …] [Section: …]` context
//! carrying the entity name, recovers it (`enriched > raw` on that fixture).
//!
//! # Determinism (gold chunk ids)
//!
//! recall@5 is measured by **`chunk_id` membership**, not text substring. Gold ids
//! only stay valid run-to-run if the chunker produces reproducible ids — so this
//! harness uses [`chunk_blocks_deterministic`] (content-derived ids), NOT the
//! production `ingest_source` pipeline (UUIDv7). The synthesized summary node also
//! uses a content-derived id ([`summary_node_id`]).
//!
//! # Network
//!
//! The harness embeds with the real [`FastembedEmbedder`], which downloads the
//! nomic-embed-text-v1.5 weights (~130 MB) on a cold cache.
//!
//! # Usage
//!
//! ```text
//! cargo run -p lens-core --bin eval               # raw recall@5 on the main corpus, exit 0/1
//! cargo run -p lens-core --bin eval -- --enriched # A/B: raw vs enriched on BOTH corpora + gates
//! cargo run -p lens-core --bin eval -- --print-ids  # dump deterministic ids per doc
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lens_core::chunk::{Chunk, chunk_blocks_deterministic};
use lens_core::embedder::{EMBED_DIM, EMBED_MODEL_ID, Embedder, FastembedEmbedder};
use lens_core::enrichment::{CorefStrategy, compose_embedding_text, compose_prefix};
use lens_core::parse::{SourceKind, parse_blocks};
use lens_core::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use lens_core::{LensEngine, LensError};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Recall gates
// ---------------------------------------------------------------------------

/// Measured raw recall@5 on the MAIN corpus (`queries.json`, 4 docs). Each query
/// retrieves its gold chunk in the top-5 → 5/5 = 1.00. Used only as a sanity floor
/// on the raw path; the A/B gate (`enriched >= raw`) is what guards the main corpus
/// under `--enriched`.
const MAIN_BASELINE_RECALL: f32 = 1.00;

/// Margin subtracted from the main-corpus baseline to set the raw pass floor. A
/// single query regressing out of the top-5 on this 4-doc / ~12-chunk corpus drops
/// recall by 0.20, so a 0.25 margin tolerates exactly one such regression.
const MAIN_MARGIN: f32 = 0.25;

/// The raw pass floor on the main corpus.
const MAIN_RECALL_FLOOR: f32 = MAIN_BASELINE_RECALL - MAIN_MARGIN;

/// **Measured** raw recall@5 on the pronoun/coref fixture (`pronoun_context.md` +
/// `pronoun_queries.json`). This MUST be `< 1.00`: the fixture is authored so the
/// gold chunks name their subject only in the document title, while the chunk
/// bodies use bare pronouns ("It used …"). Embedding the raw body alone genuinely
/// misses at least one gold chunk against an entity-named query, so raw recall sits
/// below 1.00 — which is exactly what makes the enrichment improvement measurable.
///
/// Last measured 2026-06-26 via `cargo run -p lens-core --bin eval -- --enriched`:
/// raw recall@5 on the 3-query pronoun fixture = 0.6667 (2/3 — one pronoun query
/// misses its gold child because the body never names "Antikythera mechanism").
/// The enriched path recovers it (3/3 = 1.0000). Recorded here so the gate is a
/// MEASURED baseline, not a hand-picked constant; if the chunker or embedder
/// changes this number, re-measure and update it (it must stay `< 1.00`).
const RAW_RECALL_PRONOUN_FIXTURE: f32 = 0.6667;

/// k is pinned at 5 for recall@5.
const K: usize = 5;

/// Title for the eval-corpus notebook (a REAL notebook row is created at runtime
/// because `embedding_index.notebook_id` has a FK to `notebooks(id)`).
const EVAL_NOTEBOOK_TITLE: &str = "eval-corpus";

/// The coref strategy the enriched eval path composes with — the production default
/// ([`CorefStrategy::LlmInline`], `config.rs` enrichment default).
const EVAL_COREF: CorefStrategy = CorefStrategy::LlmInline;

// ---------------------------------------------------------------------------
// queries.json schema
// ---------------------------------------------------------------------------

/// One canned query plus the gold chunk ids it should retrieve.
#[derive(Debug, Deserialize)]
struct Query {
    /// The natural-language query string.
    query: String,
    /// The deterministic `chunk_id`(s) that answer this query. recall@5 counts a
    /// query as a hit if ANY of these appears in the top-5 search results.
    gold_chunk_ids: Vec<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("eval harness failed: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<ExitCode, LensError> {
    let fixtures_dir = eval_fixtures_dir();
    let enriched_mode = std::env::args().any(|a| a == "--enriched");

    // Authoring aid: dump deterministic ids and exit (no embedding/search).
    if std::env::args().any(|a| a == "--print-ids") {
        let dir = tempfile::tempdir().map_err(|e| LensError::Io(e.to_string()))?;
        let engine = LensEngine::init(dir.path()).await?;
        let tokenizer = load_tokenizer(&engine).await?;
        let main = load_corpus(&fixtures_dir, MAIN_DOCS)?;
        let pronoun = load_corpus(&fixtures_dir, PRONOUN_DOCS)?;
        print_ids(&main, &tokenizer, "main")?;
        print_ids(&pronoun, &tokenizer, "pronoun")?;
        return Ok(ExitCode::SUCCESS);
    }

    // ── Shared infra: one temp engine + embedder + tokenizer for the whole run.
    let dir = tempfile::tempdir().map_err(|e| LensError::Io(e.to_string()))?;
    let engine = LensEngine::init(dir.path()).await?;
    let data_dir = dir.path();
    let pool = engine.pool().await;

    println!("Building embedder ({EMBED_MODEL_ID}); first run downloads ~130 MB from HuggingFace…");
    let embedder = FastembedEmbedder::new(data_dir)?;
    let tokenizer = load_tokenizer(&engine).await?;

    // ── Main corpus: raw path always; enriched path only under --enriched. ──
    let main_docs = load_corpus(&fixtures_dir, MAIN_DOCS)?;
    let main_queries = load_queries(&fixtures_dir, "queries.json")?;

    if !enriched_mode {
        // Default mode: raw recall on the main corpus with the floor gate (the
        // historical CI guard).
        println!("\n############ MAIN CORPUS — RAW ############");
        let raw = measure(
            &engine,
            &embedder,
            &tokenizer,
            data_dir,
            &pool,
            &main_docs,
            &main_queries,
            EmbedMode::Raw,
        )
        .await?;
        report_recall("raw", raw.hits, raw.total);
        println!("baseline    : {MAIN_BASELINE_RECALL:.4}");
        println!("margin      : {MAIN_MARGIN:.4}");
        println!("floor       : {MAIN_RECALL_FLOOR:.4}");
        if raw.recall() + 1e-6 < MAIN_RECALL_FLOOR {
            eprintln!(
                "\nFAIL: raw recall@{K} {:.4} is below the floor {MAIN_RECALL_FLOOR:.4}",
                raw.recall()
            );
            return Ok(ExitCode::FAILURE);
        }
        println!(
            "\nPASS: raw recall@{K} {:.4} ≥ floor {MAIN_RECALL_FLOOR:.4}",
            raw.recall()
        );
        println!("\n(run with --enriched for the A/B enrichment measurement + gates)");
        return Ok(ExitCode::SUCCESS);
    }

    // ── A/B mode (--enriched): raw vs enriched on BOTH corpora + the AC15 gates. ──
    println!("\n############ MAIN CORPUS (no-regression gate) ############");
    println!("\n--- raw ---");
    let main_raw = measure(
        &engine,
        &embedder,
        &tokenizer,
        data_dir,
        &pool,
        &main_docs,
        &main_queries,
        EmbedMode::Raw,
    )
    .await?;
    report_recall("raw", main_raw.hits, main_raw.total);

    println!("\n--- enriched ---");
    let main_enr = measure(
        &engine,
        &embedder,
        &tokenizer,
        data_dir,
        &pool,
        &main_docs,
        &main_queries,
        EmbedMode::Enriched,
    )
    .await?;
    report_recall("enriched", main_enr.hits, main_enr.total);

    let pronoun_docs = load_corpus(&fixtures_dir, PRONOUN_DOCS)?;
    let pronoun_queries = load_queries(&fixtures_dir, "pronoun_queries.json")?;

    println!("\n############ PRONOUN/COREF FIXTURE (improvement gate) ############");
    println!("\n--- raw ---");
    let pron_raw = measure(
        &engine,
        &embedder,
        &tokenizer,
        data_dir,
        &pool,
        &pronoun_docs,
        &pronoun_queries,
        EmbedMode::Raw,
    )
    .await?;
    report_recall("raw", pron_raw.hits, pron_raw.total);

    println!("\n--- enriched ---");
    let pron_enr = measure(
        &engine,
        &embedder,
        &tokenizer,
        data_dir,
        &pool,
        &pronoun_docs,
        &pronoun_queries,
        EmbedMode::Enriched,
    )
    .await?;
    report_recall("enriched", pron_enr.hits, pron_enr.total);

    // ── AC15 gates ──────────────────────────────────────────────────────────
    println!("\n=================== AC15 GATES ===================");
    println!(
        "main  : raw {:.4}  enriched {:.4}",
        main_raw.recall(),
        main_enr.recall()
    );
    println!(
        "pronoun: raw {:.4}  enriched {:.4}  (recorded RAW_RECALL_PRONOUN_FIXTURE = {RAW_RECALL_PRONOUN_FIXTURE:.4})",
        pron_raw.recall(),
        pron_enr.recall()
    );

    let mut failed = false;

    // Gate 1 — no regression on the main corpus: enriched >= raw.
    if main_enr.recall() + 1e-6 < main_raw.recall() {
        eprintln!(
            "\nFAIL [no-regression]: main enriched recall@{K} {:.4} < raw {:.4}",
            main_enr.recall(),
            main_raw.recall()
        );
        failed = true;
    } else {
        println!(
            "PASS [no-regression]: main enriched {:.4} ≥ raw {:.4}",
            main_enr.recall(),
            main_raw.recall()
        );
    }

    // Sanity: the recorded const must honestly match the measured raw recall of the
    // pronoun fixture AND be < 1.00 (else the fixture proves nothing).
    if (pron_raw.recall() - RAW_RECALL_PRONOUN_FIXTURE).abs() > 1e-3 {
        eprintln!(
            "\nFAIL [fixture drift]: measured pronoun raw recall {:.4} != recorded RAW_RECALL_PRONOUN_FIXTURE {RAW_RECALL_PRONOUN_FIXTURE:.4}; re-measure and update the const",
            pron_raw.recall()
        );
        failed = true;
    }
    if RAW_RECALL_PRONOUN_FIXTURE >= 1.0 {
        eprintln!(
            "\nFAIL [fixture invalid]: RAW_RECALL_PRONOUN_FIXTURE {RAW_RECALL_PRONOUN_FIXTURE:.4} is saturated (>= 1.00); the fixture cannot demonstrate improvement"
        );
        failed = true;
    }

    // Gate 2 — strict improvement on the pronoun fixture: enriched > recorded raw.
    if pron_enr.recall() <= RAW_RECALL_PRONOUN_FIXTURE + 1e-6 {
        eprintln!(
            "\nFAIL [improvement]: pronoun enriched recall@{K} {:.4} does NOT exceed RAW_RECALL_PRONOUN_FIXTURE {RAW_RECALL_PRONOUN_FIXTURE:.4}",
            pron_enr.recall()
        );
        failed = true;
    } else {
        println!(
            "PASS [improvement]: pronoun enriched {:.4} > RAW_RECALL_PRONOUN_FIXTURE {RAW_RECALL_PRONOUN_FIXTURE:.4}",
            pron_enr.recall()
        );
    }

    if failed {
        return Ok(ExitCode::FAILURE);
    }
    println!("\nPASS: all AC15 gates green.");
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------
// Measurement core
// ---------------------------------------------------------------------------

/// Which text each chunk contributes to its embedding.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EmbedMode {
    /// Embed the canonical `chunk.text` (the Phase-1/2 raw floor).
    Raw,
    /// Embed `COALESCE(embedding_text, text)` — the production re-embed input —
    /// plus a synthesized doc-summary node per doc.
    Enriched,
}

/// The outcome of one measurement pass.
struct Recall {
    hits: usize,
    total: usize,
}

impl Recall {
    fn recall(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.hits as f32 / self.total as f32
        }
    }
}

/// Ingests `docs` under `mode` into a FRESH per-pass notebook + Lance table, runs
/// `queries`, prints the result table, and returns recall@5. Each pass uses its own
/// notebook so the raw and enriched passes never share vectors.
#[allow(clippy::too_many_arguments)]
async fn measure(
    engine: &LensEngine,
    embedder: &FastembedEmbedder,
    tokenizer: &Tokenizer,
    data_dir: &Path,
    pool: &sqlx::SqlitePool,
    docs: &[Doc],
    queries: &[Query],
    mode: EmbedMode,
) -> Result<Recall, LensError> {
    let store = LanceVectorStore::new(data_dir, pool.clone());
    let notebook = engine
        .create_notebook(EVAL_NOTEBOOK_TITLE, None, None)
        .await?;
    let notebook_id = notebook.id.as_str();

    // chunk_id -> snippet for the printed result table.
    let mut chunk_text: HashMap<String, String> = HashMap::new();

    for doc in docs {
        let blocks = parse_blocks(&doc.text, SourceKind::Markdown);
        let chunks = chunk_blocks_deterministic(&doc.text, &blocks, tokenizer)?;

        // The deterministic per-doc "summary" stand-in for the structural map: the
        // document title + lead sentence. Carries the document's named entity/topic
        // — exactly the contextual signal the LLM structural map provides — so the
        // composed `[Document: …]` prefix anchors pronoun/coref queries.
        let doc_summary = derive_doc_summary(&chunks);

        let mut rows: Vec<VectorRow> = Vec::with_capacity(chunks.len() + 1);
        let mut texts: Vec<String> = Vec::with_capacity(chunks.len() + 1);
        let mut ids: Vec<String> = Vec::with_capacity(chunks.len() + 1);
        let mut levels: Vec<i32> = Vec::with_capacity(chunks.len() + 1);

        for chunk in &chunks {
            chunk_text.insert(chunk.id.clone(), snippet(&chunk.text));
            let embed_text = match mode {
                EmbedMode::Raw => chunk.text.clone(),
                EmbedMode::Enriched => {
                    // EXACT production composition: compose_prefix → compose_embedding_text.
                    let prefix = compose_prefix(&doc_summary, &chunk.section_path, EVAL_COREF);
                    compose_embedding_text(&prefix, &chunk.text, Some(tokenizer))
                }
            };
            ids.push(chunk.id.clone());
            levels.push(chunk.level);
            texts.push(embed_text);
        }

        // The synthesized doc-summary RAPTOR node (AC6): enriched mode only, and
        // only when a non-empty summary exists (mirrors reembed.rs:54).
        if mode == EmbedMode::Enriched && !doc_summary.trim().is_empty() {
            let sid = summary_node_id(&doc.name, &doc_summary);
            chunk_text.insert(sid.clone(), snippet(&doc_summary));
            ids.push(sid);
            levels.push(2);
            texts.push(doc_summary.clone());
        }

        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vectors = embedder.embed_documents(&text_refs)?;
        for ((id, level), vector) in ids.into_iter().zip(levels).zip(vectors.into_iter()) {
            rows.push(VectorRow {
                chunk_id: id,
                source_id: doc.name.clone(),
                notebook_id: notebook_id.to_string(),
                level,
                vector,
            });
        }
        let n = rows.len();
        store
            .add(notebook_id, EMBED_MODEL_ID, EMBED_DIM, rows)
            .await?;
        println!("ingested {} ({n} rows)", doc.name);
    }

    let mut hits = 0usize;
    println!("=== Retrieval (k = {K}) ===");
    for q in queries {
        let qvec = embedder.embed_query(&q.query)?;
        let results = store
            .search(notebook_id, EMBED_MODEL_ID, EMBED_DIM, &qvec, K)
            .await?;
        let top_ids: Vec<&str> = results.iter().map(|h| h.chunk_id.as_str()).collect();
        let hit = q
            .gold_chunk_ids
            .iter()
            .any(|g| top_ids.contains(&g.as_str()));
        if hit {
            hits += 1;
        }
        println!(
            "\nquery: {:?}  -> {}",
            q.query,
            if hit { "HIT" } else { "MISS" }
        );
        for (rank, h) in results.iter().enumerate() {
            let gold_mark = if q.gold_chunk_ids.contains(&h.chunk_id) {
                " *gold*"
            } else {
                ""
            };
            let snip = chunk_text
                .get(&h.chunk_id)
                .map(String::as_str)
                .unwrap_or("?");
            println!(
                "  {}. {} (d={:.4}){}  {}",
                rank + 1,
                short_id(&h.chunk_id),
                h.distance,
                gold_mark,
                snip
            );
        }
    }

    Ok(Recall {
        hits,
        total: queries.len(),
    })
}

/// Prints a recall@k summary line block.
fn report_recall(label: &str, hits: usize, total: usize) {
    let recall = if total == 0 {
        0.0
    } else {
        hits as f32 / total as f32
    };
    println!("\n=== {label} recall@{K} ===");
    println!("hits        : {hits}/{total}");
    println!("recall@{K}    : {recall:.4}");
}

/// Derives the deterministic per-doc "summary" stand-in for the LLM structural map.
///
/// The production worker composes `embedding_text` from the structural map's
/// `summary` (an LLM-authored sentence naming the document's entities/topic). The
/// eval is deterministic (no LLM), so it derives the same KIND of signal from the
/// document itself: the H1 title (first level-0 parent's `section_path`) plus the
/// lead sentence of the first parent body. This carries the named entity ("The
/// Antikythera Mechanism") that the pronoun-bearing chunk bodies omit — the exact
/// context the structural-map-derived prefix supplies in production.
fn derive_doc_summary(chunks: &[Chunk]) -> String {
    let Some(first_parent) = chunks.iter().find(|c| c.level == 0) else {
        return String::new();
    };
    // The H1 / top heading is the leading segment of the section path.
    let title = first_parent
        .section_path
        .split('>')
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    // The lead sentence of the first parent body (up to the first period).
    let lead = first_parent
        .text
        .split(['.', '\n'])
        .map(str::trim)
        .find(|s| !s.is_empty());

    match (title, lead) {
        // Lead already names the title's subject → the lead sentence alone carries
        // the entity. Otherwise prepend the title so the entity is always present.
        (Some(t), Some(l)) if l.starts_with(t) => format!("{l}."),
        (Some(t), Some(l)) => format!("{t}. {l}."),
        (Some(t), None) => t.to_string(),
        (None, Some(l)) => format!("{l}."),
        (None, None) => String::new(),
    }
}

/// Content-derived id for the synthesized doc-summary node, mirroring the
/// `chunk_blocks_deterministic` scheme (level=2, empty section_path, summary text,
/// ordinal 0) so the id is stable run-to-run and unique per document.
fn summary_node_id(doc_name: &str, summary: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(2i32.to_le_bytes());
    hasher.update(b"\x00");
    hasher.update(doc_name.as_bytes());
    hasher.update(b"\x00");
    hasher.update(summary.as_bytes());
    hasher.update(b"\x00summary");
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Corpus / query loading
// ---------------------------------------------------------------------------

/// The MAIN corpus doc stems (the saturated 4-doc set; `queries.json`).
const MAIN_DOCS: &[&str] = &["espresso", "photosynthesis", "rust_ownership", "tides"];

/// The pronoun/coref fixture corpus (`pronoun_queries.json`). The 4 main docs are
/// included as DISTRACTORS so the top-5 is genuinely contested: with a single
/// short doc (≤5 chunks) every chunk would trivially land in the top-5 and recall
/// would be a meaningless 1.00. The distractors force the pronoun-bearing gold
/// chunk to actually out-rank unrelated chunks, so a raw miss is real.
const PRONOUN_DOCS: &[&str] = &[
    "pronoun_context",
    "tide_prediction",
    "espresso",
    "photosynthesis",
    "rust_ownership",
    "tides",
];

/// A loaded fixture document (file stem + verbatim text).
struct Doc {
    name: String,
    text: String,
}

/// Resolves `tests/fixtures/eval/` relative to the crate manifest dir.
fn eval_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("eval")
}

/// Loads the named `*.md` fixtures (by stem) from `dir`, in the order given so the
/// corpus is deterministic.
fn load_corpus(dir: &Path, stems: &[&str]) -> Result<Vec<Doc>, LensError> {
    let mut docs = Vec::with_capacity(stems.len());
    for stem in stems {
        let path = dir.join(format!("{stem}.md"));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
        docs.push(Doc {
            name: (*stem).to_string(),
            text,
        });
    }
    if docs.is_empty() {
        return Err(LensError::Validation(format!(
            "no fixtures loaded from {}",
            dir.display()
        )));
    }
    Ok(docs)
}

/// Loads and parses a queries JSON file by name.
fn load_queries(dir: &Path, file: &str) -> Result<Vec<Query>, LensError> {
    let path = dir.join(file);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
    let queries: Vec<Query> = serde_json::from_str(&raw)?;
    Ok(queries)
}

/// Loads the nomic tokenizer via the engine's data dir, reusing the ingest
/// pipeline's shared resolver.
async fn load_tokenizer(engine: &LensEngine) -> Result<Tokenizer, LensError> {
    let data_dir = PathBuf::from(engine.config().await.paths.data_dir);
    lens_core::resolve_nomic_tokenizer(&data_dir).await
}

// ---------------------------------------------------------------------------
// --print-ids authoring aid
// ---------------------------------------------------------------------------

/// Prints every chunk's deterministic id + section path + snippet, plus the
/// synthesized summary-node id, so gold sets can be authored against stable ids.
fn print_ids(docs: &[Doc], tokenizer: &Tokenizer, corpus: &str) -> Result<(), LensError> {
    println!("\n######## corpus: {corpus} ########");
    for doc in docs {
        println!("\n## {} ##", doc.name);
        let blocks = parse_blocks(&doc.text, SourceKind::Markdown);
        let chunks = chunk_blocks_deterministic(&doc.text, &blocks, tokenizer)?;
        for c in &chunks {
            print_chunk_line(c);
        }
        let summary = derive_doc_summary(&chunks);
        if !summary.trim().is_empty() {
            println!(
                "  [S] {}  summary={:?}",
                summary_node_id(&doc.name, &summary),
                snippet(&summary)
            );
        }
    }
    Ok(())
}

fn print_chunk_line(c: &Chunk) {
    println!(
        "  [{}] {}  section={:?}  {}",
        if c.level == 0 { "P" } else { "c" },
        c.id,
        c.section_path,
        snippet(&c.text)
    );
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// First ~70 chars of `text`, single-lined, for readable output.
fn snippet(text: &str) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > 70 {
        let truncated: String = one_line.chars().take(67).collect();
        format!("{truncated}…")
    } else {
        one_line
    }
}

/// First 12 chars of a chunk id (the deterministic ids are 64-hex).
fn short_id(id: &str) -> String {
    id.chars().take(12).collect()
}
