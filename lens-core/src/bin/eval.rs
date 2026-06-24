//! Retrieval-quality eval harness (M4 Phase 1, Group g.2).
//!
//! Ingests the fixture corpus under `tests/fixtures/eval/*.md` into a temporary
//! LanceDB, runs the canned queries from `tests/fixtures/eval/queries.json`,
//! prints the top-5 chunk ids + a text snippet per query, and computes
//! **recall@5** against a documented baseline floor. Exits non-zero if recall
//! falls below the floor so the harness doubles as a CI regression guard (and,
//! later, the enrichment regression guard).
//!
//! # Determinism (gold chunk ids)
//!
//! recall@5 is measured by **`chunk_id` membership**, not text substring. Gold
//! `chunk_id`s only stay valid run-to-run if the chunker produces reproducible
//! ids — so this harness uses [`chunk_blocks_deterministic`] (content-derived
//! ids), NOT the production `ingest_source` pipeline (UUIDv7, time-random). The
//! `queries.json` gold sets are authored against those deterministic ids.
//!
//! # Network
//!
//! The harness embeds with the real [`FastembedEmbedder`], which downloads the
//! nomic-embed-text-v1.5 weights (~130 MB) on a cold cache. First run needs
//! network; subsequent runs reuse the cache under the temp dir's
//! `models/fastembed/`.
//!
//! # Usage
//!
//! ```text
//! cargo run -p lens-core --bin eval               # run recall@5, exit 0/1
//! cargo run -p lens-core --bin eval -- --print-ids  # dump deterministic ids per doc
//! ```
//!
//! `--print-ids` is the authoring aid: it prints every child chunk's
//! deterministic id + section path + snippet so the `queries.json` gold sets
//! can be filled in. It does not embed or search.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lens_core::chunk::{Chunk, chunk_blocks_deterministic};
use lens_core::embedder::{EMBED_DIM, EMBED_MODEL_ID, Embedder, FastembedEmbedder};
use lens_core::parse::{SourceKind, parse_blocks};
use lens_core::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use lens_core::{LensEngine, LensError};
use serde::Deserialize;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Recall gate (measured-baseline-minus-margin — see plan §4 / Step g.2)
// ---------------------------------------------------------------------------

/// Measured recall@5 on the fixture corpus with the real nomic model. Recorded
/// here so the gate is "measured baseline minus margin", not a hand-picked
/// constant. Each of the 5 canned queries retrieves its gold chunk in the top-5
/// → 5/5 = 1.00. Last measured 2026-06-24 (`cargo run -p lens-core --bin eval`,
/// hits 5/5, recall@5 1.0000) — re-confirmed after the M4 robustness fixes; the
/// deterministic chunk ids in `queries.json` were unchanged (no fixture block
/// exceeds the parent token bound, so the oversized-parent split never fires).
const BASELINE_RECALL: f32 = 1.00;

/// Margin subtracted from the baseline to set the pass floor. A single query
/// regressing out of the top-5 on this 4-doc / ~12-chunk corpus drops recall by
/// 0.20, so a 0.25 margin tolerates exactly one such regression before the gate
/// fails — tight enough to catch a real retrieval-quality break (e.g. a dropped
/// nomic prefix or a broken cosine pin) without flapping on a single borderline
/// query.
const MARGIN: f32 = 0.25;

/// The pass floor: recall@5 below this fails the harness (non-zero exit).
const RECALL_FLOOR: f32 = BASELINE_RECALL - MARGIN;

/// k is pinned at 5 for recall@5 (plan §4 / Step g.2).
const K: usize = 5;

/// Title for the single eval-corpus notebook. A REAL notebook row is created at
/// runtime (below) because `embedding_index.notebook_id` has a FK to
/// `notebooks(id)` — the registry `register()` inside `store.add` would fail
/// against a literal id that has no backing notebook row. Isolation across
/// notebooks is exercised by the integration tests, not here.
const EVAL_NOTEBOOK_TITLE: &str = "eval-corpus";

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
    let docs = load_corpus(&fixtures_dir)?;

    // Authoring aid: dump deterministic ids and exit (no embedding/search).
    if std::env::args().any(|a| a == "--print-ids") {
        // The tokenizer needs the real fastembed cache; build the engine to get
        // its data dir + the embedder (which downloads/locates the tokenizer).
        let dir = tempfile::tempdir().map_err(|e| LensError::Io(e.to_string()))?;
        let engine = LensEngine::init(dir.path()).await?;
        let tokenizer = load_tokenizer(&engine).await?;
        print_ids(&docs, &tokenizer)?;
        return Ok(ExitCode::SUCCESS);
    }

    // ── Build a temp engine + on-disk LanceDB ──────────────────────────────
    let dir = tempfile::tempdir().map_err(|e| LensError::Io(e.to_string()))?;
    let engine = LensEngine::init(dir.path()).await?;
    let data_dir = dir.path();
    let pool = engine.pool().await;
    let store = LanceVectorStore::new(data_dir, pool);

    // Create a REAL notebook so the registry FK (embedding_index.notebook_id →
    // notebooks.id) is satisfied when store.add registers the table.
    let notebook = engine
        .create_notebook(EVAL_NOTEBOOK_TITLE, None, None)
        .await?;
    let notebook_id = notebook.id.as_str();

    // The embedder is the real nomic model (downloads on cold cache). We build
    // it directly here so the eval harness does not depend on the engine's
    // private cache seam.
    println!("Building embedder ({EMBED_MODEL_ID}); first run downloads ~130 MB from HuggingFace…");
    let embedder = FastembedEmbedder::new(data_dir)?;
    let tokenizer = load_tokenizer(&engine).await?;

    // ── Ingest the corpus with DETERMINISTIC ids ───────────────────────────
    // chunk_id -> (text snippet) for printing later.
    let mut chunk_text: HashMap<String, String> = HashMap::new();
    for doc in &docs {
        let blocks = parse_blocks(&doc.text, SourceKind::Markdown);
        let chunks = chunk_blocks_deterministic(&doc.text, &blocks, &tokenizer)?;

        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let vectors = embedder.embed_documents(&texts)?;

        let mut rows = Vec::with_capacity(chunks.len());
        for (chunk, vector) in chunks.iter().zip(vectors.into_iter()) {
            chunk_text.insert(chunk.id.clone(), snippet(&chunk.text));
            rows.push(VectorRow {
                chunk_id: chunk.id.clone(),
                source_id: doc.name.clone(),
                notebook_id: notebook_id.to_string(),
                level: chunk.level,
                vector,
            });
        }
        store
            .add(notebook_id, EMBED_MODEL_ID, EMBED_DIM, rows)
            .await?;
        println!("ingested {} ({} chunks)", doc.name, chunks.len());
    }

    // ── Run queries + compute recall@5 ─────────────────────────────────────
    let queries = load_queries(&fixtures_dir)?;
    let mut hits = 0usize;
    println!("\n=== Retrieval results (k = {K}) ===");
    for q in &queries {
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

    let recall = if queries.is_empty() {
        0.0
    } else {
        hits as f32 / queries.len() as f32
    };

    println!("\n=== recall@{K} ===");
    println!("hits        : {hits}/{}", queries.len());
    println!("recall@{K}    : {recall:.4}");
    println!("baseline    : {BASELINE_RECALL:.4}");
    println!("margin      : {MARGIN:.4}");
    println!("floor       : {RECALL_FLOOR:.4}");

    if recall + 1e-6 < RECALL_FLOOR {
        eprintln!(
            "\nFAIL: recall@{K} {recall:.4} is below the floor {RECALL_FLOOR:.4} (baseline {BASELINE_RECALL:.4} − margin {MARGIN:.4})"
        );
        return Ok(ExitCode::FAILURE);
    }
    println!("\nPASS: recall@{K} {recall:.4} ≥ floor {RECALL_FLOOR:.4}");
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------
// Corpus / query loading
// ---------------------------------------------------------------------------

/// A loaded fixture document (file stem + verbatim text).
struct Doc {
    name: String,
    text: String,
}

/// Resolves `tests/fixtures/eval/` relative to the crate manifest dir so the
/// harness works regardless of the process CWD.
fn eval_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("eval")
}

/// Loads every `*.md` file in `dir`, sorted by file name for determinism.
fn load_corpus(dir: &Path) -> Result<Vec<Doc>, LensError> {
    let mut docs = Vec::new();
    let entries =
        std::fs::read_dir(dir).map_err(|e| LensError::Io(format!("{}: {e}", dir.display())))?;
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path)
            .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
        docs.push(Doc { name, text });
    }
    if docs.is_empty() {
        return Err(LensError::Validation(format!(
            "no .md fixtures in {}",
            dir.display()
        )));
    }
    Ok(docs)
}

/// Loads and parses `queries.json`.
fn load_queries(dir: &Path) -> Result<Vec<Query>, LensError> {
    let path = dir.join("queries.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
    let queries: Vec<Query> = serde_json::from_str(&raw)?;
    Ok(queries)
}

/// Loads the nomic tokenizer via the engine's data dir, reusing the ingest
/// pipeline's shared resolver ([`lens_core::resolve_nomic_tokenizer`]) so the
/// eval harness and production share one 3-step resolution + one atomic
/// `.part`→rename download. It locates a `tokenizer.json` already laid down by a
/// `FastembedEmbedder::new` build (the `run` path builds the embedder before
/// calling this), or downloads it once if the fastembed layout omitted one. It
/// does NOT construct a second embedder.
async fn load_tokenizer(engine: &LensEngine) -> Result<Tokenizer, LensError> {
    let data_dir = PathBuf::from(engine.config().await.paths.data_dir);
    lens_core::resolve_nomic_tokenizer(&data_dir).await
}

// ---------------------------------------------------------------------------
// --print-ids authoring aid
// ---------------------------------------------------------------------------

/// Prints every child chunk's deterministic id + section path + snippet so
/// `queries.json` gold sets can be authored against stable ids.
fn print_ids(docs: &[Doc], tokenizer: &Tokenizer) -> Result<(), LensError> {
    for doc in docs {
        println!("\n## {} ##", doc.name);
        let blocks = parse_blocks(&doc.text, SourceKind::Markdown);
        let chunks = chunk_blocks_deterministic(&doc.text, &blocks, tokenizer)?;
        for c in &chunks {
            print_chunk_line(c);
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

/// First 12 chars of a chunk id (the deterministic ids are 64-hex; the prefix is
/// plenty for human-readable result tables).
fn short_id(id: &str) -> String {
    id.chars().take(12).collect()
}
