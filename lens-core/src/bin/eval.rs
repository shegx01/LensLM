//! Retrieval-quality eval harness (M4 Phase 3, AC15 — three-way enrichment measurement).
//!
//! Ingests fixture corpora under `tests/fixtures/eval/*.md` into a temporary
//! LanceDB, runs canned queries, prints the top-5 chunk ids + a text snippet per
//! query, and computes **recall@5**. THREE embedding paths are measured on the SAME
//! corpus so the enrichment delta — and specifically the coref delta — is visible
//! in one run:
//!
//! - **raw**: embeds each chunk's canonical `chunk.text` (the Phase-1/2 floor).
//! - **prefix-only**: embeds `prefix + chunk.text`, where `prefix` is the
//!   `[Document: …] [Section: …]` context composed EXACTLY as the production worker
//!   composes it ([`lens_core::enrichment::compose_prefix`] +
//!   [`compose_embedding_text`]), plus a synthesized doc-summary RAPTOR node. This is
//!   the context-prefix enrichment WITHOUT any coreference resolution.
//! - **prefix+coref**: identical to prefix-only EXCEPT the chunk body is first run
//!   through the PRODUCTION [`lens_core::enrichment::coref::apply_substitutions`] to
//!   resolve validated referential mentions (pronouns / definite descriptions) to
//!   their named-entity antecedents before composing `embedding_text`. This is the
//!   full production re-embed input.
//!
//! This is a DETERMINISTIC stand-in for the live pipeline: NO live LLM, NO mpsc, NO
//! table flip. Two LLM-derived signals are supplied deterministically from the
//! fixtures themselves:
//!
//! - the per-doc "summary" the worker would get from the LLM structural map is
//!   derived from the document's own title + lead sentence (the same contextual
//!   signal the structural map carries: the document's named entity/topic);
//! - the coref SUBSTITUTIONS the worker would get from the LLM are supplied as a
//!   small AUTHORED coref map per fixture (the exact `(mention → antecedent)` edits a
//!   perfect coref model would emit), with the doc entities passed as
//!   `allowed_antecedents`. The eval then calls the SAME production
//!   `apply_substitutions` the worker calls — so the prefix+coref recall reflects the
//!   REAL substitution code, deterministically and with no LLM.
//!
//! # Why three corpora
//!
//! The original 4-doc corpus (`queries.json`) is already SATURATED at raw
//! recall@5 = 1.00, so it can only prove *no regression* (`enriched >= raw`).
//!
//! The pronoun fixture (`pronoun_context.md` + `pronoun_queries.json`) proves the
//! CONTEXT PREFIX adds value: its gold chunks name their subject only in the
//! document title, so RAW embedding of the body alone misses
//! ([`RAW_RECALL_PRONOUN_FIXTURE`] < 1.00) and the prefix recovers it.
//!
//! The coref fixture (`golden_record.md` + `coref_queries.json`) proves COREF adds
//! value BEYOND the prefix: its gold "Who chose what it carried" chunk's only link to
//! the query "Who chose the recordings and contents of the Voyager Golden Record?" is
//! the definite description "The disc", whose antecedent "Voyager Golden Record"
//! appears throughout the document body but NOT in the gold chunk itself and NOT in
//! the title ("A Message to the Stars") the prefix is derived from. Five OTHER
//! golden_record chunks DO name the record and fill the top-5, so prefix-only STILL
//! misses ([`PREFIX_ONLY_RECALL_COREF_FIXTURE`] < 1.00); only resolving "The disc" to
//! "The Voyager Golden Record" — the prefix+coref path running the production
//! `apply_substitutions` — recovers it.
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
//! cargo run -p lens-core --bin eval -- --enriched # three-way A/B/C on all corpora + gates
//! cargo run -p lens-core --bin eval -- --print-ids  # dump deterministic ids per doc
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lens_core::chunk::{Chunk, chunk_blocks_deterministic};
use lens_core::embedder::{
    DEFAULT_EMBED_MODEL_ID, Embedder, EmbeddingBackend, FastembedEmbedder, OllamaEmbedder,
};
use lens_core::enrichment::{
    CorefSub, apply_substitutions, compose_embedding_text, compose_prefix,
};
use lens_core::parse::{SourceKind, parse_blocks};
use lens_core::vector_store::{LanceVectorStore, VectorRow, VectorStore};
use lens_core::{EmbeddingModelSpec, LensEngine, LensError};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

/// Measured raw recall@5 on the main corpus (`queries.json`, 4 docs). Used as a
/// sanity floor; the A/B gate (`prefix-only >= raw`) guards the corpus under `--enriched`.
const MAIN_BASELINE_RECALL: f32 = 1.00;

/// Margin from baseline to set the raw pass floor (0.25 tolerates one regression out
/// of the top-5 on this ~12-chunk corpus, which drops recall by 0.20).
const MAIN_MARGIN: f32 = 0.25;

/// The raw pass floor on the main corpus.
const MAIN_RECALL_FLOOR: f32 = MAIN_BASELINE_RECALL - MAIN_MARGIN;

/// Measured raw recall@5 on the pronoun fixture. MUST be `< 1.00`: gold chunks name
/// their subject only in the title; raw body embedding misses an entity-named query.
/// Last measured 2026-06-26: 0.6667 (2/3 — pronoun body never names "Antikythera
/// mechanism"; prefix-only recovers to 1.0000). Re-measure and update if chunker/
/// embedder changes; it must stay `< 1.00`.
const RAW_RECALL_PRONOUN_FIXTURE: f32 = 0.6667;

/// Measured prefix-only recall@5 on the coref fixture. MUST be `< 1.00`: the gold
/// chunk's only link to the query is "The disc" — whose antecedent "Voyager Golden
/// Record" is not in the title/prefix. Prefix alone cannot supply it; only
/// `apply_substitutions` (prefix+coref) recovers it (2/2 = 1.0000).
/// Last measured 2026-06-26: 0.5000 (1/2). Re-measure if chunker/embedder changes;
/// must stay `< 1.00` AND less than the coref recall.
const PREFIX_ONLY_RECALL_COREF_FIXTURE: f32 = 0.5000;

const K: usize = 5;

/// A real notebook row is created at runtime because `embedding_index.notebook_id`
/// has a FK to `notebooks(id)`.
const EVAL_NOTEBOOK_TITLE: &str = "eval-corpus";

/// One canned query plus the gold chunk ids it should retrieve (a hit if ANY appears
/// in the top-5 results).
#[derive(Debug, Deserialize)]
struct Query {
    query: String,
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

    let model_id: String = {
        let args: Vec<String> = std::env::args().collect();
        args.windows(2)
            .find(|w| w[0] == "--model")
            .map(|w| w[1].clone())
            .unwrap_or_else(|| DEFAULT_EMBED_MODEL_ID.to_string())
    };
    let backend: EmbeddingBackend = {
        let args: Vec<String> = std::env::args().collect();
        let raw = args
            .windows(2)
            .find(|w| w[0] == "--backend")
            .map(|w| w[1].clone());
        EmbeddingBackend::from_opt_str(raw.as_deref())
    };
    // Rejects unknown ids so a typo'd `--model` fails loudly (mirrors
    // `set_notebook_embedding_model`). The legacy alias is accepted.
    let spec: &'static EmbeddingModelSpec = match lens_core::resolve_opt(&model_id) {
        Some(spec) => spec,
        None => {
            let known: Vec<String> = lens_core::REGISTRY
                .iter()
                .map(|s| {
                    let backends: Vec<&str> = s.backends.iter().map(|b| b.as_str()).collect();
                    format!("{} [{}]", s.id, backends.join(","))
                })
                .collect();
            eprintln!(
                "error: unknown --model {model_id:?}; known ids (alias nomic-embed-text → \
                 nomic-embed-text-v1.5): {}",
                known.join(", ")
            );
            std::process::exit(2);
        }
    };

    if std::env::args().any(|a| a == "--print-ids") {
        let dir = tempfile::tempdir().map_err(|e| LensError::Io(e.to_string()))?;
        let engine = LensEngine::init(dir.path()).await?;
        let tokenizer = load_tokenizer(&engine).await?;
        let main = load_corpus(&fixtures_dir, MAIN_DOCS)?;
        let pronoun = load_corpus(&fixtures_dir, PRONOUN_DOCS)?;
        let coref = load_corpus(&fixtures_dir, COREF_DOCS)?;
        print_ids(&main, &tokenizer, "main")?;
        print_ids(&pronoun, &tokenizer, "pronoun")?;
        print_ids(&coref, &tokenizer, "coref")?;
        return Ok(ExitCode::SUCCESS);
    }

    let dir = tempfile::tempdir().map_err(|e| LensError::Io(e.to_string()))?;
    let engine = LensEngine::init(dir.path()).await?;
    let data_dir = dir.path();
    let pool = engine.pool().await;

    println!(
        "Building embedder ({} via {}); a cold fastembed run downloads weights from HuggingFace…",
        spec.id,
        backend.as_str()
    );
    let embedder: Box<dyn Embedder> = match backend {
        EmbeddingBackend::Fastembed => Box::new(FastembedEmbedder::new_with_spec(data_dir, spec)?),
        EmbeddingBackend::Ollama => {
            let base_url = lens_core::system_check::ollama_base_url(&engine.config().await);
            Box::new(OllamaEmbedder::new(&base_url, spec)?)
        }
    };
    let embedder: &dyn Embedder = embedder.as_ref();
    let tokenizer = load_tokenizer(&engine).await?;

    println!(
        "Active embedding model : {} (dim={}, backend={})",
        spec.id,
        spec.dim,
        backend.as_str()
    );

    let main_docs = load_corpus(&fixtures_dir, MAIN_DOCS)?;
    let main_queries = load_queries(&fixtures_dir, "queries.json")?;

    if !enriched_mode {
        println!("\n############ MAIN CORPUS — RAW ############");
        let raw = measure(
            &engine,
            embedder,
            backend,
            &tokenizer,
            data_dir,
            &pool,
            &main_docs,
            &main_queries,
            EmbedMode::Raw,
            spec,
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
        println!("\n(run with --enriched for the three-way enrichment measurement + gates)");
        return Ok(ExitCode::SUCCESS);
    }

    println!("\n############ MAIN CORPUS (no-regression gate) ############");
    println!("\n--- raw ---");
    let main_raw = measure(
        &engine,
        embedder,
        backend,
        &tokenizer,
        data_dir,
        &pool,
        &main_docs,
        &main_queries,
        EmbedMode::Raw,
        spec,
    )
    .await?;
    report_recall("raw", main_raw.hits, main_raw.total);

    println!("\n--- prefix-only ---");
    let main_prefix = measure(
        &engine,
        embedder,
        backend,
        &tokenizer,
        data_dir,
        &pool,
        &main_docs,
        &main_queries,
        EmbedMode::PrefixOnly,
        spec,
    )
    .await?;
    report_recall("prefix-only", main_prefix.hits, main_prefix.total);

    let pronoun_docs = load_corpus(&fixtures_dir, PRONOUN_DOCS)?;
    let pronoun_queries = load_queries(&fixtures_dir, "pronoun_queries.json")?;

    println!("\n############ PRONOUN FIXTURE (prefix-lift gate) ############");
    println!("\n--- raw ---");
    let pron_raw = measure(
        &engine,
        embedder,
        backend,
        &tokenizer,
        data_dir,
        &pool,
        &pronoun_docs,
        &pronoun_queries,
        EmbedMode::Raw,
        spec,
    )
    .await?;
    report_recall("raw", pron_raw.hits, pron_raw.total);

    println!("\n--- prefix-only ---");
    let pron_prefix = measure(
        &engine,
        embedder,
        backend,
        &tokenizer,
        data_dir,
        &pool,
        &pronoun_docs,
        &pronoun_queries,
        EmbedMode::PrefixOnly,
        spec,
    )
    .await?;
    report_recall("prefix-only", pron_prefix.hits, pron_prefix.total);

    let coref_docs = load_corpus(&fixtures_dir, COREF_DOCS)?;
    let coref_queries = load_queries(&fixtures_dir, "coref_queries.json")?;

    println!("\n############ COREF FIXTURE (coref-lift gate) ############");
    println!("\n--- raw ---");
    let coref_raw = measure(
        &engine,
        embedder,
        backend,
        &tokenizer,
        data_dir,
        &pool,
        &coref_docs,
        &coref_queries,
        EmbedMode::Raw,
        spec,
    )
    .await?;
    report_recall("raw", coref_raw.hits, coref_raw.total);

    println!("\n--- prefix-only (NO coref) ---");
    let coref_prefix = measure(
        &engine,
        embedder,
        backend,
        &tokenizer,
        data_dir,
        &pool,
        &coref_docs,
        &coref_queries,
        EmbedMode::PrefixOnly,
        spec,
    )
    .await?;
    report_recall("prefix-only", coref_prefix.hits, coref_prefix.total);

    println!("\n--- prefix+coref (production apply_substitutions) ---");
    let coref_full = measure(
        &engine,
        embedder,
        backend,
        &tokenizer,
        data_dir,
        &pool,
        &coref_docs,
        &coref_queries,
        EmbedMode::PrefixCoref,
        spec,
    )
    .await?;
    report_recall("prefix+coref", coref_full.hits, coref_full.total);

    println!("\n=================== AC15 GATES ===================");
    println!(
        "main   : raw {:.4}  prefix-only {:.4}",
        main_raw.recall(),
        main_prefix.recall()
    );
    println!(
        "pronoun: raw {:.4}  prefix-only {:.4}  (recorded RAW_RECALL_PRONOUN_FIXTURE = {RAW_RECALL_PRONOUN_FIXTURE:.4})",
        pron_raw.recall(),
        pron_prefix.recall()
    );
    println!(
        "coref  : raw {:.4}  prefix-only {:.4}  prefix+coref {:.4}  (recorded PREFIX_ONLY_RECALL_COREF_FIXTURE = {PREFIX_ONLY_RECALL_COREF_FIXTURE:.4})",
        coref_raw.recall(),
        coref_prefix.recall(),
        coref_full.recall()
    );

    let mut failed = false;

    if main_prefix.recall() + 1e-6 < main_raw.recall() {
        eprintln!(
            "\nFAIL [no-regression]: main prefix-only recall@{K} {:.4} < raw {:.4}",
            main_prefix.recall(),
            main_raw.recall()
        );
        failed = true;
    } else {
        println!(
            "PASS [no-regression]: main prefix-only {:.4} ≥ raw {:.4}",
            main_prefix.recall(),
            main_raw.recall()
        );
    }

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

    if pron_prefix.recall() <= RAW_RECALL_PRONOUN_FIXTURE + 1e-6 {
        eprintln!(
            "\nFAIL [prefix-lift]: pronoun prefix-only recall@{K} {:.4} does NOT exceed RAW_RECALL_PRONOUN_FIXTURE {RAW_RECALL_PRONOUN_FIXTURE:.4}",
            pron_prefix.recall()
        );
        failed = true;
    } else {
        println!(
            "PASS [prefix-lift]: pronoun prefix-only {:.4} > RAW_RECALL_PRONOUN_FIXTURE {RAW_RECALL_PRONOUN_FIXTURE:.4}",
            pron_prefix.recall()
        );
    }

    if (coref_prefix.recall() - PREFIX_ONLY_RECALL_COREF_FIXTURE).abs() > 1e-3 {
        eprintln!(
            "\nFAIL [fixture drift]: measured coref prefix-only recall {:.4} != recorded PREFIX_ONLY_RECALL_COREF_FIXTURE {PREFIX_ONLY_RECALL_COREF_FIXTURE:.4}; re-measure and update the const",
            coref_prefix.recall()
        );
        failed = true;
    }
    if PREFIX_ONLY_RECALL_COREF_FIXTURE >= 1.0 {
        eprintln!(
            "\nFAIL [fixture invalid]: PREFIX_ONLY_RECALL_COREF_FIXTURE {PREFIX_ONLY_RECALL_COREF_FIXTURE:.4} is saturated (>= 1.00); the fixture cannot demonstrate a coref lift beyond the prefix"
        );
        failed = true;
    }

    // Coref must recover a gold chunk the prefix alone does NOT — the AC15 honesty gate.
    if coref_full.recall() <= PREFIX_ONLY_RECALL_COREF_FIXTURE + 1e-6 {
        eprintln!(
            "\nFAIL [coref-lift]: coref prefix+coref recall@{K} {:.4} does NOT exceed prefix-only {PREFIX_ONLY_RECALL_COREF_FIXTURE:.4}; coref adds no value beyond the prefix on this fixture",
            coref_full.recall()
        );
        failed = true;
    } else {
        println!(
            "PASS [coref-lift]: coref prefix+coref {:.4} > prefix-only {PREFIX_ONLY_RECALL_COREF_FIXTURE:.4}",
            coref_full.recall()
        );
    }

    if failed {
        return Ok(ExitCode::FAILURE);
    }
    println!("\nPASS: all AC15 gates green.");
    Ok(ExitCode::SUCCESS)
}

/// Which text each chunk contributes to its embedding.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EmbedMode {
    /// Embed raw `chunk.text`.
    Raw,
    /// Embed `prefix + chunk.text` (context prefix, no coref) + synthesized summary node.
    PrefixOnly,
    /// Embed `prefix + apply_substitutions(chunk.text)` — the full production re-embed input.
    PrefixCoref,
}

impl EmbedMode {
    fn is_enriched(self) -> bool {
        matches!(self, EmbedMode::PrefixOnly | EmbedMode::PrefixCoref)
    }
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

/// Ingests `docs` under `mode` into a fresh per-pass notebook + Lance table, runs
/// `queries`, prints the result table, and returns recall@5.
#[allow(clippy::too_many_arguments)]
async fn measure(
    engine: &LensEngine,
    embedder: &dyn Embedder,
    backend: EmbeddingBackend,
    tokenizer: &Tokenizer,
    data_dir: &Path,
    pool: &sqlx::SqlitePool,
    docs: &[Doc],
    queries: &[Query],
    mode: EmbedMode,
    spec: &'static EmbeddingModelSpec,
) -> Result<Recall, LensError> {
    let store = LanceVectorStore::new(data_dir, pool.clone());
    let notebook = engine
        .create_notebook(EVAL_NOTEBOOK_TITLE, None, None)
        .await?;
    let notebook_id = notebook.id.as_str();
    let coord = lens_core::vector_store::Coordinate::new(
        notebook_id.to_string(),
        backend,
        spec.id.to_string(),
        spec.dim,
    );

    // chunk_id -> snippet for the printed result table.
    let mut chunk_text: HashMap<String, String> = HashMap::new();

    for doc in docs {
        let blocks = parse_blocks(&doc.text, SourceKind::Markdown);
        let chunks = chunk_blocks_deterministic(&doc.text, &blocks, tokenizer)?;

        let doc_summary = derive_doc_summary(&chunks);
        let allowed_antecedents = fixture_entities(&doc.name);

        let mut rows: Vec<VectorRow> = Vec::with_capacity(chunks.len() + 1);
        let mut texts: Vec<String> = Vec::with_capacity(chunks.len() + 1);
        let mut ids: Vec<String> = Vec::with_capacity(chunks.len() + 1);
        let mut levels: Vec<i32> = Vec::with_capacity(chunks.len() + 1);

        for chunk in &chunks {
            chunk_text.insert(chunk.id.clone(), snippet(&chunk.text));
            let embed_text = match mode {
                EmbedMode::Raw => chunk.text.clone(),
                EmbedMode::PrefixOnly => {
                    let prefix = compose_prefix(&doc_summary, &chunk.section_path);
                    compose_embedding_text(&prefix, &chunk.text, Some(tokenizer))
                }
                EmbedMode::PrefixCoref => {
                    let subs = fixture_coref_subs(&doc.name, &chunk.text);
                    let resolved_body =
                        apply_substitutions(&chunk.text, &subs, &allowed_antecedents);
                    let prefix = compose_prefix(&doc_summary, &chunk.section_path);
                    compose_embedding_text(&prefix, &resolved_body, Some(tokenizer))
                }
            };
            ids.push(chunk.id.clone());
            levels.push(chunk.level);
            texts.push(embed_text);
        }

        // Synthesized doc-summary RAPTOR node (AC6): enriched modes only.
        if mode.is_enriched() && !doc_summary.trim().is_empty() {
            let sid = summary_node_id(&doc.name, &doc_summary);
            chunk_text.insert(sid.clone(), snippet(&doc_summary));
            ids.push(sid);
            levels.push(2);
            texts.push(doc_summary.clone());
        }

        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        // `block_in_place` so the Ollama backend's `Handle::block_on` doesn't panic on
        // this async thread; a no-op for the CPU-bound fastembed backend.
        let vectors = tokio::task::block_in_place(|| embedder.embed_documents(&text_refs))?;
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
        store.add(&coord, rows).await?;
        println!("ingested {} ({n} rows)", doc.name);
    }

    let mut hits = 0usize;
    println!("=== Retrieval (k = {K}) ===");
    for q in queries {
        let qvec = tokio::task::block_in_place(|| embedder.embed_query(&q.query))?;
        let results = store.search(&coord, &qvec, K).await?;
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

/// Derives the deterministic doc summary stand-in for the LLM structural map: the H1
/// title + lead sentence of the first parent body. Carries the named entity the pronoun
/// fixture bodies omit. For the coref fixture the title/lead deliberately omit the query
/// entity so only coref can supply it.
fn derive_doc_summary(chunks: &[Chunk]) -> String {
    let Some(first_parent) = chunks.iter().find(|c| c.level == 0) else {
        return String::new();
    };
    let title = first_parent
        .section_path
        .split('>')
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let lead = first_parent
        .text
        .split(['.', '\n'])
        .map(str::trim)
        .find(|s| !s.is_empty());

    match (title, lead) {
        // Lead already names the subject → omit the title to avoid redundancy.
        (Some(t), Some(l)) if l.starts_with(t) => format!("{l}."),
        (Some(t), Some(l)) => format!("{t}. {l}."),
        (Some(t), None) => t.to_string(),
        (None, Some(l)) => format!("{l}."),
        (None, None) => String::new(),
    }
}

/// Content-derived id for the synthesized doc-summary node (mirrors
/// `chunk_blocks_deterministic`: level=2, empty section_path, ordinal 0).
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

/// Doc entities for `allowed_antecedents` — the same allow-list the structural map's
/// `entities` field supplies in production. Authored per fixture.
fn fixture_entities(doc_name: &str) -> Vec<String> {
    COREF_MAP
        .iter()
        .find(|(name, _)| *name == doc_name)
        .map(|(_, edits)| {
            let mut ents: Vec<String> = edits
                .iter()
                .map(|(_, antecedent)| antecedent.to_string())
                .collect();
            ents.sort();
            ents.dedup();
            ents
        })
        .unwrap_or_default()
}

/// Builds production-shape [`CorefSub`]s for `chunk_text` from the authored edits.
/// Byte positions from `str::find` are converted to codepoint indices to match the
/// real `CorefSub` contract that [`apply_substitutions`] converts to bytes internally.
fn fixture_coref_subs(doc_name: &str, chunk_text: &str) -> Vec<CorefSub> {
    let Some((_, edits)) = COREF_MAP.iter().find(|(name, _)| *name == doc_name) else {
        return Vec::new();
    };
    let mut subs: Vec<CorefSub> = Vec::new();
    for (mention, antecedent) in *edits {
        let mut from = 0usize;
        while let Some(rel) = chunk_text[from..].find(mention) {
            let byte_start = from + rel;
            let byte_end = byte_start + mention.len();
            let char_start = chunk_text[..byte_start].chars().count();
            let char_end = chunk_text[..byte_end].chars().count();
            subs.push(CorefSub {
                mention: (*mention).to_string(),
                char_start,
                char_end,
                antecedent: (*antecedent).to_string(),
            });
            from = byte_end;
        }
    }
    subs
}

/// Authored coref map: `(mention, antecedent)` edits per fixture-doc. `apply_substitutions`
/// validates each against the real chunk text before applying; a mention absent from a
/// chunk is a no-op. `golden_record`: resolving "The disc" → "Voyager Golden Record" is
/// the only thing that links the gold chunk to the query — the prefix title "A Message
/// to the Stars" never names the record. Other entries make the pass a realistic
/// whole-doc resolution, not a single rigged edit.
const COREF_MAP: &[(&str, &[(&str, &str)])] = &[(
    "golden_record",
    &[
        ("The disc", "The Voyager Golden Record"),
        ("the disc", "the Voyager Golden Record"),
        ("The artifact", "The Voyager Golden Record"),
        ("the artifact", "the Voyager Golden Record"),
        ("the plate", "the Voyager Golden Record"),
    ],
)];

/// Main corpus doc stems (saturated 4-doc set; `queries.json`).
const MAIN_DOCS: &[&str] = &["espresso", "photosynthesis", "rust_ownership", "tides"];

/// Pronoun fixture corpus. Main docs are distractors so the gold chunk must genuinely
/// out-rank unrelated chunks — without them, every chunk trivially lands in the top-5.
const PRONOUN_DOCS: &[&str] = &[
    "pronoun_context",
    "tide_prediction",
    "espresso",
    "photosynthesis",
    "rust_ownership",
    "tides",
];

/// Coref fixture corpus. `golden_record` is the coref-dependent doc; the rest are
/// distractors that also describe things "selected and arranged" — so the gold chunk's
/// raw body (same phrase, no record name) cannot win without coref resolution.
const COREF_DOCS: &[&str] = &[
    "golden_record",
    "arecibo",
    "time_capsule",
    "westinghouse",
    "tide_prediction",
    "espresso",
    "photosynthesis",
    "rust_ownership",
    "tides",
];

struct Doc {
    name: String,
    text: String,
}

fn eval_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("eval")
}

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

fn load_queries(dir: &Path, file: &str) -> Result<Vec<Query>, LensError> {
    let path = dir.join(file);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
    let queries: Vec<Query> = serde_json::from_str(&raw)?;
    Ok(queries)
}

async fn load_tokenizer(engine: &LensEngine) -> Result<Tokenizer, LensError> {
    let data_dir = PathBuf::from(engine.config().await.paths.data_dir);
    lens_core::resolve_nomic_tokenizer(&data_dir).await
}

/// Prints each chunk's deterministic id + section path + snippet plus the summary-node
/// id, so gold sets can be authored against stable ids. For coref-mapped docs also
/// prints each chunk's resolved body for eyeballing the `(mention → antecedent)` edits.
fn print_ids(docs: &[Doc], tokenizer: &Tokenizer, corpus: &str) -> Result<(), LensError> {
    println!("\n######## corpus: {corpus} ########");
    for doc in docs {
        println!("\n## {} ##", doc.name);
        let blocks = parse_blocks(&doc.text, SourceKind::Markdown);
        let chunks = chunk_blocks_deterministic(&doc.text, &blocks, tokenizer)?;
        let allowed = fixture_entities(&doc.name);
        for c in &chunks {
            print_chunk_line(c);
            if !allowed.is_empty() {
                let subs = fixture_coref_subs(&doc.name, &c.text);
                if !subs.is_empty() {
                    let resolved = apply_substitutions(&c.text, &subs, &allowed);
                    println!("      coref-resolved -> {}", snippet(&resolved));
                }
            }
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

/// First ~70 chars of `text`, single-lined.
fn snippet(text: &str) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > 70 {
        let truncated: String = one_line.chars().take(67).collect();
        format!("{truncated}…")
    } else {
        one_line
    }
}

fn short_id(id: &str) -> String {
    id.chars().take(12).collect()
}
