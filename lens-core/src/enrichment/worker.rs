//! The background enrichment worker task and its job message.
//!
//! Step 4 (LLM pipeline — TEXT columns only). The worker consumes the engine's
//! enrichment `mpsc` receiver and walks each source through:
//!
//! 1. dedup + existence re-check (AC13 a/b);
//! 2. composite cache-key short-circuit (AC9) — a matching key ⇒ ZERO LLM calls;
//! 3. size-gate (Decision C) + kind-awareness (sub-decision b) — non-prose /
//!    too-small ⇒ `skipped` with a context-prefix `embedding_text` still applied;
//! 4. graceful degrade (AC10) — no reachable provider ⇒ stays `pending`;
//! 5. the structural-map map-reduce (AC4) with budget + circuit-break (AC11);
//! 6. contextual `embedding_text` composition (AC5);
//! 7. write the structural map + `embedding_text` to the TEXT columns
//!    (Decision D) and leave the source in the `enriching` HANDOFF state.
//!
//! **Handoff to Step 5 (architect-ratified):** Step 4 writes TEXT columns ONLY —
//! it NEVER embeds. `enriched` is set only after Step 5's re-embed flip; a source
//! whose text columns are written but not yet re-embedded stays `enriching`
//! (mid-pipeline). Crash-recovery resets `enriching → pending`; the cache-key
//! short-circuit then makes the re-run cheap (zero LLM calls) before Step 5
//! re-embeds. Terminal Step-4 states: `skipped` (non-prose), `failed`
//! (budget/LLM death), and `enriched` (only via the cache-key short-circuit of an
//! already-fully-enriched source).

use std::collections::HashMap;

use tokio::sync::mpsc;

use crate::LensEngine;
use crate::notebooks::{ChunkEnrichmentUpdate, EnrichmentChunk, EnrichmentStatus, NotebookRepo};

use super::coref::{CorefSub, apply_substitutions, resolve_coref_batch};
use super::embedding_text::{CorefStrategy, compose_embedding_text, compose_prefix, count_tokens};
use super::map::{MapError, MapOutcome, build_structural_map};
use super::meta::{
    Budget, CacheKeyParts, ENRICHMENT_MAX_TOKENS_PER_JOB, ENRICHMENT_SIZE_GATE_TOKENS,
    EnrichmentMeta, MAP_QUALITY_FALLBACK, MAP_QUALITY_OK, MAP_QUALITY_SKIPPED, SessionBudget,
};

/// A unit of background enrichment work: enrich the source identified by
/// `source_id`. Cheap to clone/send; the worker re-loads the live source row from
/// SQLite when it dequeues the job (so a purge mid-flight is handled by re-checking
/// existence rather than carrying a stale snapshot).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrichmentJob {
    /// The `sources.id` to enrich.
    pub source_id: String,
}

/// Spawns the dedicated background enrichment worker task.
///
/// Concurrency = 1 for local providers (lock #3): a single task drains the
/// receiver sequentially. The task owns the `Receiver`; dropping every `Sender`
/// clone (Tauri runtime teardown) closes the channel → `recv()` returns `None` →
/// the task exits cleanly (no explicit shutdown token in Phase 3).
///
/// The per-SESSION budget counters ([`SessionBudget`], AC11) are created HERE —
/// once, when the worker spawns — and accumulate across every job for the
/// worker's lifetime (reset-on-start). The worker holds NO `ingest_lock` permit
/// during [`process_job`] (lock #3); a concurrent `purge_source` is therefore
/// never blocked by an in-flight job (the flip-only lock window arrives in Step 5).
pub fn spawn_worker(engine: LensEngine, mut rx: mpsc::Receiver<EnrichmentJob>) {
    tokio::spawn(async move {
        tracing::debug!("enrichment worker started");
        // Per-session budget: reset once at spawn, shared across all jobs (AC11).
        let session_budget = SessionBudget::new();
        while let Some(job) = rx.recv().await {
            if let Err(e) = process_job(&engine, &job, &session_budget).await {
                tracing::warn!(
                    source_id = %job.source_id,
                    "enrichment job failed: {e}"
                );
            }
        }
        tracing::debug!("enrichment worker stopped (channel closed)");
    });
}

/// Block types whose chunks are NON-prose and skip the structural map
/// (sub-decision b). They still receive a context-prefix `embedding_text`.
fn is_nonprose_block(block_type: Option<&str>) -> bool {
    matches!(block_type, Some("code") | Some("table") | Some("html"))
}

/// Decides whether a source is prose-eligible for the structural map: at least
/// one level-0 parent must be prose (not code/table/html). A source whose every
/// parent is non-prose is `skipped` (sub-decision b).
fn source_is_prose(chunks: &[EnrichmentChunk]) -> bool {
    chunks
        .iter()
        .filter(|c| c.level == 0)
        .any(|c| !is_nonprose_block(c.block_type.as_deref()))
}

/// Processes a single enrichment job (Step 4 — TEXT columns only).
async fn process_job(
    engine: &LensEngine,
    job: &EnrichmentJob,
    session_budget: &SessionBudget,
) -> Result<(), crate::LensError> {
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    // ── Step-6 wiring: source the enrichment policy from the REAL config
    // (`AppConfig.enrichment`) instead of hardcoded defaults. When enrichment is
    // disabled the worker does nothing — the source stays on raw vectors
    // (`none`/`pending`), the same graceful no-op as having no reachable provider
    // (AC10/AC14). The coref strategy is the typed config value (no longer pinned
    // to `LlmInline`).
    let enrichment_cfg = engine.config().await.enrichment;
    if !enrichment_cfg.enabled {
        tracing::debug!(
            source_id = %job.source_id,
            "enrichment disabled in config; leaving source on raw vectors"
        );
        return Ok(());
    }

    // AC13(b): re-check the source exists. A purge mid-flight cascades the row
    // away; drop the job silently rather than erroring on a missing source.
    let source = match repo.get_source(&job.source_id).await? {
        Some(s) => s,
        None => {
            tracing::debug!(source_id = %job.source_id, "enrichment: source gone, dropping job");
            return Ok(());
        }
    };

    // AC13(a) dedup: skip a source already being enriched (concurrency=1 makes a
    // racing `enriching` impossible, but the guard is cheap + future-proof).
    // `enriched` is NOT skipped here unconditionally — the cache-key check below
    // decides whether an already-enriched source needs a re-run.
    let current = EnrichmentStatus::from_db(source.enrichment_status.as_deref())?;
    if matches!(current, EnrichmentStatus::Enriching) {
        tracing::debug!(source_id = %job.source_id, "enrichment: already enriching, skipping");
        return Ok(());
    }

    // ── Resolve the provider + coref/cloud config from `AppConfig.enrichment`
    // (Step 6). The coref strategy is the typed config value; the cloud-consent
    // gate lives in the factory that installed `llm_provider` (a cloud provider
    // only installs when `enrichment.cloud_consent` is true, AC11), so a `None`
    // provider here ⇒ graceful degrade.
    let coref = enrichment_cfg.coref_strategy;
    let provider = engine.llm_provider().await;

    // ── AC9 composite cache key. Computed from the live source + provider +
    // prompt version + coref. A matching persisted key short-circuits the LLM.
    let content_hash = source.content_hash.clone().unwrap_or_default();
    let cache_parts = provider.as_ref().map(|p| CacheKeyParts {
        content_hash: content_hash.clone(),
        llm_model_id: p.model_id().to_string(),
        prompt_version: super::meta::ENRICHMENT_PROMPT_VERSION,
        coref_strategy: coref.as_str().to_string(),
    });
    let cache_key = cache_parts.as_ref().map(|p| p.compute());

    if let (Some(key), Some(meta_json)) = (&cache_key, &source.enrichment_meta)
        && let Ok(meta) = serde_json::from_str::<EnrichmentMeta>(meta_json)
        && meta.cache_key == *key
        && current == EnrichmentStatus::Enriched
    {
        // AC9: identical config + already enriched ⇒ ZERO LLM calls.
        tracing::debug!(
            source_id = %job.source_id,
            "enrichment: cache-key hit, skipping LLM (already enriched)"
        );
        return Ok(());
    }

    // ── Mark enriching (the Step-4→Step-5 handoff/in-flight state).
    repo.update_enrichment_status(&job.source_id, EnrichmentStatus::Enriching)
        .await?;

    // Optional AC3 test gate: pin a job "in flight" to assert the worker holds no
    // ingest_lock permit during the body (compiled out of production builds).
    #[cfg(feature = "test-util")]
    engine.enrichment_job_gate().await;

    // Re-check existence after the (possibly long) gate / status write (AC13 b).
    if repo.get_source(&job.source_id).await?.is_none() {
        tracing::debug!(source_id = %job.source_id, "enrichment: source purged mid-job, dropping");
        return Ok(());
    }

    // ── Read the chunks. No chunks ⇒ nothing to enrich; degrade to `pending`
    // (raw vectors serve; a future re-ingest/rescan re-drives it).
    let chunks = repo.list_chunks_for_enrichment(&job.source_id).await?;
    if chunks.is_empty() {
        tracing::debug!(source_id = %job.source_id, "enrichment: no chunks, degrading to pending");
        repo.update_enrichment_status(&job.source_id, EnrichmentStatus::Pending)
            .await?;
        return Ok(());
    }

    // ── Size-gate (Decision C): skip docs below the token threshold. Token count
    // uses the tokenizer when available (production) else a word approximation.
    let tokenizer = engine.tokenizer().await.ok();
    let total_tokens: usize = chunks
        .iter()
        .filter(|c| c.level == 0)
        .map(|c| count_tokens(&c.text, tokenizer.as_deref()))
        .sum();
    let below_size_gate = total_tokens < ENRICHMENT_SIZE_GATE_TOKENS;

    // ── Kind-awareness (sub-decision b): a non-prose source skips the map.
    let is_prose = source_is_prose(&chunks);

    // ── Graceful degrade (AC10): no reachable provider ⇒ stays `pending`.
    let provider = match provider {
        Some(p) if p.reachable().await => p,
        _ => {
            tracing::debug!(
                source_id = %job.source_id,
                "enrichment: no reachable provider, degrading to pending"
            );
            repo.update_enrichment_status(&job.source_id, EnrichmentStatus::Pending)
                .await?;
            return Ok(());
        }
    };

    // ── Per-task model providers (M4 Phase 3, Stage 3). When `coref_model` /
    // `map_model` is set, build a sibling provider pinned to that exact
    // `(provider, model)` — reusing the base provider's ONE genai client (only the
    // pinned target differs). An unset override (or one failing its consent/catalog
    // gate) falls back to the base provider. Example: a user picks Qwen2.5-Coder for
    // coref while enrichment defaults to Qwen2.5-Instruct — coref runs on the coder.
    let cfg_models = engine.config().await.models;
    let map_provider = crate::llm::task_provider_from_config(
        &provider,
        enrichment_cfg.map_model.as_ref(),
        &cfg_models,
        enrichment_cfg.cloud_consent,
    );
    let coref_provider = crate::llm::task_provider_from_config(
        &provider,
        enrichment_cfg.coref_model.as_ref(),
        &cfg_models,
        enrichment_cfg.cloud_consent,
    );

    // Cache parts are guaranteed Some here (provider is Some).
    let cache_key = cache_key.unwrap_or_default();
    // Per-job budget over the shared session counters. The per-job call ceiling is
    // the production default unless a test tightened it (AC11 budget seam).
    let mut budget = Budget::with_caps(
        session_budget.clone(),
        ENRICHMENT_MAX_TOKENS_PER_JOB,
        engine.enrichment_max_calls_per_job(),
    );

    // ── SKIPPED path (non-prose OR size-gated): no structural map, but STILL
    // compose a lightweight context-prefix `embedding_text` (Decision B). Zero
    // LLM calls. Terminal status `skipped`.
    if !is_prose || below_size_gate {
        write_prefix_only(&repo, &chunks, "", tokenizer.as_deref()).await?;
        let meta = EnrichmentMeta {
            cache_key,
            map_quality: MAP_QUALITY_SKIPPED.to_string(),
            budget_exceeded: false,
            tokens_spent: budget.job_tokens(),
            calls_made: budget.job_calls(),
        };
        persist_meta(&repo, &job.source_id, EnrichmentStatus::Skipped, &meta).await?;
        tracing::debug!(
            source_id = %job.source_id,
            prose = is_prose,
            below_size_gate,
            "enrichment: skipped structural map; prefix-only embedding_text applied"
        );
        return Ok(());
    }

    // ── PROSE path: structural map over level-0 parents (AC4) + budget (AC11).
    let parent_texts: Vec<String> = chunks
        .iter()
        .filter(|c| c.level == 0)
        .map(|c| c.text.clone())
        .collect();

    let (map_json, doc_summary, map_entities, map_quality) =
        match build_structural_map(map_provider.as_ref(), &mut budget, &parent_texts).await {
            Ok(MapOutcome::Ok(map)) => {
                let summary = map.summary.clone();
                let entities = map.entities.clone();
                let json = serde_json::to_string(&map)?;
                (Some(json), summary, entities, MAP_QUALITY_OK)
            }
            // AC4: persistent malformed output ⇒ degrade to context-prefix-only,
            // source NOT failed.
            Ok(MapOutcome::Fallback) => (None, String::new(), Vec::new(), MAP_QUALITY_FALLBACK),
            // AC11: a budget circuit-break ⇒ `failed` + `budget_exceeded`, never
            // silent. Raw vectors are untouched (no text columns written).
            Err(MapError::BudgetExceeded) => {
                let meta = EnrichmentMeta {
                    cache_key,
                    map_quality: String::new(),
                    budget_exceeded: true,
                    tokens_spent: budget.job_tokens(),
                    calls_made: budget.job_calls(),
                };
                persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &meta).await?;
                tracing::warn!(
                    source_id = %job.source_id,
                    calls = budget.job_calls(),
                    "enrichment: budget exceeded, circuit-broke to failed"
                );
                return Ok(());
            }
            // AC13(d): LLM death/429 ⇒ `failed`, raw vectors untouched, eligible
            // for re-enqueue (the queue-rebuild scans `failed`).
            Err(MapError::Llm(e)) => {
                let meta = EnrichmentMeta {
                    cache_key,
                    map_quality: String::new(),
                    budget_exceeded: false,
                    tokens_spent: budget.job_tokens(),
                    calls_made: budget.job_calls(),
                };
                persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &meta).await?;
                tracing::warn!(source_id = %job.source_id, "enrichment: LLM error, failed: {e}");
                return Ok(());
            }
        };

    // ── Coref resolution (AC5, `LlmInline`). When the map is `ok` (entities are
    // available) and the strategy is `LlmInline`, run real LLM-driven coref over
    // the chunks being enriched, sharing the SAME `Budget` instance as the map so
    // the per-job circuit-break (AC11) covers BOTH passes. The LLM only IDENTIFIES
    // substitutions; `apply_substitutions` applies them deterministically to each
    // chunk's BODY (the canonical `chunks.text` is never touched). A coref budget
    // breach fails the source exactly like the map; any other coref miss
    // (malformed/transport) DEGRADES to empty subs so the body falls back to raw —
    // coref is strictly additive and never fails the source on its own.
    let coref_subs: HashMap<usize, Vec<CorefSub>> = if coref == CorefStrategy::LlmInline
        && map_quality == MAP_QUALITY_OK
        && !map_entities.is_empty()
    {
        let coref_chunks: Vec<(usize, &str)> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.text.as_str()))
            .collect();
        match resolve_coref_batch(
            coref_provider.as_ref(),
            &mut budget,
            &coref_chunks,
            &map_entities,
        )
        .await
        {
            Ok(subs) => subs,
            // AC11: a budget circuit-break during coref ⇒ `failed` + `budget_exceeded`
            // (the shared budget makes this identical to a map breach).
            Err(MapError::BudgetExceeded) => {
                let meta = EnrichmentMeta {
                    cache_key,
                    map_quality: String::new(),
                    budget_exceeded: true,
                    tokens_spent: budget.job_tokens(),
                    calls_made: budget.job_calls(),
                };
                persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &meta).await?;
                tracing::warn!(
                    source_id = %job.source_id,
                    calls = budget.job_calls(),
                    "enrichment: coref budget exceeded, circuit-broke to failed"
                );
                return Ok(());
            }
            // A coref transport error is NOT a source failure — coref is additive.
            // Degrade to empty subs (every body falls back to raw).
            Err(MapError::Llm(e)) => {
                tracing::debug!(
                    source_id = %job.source_id,
                    "enrichment: coref LLM error, degrading to raw bodies: {e}"
                );
                HashMap::new()
            }
        }
    } else {
        HashMap::new()
    };

    // ── Compose `embedding_text` for every chunk (AC5) + attach the map JSON to
    // the FIRST level-0 parent row, then write the TEXT columns in one txn.
    let mut updates: Vec<ChunkEnrichmentUpdate> = Vec::with_capacity(chunks.len());
    let mut map_attached = map_json.is_none();
    for (i, chunk) in chunks.iter().enumerate() {
        let prefix = compose_prefix(&doc_summary, &chunk.section_path);
        // The body sourced into `embedding_text` is the canonical text with this
        // chunk's surviving coref substitutions applied (RIGHT-TO-LEFT, validated).
        // `chunks.text` itself stays byte-identical — only `embedding_text` carries
        // the resolved body. On no subs (None strategy / degrade) this is the raw
        // body unchanged.
        let resolved_body = match coref_subs.get(&i) {
            Some(subs) if !subs.is_empty() => apply_substitutions(&chunk.text, subs, &map_entities),
            _ => chunk.text.clone(),
        };
        let embedding_text = compose_embedding_text(&prefix, &resolved_body, tokenizer.as_deref());
        // Attach the per-doc map to the first parent row only (AC4).
        let enrichment_json = if !map_attached && chunk.parent_id.is_none() {
            map_attached = true;
            map_json.clone()
        } else {
            None
        };
        updates.push(ChunkEnrichmentUpdate {
            chunk_id: chunk.id.clone(),
            embedding_text,
            enrichment_json,
        });
    }
    repo.write_chunk_enrichment(&updates).await?;

    // ── Persist meta. Status STAYS `enriching` — the Step-4→Step-5 handoff: the
    // text columns are written but the re-embed flip below is what advances it to
    // `enriched`. (No new enum variant; `enriching` is the mid-pipeline state.)
    let meta = EnrichmentMeta {
        cache_key: cache_key.clone(),
        map_quality: map_quality.to_string(),
        budget_exceeded: false,
        tokens_spent: budget.job_tokens(),
        calls_made: budget.job_calls(),
    };
    // Re-check existence before the terminal-ish write (the body may have spanned
    // a purge): a vanished source means the job is moot (AC13 b).
    if repo.get_source(&job.source_id).await?.is_none() {
        tracing::debug!(source_id = %job.source_id, "enrichment: source purged before meta write, dropping");
        return Ok(());
    }
    persist_meta(&repo, &job.source_id, EnrichmentStatus::Enriching, &meta).await?;
    tracing::debug!(
        source_id = %job.source_id,
        map_quality,
        calls = budget.job_calls(),
        "enrichment: Step-4 text columns written; entering Step-5 re-embed flip"
    );

    // ── STEP 5 — re-embed new-table-flip (AC6/AC7/AC8). Embed
    // `COALESCE(embedding_text, text)` for every chunk + the doc-summary RAPTOR
    // node into a PRIVATE gen-suffixed building table (lock-free), then flip it
    // active under the FLIP-ONLY `ingest_lock` window, then mark `enriched`. On
    // ANY failure BEFORE the flip the raw `active` vectors are untouched (the flip
    // is crash-safe by construction) — degrade to `failed`, eligible for re-enqueue.
    if let Err(e) =
        super::reembed::reembed_and_flip(engine, &job.source_id, &source.notebook_id, &doc_summary)
            .await
    {
        // A purge mid-flip cascades the source away; that is not a failure.
        if repo.get_source(&job.source_id).await?.is_none() {
            tracing::debug!(source_id = %job.source_id, "enrichment: source purged during re-embed, dropping");
            return Ok(());
        }
        let fail_meta = EnrichmentMeta {
            cache_key,
            map_quality: map_quality.to_string(),
            budget_exceeded: false,
            tokens_spent: budget.job_tokens(),
            calls_made: budget.job_calls(),
        };
        persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &fail_meta).await?;
        tracing::warn!(source_id = %job.source_id, "enrichment: re-embed flip failed: {e}");
        return Ok(());
    }

    tracing::debug!(
        source_id = %job.source_id,
        map_quality,
        "enrichment: re-embed flip complete (enriched)"
    );
    Ok(())
}

/// Serializes `meta` and writes it alongside the terminal-ish `status` for the
/// source, in one place (DRY across the worker's budget-exceeded / llm-error /
/// skipped / success / re-embed-fallback paths — the construct→serialize→update
/// triple was copy-pasted across ~6 sites).
///
/// Preserves the underlying no-op-on-missing-source semantics: a purge mid-flight
/// makes [`NotebookRepo::update_enrichment_status_and_meta`] a no-op rather than an
/// error, so callers that already re-checked existence keep their behavior.
async fn persist_meta(
    repo: &NotebookRepo<'_>,
    source_id: &str,
    status: EnrichmentStatus,
    meta: &EnrichmentMeta,
) -> Result<(), crate::LensError> {
    repo.update_enrichment_status_and_meta(source_id, status, &serde_json::to_string(meta)?)
        .await
}

/// Writes a context-prefix-only `embedding_text` to every chunk (the SKIPPED /
/// non-prose / size-gated path — Decision B). No structural map; zero LLM calls.
async fn write_prefix_only(
    repo: &NotebookRepo<'_>,
    chunks: &[EnrichmentChunk],
    doc_summary: &str,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<(), crate::LensError> {
    let updates: Vec<ChunkEnrichmentUpdate> = chunks
        .iter()
        .map(|chunk| {
            let prefix = compose_prefix(doc_summary, &chunk.section_path);
            let embedding_text = compose_embedding_text(&prefix, &chunk.text, tokenizer);
            ChunkEnrichmentUpdate {
                chunk_id: chunk.id.clone(),
                embedding_text,
                enrichment_json: None,
            }
        })
        .collect();
    repo.write_chunk_enrichment(&updates).await
}

#[cfg(test)]
mod tests {
    use crate::config::{AppConfig, EnrichmentConfig, ModelConfig, TaskModel};
    use crate::enrichment::coref::resolve_coref_batch;
    use crate::enrichment::meta::{Budget, SessionBudget};
    use crate::llm::{GenaiProvider, provider_from_config, task_provider_from_config};

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// An Ollama `/api/chat` body carrying a model field + the given assistant text.
    fn ollama_body(model: &str, content: &str) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "message": { "role": "assistant", "content": content },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5
        })
    }

    /// A valid coref response body (empty subs — the parse succeeds, no mutation).
    fn coref_ok() -> &'static str {
        r#"{"results":[{"id":0,"subs":[]}]}"#
    }

    /// The worker's per-task resolution, in isolation: a `coref_model` override
    /// must make the COREF pass dispatch against the override model, while the MAP
    /// pass (no override) keeps the routing-default model. This is the Stage-3
    /// product ask: coref on a coder model, enrichment default on a generalist.
    #[tokio::test]
    async fn coref_pass_uses_coref_model_override_when_set() {
        // Base/default model is served by `instruct_server`; the coref override
        // model is served by `coder_server`. The coder server is the ONLY one
        // mounted to answer a coref call, and it asserts EXACTLY one hit — so the
        // test fails if coref dispatched against the base instead of the override.
        let instruct_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(ollama_body("qwen2.5-instruct", coref_ok())),
            )
            .expect(0) // the base must NOT be hit by the coref pass
            .mount(&instruct_server)
            .await;

        let coder_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(ollama_body("qwen2.5-coder", coref_ok())),
            )
            .expect(1) // the coref pass dispatches here exactly once
            .mount(&coder_server)
            .await;

        // Config: routing default → the instruct entry; coref_model → the coder
        // entry (a distinct local Ollama endpoint).
        let config = AppConfig {
            models: vec![
                ModelConfig {
                    provider: "ollama".to_string(),
                    base_url: instruct_server.uri(),
                    model: "qwen2.5-instruct".to_string(),
                    ..ModelConfig::default()
                },
                ModelConfig {
                    provider: "ollama".to_string(),
                    base_url: coder_server.uri(),
                    model: "qwen2.5-coder".to_string(),
                    ..ModelConfig::default()
                },
            ],
            enrichment: EnrichmentConfig {
                enabled: true,
                coref_model: Some(TaskModel {
                    provider: "ollama".to_string(),
                    model: "qwen2.5-coder".to_string(),
                }),
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };

        // The base provider the worker resolves from routing (the instruct entry).
        let base = provider_from_config(&config, false).expect("base provider");
        assert_eq!(base.model_id(), "qwen2.5-instruct");

        // The worker's per-task resolution (mirrors `process_job`).
        let map_provider = task_provider_from_config(
            &base,
            config.enrichment.map_model.as_ref(),
            &config.models,
            config.enrichment.cloud_consent,
        );
        let coref_provider = task_provider_from_config(
            &base,
            config.enrichment.coref_model.as_ref(),
            &config.models,
            config.enrichment.cloud_consent,
        );
        // map has no override ⇒ default; coref is pinned to the coder.
        assert_eq!(map_provider.model_id(), "qwen2.5-instruct");
        assert_eq!(coref_provider.model_id(), "qwen2.5-coder");

        // Run the actual coref pass against the resolved coref provider: it MUST hit
        // the coder server (override), proven by the coder server's expect(1) and the
        // instruct server's expect(0), verified on drop.
        let mut budget = Budget::new(SessionBudget::new());
        let chunks: Vec<(usize, &str)> = vec![(0, "Ada built the engine. She wrote notes.")];
        let entities = vec!["Ada".to_string()];
        let subs = resolve_coref_batch(coref_provider.as_ref(), &mut budget, &chunks, &entities)
            .await
            .expect("coref pass dispatches against the override");
        // Empty subs (the mock returned an empty set) — the point is WHICH server
        // was hit, asserted by the mock expectations on drop.
        assert!(subs.is_empty() || subs.values().all(|v| v.is_empty()));

        // Sanity: the shared genai client was reused (siblings share the Arc-backed
        // client), so no second genai Client was constructed for the override.
        let base_genai = base.as_any().downcast_ref::<GenaiProvider>();
        let coref_genai = coref_provider.as_any().downcast_ref::<GenaiProvider>();
        assert!(base_genai.is_some() && coref_genai.is_some());

        drop(instruct_server);
        drop(coder_server);
    }
}
