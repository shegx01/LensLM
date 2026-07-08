//! The background enrichment worker task and its job message (Step 4 — TEXT columns only).
//!
//! Walks each source through: dedup + existence re-check (AC13); composite cache-key
//! short-circuit (AC9); size-gate + kind-awareness; graceful degrade (AC10); structural-map
//! map-reduce (AC4) with budget + circuit-break (AC11); `embedding_text` composition (AC5).
//! Step 4 writes TEXT columns ONLY — `enriched` is set by Step 5's re-embed flip.

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

/// A unit of background enrichment work. The worker re-loads the live source row
/// from SQLite on dequeue so a purge mid-flight is handled by re-checking existence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrichmentJob {
    pub source_id: String,
}

/// Spawns the dedicated background enrichment worker task.
///
/// Concurrency = 1 (lock #3): drains the receiver sequentially. Per-session budget
/// counters (AC11) are created once at spawn and accumulate across all jobs. The
/// worker holds NO `ingest_lock` during `process_job`; the flip-only lock window
/// arrives in Step 5.
pub fn spawn_worker(engine: LensEngine, mut rx: mpsc::Receiver<EnrichmentJob>) {
    tokio::spawn(async move {
        tracing::debug!("enrichment worker started");
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

/// Returns true if at least one level-0 parent chunk is prose (not code/table/html).
fn source_is_prose(chunks: &[EnrichmentChunk]) -> bool {
    chunks
        .iter()
        .filter(|c| c.level == 0)
        .any(|c| !super::is_nonprose_block(c.block_type.as_deref()))
}

async fn process_job(
    engine: &LensEngine,
    job: &EnrichmentJob,
    session_budget: &SessionBudget,
) -> Result<(), crate::LensError> {
    let pool = engine.pool().await;
    let repo = NotebookRepo::new(&pool);

    let enrichment_cfg = engine.config().await.enrichment;
    if !enrichment_cfg.enabled {
        tracing::debug!(
            source_id = %job.source_id,
            "enrichment disabled in config; leaving source on raw vectors"
        );
        return Ok(());
    }

    // AC13(b): re-check existence — a purge mid-flight cascades the row away.
    let source = match repo.get_source(&job.source_id).await? {
        Some(s) => s,
        None => {
            tracing::debug!(source_id = %job.source_id, "enrichment: source gone, dropping job");
            return Ok(());
        }
    };

    // AC13(a): skip a source already mid-enrichment. `enriched` is not skipped
    // unconditionally — the cache-key check below decides whether to re-run.
    let current = EnrichmentStatus::from_db(source.enrichment_status.as_deref())?;
    if matches!(current, EnrichmentStatus::Enriching) {
        tracing::debug!(source_id = %job.source_id, "enrichment: already enriching, skipping");
        return Ok(());
    }

    let coref = enrichment_cfg.coref_strategy;
    let provider = engine.llm_provider().await;

    // AC9: composite cache key — content_hash + model + prompt_version + coref.
    // A matching persisted key short-circuits the LLM entirely.
    let content_hash = source.content_hash.clone().unwrap_or_default();
    let cache_parts = provider.as_ref().map(|p| CacheKeyParts {
        content_hash: content_hash.clone(),
        llm_model_id: p.model_id().to_string(),
        prompt_version: super::meta::ENRICHMENT_PROMPT_VERSION,
        coref_strategy: coref.as_str().to_string(),
        relations_strategy: enrichment_cfg.relations_strategy.as_str().to_string(),
    });
    let cache_key = cache_parts.as_ref().map(|p| p.compute());

    if let (Some(key), Some(meta_json)) = (&cache_key, &source.enrichment_meta)
        && let Ok(meta) = serde_json::from_str::<EnrichmentMeta>(meta_json)
        && meta.cache_key == *key
        && current == EnrichmentStatus::Enriched
    {
        tracing::debug!(
            source_id = %job.source_id,
            "enrichment: cache-key hit, skipping LLM (already enriched)"
        );
        return Ok(());
    }

    // Preflight (issue #90): fail-fast with a human-readable reason when the model is
    // known-bad BEFORE marking `Enriching` — distinct from a temporarily-unreachable
    // provider, which degrades to `pending` via the downstream `reachable()` check.
    let ollama_base = crate::system_check::ollama_base_url(&engine.config().await);
    if let PreflightOutcome::Fail(reason) =
        preflight_enrichment_provider(provider.as_ref(), &ollama_base).await
    {
        let meta = EnrichmentMeta {
            failure_reason: Some(reason.clone()),
            ..EnrichmentMeta::default()
        };
        persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &meta).await?;
        tracing::warn!(
            source_id = %job.source_id,
            reason = %reason,
            "enrichment: preflight failed; source marked failed before enriching"
        );
        return Ok(());
    }

    repo.update_enrichment_status(&job.source_id, EnrichmentStatus::Enriching)
        .await?;

    #[cfg(feature = "test-util")]
    engine.enrichment_job_gate().await;

    // AC13(b): re-check existence after the status write.
    if repo.get_source(&job.source_id).await?.is_none() {
        tracing::debug!(source_id = %job.source_id, "enrichment: source purged mid-job, dropping");
        return Ok(());
    }

    let chunks = repo.list_chunks_for_enrichment(&job.source_id).await?;
    if chunks.is_empty() {
        tracing::debug!(source_id = %job.source_id, "enrichment: no chunks, degrading to pending");
        repo.update_enrichment_status(&job.source_id, EnrichmentStatus::Pending)
            .await?;
        return Ok(());
    }

    let tokenizer = engine.tokenizer().await.ok();
    let total_tokens: usize = chunks
        .iter()
        .filter(|c| c.level == 0)
        .map(|c| count_tokens(&c.text, tokenizer.as_deref()))
        .sum();
    let below_size_gate = total_tokens < ENRICHMENT_SIZE_GATE_TOKENS;
    let is_prose = source_is_prose(&chunks);

    // AC10: no reachable provider ⇒ degrade to `pending`.
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

    // Per-task model overrides: a set `coref_model` / `map_model` builds a sibling
    // provider pinned to that model while reusing the base genai client.
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

    let cache_key = cache_key.unwrap_or_default();
    let mut budget = Budget::with_caps(
        session_budget.clone(),
        ENRICHMENT_MAX_TOKENS_PER_JOB,
        engine.enrichment_max_calls_per_job(),
    );

    // Non-prose or size-gated: no structural map, but still compose a context-prefix
    // `embedding_text` (Decision B). Zero LLM calls. Terminal status `skipped`.
    if !is_prose || below_size_gate {
        write_prefix_only(&repo, &chunks, "", tokenizer.as_deref()).await?;
        let meta = EnrichmentMeta {
            cache_key,
            map_quality: MAP_QUALITY_SKIPPED.to_string(),
            budget_exceeded: false,
            tokens_spent: budget.job_tokens(),
            calls_made: budget.job_calls(),
            failure_reason: None,
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

    let parent_texts: Vec<String> = chunks
        .iter()
        .filter(|c| c.level == 0)
        .map(|c| c.text.clone())
        .collect();

    let (map_json, doc_summary, map_entities, map_definitions, map_dates, map_quality) =
        match build_structural_map(map_provider.as_ref(), &mut budget, &parent_texts).await {
            Ok(MapOutcome::Ok(map)) => {
                let summary = map.summary.clone();
                let entities = map.entities.clone();
                let definitions = map.definitions.clone();
                let dates = map.dates.clone();
                let json = serde_json::to_string(&map)?;
                (
                    Some(json),
                    summary,
                    entities,
                    definitions,
                    dates,
                    MAP_QUALITY_OK,
                )
            }
            // AC4: persistent malformed output ⇒ degrade to context-prefix-only.
            Ok(MapOutcome::Fallback) => (
                None,
                String::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                MAP_QUALITY_FALLBACK,
            ),
            // AC11: budget circuit-break ⇒ `failed` + `budget_exceeded`.
            Err(MapError::BudgetExceeded) => {
                let meta = EnrichmentMeta {
                    cache_key,
                    map_quality: String::new(),
                    budget_exceeded: true,
                    tokens_spent: budget.job_tokens(),
                    calls_made: budget.job_calls(),
                    failure_reason: None,
                };
                persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &meta).await?;
                tracing::warn!(
                    source_id = %job.source_id,
                    calls = budget.job_calls(),
                    "enrichment: budget exceeded, circuit-broke to failed"
                );
                return Ok(());
            }
            // AC13(d): LLM error ⇒ `failed`, raw vectors untouched.
            Err(MapError::Llm(e)) => {
                let meta = EnrichmentMeta {
                    cache_key,
                    map_quality: String::new(),
                    budget_exceeded: false,
                    tokens_spent: budget.job_tokens(),
                    calls_made: budget.job_calls(),
                    failure_reason: None,
                };
                persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &meta).await?;
                tracing::warn!(source_id = %job.source_id, "enrichment: LLM error, failed: {e}");
                return Ok(());
            }
        };

    // Coref (AC5): shares the same Budget as the map (AC11 covers both passes).
    // A budget breach fails the source; any other miss degrades to empty subs (coref
    // is additive — never fails the source on its own).
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
            // AC11: budget breach during coref ⇒ `failed` + `budget_exceeded`.
            Err(MapError::BudgetExceeded) => {
                let meta = EnrichmentMeta {
                    cache_key,
                    map_quality: String::new(),
                    budget_exceeded: true,
                    tokens_spent: budget.job_tokens(),
                    calls_made: budget.job_calls(),
                    failure_reason: None,
                };
                persist_meta(&repo, &job.source_id, EnrichmentStatus::Failed, &meta).await?;
                tracing::warn!(
                    source_id = %job.source_id,
                    calls = budget.job_calls(),
                    "enrichment: coref budget exceeded, circuit-broke to failed"
                );
                return Ok(());
            }
            // Transport error: degrade to empty subs (coref is additive, not a source failure).
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

    // Compose `embedding_text` for every chunk (AC5); attach the map JSON to the
    // first level-0 parent row; write TEXT columns in one txn.
    let mut updates: Vec<ChunkEnrichmentUpdate> = Vec::with_capacity(chunks.len());
    let mut map_attached = map_json.is_none();
    for (i, chunk) in chunks.iter().enumerate() {
        let prefix = compose_prefix(&doc_summary, &chunk.section_path);
        // `chunks.text` is never mutated — only `embedding_text` carries coref-resolved body.
        let resolved_body = match coref_subs.get(&i) {
            Some(subs) if !subs.is_empty() => apply_substitutions(&chunk.text, subs, &map_entities),
            _ => chunk.text.clone(),
        };
        let embedding_text = compose_embedding_text(&prefix, &resolved_body, tokenizer.as_deref());
        // Attach the per-doc map to the first parent row only.
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
    // M13: build the entity graph from in-memory enrichment outputs (ZERO new LLM
    // calls) and persist it in the SAME transaction as the chunk-enrichment writes.
    let graph_rows = crate::graph::build_entity_graph_rows(
        &source.notebook_id,
        &job.source_id,
        &chunks,
        &map_entities,
        &map_definitions,
        &map_dates,
        &coref_subs,
        &mut || uuid::Uuid::now_v7().to_string(),
        &chrono::Utc::now().to_rfc3339(),
        crate::enrichment::meta::CO_OCCURRENCE_MAX_ENTITIES,
    );
    if graph_rows.dropped_cooccurrence > 0 {
        tracing::debug!(
            source_id = %job.source_id,
            dropped = graph_rows.dropped_cooccurrence,
            "entity graph: co-occurrence entities dropped over per-chunk cap"
        );
    }
    repo.write_enrichment_and_graph(&updates, &graph_rows)
        .await?;

    // Status stays `enriching` (Step-4→Step-5 handoff): text columns written,
    // re-embed flip (below) advances it to `enriched`.
    let meta = EnrichmentMeta {
        cache_key: cache_key.clone(),
        map_quality: map_quality.to_string(),
        budget_exceeded: false,
        tokens_spent: budget.job_tokens(),
        calls_made: budget.job_calls(),
        failure_reason: None,
    };
    // AC13(b): re-check before the terminal write (AC13 b).
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

    // Step 5 (AC6–AC8): re-embed into a private building table, flip active under
    // `ingest_lock`, mark `enriched`. Any failure before the flip leaves raw vectors
    // untouched (crash-safe) and degrades to `failed`.
    if let Err(e) =
        super::reembed::reembed_and_flip(engine, &job.source_id, &source.notebook_id, &doc_summary)
            .await
    {
        // A purge mid-flip is not a failure.
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
            failure_reason: None,
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

/// Serializes `meta` and writes it alongside `status` in one call (DRY across all
/// budget-exceeded / llm-error / skipped / success / re-embed-fallback paths).
async fn persist_meta(
    repo: &NotebookRepo<'_>,
    source_id: &str,
    status: EnrichmentStatus,
    meta: &EnrichmentMeta,
) -> Result<(), crate::LensError> {
    repo.update_enrichment_status_and_meta(source_id, status, &serde_json::to_string(meta)?)
        .await
}

/// Writes a context-prefix-only `embedding_text` to every chunk (skipped / non-prose /
/// size-gated path). No structural map; zero LLM calls.
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

/// Outcome of the worker's pre-enrichment preflight (issue #90).
///
/// `Proceed` = model is usable or the situation is a normal AC10 degrade (no provider /
/// unreachable runtime). `Fail(reason)` = model is known-bad (runtime reachable but
/// model absent); the worker persists `Failed` with the human-readable reason.
#[derive(Debug, PartialEq, Eq)]
enum PreflightOutcome {
    Proceed,
    Fail(String),
}

/// Catches a known-bad local Ollama model BEFORE marking `Enriching` (issue #90).
///
/// `None` provider ⇒ `Proceed` (normal first-launch state; AC10 degrade handles it —
/// architect-ratified deviation from the plan's literal "None ⇒ Failed"). Unreachable
/// Ollama ⇒ `Proceed` (runtime flap; downstream `reachable()` degrades to `Pending`).
/// Reachable Ollama with model absent from `/api/tags` ⇒ `Fail`. Cloud providers ⇒
/// `Proceed` (no `/api/tags` equivalent; onboarding gate + `reachable()` cover them).
async fn preflight_enrichment_provider(
    provider: Option<&std::sync::Arc<dyn crate::llm::LlmProvider>>,
    ollama_base_url: &str,
) -> PreflightOutcome {
    let Some(provider) = provider else {
        return PreflightOutcome::Proceed;
    };

    // Only a local Ollama `GenaiProvider` gets a tags-membership check.
    let is_ollama = provider
        .as_any()
        .downcast_ref::<crate::llm::GenaiProvider>()
        .is_some_and(|p| p.is_ollama());
    if !is_ollama {
        return PreflightOutcome::Proceed;
    }

    // An unreachable Ollama also returns empty `/api/tags` — gate on reachability
    // first so a runtime flap is not wrongly failed as a missing model.
    if !provider.reachable().await {
        return PreflightOutcome::Proceed;
    }

    let model = provider.model_id().to_string();
    let installed = crate::list_ollama_models(ollama_base_url).await;
    if installed.iter().any(|m| m == &model) {
        PreflightOutcome::Proceed
    } else {
        PreflightOutcome::Fail(crate::system_check::ollama_model_missing_reason(&model))
    }
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

    /// `coref_model` override routes the coref pass to the override model; the map
    /// pass (no override) keeps the routing-default model.
    #[tokio::test]
    async fn coref_pass_uses_coref_model_override_when_set() {
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

        let base = provider_from_config(&config, false).expect("base provider");
        assert_eq!(base.model_id(), "qwen2.5-instruct");

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
        assert_eq!(map_provider.model_id(), "qwen2.5-instruct");
        assert_eq!(coref_provider.model_id(), "qwen2.5-coder");

        let mut budget = Budget::new(SessionBudget::new());
        let chunks: Vec<(usize, &str)> = vec![(0, "Ada built the engine. She wrote notes.")];
        let entities = vec!["Ada".to_string()];
        let subs = resolve_coref_batch(coref_provider.as_ref(), &mut budget, &chunks, &entities)
            .await
            .expect("coref pass dispatches against the override");
        assert!(subs.is_empty() || subs.values().all(|v| v.is_empty()));

        let base_genai = base.as_any().downcast_ref::<GenaiProvider>();
        let coref_genai = coref_provider.as_any().downcast_ref::<GenaiProvider>();
        assert!(base_genai.is_some() && coref_genai.is_some());

        drop(instruct_server);
        drop(coder_server);
    }

    use super::{PreflightOutcome, preflight_enrichment_provider};
    use crate::llm::LlmProvider;
    use std::sync::Arc;

    /// Non-Ollama mock provider: `as_any` never downcasts to `GenaiProvider` → preflight Proceeds.
    struct MockCloudProvider;
    #[async_trait::async_trait]
    impl LlmProvider for MockCloudProvider {
        fn model_id(&self) -> &str {
            "mock-cloud-model"
        }
        async fn reachable(&self) -> bool {
            true
        }
        async fn generate(
            &self,
            _req: &crate::llm::LlmRequest,
        ) -> Result<crate::llm::LlmResponse, crate::LensError> {
            Ok(crate::llm::LlmResponse {
                text: "ok".to_string(),
                tokens_used: 1,
            })
        }
    }

    #[tokio::test]
    async fn worker_preflight_no_provider() {
        let outcome = preflight_enrichment_provider(None, "http://127.0.0.1:1").await;
        assert!(matches!(outcome, PreflightOutcome::Proceed));
    }

    #[tokio::test]
    async fn worker_preflight_local_unreachable_proceeds() {
        let config = AppConfig {
            models: vec![ModelConfig {
                provider: "ollama".to_string(),
                base_url: "http://127.0.0.1:1".to_string(),
                model: "llama3.2:3b".to_string(),
                ..ModelConfig::default()
            }],
            enrichment: EnrichmentConfig {
                enabled: true,
                routing: crate::llm::LlmRouting::LocalFirst,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        let provider = provider_from_config(&config, false).expect("local provider");
        let outcome = preflight_enrichment_provider(Some(&provider), "http://127.0.0.1:1").await;
        assert!(matches!(outcome, PreflightOutcome::Proceed));
    }

    #[tokio::test]
    async fn worker_preflight_local_model_missing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.4.1"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{ "name": "some-other-model:latest" }]
            })))
            .mount(&server)
            .await;

        let config = AppConfig {
            models: vec![ModelConfig {
                provider: "ollama".to_string(),
                base_url: server.uri(),
                model: "llama3.2:3b".to_string(),
                ..ModelConfig::default()
            }],
            enrichment: EnrichmentConfig {
                enabled: true,
                routing: crate::llm::LlmRouting::LocalFirst,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        let provider = provider_from_config(&config, false).expect("local provider");
        let outcome = preflight_enrichment_provider(Some(&provider), &server.uri()).await;
        match outcome {
            PreflightOutcome::Fail(reason) => {
                assert!(reason.contains("llama3.2:3b"), "got {reason}");
                assert!(reason.contains("ollama pull"), "got {reason}");
            }
            PreflightOutcome::Proceed => panic!("expected Fail for a missing local model"),
        }
    }

    #[tokio::test]
    async fn worker_preflight_local_model_present() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.4.1"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{ "name": "llama3.2:3b" }]
            })))
            .mount(&server)
            .await;

        let config = AppConfig {
            models: vec![ModelConfig {
                provider: "ollama".to_string(),
                base_url: server.uri(),
                model: "llama3.2:3b".to_string(),
                ..ModelConfig::default()
            }],
            enrichment: EnrichmentConfig {
                enabled: true,
                routing: crate::llm::LlmRouting::LocalFirst,
                ..EnrichmentConfig::default()
            },
            ..AppConfig::default()
        };
        let provider = provider_from_config(&config, false).expect("local provider");
        let outcome = preflight_enrichment_provider(Some(&provider), &server.uri()).await;
        assert!(matches!(outcome, PreflightOutcome::Proceed));
    }

    #[tokio::test]
    async fn worker_preflight_cloud_provider_proceeds() {
        let provider: Arc<dyn LlmProvider> = Arc::new(MockCloudProvider);
        let outcome = preflight_enrichment_provider(Some(&provider), "http://127.0.0.1:1").await;
        assert!(matches!(outcome, PreflightOutcome::Proceed));
    }
}
