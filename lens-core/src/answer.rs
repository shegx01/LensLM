//! Grounded-answer orchestrator (issue #173): the "rag route". Turns a notebook
//! question into a streamed, citation-bearing answer, all headless — NO `tauri`/UI
//! types. [`answer_stream`] is a pure free fn over an owned [`AnswerCtx`]; the
//! fallible ctx-gathering lives in `LensEngine::answer_notebook` (lib.rs).
//!
//! Items are `Result<AnswerEvent, LensError>`, mirroring the engine's existing
//! streaming idiom ([`LlmProvider::generate_stream`] yields
//! `Result<StreamChunk, LensError>`). Legal sequence on a successful run:
//! `Ok(Stage(Retrieving))` → [`Ok(Stage(Thinking))` iff ≥1 `ThinkingDelta`, then
//! `Ok(ThinkingDelta)*`] → `Ok(Stage(Answering))` → `Ok(TextDelta)*` → exactly one
//! `Ok(Citations)` (may be empty) → `Ok(Done)` (always last, including the
//! empty-context path). On cancel the stream ends with NO further items. On a
//! stage failure (embed/retrieve/title/generate/mid-stream) it yields a single
//! terminal `Err(LensError)` then ends — NO `Citations`/`Done` (a truncated answer
//! is never cited); the command maps that `Err` onto `StreamEvent::Failed`.

use std::collections::HashMap;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;

use crate::citation::{
    CITATION_PROMPT_INSTRUCTION, Citation, extract_citations, hydrate_locators, load_chunk_locators,
};
use crate::config::{ModelConfig, RetrievalConfig, TierThresholds};
use crate::embedder::Embedder;
use crate::error::LensError;
use crate::graph::NotebookGraph;
use crate::llm::{LlmProvider, LlmRequest, StreamChunk};
use crate::retrieval::Reranker;
use crate::retrieval::router::{ContextUnit, RESERVED_OUTPUT, tiered_search};
use crate::vector_store::{Coordinate, VectorStore};

/// Low, near-deterministic sampling for grounded answers.
const ANSWER_TEMPERATURE: f32 = 0.1;

/// Fixed grounded refusal emitted when retrieval finds no supporting sources. The
/// LLM is never called on this path (cannot hallucinate with nothing to cite).
const NO_SOURCES_MSG: &str =
    "I couldn't find anything in this notebook's selected sources to answer that.";

/// Coarse pipeline phase carried by [`AnswerEvent::Stage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnswerStage {
    /// Embedding the query + running tiered retrieval.
    Retrieving,
    /// The model is emitting reasoning (`ThinkingDelta`) tokens.
    Thinking,
    /// The model is emitting the final answer (`TextDelta`) tokens.
    Answering,
}

/// One event streamed by [`answer_stream`]. See the module header for the legal
/// ordering. `Citations` carries the citations extracted over the accumulated
/// `TextDelta` text (empty when the answer cited nothing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnswerEvent {
    Stage(AnswerStage),
    ThinkingDelta(String),
    TextDelta(String),
    Citations(Vec<Citation>),
    Done { tokens_used: u32 },
}

/// Owned, `Send` bundle the pure [`answer_stream`] needs. Every field is owned so
/// the generated `stream!` future is `Send + 'static` — nothing borrows `&self`.
/// `tiered_search`'s by-reference params are satisfied via `&*arc`/`&coord`/
/// `.as_deref()` at the call site.
pub struct AnswerCtx {
    pub provider: Arc<dyn LlmProvider>,
    pub store: Arc<dyn VectorStore>,
    pub embedder: Arc<dyn Embedder>,
    /// Owned — `Reranker::new` is cheap and never inits a model unless enabled.
    pub reranker: Reranker,
    pub graph: Option<Arc<NotebookGraph>>,
    pub pool: SqlitePool,
    pub coord: Coordinate,
    pub model: ModelConfig,
    pub retrieval: RetrievalConfig,
    pub thresholds: TierThresholds,
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub question: String,
}

/// Builds the `(system, user)` prompt from the retrieved units. Units are numbered
/// by **Vec slice position** (`[i+1]`), matching #23a's positional citation
/// contract — NEVER keyed off `order_index`. `title` falls back to the raw
/// `source_id` when absent so assembly can never fail.
fn build_grounded_prompt(
    units: &[ContextUnit],
    titles: &HashMap<String, String>,
    question: &str,
) -> (String, String) {
    let mut blocks = String::new();
    for (i, u) in units.iter().enumerate() {
        let title = titles.get(&u.source_id).unwrap_or(&u.source_id);
        blocks.push_str(&format!("[{}] ({}): {}\n", i + 1, title, u.text));
    }
    let system = format!(
        "You answer strictly from the numbered source units below. If they do not \
         support an answer, say so.\n\n{CITATION_PROMPT_INSTRUCTION}\n\nSource units:\n{blocks}"
    );
    (system, question.to_string())
}

/// Batched `SELECT id, title` over the distinct `source_id`s in `units`, into a
/// `HashMap`. Chunks the `IN (…)` list under the SQLite bind limit. Absent titles
/// are simply missing (the prompt builder falls back to `source_id`).
async fn load_source_titles(
    pool: &SqlitePool,
    units: &[ContextUnit],
) -> Result<HashMap<String, String>, LensError> {
    // SQLite default bind-variable limit is 999; stay well under it.
    const TITLE_LOAD_BATCH: usize = 500;

    let mut ids: Vec<&str> = units.iter().map(|u| u.source_id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();

    let mut out = HashMap::new();
    if ids.is_empty() {
        return Ok(out);
    }
    for batch in ids.chunks(TITLE_LOAD_BATCH) {
        let placeholders = std::iter::repeat_n("?", batch.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!("SELECT id, title FROM sources WHERE id IN ({placeholders})");
        let mut q = sqlx::query_as::<_, (String, String)>(&sql);
        for id in batch {
            q = q.bind(*id);
        }
        for (id, title) in q.fetch_all(pool).await? {
            out.insert(id, title);
        }
    }
    Ok(out)
}

/// The grounded-answer stream. Pure over the owned [`AnswerCtx`] so the returned
/// future is `Send + 'static`. See the module header for the event contract.
///
/// Errors surface as a terminal [`AnswerEvent`]-less end: the caller (the command
/// glue) maps the absence-of-`Done` plus the yielded terminal marker onto
/// `StreamEvent::Failed`. Within the engine, an error mid-stream stops emission
/// after the last successful delta with NO `Citations`/`Done` (a truncated answer
/// is never cited).
pub fn answer_stream(
    ctx: AnswerCtx,
    cancel: CancellationToken,
) -> impl Stream<Item = Result<AnswerEvent, LensError>> + Send {
    async_stream::stream! {
        yield Ok(AnswerEvent::Stage(AnswerStage::Retrieving));

        if cancel.is_cancelled() {
            return;
        }

        // Embed the query fully OFF the async runtime, returning the owned Vec
        // before any await — the fastembed `std::sync::Mutex` guard must never
        // straddle an await (Send/deadlock hazard, R1). Terminal-error on failure,
        // never `.unwrap()`.
        let embedder = ctx.embedder.clone();
        let question = ctx.question.clone();
        let qvec = match tokio::task::spawn_blocking(move || embedder.embed_query(&question)).await {
            Ok(Ok(v)) => v,
            Ok(Err(err)) => {
                yield Err(err);
                return;
            }
            Err(join) => {
                yield Err(LensError::from(join));
                return;
            }
        };

        if cancel.is_cancelled() {
            return;
        }

        let out = match tiered_search(
            &ctx.pool,
            &*ctx.store,
            &ctx.reranker,
            ctx.graph.as_deref(),
            &ctx.coord,
            &ctx.question,
            &qvec,
            &ctx.model,
            ctx.retrieval.answer_candidate_pool,
            &ctx.retrieval,
            Some(ctx.retrieval.graph_retrieval_enabled),
            &ctx.thresholds,
            ctx.tokenizer.as_deref(),
        )
        .await
        {
            Ok(o) => o,
            Err(err) => {
                yield Err(err);
                return;
            }
        };

        if cancel.is_cancelled() {
            return;
        }

        // Empty selected+live corpus → deterministic grounded refusal, no LLM call.
        if out.units.is_empty() {
            yield Ok(AnswerEvent::Stage(AnswerStage::Answering));
            yield Ok(AnswerEvent::TextDelta(NO_SOURCES_MSG.to_string()));
            yield Ok(AnswerEvent::Citations(Vec::new()));
            yield Ok(AnswerEvent::Done { tokens_used: 0 });
            return;
        }

        let titles = match load_source_titles(&ctx.pool, &out.units).await {
            Ok(t) => t,
            Err(err) => {
                yield Err(err);
                return;
            }
        };

        if cancel.is_cancelled() {
            return;
        }

        let (system, prompt) = build_grounded_prompt(&out.units, &titles, &ctx.question);
        let req = LlmRequest {
            system: Some(system),
            prompt,
            max_tokens: RESERVED_OUTPUT,
            temperature: ANSWER_TEMPERATURE,
            json: false,
            thinking: false,
            reasoning_effort: None,
        };

        let mut stream = match ctx.provider.generate_stream(&req).await {
            Ok(s) => s,
            Err(err) => {
                yield Err(err);
                return;
            }
        };

        let mut answer_text = String::new();
        let mut thinking_started = false;
        let mut answering_started = false;
        let mut tokens_used: u32 = 0;

        while let Some(item) = stream.next().await {
            if cancel.is_cancelled() {
                return;
            }
            match item {
                Ok(StreamChunk::ThinkingDelta(s)) => {
                    if !thinking_started {
                        thinking_started = true;
                        yield Ok(AnswerEvent::Stage(AnswerStage::Thinking));
                    }
                    yield Ok(AnswerEvent::ThinkingDelta(s));
                }
                Ok(StreamChunk::TextDelta(s)) => {
                    if !answering_started {
                        answering_started = true;
                        yield Ok(AnswerEvent::Stage(AnswerStage::Answering));
                    }
                    answer_text.push_str(&s);
                    yield Ok(AnswerEvent::TextDelta(s));
                }
                Ok(StreamChunk::Done { tokens_used: t }) => {
                    tokens_used = t;
                }
                Err(err) => {
                    // Mid-stream item error: stop relaying, do NOT cite a truncated
                    // answer (no Citations/Done), surface terminally (OQ-3).
                    yield Err(err);
                    return;
                }
            }
        }

        if cancel.is_cancelled() {
            return;
        }

        // Extract citations over the accumulated answer text only (never thinking),
        // hydrate their locators engine-side, then emit one Citations + Done.
        let mut cites = extract_citations(&answer_text, &out.units);
        let chunk_ids = distinct_chunk_ids(&cites);
        match load_chunk_locators(&ctx.pool, &chunk_ids).await {
            Ok(rows) => hydrate_locators(&mut cites, &rows),
            Err(err) => {
                yield Err(err);
                return;
            }
        }
        yield Ok(AnswerEvent::Citations(cites));
        yield Ok(AnswerEvent::Done { tokens_used });
    }
}

/// Distinct `chunk_id`s across every citation's locators, for the locator-hydration
/// batch load.
fn distinct_chunk_ids(citations: &[Citation]) -> Vec<String> {
    let mut ids: Vec<String> = citations
        .iter()
        .flat_map(|c| c.locators.iter().map(|l| l.chunk_id.clone()))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::HitSource;
    use crate::retrieval::router::Provenance;

    fn unit(source_id: &str, chunk_id: &str, text: &str, order_index: usize) -> ContextUnit {
        ContextUnit {
            text: text.to_string(),
            source_id: source_id.to_string(),
            chunk_id: chunk_id.to_string(),
            parent_id: None,
            locator: None,
            order_index,
            provenance: Provenance {
                source: HitSource::Dense,
                graph_confidence: None,
            },
        }
    }

    #[test]
    fn prompt_numbers_units_one_based_by_position() {
        let units = vec![unit("sA", "c1", "alpha", 0), unit("sB", "c2", "beta", 1)];
        let titles = HashMap::new();
        let (system, _user) = build_grounded_prompt(&units, &titles, "q");
        assert!(system.contains("[1] (sA): alpha"));
        assert!(system.contains("[2] (sB): beta"));
    }

    #[test]
    fn prompt_numbers_by_position_ignoring_scrambled_order_index() {
        // order_index is deliberately reversed; numbering must follow Vec position.
        let units = vec![unit("sA", "c1", "alpha", 9), unit("sB", "c2", "beta", 3)];
        let titles = HashMap::new();
        let (system, _user) = build_grounded_prompt(&units, &titles, "q");
        assert!(system.contains("[1] (sA): alpha"));
        assert!(system.contains("[2] (sB): beta"));
    }

    #[test]
    fn prompt_title_falls_back_to_source_id() {
        let units = vec![unit("src-xyz", "c1", "body", 0)];
        let mut titles = HashMap::new();
        titles.insert("src-xyz".to_string(), "My Title".to_string());
        let (with_title, _) = build_grounded_prompt(&units, &titles, "q");
        assert!(with_title.contains("[1] (My Title): body"));

        let (fallback, _) = build_grounded_prompt(&units, &HashMap::new(), "q");
        assert!(fallback.contains("[1] (src-xyz): body"));
    }

    #[test]
    fn prompt_embeds_instruction_and_question() {
        let units = vec![unit("sA", "c1", "alpha", 0)];
        let (system, user) = build_grounded_prompt(&units, &HashMap::new(), "what is X?");
        assert!(system.contains(CITATION_PROMPT_INSTRUCTION));
        assert_eq!(user, "what is X?");
    }

    #[test]
    fn distinct_chunk_ids_dedups() {
        let cites = vec![Citation {
            source_id: "sA".into(),
            ordinal: 1,
            locators: vec![
                crate::citation::Locator {
                    chunk_id: "c1".into(),
                    anchor: None,
                    section_path: None,
                    page: None,
                    char_start: None,
                    char_end: None,
                },
                crate::citation::Locator {
                    chunk_id: "c1".into(),
                    anchor: None,
                    section_path: None,
                    page: None,
                    char_start: None,
                    char_end: None,
                },
            ],
        }];
        assert_eq!(distinct_chunk_ids(&cites), vec!["c1".to_string()]);
    }
}
