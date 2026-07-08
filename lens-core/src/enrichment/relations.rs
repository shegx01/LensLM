//! LLM relation-extraction pass (#154): typed semantic edges between entities
//! already in the structural map, anchored to a `chunk_id` with confidence.
//!
//! Pure-logic (sampling / prompt / parse / validate) is DB- and network-free so it
//! is unit-testable against the mock [`LlmProvider`](crate::llm::LlmProvider). The
//! LLM shape mirrors the coref pass: an object `{"relations":[...]}`, strict-parsed
//! with reprompt via [`run_llm_with_retries`]. Endpoints are resolved to existing
//! node ids by case-insensitive name; unmatched / low-confidence / uncited triples
//! and unknown predicates are dropped (additive — a mid-pass failure keeps triples
//! gathered so far).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::graph::{Relation, ResolvedNode};
use crate::llm::LlmProvider;
use crate::notebooks::EnrichmentChunk;

use super::is_nonprose_block;
use super::map::{MapError, run_llm_with_retries};
use super::meta::{
    Budget, RELATIONS_BATCH_SIZE, RELATIONS_MAX_TOKENS, RELATIONS_MIN_CONFIDENCE,
    RELATIONS_SAMPLE_CAP, RELATIONS_SAMPLE_FRACTION, extract_json_object,
};

const RELATIONS_SYSTEM_PROMPT: &str = "You extract semantic relations between named \
entities from document text. Respond with ONLY a JSON object, no prose, no markdown \
fences, with EXACTLY this shape: \
{\"relations\":[{\"from_entity\":<str>,\"to_entity\":<str>,\"predicate\":<str>,\"chunk_id\":<str>,\"confidence\":<0.0-1.0>}]}. \
Use ONLY predicates from the provided list. Use ONLY entity names from the provided \
list. Each chunk_id MUST be one of the provided ids. confidence is your certainty the \
relation is stated or strongly implied in the cited chunk. If nothing qualifies, return \
an empty relations array. Do not add any other keys.";

/// Raw LLM triple (unvalidated). `deny_unknown_fields` triggers a reprompt on a
/// garbled shape rather than silent acceptance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawTriple {
    pub from_entity: String,
    pub to_entity: String,
    pub predicate: String,
    pub chunk_id: String,
    pub confidence: f64,
}

/// Object wrapper for the LLM response (mirrors `CorefResponse`'s `{"results":[...]}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationsResponse {
    pub relations: Vec<RawTriple>,
}

/// A validated triple: endpoints are resolved to existing node ids and the predicate
/// is a known [`Relation`].
#[derive(Debug, Clone, PartialEq)]
pub struct RelationTriple {
    pub from_node: String,
    pub to_node: String,
    pub relation: Relation,
    pub chunk_id: String,
    pub confidence: f64,
}

/// Parses + validates the LLM response, tolerating markdown fences / preamble.
/// Returns `Err` on any parse miss so the caller can reprompt.
pub fn parse_relations_response(body: &str) -> Result<Vec<RawTriple>, crate::LensError> {
    let json = extract_json_object(body).unwrap_or(body);
    let resp = serde_json::from_str::<RelationsResponse>(json)
        .map_err(|e| crate::LensError::Parse(format!("relations response invalid: {e}")))?;
    Ok(resp.relations)
}

/// Prose chunks (`level > 0`, not a non-prose block) ranked by entity density
/// (case-insensitive substring count over the provided entity names), taking the
/// top `min(ceil(n * FRACTION), CAP)`. Returns `(index, chunk)` pairs; the index
/// is the positional id used in the prompt and validated against on return.
pub fn sample_chunks<'a>(
    chunks: &'a [EnrichmentChunk],
    entities: &[String],
) -> Vec<(usize, &'a EnrichmentChunk)> {
    let lower_entities: Vec<String> = entities.iter().map(|e| e.to_lowercase()).collect();

    let mut scored: Vec<(usize, &EnrichmentChunk, usize)> = chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.level > 0 && !is_nonprose_block(c.block_type.as_deref()))
        .map(|(i, c)| {
            let lower = c.text.to_lowercase();
            let density = lower_entities
                .iter()
                .filter(|e| !e.is_empty())
                .map(|e| lower.matches(e.as_str()).count())
                .sum();
            (i, c, density)
        })
        .collect();

    // Sort by density desc; ties keep original order (stable sort on index asc first).
    scored.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));

    let prose_count = scored.len();
    let proportional = (prose_count as f64 * RELATIONS_SAMPLE_FRACTION).ceil() as usize;
    let take = proportional.min(RELATIONS_SAMPLE_CAP);

    scored
        .into_iter()
        .take(take)
        .map(|(i, c, _)| (i, c))
        .collect()
}

/// Renders the per-batch user prompt: entity list, predicate list, valid chunk ids,
/// then each chunk keyed by its positional id.
pub fn build_relations_prompt(
    batch: &[(usize, &EnrichmentChunk)],
    entities: &[String],
    predicates: &[String],
) -> String {
    let entity_line = if entities.is_empty() {
        "(none)".to_string()
    } else {
        entities.join(", ")
    };
    let predicate_line = if predicates.is_empty() {
        "(none)".to_string()
    } else {
        predicates.join(", ")
    };
    let chunk_ids: Vec<String> = batch.iter().map(|(i, _)| i.to_string()).collect();

    let mut prompt = String::new();
    prompt.push_str("Entity names: ");
    prompt.push_str(&entity_line);
    prompt.push_str("\nPredicates: ");
    prompt.push_str(&predicate_line);
    prompt.push_str("\nValid chunk_id values: ");
    prompt.push_str(&chunk_ids.join(", "));
    prompt.push_str("\n\nExtract relations from these chunks:\n");
    for (id, chunk) in batch {
        prompt.push_str(&format!("[chunk_id={id}]\n{}\n\n", chunk.text));
    }
    prompt
}

/// Drops triples that fail the quality floor and resolves surviving endpoints to
/// existing node ids and the cited chunk to its real DB id. A triple is kept iff:
/// `confidence >= MIN_CONFIDENCE`; its positional `chunk_id` is in `chunk_id_map`;
/// both endpoints resolve to a node in `node_index` (case-insensitive name); and its
/// predicate maps to a known [`Relation`] (canonical or alias → canonical). The
/// resulting `RelationTriple.chunk_id` is the real `chunks.id` (FK-ready).
pub fn validate_triples(
    raw: Vec<RawTriple>,
    node_index: &std::collections::HashMap<String, ResolvedNode>,
    chunk_id_map: &std::collections::HashMap<String, String>,
    predicate_vocab: &HashSet<String>,
    aliases: &std::collections::HashMap<String, String>,
) -> Vec<RelationTriple> {
    let mut out = Vec::new();
    for t in raw {
        if t.confidence < RELATIONS_MIN_CONFIDENCE {
            continue;
        }
        let Some(real_chunk_id) = chunk_id_map.get(&t.chunk_id) else {
            continue;
        };
        let Some(from) = node_index.get(&t.from_entity.to_lowercase()) else {
            continue;
        };
        let Some(to) = node_index.get(&t.to_entity.to_lowercase()) else {
            continue;
        };
        // Resolve alias → canonical, then gate against the known vocab.
        let canonical = aliases.get(&t.predicate).cloned().unwrap_or(t.predicate);
        let Ok(relation) = Relation::semantic(&canonical, predicate_vocab) else {
            continue;
        };
        out.push(RelationTriple {
            from_node: from.id.clone(),
            to_node: to.id.clone(),
            relation,
            chunk_id: real_chunk_id.clone(),
            confidence: t.confidence,
        });
    }
    out
}

/// Runs the relation-extraction pass: sample prose chunks, batch by
/// [`RELATIONS_BATCH_SIZE`], extract + validate per batch. Accumulate-then-persist:
/// a budget breach or transport error after some batches succeed keeps the triples
/// gathered so far (budget breach still surfaces so the worker can flip `failed`).
#[allow(clippy::too_many_arguments)]
pub async fn extract_relations(
    provider: &dyn LlmProvider,
    budget: &mut Budget,
    chunks: &[EnrichmentChunk],
    entities: &[String],
    node_index: &std::collections::HashMap<String, ResolvedNode>,
    predicate_vocab: &HashSet<String>,
    aliases: &std::collections::HashMap<String, String>,
) -> Result<Vec<RelationTriple>, MapError> {
    let sampled = sample_chunks(chunks, entities);
    if sampled.is_empty() {
        return Ok(Vec::new());
    }

    let mut all: Vec<RelationTriple> = Vec::new();
    for batch in sampled.chunks(RELATIONS_BATCH_SIZE) {
        // Positional prompt id (string) -> real chunks.id, for validate + FK.
        let chunk_id_map: std::collections::HashMap<String, String> = batch
            .iter()
            .map(|(i, c)| (i.to_string(), c.id.clone()))
            .collect();
        let prompt = build_relations_prompt(batch, entities, &sorted_vocab(predicate_vocab));

        let raw = match run_llm_with_retries(
            provider,
            budget,
            RELATIONS_SYSTEM_PROMPT,
            &prompt,
            RELATIONS_MAX_TOKENS,
            parse_relations_response,
        )
        .await
        {
            Ok(Some(raw)) => raw,
            // Exhausted reprompts on this batch: skip it, keep prior triples.
            Ok(None) => continue,
            // Budget breach surfaces (worker flips failed); prior triples are still gathered.
            Err(e @ MapError::BudgetExceeded) => return Err(e),
            // Transport error: keep what we have (additive) — surface so the worker degrades.
            Err(e @ MapError::Llm(_)) => return Err(e),
        };
        all.extend(validate_triples(
            raw,
            node_index,
            &chunk_id_map,
            predicate_vocab,
            aliases,
        ));
    }
    Ok(all)
}

/// Deterministic, order-stable predicate list for the prompt (a `HashSet` has no
/// stable iteration order; sorting keeps prompts cacheable).
fn sorted_vocab(vocab: &HashSet<String>) -> Vec<String> {
    let mut v: Vec<String> = vocab.iter().cloned().collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrichment::meta::{Budget, SessionBudget};
    use crate::enrichment::test_util::ScriptedProvider;
    use crate::graph::{EntityKind, build_node_index};
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    fn chunk(id: &str, level: i32, block_type: Option<&str>, text: &str) -> EnrichmentChunk {
        EnrichmentChunk {
            id: id.to_string(),
            parent_id: None,
            kind: "child".to_string(),
            level,
            section_path: "[]".to_string(),
            text: text.to_string(),
            block_type: block_type.map(|s| s.to_string()),
        }
    }

    fn vocab(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn sample_ranks_by_density_and_filters_nonprose() {
        let entities = vec!["Ada".to_string(), "Babbage".to_string()];
        let chunks = vec![
            chunk("c0", 0, None, "Ada Babbage parent level 0"), // level 0 excluded
            chunk("c1", 1, Some("code"), "Ada Babbage Ada"),    // non-prose excluded
            chunk("c2", 1, None, "Ada mentions Babbage and Ada again"), // density 3
            chunk("c3", 1, None, "nothing here"),               // density 0
        ];
        let sampled = sample_chunks(&chunks, &entities);
        // ceil(2 prose * 0.10) = 1 -> top chunk by density is c2 (index 2).
        assert_eq!(sampled.len(), 1);
        assert_eq!(sampled[0].0, 2);
        assert_eq!(sampled[0].1.id, "c2");
    }

    #[test]
    fn sample_large_doc_caps_at_24() {
        // 15,000 prose chunks; ceil(15000*0.10)=1500 but capped at 24.
        let entities = vec!["X".to_string()];
        let chunks: Vec<EnrichmentChunk> = (0..15_000)
            .map(|i| chunk(&format!("c{i}"), 1, None, "X X X"))
            .collect();
        let sampled = sample_chunks(&chunks, &entities);
        assert_eq!(sampled.len(), RELATIONS_SAMPLE_CAP);
        assert_eq!(RELATIONS_SAMPLE_CAP, 24);
        // ceil(24 / batch 8) = 3 calls.
        assert_eq!(sampled.chunks(RELATIONS_BATCH_SIZE).count(), 3);
    }

    #[test]
    fn parse_object_fenced_malformed_empty() {
        let valid = r#"{"relations":[{"from_entity":"A","to_entity":"B","predicate":"founded","chunk_id":"0","confidence":0.9}]}"#;
        assert_eq!(parse_relations_response(valid).unwrap().len(), 1);

        let fenced = format!("Sure:\n```json\n{valid}\n```");
        assert_eq!(parse_relations_response(&fenced).unwrap().len(), 1);

        assert!(parse_relations_response("not json").is_err());
        // A bare array is NOT accepted — must be the object wrapper.
        assert!(parse_relations_response(r#"[{"from_entity":"A"}]"#).is_err());

        let empty = r#"{"relations":[]}"#;
        assert!(parse_relations_response(empty).unwrap().is_empty());
    }

    #[test]
    fn validate_drops_lowconf_uncited_unknown_and_resolves_ids() {
        let entities = vec!["Ada".to_string(), "Babbage".to_string()];
        let node_index = build_node_index("src", &entities, &[], &[]);
        let mut chunk_id_map = HashMap::new();
        chunk_id_map.insert("0".to_string(), "real-chunk-0".to_string());
        let predicate_vocab = vocab(&["founded", "employed_by"]);
        let mut aliases = HashMap::new();
        aliases.insert("works_at".to_string(), "employed_by".to_string());

        let raw = vec![
            // kept: valid, alias resolves to employed_by
            RawTriple {
                from_entity: "ada".to_string(),
                to_entity: "Babbage".to_string(),
                predicate: "works_at".to_string(),
                chunk_id: "0".to_string(),
                confidence: 0.9,
            },
            // dropped: confidence below floor
            RawTriple {
                from_entity: "Ada".to_string(),
                to_entity: "Babbage".to_string(),
                predicate: "founded".to_string(),
                chunk_id: "0".to_string(),
                confidence: 0.3,
            },
            // dropped: chunk_id not sampled
            RawTriple {
                from_entity: "Ada".to_string(),
                to_entity: "Babbage".to_string(),
                predicate: "founded".to_string(),
                chunk_id: "99".to_string(),
                confidence: 0.9,
            },
            // dropped: unknown predicate
            RawTriple {
                from_entity: "Ada".to_string(),
                to_entity: "Babbage".to_string(),
                predicate: "bogus".to_string(),
                chunk_id: "0".to_string(),
                confidence: 0.9,
            },
            // dropped: unknown entity endpoint
            RawTriple {
                from_entity: "Nobody".to_string(),
                to_entity: "Babbage".to_string(),
                predicate: "founded".to_string(),
                chunk_id: "0".to_string(),
                confidence: 0.9,
            },
        ];
        let out = validate_triples(
            raw,
            &node_index,
            &chunk_id_map,
            &predicate_vocab,
            &aliases,
        );
        assert_eq!(out.len(), 1, "only the valid alias-resolved triple survives");
        assert_eq!(out[0].relation, Relation::Semantic("employed_by".to_string()));
        // Endpoint ids are the resolved node ids (not names); chunk id is the real DB id.
        assert_eq!(out[0].from_node, node_index["ada"].id);
        assert_eq!(out[0].to_node, node_index["babbage"].id);
        assert_eq!(out[0].chunk_id, "real-chunk-0");
        assert_eq!(node_index["ada"].kind, EntityKind::Concept);
    }

    #[test]
    fn validate_empty_vocab_drops_all() {
        let entities = vec!["Ada".to_string(), "Babbage".to_string()];
        let node_index = build_node_index("src", &entities, &[], &[]);
        let mut chunk_id_map = HashMap::new();
        chunk_id_map.insert("0".to_string(), "real-chunk-0".to_string());
        let raw = vec![RawTriple {
            from_entity: "Ada".to_string(),
            to_entity: "Babbage".to_string(),
            predicate: "founded".to_string(),
            chunk_id: "0".to_string(),
            confidence: 0.9,
        }];
        let out = validate_triples(
            raw,
            &node_index,
            &chunk_id_map,
            &HashSet::new(),
            &HashMap::new(),
        );
        assert!(out.is_empty(), "empty vocab drops every triple");
    }

    #[test]
    fn prompt_contains_entities_predicates_chunk_ids_and_object_hint() {
        let entities = vec!["Ada".to_string(), "Babbage".to_string()];
        let predicates = vec!["founded".to_string(), "employed_by".to_string()];
        let chunks = vec![chunk("c0", 1, None, "Ada founded it.")];
        let batch: Vec<(usize, &EnrichmentChunk)> = vec![(0, &chunks[0])];
        let prompt = build_relations_prompt(&batch, &entities, &predicates);
        assert!(prompt.contains("Ada"));
        assert!(prompt.contains("Babbage"));
        assert!(prompt.contains("founded"));
        assert!(prompt.contains("employed_by"));
        assert!(prompt.contains("chunk_id=0"));
        assert!(RELATIONS_SYSTEM_PROMPT.contains("\"relations\""));
    }

    #[tokio::test]
    async fn extract_returns_validated_triples_one_call() {
        let entities = vec!["Ada".to_string(), "Babbage".to_string()];
        let node_index = build_node_index("src", &entities, &[], &[]);
        let predicate_vocab = vocab(&["founded"]);
        let chunks = vec![chunk("c0", 1, None, "Ada founded the company with Babbage.")];
        let body = r#"{"relations":[{"from_entity":"Ada","to_entity":"Babbage","predicate":"founded","chunk_id":"0","confidence":0.9}]}"#;
        let (provider, calls) = ScriptedProvider::new(vec![body]);
        let mut budget = Budget::new(SessionBudget::new());
        let out = extract_relations(
            &provider,
            &mut budget,
            &chunks,
            &entities,
            &node_index,
            &predicate_vocab,
            &HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].relation, Relation::Semantic("founded".to_string()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn extract_empty_sample_makes_no_calls() {
        let entities = vec!["Ada".to_string()];
        let node_index = build_node_index("src", &entities, &[], &[]);
        // Only a level-0 chunk → nothing sampled.
        let chunks = vec![chunk("c0", 0, None, "Ada parent")];
        let (provider, calls) = ScriptedProvider::new(vec!["{}"]);
        let mut budget = Budget::new(SessionBudget::new());
        let out = extract_relations(
            &provider,
            &mut budget,
            &chunks,
            &entities,
            &node_index,
            &vocab(&["founded"]),
            &HashMap::new(),
        )
        .await
        .unwrap();
        assert!(out.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}
