//! Pure row-builder for the M13 entity graph. No DB, no LLM, not async: it maps
//! in-memory enrichment outputs to [`EntityGraphRows`]. `ids`/`created_at` are
//! injected so tests are fully deterministic.

use std::collections::{BTreeMap, HashMap};

use sha2::{Digest, Sha256};

use super::{EntityEdge, EntityGraphRows, EntityKind, EntityMention, EntityNode, Relation};
use crate::enrichment::coref::CorefSub;
use crate::enrichment::is_nonprose_block;
use crate::enrichment::meta::Definition;
use crate::notebooks::EnrichmentChunk;

/// Deterministic node id derived from `(source_id, lowercased name, kind)` — the
/// node's UNIQUE key. STABILITY IS LOAD-BEARING: re-enrichment regenerates rows, and
/// `INSERT OR IGNORE` keeps the FIRST-seen node row. If node ids were random, edges
/// and mentions built in the SAME re-run would reference the fresh (ignored) ids and
/// violate their FK to `entity_nodes`. A stable id makes the regenerated node id
/// equal the persisted one, so edges/mentions always reference a row that exists.
pub(crate) fn node_id(source_id: &str, lower_name: &str, kind: EntityKind) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source_id.as_bytes());
    hasher.update([0u8]);
    hasher.update(lower_name.as_bytes());
    hasher.update([0u8]);
    hasher.update(kind.as_str().as_bytes());
    crate::hex_encode(&hasher.finalize())
}

/// A resolved graph node keyed by its lowercased name: the deterministic node id
/// plus kind. Built by [`build_node_index`] with the SAME dedup rules as
/// [`build_entity_graph_rows`] so relation endpoints resolve to real node ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedNode {
    pub id: String,
    pub kind: EntityKind,
}

/// Maps lowercased name to `(node id, kind)` mirroring the dedup rules in
/// [`build_entity_graph_rows`]. Used by the relation pass to resolve LLM-named
/// endpoints to existing node ids by case-insensitive name.
pub fn build_node_index(
    source_id: &str,
    entities: &[String],
    definitions: &[Definition],
    dates: &[String],
) -> HashMap<String, ResolvedNode> {
    let mut index: HashMap<String, ResolvedNode> = HashMap::new();
    for entity in entities {
        let key = entity.to_lowercase();
        index.entry(key.clone()).or_insert_with(|| ResolvedNode {
            id: node_id(source_id, &key, EntityKind::Concept),
            kind: EntityKind::Concept,
        });
    }
    for date in dates {
        let key = date.to_lowercase();
        index.entry(key.clone()).or_insert_with(|| ResolvedNode {
            id: node_id(source_id, &key, EntityKind::Date),
            kind: EntityKind::Date,
        });
    }
    for def in definitions {
        let key = def.term.to_lowercase();
        index.entry(key.clone()).or_insert_with(|| ResolvedNode {
            id: node_id(source_id, &key, EntityKind::Concept),
            kind: EntityKind::Concept,
        });
    }
    index
}

/// Builds nodes / chunk-anchored mentions / bounded co-occurrence edges from the
/// in-memory enrichment outputs of a single source. Pure + deterministic:
///
/// - Case-insensitive (`.to_lowercase()`) matching for node dedup, definition-term
///   fill, string-match presence, and coref-antecedent lookup. Nodes keep first-seen casing.
/// - Kind-collision: a string appearing as both entity and date keeps only the Concept.
/// - Prose-leaf filter: mentions/co-occurrence confined to `level > 0` non-prose-free chunks.
/// - Codepoint (not byte) offsets for string-match mentions.
/// - Deterministic co-occurrence via a `BTreeMap` keyed on the canonical (min,max) id pair.
/// - Per-chunk co-occurrence cap; overflow entities are counted in `dropped_cooccurrence`.
#[allow(clippy::too_many_arguments)]
pub fn build_entity_graph_rows(
    notebook_id: &str,
    source_id: &str,
    chunks: &[EnrichmentChunk],
    entities: &[String],
    definitions: &[Definition],
    dates: &[String],
    coref_subs: &HashMap<usize, Vec<CorefSub>>,
    ids: &mut dyn FnMut() -> String,
    created_at: &str,
    max_entities_per_chunk: usize,
) -> EntityGraphRows {
    let mut nodes: Vec<EntityNode> = Vec::new();
    // lowercase name -> index into `nodes`.
    let mut index: HashMap<String, usize> = HashMap::new();

    // Entities → Concept nodes (first-seen casing wins).
    for entity in entities {
        let key = entity.to_lowercase();
        if index.contains_key(&key) {
            continue;
        }
        index.insert(key.clone(), nodes.len());
        nodes.push(EntityNode {
            id: node_id(source_id, &key, EntityKind::Concept),
            notebook_id: notebook_id.to_string(),
            source_id: source_id.to_string(),
            kind: EntityKind::Concept,
            name: entity.clone(),
            canonical_name: None,
            definition: None,
            resolution_conf: None,
            resolution_prompt_version: None,
            created_at: created_at.to_string(),
        });
    }

    // Dates → Date nodes, unless the lowercased name already exists (Concept wins).
    for date in dates {
        let key = date.to_lowercase();
        if index.contains_key(&key) {
            continue;
        }
        index.insert(key.clone(), nodes.len());
        nodes.push(EntityNode {
            id: node_id(source_id, &key, EntityKind::Date),
            notebook_id: notebook_id.to_string(),
            source_id: source_id.to_string(),
            kind: EntityKind::Date,
            name: date.clone(),
            canonical_name: None,
            definition: None,
            resolution_conf: None,
            resolution_prompt_version: None,
            created_at: created_at.to_string(),
        });
    }

    // Definitions: fill `definition` on an existing node, else create a Concept node.
    for def in definitions {
        let key = def.term.to_lowercase();
        if let Some(&i) = index.get(&key) {
            if nodes[i].definition.is_none() {
                nodes[i].definition = Some(def.definition.clone());
            }
        } else {
            index.insert(key.clone(), nodes.len());
            nodes.push(EntityNode {
                id: node_id(source_id, &key, EntityKind::Concept),
                notebook_id: notebook_id.to_string(),
                source_id: source_id.to_string(),
                kind: EntityKind::Concept,
                name: def.term.clone(),
                canonical_name: None,
                definition: Some(def.definition.clone()),
                resolution_conf: None,
                resolution_prompt_version: None,
                created_at: created_at.to_string(),
            });
        }
    }

    // Lowercased node names, computed once and reused across the per-chunk scan.
    let lower_names: Vec<String> = nodes.iter().map(|n| n.name.to_lowercase()).collect();

    let mut mentions: Vec<EntityMention> = Vec::new();
    // Canonical (min,max) node-id pair -> (count, first co-occurring chunk_id).
    let mut pairs: BTreeMap<(String, String), (u32, String)> = BTreeMap::new();
    let mut dropped_cooccurrence: usize = 0;

    for (i, chunk) in chunks.iter().enumerate() {
        // Prose-leaf filter: level > 0 and not a non-prose block.
        if chunk.level <= 0 || is_nonprose_block(chunk.block_type.as_deref()) {
            continue;
        }

        // String-match mentions + presence set (in stable node-insertion order).
        let mut present: Vec<usize> = Vec::new();
        let lower_text = chunk.text.to_lowercase();
        for (node_i, node) in nodes.iter().enumerate() {
            let name_lower = &lower_names[node_i];
            if name_lower.is_empty() {
                continue;
            }
            let occurrences = find_word_boundary_occurrences(&lower_text, name_lower);
            // A node joins the chunk's co-occurrence set iff its name literally occurs here.
            if !occurrences.is_empty() {
                present.push(node_i);
            }
            for (cp_start, cp_end) in occurrences {
                mentions.push(EntityMention {
                    id: ids(),
                    notebook_id: notebook_id.to_string(),
                    entity_node_id: node.id.clone(),
                    chunk_id: chunk.id.clone(),
                    char_start: cp_start as i64,
                    char_end: cp_end as i64,
                    created_at: created_at.to_string(),
                });
            }
        }

        // Coref mentions: antecedent → node, at the sub's codepoint offsets.
        if let Some(subs) = coref_subs.get(&i) {
            for sub in subs {
                let key = sub.antecedent.to_lowercase();
                if let Some(&node_i) = index.get(&key) {
                    mentions.push(EntityMention {
                        id: ids(),
                        notebook_id: notebook_id.to_string(),
                        entity_node_id: nodes[node_i].id.clone(),
                        chunk_id: chunk.id.clone(),
                        char_start: sub.char_start as i64,
                        char_end: sub.char_end as i64,
                        created_at: created_at.to_string(),
                    });
                }
            }
        }

        // Co-occurrence: cap the present set, count the overflow, aggregate pairs.
        if present.len() > max_entities_per_chunk {
            dropped_cooccurrence += present.len() - max_entities_per_chunk;
            present.truncate(max_entities_per_chunk);
        }
        for a in 0..present.len() {
            for b in (a + 1)..present.len() {
                let id_a = &nodes[present[a]].id;
                let id_b = &nodes[present[b]].id;
                let key = if id_a <= id_b {
                    (id_a.clone(), id_b.clone())
                } else {
                    (id_b.clone(), id_a.clone())
                };
                pairs
                    .entry(key)
                    .and_modify(|(count, _)| *count += 1)
                    .or_insert((1, chunk.id.clone()));
            }
        }
    }

    let edges: Vec<EntityEdge> = pairs
        .into_iter()
        .map(|((from, to), (count, chunk_id))| EntityEdge {
            id: ids(),
            notebook_id: notebook_id.to_string(),
            source_id: source_id.to_string(),
            chunk_id,
            from_node: from,
            to_node: to,
            relation: Relation::CoOccurs,
            weight: Some(count as f64),
            confidence: None,
            created_at: created_at.to_string(),
        })
        .collect();

    EntityGraphRows {
        source_id: source_id.to_string(),
        nodes,
        edges,
        mentions,
        dropped_cooccurrence,
    }
}

/// Finds literal occurrences of `needle` in `haystack` (both already lowercased)
/// on Unicode word boundaries, returning `[char_start, char_end)` **codepoint**
/// ranges. A boundary requires the char before/after the match to be non-alphanumeric.
fn find_word_boundary_occurrences(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if needle.is_empty() {
        return out;
    }
    // Codepoint view of the haystack for boundary checks and offset math.
    let chars: Vec<char> = haystack.chars().collect();
    let needle_chars: Vec<char> = needle.chars().collect();
    let n = chars.len();
    let m = needle_chars.len();
    if m == 0 || m > n {
        return out;
    }
    let mut i = 0;
    while i + m <= n {
        if chars[i..i + m] == needle_chars[..] {
            let before_ok = i == 0 || !chars[i - 1].is_alphanumeric();
            let after_ok = i + m == n || !chars[i + m].is_alphanumeric();
            if before_ok && after_ok {
                out.push((i, i + m));
                i += m;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter() -> impl FnMut() -> String {
        let mut n = 0u64;
        move || {
            n += 1;
            format!("id-{n}")
        }
    }

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

    fn def(term: &str, definition: &str) -> Definition {
        Definition {
            term: term.to_string(),
            definition: definition.to_string(),
        }
    }

    fn build(
        chunks: &[EnrichmentChunk],
        entities: &[String],
        definitions: &[Definition],
        dates: &[String],
        coref_subs: &HashMap<usize, Vec<CorefSub>>,
        cap: usize,
    ) -> EntityGraphRows {
        let mut ids = counter();
        build_entity_graph_rows(
            "nb",
            "src",
            chunks,
            entities,
            definitions,
            dates,
            coref_subs,
            &mut ids,
            "2026-01-01T00:00:00Z",
            cap,
        )
    }

    #[test]
    fn definition_creates_node_when_term_absent() {
        let rows = build(
            &[],
            &[],
            &[def("Analytical Engine", "a mechanical computer")],
            &[],
            &HashMap::new(),
            30,
        );
        assert_eq!(rows.nodes.len(), 1);
        assert_eq!(rows.nodes[0].name, "Analytical Engine");
        assert_eq!(rows.nodes[0].kind, EntityKind::Concept);
        assert_eq!(
            rows.nodes[0].definition.as_deref(),
            Some("a mechanical computer")
        );
    }

    #[test]
    fn definition_fill_case_insensitive() {
        let rows = build(
            &[],
            &["React".to_string()],
            &[def("react", "a UI library")],
            &[],
            &HashMap::new(),
            30,
        );
        assert_eq!(rows.nodes.len(), 1, "no duplicate node created");
        assert_eq!(rows.nodes[0].name, "React", "first-seen casing kept");
        assert_eq!(rows.nodes[0].definition.as_deref(), Some("a UI library"));
    }

    #[test]
    fn kind_collision_dedupe() {
        let rows = build(
            &[],
            &["1984".to_string()],
            &[],
            &["1984".to_string()],
            &HashMap::new(),
            30,
        );
        assert_eq!(rows.nodes.len(), 1, "Date suppressed, single node");
        assert_eq!(rows.nodes[0].kind, EntityKind::Concept, "Concept wins");
    }

    #[test]
    fn coref_antecedent_case_insensitive() {
        let mut coref = HashMap::new();
        // "She" @ codepoints [0,3) resolves to antecedent "ada".
        coref.insert(
            0usize,
            vec![CorefSub {
                mention: "She".to_string(),
                char_start: 0,
                char_end: 3,
                antecedent: "ada".to_string(),
            }],
        );
        let rows = build(
            &[chunk("c0", 1, None, "She wrote code.")],
            &["Ada".to_string()],
            &[],
            &[],
            &coref,
            30,
        );
        let ada_id = &rows.nodes[0].id;
        // One coref mention on the Ada node at [0,3). ("She" does not string-match "Ada".)
        let coref_mention = rows
            .mentions
            .iter()
            .find(|m| m.char_start == 0 && m.char_end == 3);
        let m = coref_mention.expect("coref mention present");
        assert_eq!(&m.entity_node_id, ada_id);
    }

    #[test]
    fn string_match_multibyte_offsets_are_codepoints() {
        // "café" is 4 codepoints (é is 2 bytes). Preceded by multibyte prefix.
        // Text: "Zoé aime café ici." — find "café".
        let text = "Zoé aime café ici.";
        let rows = build(
            &[chunk("c0", 1, None, text)],
            &["café".to_string()],
            &[],
            &[],
            &HashMap::new(),
            30,
        );
        // codepoints: Z(0)o(1)é(2) (3)a(4)i(5)m(6)e(7) (8)c(9)a(10)f(11)é(12)
        // so "café" is [9,13).
        let m = rows
            .mentions
            .iter()
            .find(|m| m.entity_node_id == rows.nodes[0].id)
            .expect("mention present");
        assert_eq!((m.char_start, m.char_end), (9, 13));
    }

    #[test]
    fn string_match_word_boundary() {
        let rows = build(
            &[
                chunk("c0", 1, None, "I like React."),
                chunk("c1", 1, None, "A Reactor hums."),
            ],
            &["React".to_string()],
            &[],
            &[],
            &HashMap::new(),
            30,
        );
        // Matches in c0 ("React.") but NOT in c1 ("Reactor").
        assert_eq!(rows.mentions.len(), 1);
        assert_eq!(rows.mentions[0].chunk_id, "c0");
    }

    #[test]
    fn pure_nonprose_source_yields_zero_rows() {
        let rows = build(
            &[
                chunk("c0", 1, Some("code"), "React React React"),
                chunk("c1", 1, Some("table"), "React"),
                chunk("c2", 1, Some("html"), "React"),
                chunk("c3", 0, None, "React parent level 0"),
            ],
            &["React".to_string()],
            &[],
            &[],
            &HashMap::new(),
            30,
        );
        // The node still exists (from entities), but NO mentions/edges from non-prose
        // or level-0 chunks.
        assert!(rows.mentions.is_empty(), "no mentions on non-prose/leaf-0");
        assert!(rows.edges.is_empty(), "no edges");
    }

    #[test]
    fn cooccurrence_cap_and_drop_count() {
        // 32 entities all present in one prose chunk; cap 30 → drop 2.
        let entities: Vec<String> = (0..32).map(|i| format!("e{i}")).collect();
        let text = entities.join(" ");
        let rows = build(
            &[chunk("c0", 1, None, &text)],
            &entities,
            &[],
            &[],
            &HashMap::new(),
            30,
        );
        assert_eq!(rows.dropped_cooccurrence, 2);
        // 30 kept → C(30,2) = 435 pairs.
        assert_eq!(rows.edges.len(), 435);
    }

    #[test]
    fn cooccurrence_cross_chunk_weight() {
        let entities = vec!["Alice".to_string(), "Bob".to_string()];
        let chunks = vec![
            chunk("c0", 1, None, "Alice met Bob."),
            chunk("c1", 1, None, "Bob called Alice."),
            chunk("c2", 1, None, "Alice and Bob again."),
        ];
        let rows = build(&chunks, &entities, &[], &[], &HashMap::new(), 30);
        assert_eq!(rows.edges.len(), 1);
        assert_eq!(rows.edges[0].weight, Some(3.0));
        assert_eq!(
            rows.edges[0].chunk_id, "c0",
            "anchored to first co-occurrence"
        );
    }

    #[test]
    fn cooccurrence_deterministic_edge_order() {
        let entities = vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()];
        let chunks = vec![chunk("c0", 1, None, "Alice Bob Carol together.")];
        let coref = HashMap::new();
        let mut ids1 = counter();
        let first = build_entity_graph_rows(
            "nb",
            "src",
            &chunks,
            &entities,
            &[],
            &[],
            &coref,
            &mut ids1,
            "2026-01-01T00:00:00Z",
            30,
        );
        let mut ids2 = counter();
        let second = build_entity_graph_rows(
            "nb",
            "src",
            &chunks,
            &entities,
            &[],
            &[],
            &coref,
            &mut ids2,
            "2026-01-01T00:00:00Z",
            30,
        );
        let e1: Vec<_> = first
            .edges
            .iter()
            .map(|e| (e.from_node.clone(), e.to_node.clone()))
            .collect();
        let e2: Vec<_> = second
            .edges
            .iter()
            .map(|e| (e.from_node.clone(), e.to_node.clone()))
            .collect();
        assert_eq!(e1, e2, "edge (from,to) order identical across runs");
        assert_eq!(first.edges, second.edges, "full edge vectors identical");
    }
}
