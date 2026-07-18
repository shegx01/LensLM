//! Offline, deterministic cascade tests: hand-built vectors, a mock [`VectorStore`]
//! returning scripted neighbours, a mock [`LlmProvider`] ([`ScriptedProvider`]), and
//! a counting mock [`AdjudicationCache`]. No embedder, DB, or network in the loop.

use super::*;

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::embedder::EmbeddingBackend;
use crate::enrichment::meta::SessionBudget;
use crate::enrichment::test_util::ScriptedProvider;
use crate::graph::EntityKind;
use crate::vector_store::{Coordinate, EntityVectorRow, Hit, VectorRow};

// --- fixtures ---------------------------------------------------------------

fn node(id: &str, kind: EntityKind, name: &str, def: Option<&str>) -> EntityNode {
    EntityNode {
        id: id.to_string(),
        notebook_id: "nb".to_string(),
        source_id: "s1".to_string(),
        kind,
        name: name.to_string(),
        canonical_name: None,
        definition: def.map(|d| d.to_string()),
        resolution_conf: None,
        resolution_prompt_version: None,
        created_at: "2026-07-09T00:00:00+00:00".to_string(),
    }
}

fn coord() -> Coordinate {
    Coordinate::new(
        "nb",
        EmbeddingBackend::Fastembed,
        "nomic-embed-text-v1.5",
        3,
    )
}

fn budget() -> Budget {
    Budget::with_caps(
        SessionBudget::new(),
        RESOLUTION_MAX_TOKENS_PER_NOTEBOOK,
        RESOLUTION_MAX_CALLS_PER_NOTEBOOK,
    )
}

/// A [`VectorStore`] whose `entity_ann` replays a scripted neighbour list per query
/// vector's node id. Only `entity_ann` is exercised; the rest are unreachable stubs.
struct MockStore {
    /// query_node_id → Vec<(neighbor_node_id, cosine_distance)>
    neighbors: HashMap<String, Vec<(String, f32)>>,
    /// node_id → its query vector, so `entity_ann` can reverse-map a query to its id.
    vec_owner: HashMap<Vec<u8>, String>,
}

impl MockStore {
    fn new(
        vectors: &HashMap<String, Vec<f32>>,
        neighbors: HashMap<String, Vec<(String, f32)>>,
    ) -> Self {
        let vec_owner = vectors
            .iter()
            .map(|(id, v)| (vec_key(v), id.clone()))
            .collect();
        Self {
            neighbors,
            vec_owner,
        }
    }
}

fn vec_key(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[async_trait]
impl VectorStore for MockStore {
    async fn entity_ann(
        &self,
        _coord: &Coordinate,
        query: &[f32],
        k: usize,
        kind: Option<&str>,
    ) -> Result<Vec<(String, f32)>, LensError> {
        let owner = match self.vec_owner.get(&vec_key(query)) {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };
        let mut hits = self.neighbors.get(owner).cloned().unwrap_or_default();
        // The real store applies the kind filter; the mock's scripted lists are
        // already same-kind, so the filter is a no-op here (kept for signature parity).
        let _ = kind;
        hits.truncate(k);
        Ok(hits)
    }

    // --- unused in these tests ---
    async fn add(&self, _c: &Coordinate, _r: Vec<VectorRow>) -> Result<(), LensError> {
        unreachable!("chunk-vector path not exercised")
    }
    async fn search(&self, _c: &Coordinate, _q: &[f32], _k: usize) -> Result<Vec<Hit>, LensError> {
        unreachable!()
    }
    async fn search_filtered(
        &self,
        _c: &Coordinate,
        _q: &[f32],
        _k: usize,
        _s: &[String],
    ) -> Result<Vec<Hit>, LensError> {
        unreachable!("chunk-vector search path not exercised")
    }
    async fn drop_source(&self, _c: &Coordinate, _s: &str) -> Result<(), LensError> {
        unreachable!()
    }
    async fn drop_tables(&self, _t: &[String]) -> Result<(), LensError> {
        unreachable!()
    }
    async fn create_building_table(&self, _c: &Coordinate) -> Result<String, LensError> {
        unreachable!()
    }
    async fn add_to_table(&self, _t: &str, _r: Vec<VectorRow>, _d: usize) -> Result<(), LensError> {
        unreachable!()
    }
    async fn add_to_table_no_index(
        &self,
        _t: &str,
        _r: Vec<VectorRow>,
        _d: usize,
    ) -> Result<(), LensError> {
        unreachable!()
    }
    async fn build_index_on_table(&self, _t: &str, _d: usize) -> Result<(), LensError> {
        unreachable!()
    }
    async fn flip_active(&self, _c: &Coordinate, _b: &str) -> Result<(), LensError> {
        unreachable!()
    }
    async fn retire_coordinate(&self, _c: &Coordinate) -> Result<(), LensError> {
        unreachable!()
    }
    async fn upsert_entity_vectors(
        &self,
        _c: &Coordinate,
        _r: Vec<EntityVectorRow>,
    ) -> Result<(), LensError> {
        unreachable!()
    }
    async fn drop_entity_source(&self, _c: &Coordinate, _s: &str) -> Result<(), LensError> {
        unreachable!()
    }
    async fn drop_entity_tables_for_notebook(&self, _n: &str) -> Result<(), LensError> {
        unreachable!()
    }
    async fn entity_table_names_for_notebook(&self, _n: &str) -> Result<Vec<String>, LensError> {
        unreachable!()
    }
    async fn entity_tables_with_notebook(&self) -> Result<Vec<(String, String)>, LensError> {
        unreachable!()
    }
}

/// Counting mock cache: `get` returns preseeded verdicts; `put` records them.
/// `get_calls`/`put_calls` assert cache hits vs LLM misses.
#[derive(Default)]
struct MockCache {
    seed: Mutex<HashMap<String, (bool, f64)>>,
    get_calls: AtomicU32,
    put_calls: AtomicU32,
}

impl MockCache {
    fn preseeded(pairs: Vec<(&str, bool, f64)>) -> Self {
        let seed = pairs
            .into_iter()
            .map(|(k, v, c)| (k.to_string(), (v, c)))
            .collect();
        Self {
            seed: Mutex::new(seed),
            ..Self::default()
        }
    }
}

#[async_trait]
impl AdjudicationCache for MockCache {
    async fn get(&self, key: &str, _version: &str) -> Result<Option<(bool, f64)>, LensError> {
        self.get_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.seed.lock().unwrap().get(key).copied())
    }
    async fn put(
        &self,
        key: &str,
        _version: &str,
        _nb: &str,
        verdict: bool,
        confidence: f64,
    ) -> Result<(), LensError> {
        self.put_calls.fetch_add(1, Ordering::SeqCst);
        self.seed
            .lock()
            .unwrap()
            .insert(key.to_string(), (verdict, confidence));
        Ok(())
    }
}

fn dist(sim: f64) -> f32 {
    (1.0 - sim) as f32
}

// --- normalize_name / embedding_text ----------------------------------------

#[test]
fn normalize_name_cases() {
    assert_eq!(
        normalize_name("  Amazon   Web  Services  "),
        "amazon web services"
    );
    assert_eq!(normalize_name("AWS."), "aws");
    assert_eq!(normalize_name("OpenAI!!!"), "openai");
    assert_eq!(normalize_name("Node.js"), "node.js"); // internal punctuation kept
    assert_eq!(normalize_name("The Company,"), "the company");
}

#[test]
fn embedding_text_with_and_without_definition() {
    let with = node("1", EntityKind::Org, "AWS", Some("cloud platform"));
    assert_eq!(embedding_text(&with), "org: AWS — cloud platform");
    let without = node("2", EntityKind::Concept, "RAG", None);
    assert_eq!(embedding_text(&without), "concept: RAG");
    let blank = node("3", EntityKind::Concept, "X", Some("   "));
    assert_eq!(embedding_text(&blank), "concept: X");
}

// --- Tier 1 -----------------------------------------------------------------

#[tokio::test]
async fn tier1_exact_name_union_across_sources_longest_canonical() {
    let mut a = node("a", EntityKind::Org, "Amazon Web Services", None);
    a.source_id = "s1".into();
    let mut b = node("b", EntityKind::Org, "amazon web services", None);
    b.source_id = "s2".into();
    let nodes = vec![a, b];

    let vectors = HashMap::new();
    let store = MockStore::new(&vectors, HashMap::new());
    let cache = MockCache::default();

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: None,
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &[],
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .unwrap();

    assert_eq!(updates.len(), 2, "both members aliased");
    for u in &updates {
        assert_eq!(
            u.canonical_name, "Amazon Web Services",
            "longest form is canonical"
        );
        assert_eq!(u.resolution_conf, 1.0);
    }
}

#[tokio::test]
async fn tier1_typed_veto_same_name_different_kind_not_merged() {
    let nodes = vec![
        node("a", EntityKind::Org, "Apollo", None),
        node("b", EntityKind::Person, "Apollo", None),
    ];
    let vectors = HashMap::new();
    let store = MockStore::new(&vectors, HashMap::new());
    let cache = MockCache::default();

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: None,
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &[],
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .unwrap();

    assert!(updates.is_empty(), "different kinds never merge");
}

// --- Tier 2 -----------------------------------------------------------------

#[tokio::test]
async fn tier2_auto_merge_at_high_sim_keep_separate_at_low_sim() {
    let nodes = vec![
        node("a", EntityKind::Concept, "Neural Net", None),
        node("b", EntityKind::Concept, "Neural Network", None), // near-dup, sim 0.95
        node("c", EntityKind::Concept, "Bicycle", None),        // unrelated, sim 0.30
    ];
    let mut vectors = HashMap::new();
    vectors.insert("a".to_string(), vec![1.0, 0.0, 0.0]);
    vectors.insert("b".to_string(), vec![0.0, 1.0, 0.0]);
    vectors.insert("c".to_string(), vec![0.0, 0.0, 1.0]);

    let mut neighbors = HashMap::new();
    neighbors.insert(
        "a".to_string(),
        vec![("b".to_string(), dist(0.95)), ("c".to_string(), dist(0.30))],
    );
    let store = MockStore::new(&vectors, neighbors);
    let cache = MockCache::default();

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: None,
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &[],
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .unwrap();

    // a & b merge (sim 0.95 >= 0.90); c stays separate (sim 0.30 < 0.72).
    assert_eq!(updates.len(), 2);
    let ids: std::collections::HashSet<_> =
        updates.iter().map(|u| u.entity_node_id.as_str()).collect();
    assert!(ids.contains("a") && ids.contains("b"));
    assert!(!ids.contains("c"));
    for u in &updates {
        assert_eq!(u.canonical_name, "Neural Network"); // longest
        assert!((u.resolution_conf - 0.95).abs() < 1e-6);
    }
    assert_eq!(
        cache.get_calls.load(Ordering::SeqCst),
        0,
        "high-sim never adjudicates"
    );
}

// --- Tier 3 -----------------------------------------------------------------

type MidbandFixture = (
    Vec<EntityNode>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<(String, f32)>>,
);

/// Builds a mid-band (sim 0.80) candidate pair a<->b so Tier 3 must adjudicate.
fn midband_setup() -> MidbandFixture {
    let nodes = vec![
        node(
            "a",
            EntityKind::Org,
            "IBM",
            Some("International Business Machines"),
        ),
        node("b", EntityKind::Org, "Big Blue", Some("nickname for IBM")),
    ];
    let mut vectors = HashMap::new();
    vectors.insert("a".to_string(), vec![1.0, 0.0, 0.0]);
    vectors.insert("b".to_string(), vec![0.0, 1.0, 0.0]);
    let mut neighbors = HashMap::new();
    neighbors.insert("a".to_string(), vec![("b".to_string(), dist(0.80))]);
    (nodes, vectors, neighbors)
}

#[tokio::test]
async fn tier3_llm_same_high_conf_merges() {
    let (nodes, vectors, neighbors) = midband_setup();
    let store = MockStore::new(&vectors, neighbors);
    let cache = MockCache::default();
    let (provider, calls) = ScriptedProvider::new(vec![r#"{"same": true, "confidence": 0.9}"#]);

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: Some(&provider),
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &[],
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1, "one adjudication call");
    assert_eq!(updates.len(), 2, "merged at conf 0.9 >= 0.88");
    for u in &updates {
        assert!((u.resolution_conf - 0.9).abs() < 1e-6);
        assert_eq!(u.canonical_name, "Big Blue"); // longest of {IBM, Big Blue}
    }
    assert_eq!(
        cache.put_calls.load(Ordering::SeqCst),
        1,
        "verdict persisted"
    );
}

#[tokio::test]
async fn tier3_llm_same_below_bar_not_merged() {
    let (nodes, vectors, neighbors) = midband_setup();
    let store = MockStore::new(&vectors, neighbors);
    let cache = MockCache::default();
    let (provider, _calls) = ScriptedProvider::new(vec![r#"{"same": true, "confidence": 0.80}"#]);

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: Some(&provider),
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &[],
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .unwrap();

    assert!(updates.is_empty(), "0.80 < 0.88 write bar → discarded");
}

#[tokio::test]
async fn tier3_provider_transport_error_degrades_but_pass_ok() {
    // Two exact-name pairs (Tier-1 merge, conf 1.0) PLUS a mid-band pair whose only
    // resolver is a dead provider — the pass must still return the Tier-1 result Ok.
    let nodes = vec![
        node("x1", EntityKind::Concept, "Vector Search", None),
        node("x2", EntityKind::Concept, "vector search", None), // Tier-1 exact merge
        node("a", EntityKind::Org, "IBM", Some("machines")),
        node("b", EntityKind::Org, "Big Blue", Some("nickname")),
    ];
    let mut vectors = HashMap::new();
    vectors.insert("a".to_string(), vec![1.0, 0.0, 0.0]);
    vectors.insert("b".to_string(), vec![0.0, 1.0, 0.0]);
    let mut neighbors = HashMap::new();
    neighbors.insert("a".to_string(), vec![("b".to_string(), dist(0.80))]);
    let store = MockStore::new(&vectors, neighbors);
    let cache = MockCache::default();
    let (provider, _calls) = ScriptedProvider::dead();

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: Some(&provider),
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &[],
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .expect("dead provider degrades, never errors");

    // Only the Tier-1 pair resolves; the mid-band IBM/Big Blue pair is left NULL.
    assert_eq!(updates.len(), 2);
    let ids: std::collections::HashSet<_> =
        updates.iter().map(|u| u.entity_node_id.as_str()).collect();
    assert!(ids.contains("x1") && ids.contains("x2"));
    assert!(!ids.contains("a") && !ids.contains("b"));
    assert_eq!(
        cache.put_calls.load(Ordering::SeqCst),
        0,
        "no verdict persisted on degrade"
    );
}

// --- cache hit path ---------------------------------------------------------

#[tokio::test]
async fn tier3_cache_hit_skips_llm() {
    let (nodes, vectors, neighbors) = midband_setup();
    let store = MockStore::new(&vectors, neighbors);
    // Preseed the exact key the cascade will compute for (IBM, Big Blue) as Org.
    let key = format!("{}|{}|{}", "org", "big blue", "ibm");
    let cache = MockCache::preseeded(vec![(&key, true, 0.95)]);
    let (provider, calls) = ScriptedProvider::new(vec![r#"{"same": true, "confidence": 0.9}"#]);

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: Some(&provider),
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &[],
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "cache hit ⇒ LLM never called"
    );
    assert_eq!(updates.len(), 2, "cached verdict 0.95 merges");
    for u in &updates {
        assert!((u.resolution_conf - 0.95).abs() < 1e-6);
    }
}

// --- coref seed -------------------------------------------------------------

#[tokio::test]
async fn coref_pairs_force_a_union() {
    // Two differently-named same-kind nodes with no ANN neighbours: only the coref
    // seed can merge them.
    let nodes = vec![
        node("a", EntityKind::Person, "Dr. Smith", None),
        node("b", EntityKind::Person, "Jane Smith", None),
    ];
    let vectors = HashMap::new();
    let store = MockStore::new(&vectors, HashMap::new());
    let cache = MockCache::default();
    let coref = vec![("a".to_string(), "b".to_string())];

    let updates = resolve_notebook(
        ResolveInput {
            nodes: &nodes,
            vectors: &vectors,
            store: &store,
            coord: &coord(),
            provider: None,
            cache: &cache,
            prompt_version: "v1",
            coref_pairs: &coref,
            notebook_id: "nb",
        },
        &mut budget(),
    )
    .await
    .unwrap();

    assert_eq!(updates.len(), 2, "coref seed unions them");
    for u in &updates {
        assert_eq!(u.resolution_conf, 1.0);
        assert_eq!(u.canonical_name, "Jane Smith"); // 10 vs 9 chars → longest
    }
}
