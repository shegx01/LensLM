//! M13 entity graph domain types (Phase 1, resolution-immune).
//!
//! Plain data + [`build_entity_graph_rows`], a pure row-builder consumed by the
//! `write_enrichment_and_graph` transaction seam. `canonical_name` /
//! `resolution_conf` are carried but stay `None` in Phase 1 (#155/#156 populate them).

mod build;
mod ppr;
mod tools;

pub use build::{ResolvedNode, build_entity_graph_rows, build_node_index};
pub use ppr::{NotebookGraph, ppr_expand, ppr_expand_capped_for_test};
pub use tools::{GraphEntity, entity_evidence, entity_lookup, expand_neighbors};

use crate::LensError;

/// A graph-traversal result: a neighbor entity reached from the seed set.
/// `graph_confidence` is max-normalized over the returned set (top hit ~1.0;
/// all-zero → 0.0) and is a within-call ranking signal only. `relation` is the
/// relation of the neighbor's max-weight edge (`expand_neighbors`) or `None`
/// (`ppr_expand`, where hits are ranked by a global PPR score, not one edge).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphHit {
    pub name: String,
    pub kind: EntityKind,
    pub chunk_ids: Vec<String>,
    pub graph_confidence: f32,
    pub relation: Option<String>,
}

/// Multiplier applied to a semantic edge's confidence so a typed predicate
/// outranks a raw co-occurrence of equal nominal strength. Tunable estimate.
const SEMANTIC_BOOST: f64 = 3.0;

/// Strictly-positive floor on a blended edge weight so PPR normalization never
/// divides by (or propagates) a zero weight.
const WEIGHT_FLOOR: f32 = 0.01;

/// Strictly-positive transition/ranking weight for a directed edge, shared by the
/// `expand_neighbors` post-ranking and the PPR loader. `co_occurs` weight is the
/// stored co-occurrence count, log-damped; a semantic edge uses its confidence
/// scaled by [`SEMANTIC_BOOST`]. When a logical pair carries both classes, the
/// caller takes the `max`. Floored at `0.01` so PPR normalization never divides
/// by (or propagates) a zero weight.
fn blended_edge_weight(relation: &Relation, weight: Option<f64>, confidence: Option<f64>) -> f32 {
    let raw = match relation {
        Relation::CoOccurs => weight.unwrap_or(1.0).max(0.0).ln_1p(),
        Relation::Semantic(_) => SEMANTIC_BOOST * confidence.unwrap_or(0.5),
    };
    (raw as f32).max(WEIGHT_FLOOR)
}

/// Entity node kind, stored as plain `TEXT` (no SQL CHECK; the Rust enum is the guard).
/// `Person`/`Org`/`Location`/`Other` are schema-allowed forward variants unused in Phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Concept,
    Date,
    Person,
    Org,
    Location,
    Other,
}

impl EntityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EntityKind::Concept => "concept",
            EntityKind::Date => "date",
            EntityKind::Person => "person",
            EntityKind::Org => "org",
            EntityKind::Location => "location",
            EntityKind::Other => "other",
        }
    }

    pub fn from_db(s: &str) -> Result<Self, LensError> {
        match s {
            "concept" => Ok(EntityKind::Concept),
            "date" => Ok(EntityKind::Date),
            "person" => Ok(EntityKind::Person),
            "org" => Ok(EntityKind::Org),
            "location" => Ok(EntityKind::Location),
            "other" => Ok(EntityKind::Other),
            _ => Err(LensError::Parse(format!("unknown entity kind: {s}"))),
        }
    }
}

/// Edge relation, stored as plain `TEXT` (no SQL CHECK; the Rust enum is the guard,
/// mirroring [`EntityKind`]). `Semantic` carries a canonical predicate name validated
/// against `relation_types` at construction time (#154).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Relation {
    CoOccurs,
    /// A typed semantic predicate (e.g. `founded`, `part_of`). The name is the
    /// canonical `relation_types.name`, validated by [`Relation::semantic`].
    Semantic(String),
}

impl Relation {
    pub fn as_str(&self) -> &str {
        match self {
            Relation::CoOccurs => "co_occurs",
            Relation::Semantic(name) => name.as_str(),
        }
    }

    /// Reconstructs a `Relation` from a DB value. Never errors: the value was
    /// validated at write time, so any non-`co_occurs` string is a `Semantic`.
    pub fn from_db(s: &str) -> Self {
        match s {
            "co_occurs" => Relation::CoOccurs,
            other => Relation::Semantic(other.to_string()),
        }
    }

    /// Gated constructor for a semantic predicate: succeeds only when `name` is a
    /// known `relation_types` predicate. `vocab` is the in-memory set loaded from
    /// the DB. An empty vocab rejects everything (safe: zero semantic edges).
    pub fn semantic(
        name: &str,
        vocab: &std::collections::HashSet<String>,
    ) -> Result<Self, LensError> {
        if vocab.contains(name) {
            Ok(Relation::Semantic(name.to_string()))
        } else {
            Err(LensError::Parse(format!(
                "unknown relation predicate: {name}"
            )))
        }
    }
}

/// An entity node (per-source). `canonical_name`/`resolution_conf`/
/// `resolution_prompt_version` stay `None` until the #155 resolution pass populates them.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityNode {
    pub id: String,
    pub notebook_id: String,
    pub source_id: String,
    pub kind: EntityKind,
    pub name: String,
    pub canonical_name: Option<String>,
    pub definition: Option<String>,
    pub resolution_conf: Option<f64>,
    pub resolution_prompt_version: Option<String>,
    pub created_at: String,
}

/// A co-occurrence edge between two nodes, anchored to the first co-occurring chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityEdge {
    pub id: String,
    pub notebook_id: String,
    pub source_id: String,
    pub chunk_id: String,
    pub from_node: String,
    pub to_node: String,
    pub relation: Relation,
    pub weight: Option<f64>,
    pub confidence: Option<f64>,
    pub created_at: String,
}

/// A chunk-anchored mention of a node at codepoint `[char_start, char_end)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityMention {
    pub id: String,
    pub notebook_id: String,
    pub entity_node_id: String,
    pub chunk_id: String,
    pub char_start: i64,
    pub char_end: i64,
    pub created_at: String,
}

/// Output of [`build_entity_graph_rows`]: prebuilt rows ready for the write seam,
/// plus the count of co-occurrence entities dropped over the per-chunk cap.
///
/// `source_id` scopes the rows to a single source; the write seam uses it to make
/// the graph write self-replacing (delete-then-reinsert the source's nodes), so a
/// re-enrichment that drops entities leaves no stale nodes behind (#157).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EntityGraphRows {
    pub source_id: String,
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<EntityEdge>,
    pub mentions: Vec<EntityMention>,
    pub dropped_cooccurrence: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_kind_as_str_roundtrips() {
        for (kind, s) in [
            (EntityKind::Concept, "concept"),
            (EntityKind::Date, "date"),
            (EntityKind::Person, "person"),
            (EntityKind::Org, "org"),
            (EntityKind::Location, "location"),
            (EntityKind::Other, "other"),
        ] {
            assert_eq!(kind.as_str(), s);
            assert_eq!(EntityKind::from_db(s).unwrap(), kind);
        }
    }

    #[test]
    fn entity_kind_from_db_unknown_errors() {
        assert!(matches!(
            EntityKind::from_db("bogus"),
            Err(LensError::Parse(_))
        ));
    }

    #[test]
    fn relation_as_str_roundtrips() {
        assert_eq!(Relation::CoOccurs.as_str(), "co_occurs");
        assert_eq!(Relation::from_db("co_occurs"), Relation::CoOccurs);
        assert_eq!(
            Relation::Semantic("founded".to_string()).as_str(),
            "founded"
        );
    }

    #[test]
    fn relation_from_db_unknown_is_semantic() {
        assert_eq!(
            Relation::from_db("founded"),
            Relation::Semantic("founded".to_string())
        );
    }

    #[test]
    fn relation_semantic_validates_against_vocab() {
        let mut vocab = std::collections::HashSet::new();
        vocab.insert("founded".to_string());
        assert_eq!(
            Relation::semantic("founded", &vocab).unwrap(),
            Relation::Semantic("founded".to_string())
        );
        assert!(matches!(
            Relation::semantic("nope", &vocab),
            Err(LensError::Parse(_))
        ));
        // Empty vocab rejects everything.
        assert!(Relation::semantic("founded", &std::collections::HashSet::new()).is_err());
    }

    #[test]
    fn blended_weight_cooccurs_log_damped() {
        // ln_1p(1) ≈ 0.6931; NULL weight defaults to 1.0.
        let w = blended_edge_weight(&Relation::CoOccurs, Some(1.0), None);
        assert!((w - 1.0f64.ln_1p() as f32).abs() < 1e-6);
        let d = blended_edge_weight(&Relation::CoOccurs, None, None);
        assert!((d - 1.0f64.ln_1p() as f32).abs() < 1e-6);
        // Higher count → higher (but log-damped) weight.
        let hi = blended_edge_weight(&Relation::CoOccurs, Some(10.0), None);
        assert!(hi > w);
    }

    #[test]
    fn blended_weight_semantic_boosted() {
        let s = Relation::Semantic("founded".to_string());
        // SEMANTIC_BOOST(3.0) * 0.5 default when confidence is NULL.
        let d = blended_edge_weight(&s, None, None);
        assert!((d - (SEMANTIC_BOOST * 0.5) as f32).abs() < 1e-6);
        let c = blended_edge_weight(&s, None, Some(0.8));
        assert!((c - (SEMANTIC_BOOST * 0.8) as f32).abs() < 1e-6);
    }

    #[test]
    fn blended_weight_floored_strictly_positive() {
        // A zero-count co_occurs (ln_1p(0)=0) is floored to 0.01, never 0.
        let z = blended_edge_weight(&Relation::CoOccurs, Some(0.0), None);
        assert!((z - 0.01).abs() < 1e-6);
        assert!(z > 0.0);
    }
}
