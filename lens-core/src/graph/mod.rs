//! M13 entity graph domain types (Phase 1, resolution-immune).
//!
//! Plain data + [`build_entity_graph_rows`], a pure row-builder consumed by the
//! `write_enrichment_and_graph` transaction seam. `canonical_name` /
//! `resolution_conf` are carried but stay `None` in Phase 1 (#155/#156 populate them).

mod build;
mod tools;

pub use build::{ResolvedNode, build_entity_graph_rows, build_node_index};
pub use tools::{GraphEntity, entity_evidence, entity_lookup};

use crate::LensError;

/// Entity node kind, stored as plain `TEXT` (no SQL CHECK; the Rust enum is the guard).
/// `Person`/`Org`/`Location`/`Other` are schema-allowed forward variants unused in Phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}
