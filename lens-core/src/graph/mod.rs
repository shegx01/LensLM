//! M13 entity graph domain types (Phase 1, resolution-immune).
//!
//! Plain data + [`build_entity_graph_rows`], a pure row-builder consumed by the
//! `write_enrichment_and_graph` transaction seam. `canonical_name` /
//! `resolution_conf` are carried but stay `None` in Phase 1 (#155/#156 populate them).

mod build;

pub use build::build_entity_graph_rows;

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

/// Edge relation, stored as plain `TEXT`. Single-variant in Phase 1; #154 adds
/// semantic relations with a trivial same-crate edit (no `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    CoOccurs,
}

impl Relation {
    pub fn as_str(self) -> &'static str {
        match self {
            Relation::CoOccurs => "co_occurs",
        }
    }

    pub fn from_db(s: &str) -> Result<Self, LensError> {
        match s {
            "co_occurs" => Ok(Relation::CoOccurs),
            _ => Err(LensError::Parse(format!("unknown relation: {s}"))),
        }
    }
}

/// An entity node (per-source). `canonical_name`/`resolution_conf` stay `None` in Phase 1.
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
        assert_eq!(Relation::from_db("co_occurs").unwrap(), Relation::CoOccurs);
    }

    #[test]
    fn relation_from_db_unknown_errors() {
        assert!(matches!(
            Relation::from_db("nope"),
            Err(LensError::Parse(_))
        ));
    }
}
