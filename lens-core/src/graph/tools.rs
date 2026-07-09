use sqlx::{Row, SqlitePool};

use super::EntityKind;
use crate::LensError;

/// A collapsed logical entity: one row per distinct (name COLLATE NOCASE, kind)
/// in a notebook, aggregated across its per-source nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEntity {
    pub name: String,
    pub kind: EntityKind,
    pub definition: Option<String>,
    pub source_count: u32,
    pub mention_count: u32,
}

/// Returns up to `k` entities in a notebook matching `query`, ranked
/// exact > prefix > substring. Collapses `(name COLLATE NOCASE, kind)` across
/// sources. Returns `[]` immediately for an empty query.
pub async fn entity_lookup(
    pool: &SqlitePool,
    notebook_id: &str,
    query: &str,
    k: usize,
) -> Result<Vec<GraphEntity>, LensError> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Escape LIKE metacharacters so query is treated literally in LIKE clauses.
    // Backslash must be escaped first so the replacements don't re-escape.
    let esc = query
        .replace('\\', r"\\")
        .replace('%', r"\%")
        .replace('_', r"\_");

    // #155: nodes resolved to the same entity share a `canonical_name`; group and
    // return by COALESCE(canonical_name, name) so cross-doc aliases collapse into one
    // result. Unresolved nodes (NULL canonical_name) fall back to their own name.
    let rows = sqlx::query(
        "SELECT
            COALESCE(en.canonical_name, en.name) AS name,
            en.kind,
            MIN(en.definition) AS definition,
            COUNT(DISTINCT en.source_id) AS source_count,
            COUNT(DISTINCT em.id)        AS mention_count,
            MIN(CASE
                WHEN en.name = ?1 COLLATE NOCASE
                  OR en.canonical_name = ?1 COLLATE NOCASE THEN 0
                WHEN en.name LIKE ?2 || '%' ESCAPE '\\'
                  OR en.canonical_name LIKE ?2 || '%' ESCAPE '\\' THEN 1
                ELSE 2
            END) AS tier
        FROM entity_nodes en
        JOIN sources s ON s.id = en.source_id
        LEFT JOIN entity_mentions em ON em.entity_node_id = en.id
        WHERE en.notebook_id = ?3
          AND s.trashed_at IS NULL AND s.selected = 1
          AND (
              en.name = ?1 COLLATE NOCASE
           OR en.canonical_name = ?1 COLLATE NOCASE
           OR en.name LIKE ?2 || '%'        ESCAPE '\\'
           OR en.canonical_name LIKE ?2 || '%' ESCAPE '\\'
           OR en.name LIKE '%' || ?2 || '%' ESCAPE '\\'
           OR en.canonical_name LIKE '%' || ?2 || '%' ESCAPE '\\'
          )
        GROUP BY COALESCE(en.canonical_name, en.name) COLLATE NOCASE, en.kind
        ORDER BY tier ASC, length(COALESCE(en.canonical_name, en.name)) ASC
        LIMIT ?4",
    )
    .bind(query)
    .bind(&esc)
    .bind(notebook_id)
    .bind(k as i64)
    .fetch_all(pool)
    .await?;

    rows.iter()
        .map(|row| {
            let kind_str: String = row.get("kind");
            let kind = EntityKind::from_db(&kind_str)?;
            let source_count: i64 = row.get("source_count");
            let mention_count: i64 = row.get("mention_count");
            Ok(GraphEntity {
                name: row.get("name"),
                kind,
                definition: row.get("definition"),
                source_count: source_count as u32,
                mention_count: mention_count as u32,
            })
        })
        .collect()
}

/// Returns up to `k` chunk IDs that mention `(name, kind)` in a notebook,
/// ordered by mention-count DESC then document order (level ASC, token_start ASC NULLS LAST).
pub async fn entity_evidence(
    pool: &SqlitePool,
    notebook_id: &str,
    name: &str,
    kind: EntityKind,
    k: usize,
) -> Result<Vec<String>, LensError> {
    let rows = sqlx::query(
        "SELECT em.chunk_id
        FROM entity_mentions em
        JOIN entity_nodes en ON en.id = em.entity_node_id
        JOIN sources s ON s.id = en.source_id
        JOIN chunks c ON c.id = em.chunk_id
        WHERE en.notebook_id = ?1
          AND (en.name = ?2 COLLATE NOCASE OR en.canonical_name = ?2 COLLATE NOCASE)
          AND en.kind = ?3
          AND s.trashed_at IS NULL AND s.selected = 1
        GROUP BY em.chunk_id
        ORDER BY COUNT(*) DESC, c.level ASC, c.token_start ASC NULLS LAST
        LIMIT ?4",
    )
    .bind(notebook_id)
    .bind(name)
    .bind(kind.as_str())
    .bind(k as i64)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| r.get::<String, _>("chunk_id"))
        .collect())
}
