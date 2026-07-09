use std::collections::HashMap;

use sqlx::{Row, SqlitePool};

use super::{EntityKind, GraphHit, Relation, blended_edge_weight};
use crate::LensError;

/// Max neighbor triples returned by [`expand_neighbors`] (log a warning when the
/// candidate set is truncated to this cap).
const MAX_TRIPLES: usize = 50;

/// Depth ceiling for [`expand_neighbors`]; larger requests are clamped down.
const MAX_DEPTH: usize = 2;

/// Seed ceiling for [`expand_neighbors`]; the anchor query binds `2 * seeds.len()`
/// params, so cap the seed set to stay under the SQLite bound-variable limit.
const MAX_SEEDS: usize = 64;

/// Per-neighbor chunk-evidence cap so the mention lookup stays bounded.
const MAX_CHUNKS_PER_NEIGHBOR: usize = 50;

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
    mention_chunk_ids(pool, notebook_id, name, kind, Some(k)).await
}

/// Chunk ids mentioning a logical entity (canonical-or-raw match, live-source
/// scoped), drawn from `entity_mentions`. Ordered by mention count then doc order.
/// `limit` bounds the row count; `None` means unbounded.
async fn mention_chunk_ids(
    pool: &SqlitePool,
    notebook_id: &str,
    name: &str,
    kind: EntityKind,
    limit: Option<usize>,
) -> Result<Vec<String>, LensError> {
    let mut sql = String::from(
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
        ORDER BY COUNT(*) DESC, c.level ASC, c.token_start ASC NULLS LAST",
    );
    if limit.is_some() {
        sql.push_str(" LIMIT ?4");
    }

    let mut q = sqlx::query(&sql)
        .bind(notebook_id)
        .bind(name)
        .bind(kind.as_str());
    if let Some(k) = limit {
        q = q.bind(k as i64);
    }
    let rows = q.fetch_all(pool).await?;

    Ok(rows
        .iter()
        .map(|r| r.get::<String, _>("chunk_id"))
        .collect())
}

/// A neighbor node reached by the traversal, before logical collapse.
struct NeighborRow {
    logical_name: String,
    kind: EntityKind,
    relation: Relation,
    weight: f32,
}

/// Expands the entity graph outward from `seeds` up to `depth` hops (clamped to 2),
/// returning neighbor entities ranked by blended edge weight. Traversal is over the
/// live-source portion of the notebook; results collapse to logical entities
/// (`COALESCE(canonical_name, name)`), with `chunk_ids` drawn from `entity_mentions`
/// (not `entity_edges.chunk_id`, which anchors only the first co-occurrence).
/// `graph_confidence` is max-normalized over the returned set; `relation` is the
/// neighbor's max-weight edge relation. Empty or zero-match seeds → `Ok(vec![])`.
pub async fn expand_neighbors(
    pool: &SqlitePool,
    notebook_id: &str,
    seeds: &[(String, EntityKind)],
    depth: usize,
) -> Result<Vec<GraphHit>, LensError> {
    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    let depth = depth.min(MAX_DEPTH);
    if depth == 0 {
        return Ok(Vec::new());
    }
    let seeds = if seeds.len() > MAX_SEEDS {
        tracing::warn!(
            seeds = seeds.len(),
            "expand_neighbors: seed count exceeds {MAX_SEEDS}, truncating"
        );
        &seeds[..MAX_SEEDS]
    } else {
        seeds
    };

    // Resolve seeds to per-source node ids, mirroring #156a.
    let mut anchor_sql = String::from(
        "SELECT en.id
         FROM entity_nodes en
         JOIN sources s ON s.id = en.source_id
         WHERE en.notebook_id = ?1
           AND s.trashed_at IS NULL AND s.selected = 1
           AND (",
    );
    for i in 0..seeds.len() {
        if i > 0 {
            anchor_sql.push_str(" OR ");
        }
        let name_param = 2 + i * 2;
        let kind_param = name_param + 1;
        anchor_sql.push_str(&format!(
            "((en.name = ?{name_param} COLLATE NOCASE OR en.canonical_name = ?{name_param} COLLATE NOCASE) AND en.kind = ?{kind_param})"
        ));
    }
    anchor_sql.push(')');

    let mut anchor_q = sqlx::query(&anchor_sql).bind(notebook_id);
    for (name, kind) in seeds {
        anchor_q = anchor_q.bind(name.clone()).bind(kind.as_str());
    }
    let anchor_rows = anchor_q.fetch_all(pool).await?;
    if anchor_rows.is_empty() {
        return Ok(Vec::new());
    }
    let seed_ids: Vec<String> = anchor_rows.iter().map(|r| r.get("id")).collect();

    // Recursive CTE: walk node ids up to `depth` hops. Each hop expands via a
    // UNION ALL of `from_node = frontier` and `to_node = frontier` (index-friendly,
    // not `OR`), excluding self-loops. The `path` column carries the visited node
    // ids; the join guard `instr(path, ...)` blocks revisits (SQLite has no native
    // visited-set), so cycles terminate.
    // SAFETY: values are DB-read `entity_nodes.id` UUIDs (not caller input), single-quote-escaped.
    let seed_ids_csv = seed_ids
        .iter()
        .map(|id| format!("'{}'", id.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(",");

    let cte_sql = format!(
        "WITH RECURSIVE seed_nodes(id) AS (
            SELECT id FROM entity_nodes WHERE id IN ({seed_ids_csv})
         ),
         walk(node, depth, path) AS (
            SELECT id, 0, '|' || id || '|' FROM seed_nodes
            UNION ALL
            SELECT ee.to_node, w.depth + 1, w.path || ee.to_node || '|'
            FROM walk w
            JOIN entity_edges ee ON ee.from_node = w.node
            WHERE w.depth < ?1
              AND ee.notebook_id = ?2
              AND ee.to_node <> w.node
              AND instr(w.path, '|' || ee.to_node || '|') = 0
            UNION ALL
            SELECT ee.from_node, w.depth + 1, w.path || ee.from_node || '|'
            FROM walk w
            JOIN entity_edges ee ON ee.to_node = w.node
            WHERE w.depth < ?1
              AND ee.notebook_id = ?2
              AND ee.from_node <> w.node
              AND instr(w.path, '|' || ee.from_node || '|') = 0
         )
         SELECT DISTINCT w.node AS neighbor_id,
                COALESCE(en.canonical_name, en.name) AS logical_name,
                en.kind AS kind,
                ee.relation AS relation,
                ee.weight AS weight,
                ee.confidence AS confidence
         FROM walk w
         JOIN entity_nodes en ON en.id = w.node
         JOIN sources s ON s.id = en.source_id
         JOIN entity_edges ee
              ON (ee.from_node = w.node OR ee.to_node = w.node)
             AND ee.notebook_id = ?2
         JOIN entity_nodes other
              ON other.id = CASE WHEN ee.from_node = w.node THEN ee.to_node ELSE ee.from_node END
         JOIN sources os ON os.id = other.source_id
         WHERE w.depth > 0
           AND s.trashed_at IS NULL AND s.selected = 1
           AND os.trashed_at IS NULL AND os.selected = 1
           AND ee.from_node <> ee.to_node"
    );

    let rows = sqlx::query(&cte_sql)
        .bind(depth as i64)
        .bind(notebook_id)
        .fetch_all(pool)
        .await?;

    let mut best: HashMap<(String, EntityKind), NeighborRow> = HashMap::new();
    for row in &rows {
        let logical_name: String = row.get("logical_name");
        let kind_str: String = row.get("kind");
        let kind = EntityKind::from_db(&kind_str)?;
        let relation = Relation::from_db(&row.get::<String, _>("relation"));
        let weight: Option<f64> = row.get("weight");
        let confidence: Option<f64> = row.get("confidence");
        let w = blended_edge_weight(&relation, weight, confidence);

        let key = (logical_name.clone(), kind);
        best.entry(key)
            .and_modify(|nr| {
                if w > nr.weight {
                    nr.weight = w;
                    nr.relation = relation.clone();
                }
            })
            .or_insert(NeighborRow {
                logical_name,
                kind,
                relation,
                weight: w,
            });
    }

    // A seed reached as a depth>0 neighbor from another seed must not leak into the
    // results. Exclude any collapsed row whose (name, kind) matches an input seed,
    // case-insensitively (consistent with the COLLATE NOCASE seed resolution above).
    let seed_keys: std::collections::HashSet<(String, EntityKind)> = seeds
        .iter()
        .map(|(name, kind)| (name.to_lowercase(), *kind))
        .collect();
    best.retain(|(name, kind), _| !seed_keys.contains(&(name.to_lowercase(), *kind)));

    if best.is_empty() {
        return Ok(Vec::new());
    }

    let mut neighbors: Vec<NeighborRow> = best.into_values().collect();
    neighbors.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.logical_name.cmp(&b.logical_name))
    });
    if neighbors.len() > MAX_TRIPLES {
        tracing::warn!(
            notebook_id = %notebook_id,
            candidates = neighbors.len(),
            "expand_neighbors: truncating to {MAX_TRIPLES} triples"
        );
        neighbors.truncate(MAX_TRIPLES);
    }

    let max_weight = neighbors.first().map(|n| n.weight).unwrap_or(0.0);

    let mut hits = Vec::with_capacity(neighbors.len());
    for nr in neighbors {
        let chunk_ids = mention_chunk_ids(
            pool,
            notebook_id,
            &nr.logical_name,
            nr.kind,
            Some(MAX_CHUNKS_PER_NEIGHBOR),
        )
        .await?;
        let confidence = if max_weight > 0.0 {
            nr.weight / max_weight
        } else {
            0.0
        };
        hits.push(GraphHit {
            name: nr.logical_name,
            kind: nr.kind,
            chunk_ids,
            graph_confidence: confidence,
            relation: Some(nr.relation.as_str().to_string()),
        });
    }

    Ok(hits)
}
