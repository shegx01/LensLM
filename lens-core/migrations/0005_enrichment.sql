-- M4 Phase 3: enrichment columns + registry relaxation for the re-embed flip.
--
-- Three additive, nullable columns plus a registry constraint relaxation:
--
-- 1. `chunks.embedding_text` — the contextual text the enrichment pass embeds
--    (context-prefix + optional inline coref + canonical body). NULL until the
--    background worker populates it via UPDATE. Canonical `chunks.text` is
--    UNTOUCHED (it remains the immutable citation text).
-- 2. `sources.enrichment_status` — the per-source enrichment lifecycle
--    (none|pending|enriching|enriched|failed|skipped), SEPARATE from
--    `sources.status` (SourceStatus). NULL ≡ `none` on pre-migration rows.
-- 3. `sources.enrichment_meta` — JSON composite cache key + budget reason.
--
-- SQLite `ADD COLUMN` of a nullable column with no DEFAULT is a safe, O(1)
-- metadata-only operation (no table rewrite); existing rows read NULL.
ALTER TABLE chunks ADD COLUMN embedding_text TEXT;
ALTER TABLE sources ADD COLUMN enrichment_status TEXT;
ALTER TABLE sources ADD COLUMN enrichment_meta TEXT;

-- Registry relaxation (Decision E1): the re-embed flip needs a `building` row to
-- co-exist with the `active` row for the SAME (notebook, model, dim) coordinate.
-- The original table-level `UNIQUE (notebook_id, model, dim)` (0001_init.sql:72)
-- forbids that. SQLite has NO `ALTER TABLE … DROP CONSTRAINT`, and that UNIQUE is
-- backed by the internal `sqlite_autoindex_embedding_index_1` which `DROP INDEX`
-- refuses — so the relaxation MUST use the canonical 12-step TABLE REBUILD:
-- recreate the table WITHOUT the table-level UNIQUE (keeping the outbound FK to
-- notebooks; nothing FKs INTO embedding_index), copy every row via an EXPLICIT
-- column list (never `SELECT *`, which silently breaks on column-order drift),
-- drop + rename, then recreate a PARTIAL unique index that keeps at most one
-- `active` row per coordinate while allowing unlimited `building`/`stale` rows,
-- plus the `idx_embidx_notebook` lookup index (dropped with the old table,
-- 0001_init.sql:111).
PRAGMA foreign_keys=OFF;

CREATE TABLE embedding_index_new (
    id                TEXT PRIMARY KEY NOT NULL,
    notebook_id       TEXT NOT NULL,
    model             TEXT NOT NULL,
    dim               INTEGER NOT NULL,
    prefix_convention TEXT NOT NULL,
    lance_table_name  TEXT NOT NULL,
    status            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
    -- NOTE: NO table-level UNIQUE(notebook_id, model, dim) here.
);

INSERT INTO embedding_index_new
    (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at)
SELECT id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at
FROM embedding_index;

DROP TABLE embedding_index;

ALTER TABLE embedding_index_new RENAME TO embedding_index;

-- Partial unique: at most one live index per coordinate; unlimited building/stale.
CREATE UNIQUE INDEX uq_embidx_active
    ON embedding_index (notebook_id, model, dim) WHERE status = 'active';

-- Recreate the lookup index dropped with the old table (0001_init.sql:111).
CREATE INDEX idx_embidx_notebook ON embedding_index (notebook_id);

PRAGMA foreign_keys=ON;
