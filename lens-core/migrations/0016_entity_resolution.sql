-- M13 #155: cross-document entity resolution. Adds the resolution version tag to
-- entity_nodes (populated by the resolution worker; NULL = unresolved) and a
-- persistent adjudication cache so full-notebook recompute never re-pays the LLM
-- for an already-judged pair. `canonical_name`/`resolution_conf` already exist
-- (0014). Query-time alias only — no in-place merge of node/edge rows.
ALTER TABLE entity_nodes ADD COLUMN resolution_prompt_version TEXT;

-- Persisted LLM adjudication verdicts. Keyed on the normalized candidate pair +
-- prompt version so a version bump invalidates by key (old-version rows are GC'd
-- by the resolution pass). `verdict` is 0/1 (bool). Cascades on notebook delete.
CREATE TABLE IF NOT EXISTS adjudication_cache (
    normalized_pair           TEXT NOT NULL,
    resolution_prompt_version TEXT NOT NULL,
    notebook_id               TEXT NOT NULL,
    verdict                   INTEGER NOT NULL,
    confidence                REAL NOT NULL,
    created_at                TEXT NOT NULL,
    PRIMARY KEY (normalized_pair, resolution_prompt_version),
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_adjcache_nb ON adjudication_cache (notebook_id);
