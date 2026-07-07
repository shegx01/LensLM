-- M13 entity graph (see write_entity_graph_tx for teardown scope).

CREATE TABLE IF NOT EXISTS entity_nodes (
    id              TEXT PRIMARY KEY NOT NULL,
    notebook_id     TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL COLLATE NOCASE,
    canonical_name  TEXT,
    definition      TEXT,
    resolution_conf REAL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE,
    UNIQUE (source_id, name, kind)
);

CREATE TABLE IF NOT EXISTS entity_edges (
    id          TEXT PRIMARY KEY NOT NULL,
    notebook_id TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    from_node   TEXT NOT NULL,
    to_node     TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL,
    confidence  REAL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE,
    FOREIGN KEY (from_node) REFERENCES entity_nodes (id) ON DELETE CASCADE,
    FOREIGN KEY (to_node) REFERENCES entity_nodes (id) ON DELETE CASCADE,
    UNIQUE (source_id, from_node, to_node, relation)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id             TEXT PRIMARY KEY NOT NULL,
    notebook_id    TEXT NOT NULL,
    entity_node_id TEXT NOT NULL,
    chunk_id       TEXT NOT NULL,
    char_start     INTEGER NOT NULL,
    char_end       INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE,
    FOREIGN KEY (entity_node_id) REFERENCES entity_nodes (id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE,
    UNIQUE (entity_node_id, chunk_id, char_start, char_end)
);

CREATE INDEX IF NOT EXISTS idx_gnodes_nb ON entity_nodes (notebook_id, name);
CREATE INDEX IF NOT EXISTS idx_gnodes_source ON entity_nodes (source_id);
CREATE INDEX IF NOT EXISTS idx_gedges_src ON entity_edges (from_node);
CREATE INDEX IF NOT EXISTS idx_gedges_tgt ON entity_edges (to_node);
CREATE INDEX IF NOT EXISTS idx_gedges_rel ON entity_edges (notebook_id, relation);
CREATE INDEX IF NOT EXISTS idx_gedges_chunk ON entity_edges (chunk_id);
CREATE INDEX IF NOT EXISTS idx_gmentions_entity ON entity_mentions (entity_node_id);
CREATE INDEX IF NOT EXISTS idx_gmentions_chunk ON entity_mentions (chunk_id);
