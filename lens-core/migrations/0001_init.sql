-- M0 Foundation schema (issue #4).
-- Scope: M0-M3 tables only. Feature tables (tts_voice -> M2, audio_overview -> M7)
-- arrive as additive migrations in their owning milestones. There is NO
-- app_config table: AppConfig is disk-only (config.json).
--
-- Conventions:
--   * All primary keys are TEXT (UUIDv7 generated app-side, bound as String).
--   * JSON-shaped columns are TEXT holding a JSON document.
--   * Child tables CASCADE on notebook delete (foreign_keys=ON is set on the pool).
--   * DDL is idempotent (IF NOT EXISTS) as defense-in-depth; sqlx wraps each
--     migration file in a transaction (one file = one atomic unit).

CREATE TABLE IF NOT EXISTS notebooks (
    id         TEXT PRIMARY KEY NOT NULL,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    trashed_at TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY NOT NULL,
    notebook_id  TEXT NOT NULL,
    kind         TEXT NOT NULL,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL,
    locator      TEXT NOT NULL,
    selected     INTEGER NOT NULL DEFAULT 1,
    token_count  INTEGER,
    content_hash TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
);

-- Adjacency-list chunk hierarchy: one table holds 512-token parents and
-- 128-token children; a child row points at its parent via parent_id.
-- `text` is the canonical, immutable citation text. `enrichment` is NULL until
-- the M4 enrichment pass writes a JSON object. There is NO embedding_ref column
-- (the LanceDB->SQLite link is keyed by chunk id on the LanceDB side).
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY NOT NULL,
    source_id    TEXT NOT NULL,
    parent_id    TEXT,
    kind         TEXT NOT NULL,
    level        INTEGER NOT NULL,
    section_path TEXT NOT NULL,
    text         TEXT NOT NULL,
    token_start  INTEGER,
    token_end    INTEGER,
    page         INTEGER,
    char_start   INTEGER,
    char_end     INTEGER,
    block_type   TEXT,
    enrichment   TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES chunks (id) ON DELETE CASCADE
);

-- Per-(notebook, model, dim) registry mapping to a future LanceDB table name.
-- Model-switch flow becomes an UPDATE of lance_table_name + status on one row.
CREATE TABLE IF NOT EXISTS embedding_index (
    id                TEXT PRIMARY KEY NOT NULL,
    notebook_id       TEXT NOT NULL,
    model             TEXT NOT NULL,
    dim               INTEGER NOT NULL,
    prefix_convention TEXT NOT NULL,
    lance_table_name  TEXT NOT NULL,
    status            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE,
    UNIQUE (notebook_id, model, dim)
);

CREATE TABLE IF NOT EXISTS notes (
    id          TEXT PRIMARY KEY NOT NULL,
    notebook_id TEXT NOT NULL,
    content     TEXT NOT NULL,
    origin      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY NOT NULL,
    notebook_id TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS citations (
    id         TEXT PRIMARY KEY NOT NULL,
    message_id TEXT NOT NULL,
    source_id  TEXT NOT NULL,
    chunk_id   TEXT,
    locator    TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    FOREIGN KEY (message_id) REFERENCES chat_messages (id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sources_notebook ON sources (notebook_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks (parent_id);
CREATE INDEX IF NOT EXISTS idx_messages_notebook ON chat_messages (notebook_id);
CREATE INDEX IF NOT EXISTS idx_notes_notebook ON notes (notebook_id);
CREATE INDEX IF NOT EXISTS idx_embidx_notebook ON embedding_index (notebook_id);

-- Citations are looked up by message (render a message's citations) and must be
-- found fast when a parent source/chunk cascade-deletes; index every FK column.
CREATE INDEX IF NOT EXISTS idx_citations_message ON citations (message_id);
CREATE INDEX IF NOT EXISTS idx_citations_source ON citations (source_id);
CREATE INDEX IF NOT EXISTS idx_citations_chunk ON citations (chunk_id);
