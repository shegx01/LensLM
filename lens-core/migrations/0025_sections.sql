-- Per-source document outline (#279): structure-aware retrieval resolves positional
-- queries ("chapter 2") against real ordinals here, not the lossy `section_path` string.
-- `char_start`/`char_end` share `chunks.char_start`'s coordinate space so a section's
-- chunks are selected by range containment. New-ingests-only; old sources have no rows.
CREATE TABLE IF NOT EXISTS sections (
    id         TEXT PRIMARY KEY NOT NULL,
    source_id  TEXT NOT NULL,
    level      INTEGER NOT NULL,
    ordinal    INTEGER NOT NULL,
    title      TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    page       INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sections_source ON sections (source_id);
