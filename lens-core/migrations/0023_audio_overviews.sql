-- Per-notebook Audio Overview status (#29). One row per notebook (regenerate
-- overwrites via UPSERT). See lens-core/src/audio_overview.rs for the status model.
CREATE TABLE IF NOT EXISTS audio_overviews (
    notebook_id     TEXT PRIMARY KEY REFERENCES notebooks(id) ON DELETE CASCADE,
    path            TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    status          TEXT NOT NULL,
    source_set_hash TEXT NOT NULL
);
