-- #96: raw-file-bytes SHA-256 for add-time dedup.
-- NULL for text/paste/url sources and pre-migration rows.
ALTER TABLE sources ADD COLUMN file_hash TEXT;

-- Partial unique index: at most one live (non-trashed) source per
-- (notebook, file_hash) pair.  NULL file_hash rows are excluded
-- (SQLite: NULLs are always distinct in unique indexes, but the
-- explicit WHERE makes intent clear and skips text/url sources).
CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_notebook_file_hash
  ON sources (notebook_id, file_hash)
  WHERE trashed_at IS NULL AND file_hash IS NOT NULL;
