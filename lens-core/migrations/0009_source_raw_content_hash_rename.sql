-- #100: rename file_hash -> raw_content_hash; column now hashes
-- text-paste bytes and normalized URLs too, not just file bytes.
ALTER TABLE sources RENAME COLUMN file_hash TO raw_content_hash;

-- SQLite does not cascade RENAME COLUMN into index definitions.
-- Drop the old index and recreate under the new column name.
DROP INDEX IF EXISTS idx_sources_notebook_file_hash;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_notebook_raw_content_hash
  ON sources (notebook_id, raw_content_hash)
  WHERE trashed_at IS NULL AND raw_content_hash IS NOT NULL;
