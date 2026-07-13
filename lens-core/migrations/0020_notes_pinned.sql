-- Issue #25: pin-to-top for notes. New column defaults 0 (unpinned) so
-- pre-existing rows are unpinned. ALTER ADD COLUMN is not idempotent, but sqlx
-- never replays an applied migration; only the index needs IF NOT EXISTS.
ALTER TABLE notes ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_notes_notebook_pinned ON notes (notebook_id, pinned);
