-- M4 Phase 1 source soft-delete: adds trashed_at column to sources table.
-- Mirrors the notebook trash pattern (0001_init.sql) so sources can be moved
-- to trash (recoverable) before a permanent purge. NULL = live, non-NULL =
-- trashed (RFC3339 timestamp of when the source was trashed).
ALTER TABLE sources ADD COLUMN trashed_at TEXT;
