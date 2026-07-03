-- Adds a nullable `last_activity_at` recency timestamp to `notebooks` (issue:
-- last-edited-notebook-default). It is the single source of truth for "most
-- recently active" ordering / cold-launch auto-open. Nullable for migration
-- safety; read paths use `COALESCE(last_activity_at, created_at)`.
ALTER TABLE notebooks ADD COLUMN last_activity_at TEXT;

-- Backfill existing rows to `created_at` (a neutral baseline). NOT
-- `MAX(created_at, updated_at)` — `updated_at` reflects admin actions
-- (rename/trash/restore), not user work, so it is the wrong recency signal here.
UPDATE notebooks SET last_activity_at = created_at;
