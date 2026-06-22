-- M1 Onboarding "Create notebook" personalization fields (additive).
-- Adds two nullable columns to `notebooks` captured during first-run onboarding:
--   * description — optional free-text blurb for the notebook.
--   * focus_mode  — 'research' (default in UI) | 'coding' | 'notes'.
-- Both are NULLABLE and write-only in M1 (persisted for M3 to consume; there is
-- no read/edit surface yet). M3 extends these columns; it does NOT migrate them
-- away. SQLite `ADD COLUMN` of a nullable column with no default is a safe,
-- O(1) metadata-only operation (no table rewrite).
--
-- NOTE: SQLite has no `ADD COLUMN IF NOT EXISTS`; idempotency here rests on
-- sqlx's migration ledger (`_sqlx_migrations`) never re-running an applied file,
-- consistent with the one-file-one-atomic-unit convention in 0001_init.sql.

ALTER TABLE notebooks ADD COLUMN description TEXT;
ALTER TABLE notebooks ADD COLUMN focus_mode TEXT;
