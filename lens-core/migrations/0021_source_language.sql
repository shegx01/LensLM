-- Issue #194 [161f]: per-source detected language for the engine-aware TTS guard.
--
-- Additive, nullable column. No backfill: populated at index time with the
-- whatlang ISO 639-3 code (e.g. `eng`, `cmn`) when detection is reliable; NULL
-- for pre-migration rows and for text where detection is unreliable/undetermined.
-- SQLite `ADD COLUMN` has no `IF NOT EXISTS`; re-run safety comes from sqlx
-- version tracking (mirrors 0012_source_error_meta.sql).
ALTER TABLE sources ADD COLUMN language TEXT;
