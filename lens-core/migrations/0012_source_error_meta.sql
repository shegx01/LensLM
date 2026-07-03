-- Issue #73: persist a structured ingest failure reason on the source row.
--
-- Additive, nullable JSON column. No backfill: pre-migration error rows (and
-- crash-recovery-flipped rows) keep NULL error_meta and the UI shows a graceful
-- "no details captured" fallback. Populated by `set_source_error` on the
-- ingest `Err` arm as {kind, message, timestamp, attempt_count}; cleared to
-- NULL on a successful (re-)ingest.
ALTER TABLE sources ADD COLUMN error_meta TEXT;
