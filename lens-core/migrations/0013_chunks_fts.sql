-- M5 hybrid search (issue #39): BM25 lexical index over chunk text via SQLite FTS5.
-- `chunk_id UNINDEXED` stores the TEXT UUID link back to `chunks.id` without an
-- implicit-rowid dependency (external-content FTS keys off the mutable rowid; this
-- explicit-content table does not). `text` is the sole indexed column.
-- Idempotent: virtual table + triggers are guarded so a re-run is a no-op.
--
-- FK-CASCADE CAVEAT: SQLite AFTER DELETE triggers do NOT fire on FK-cascade deletes
-- (recursive_triggers is off by default), so a notebook/source purge that cascades
-- `chunks` can leave orphan `chunks_fts` rows. This is correctness-safe because the
-- BM25 query INNER-JOINs `chunks` (orphans are invisible); `purge_source` also deletes
-- matching rows for hygiene, and `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')`
-- is the escape hatch.

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    chunk_id UNINDEXED,
    text
);

-- Backfill any pre-existing chunks (idempotent: INSERT OR IGNORE would still
-- duplicate on an FTS table which has no unique constraint, so guard on emptiness).
INSERT INTO chunks_fts (chunk_id, text)
SELECT id, text FROM chunks
WHERE (SELECT COUNT(*) FROM chunks_fts) = 0;

DROP TRIGGER IF EXISTS chunks_fts_ai;
CREATE TRIGGER chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (chunk_id, text) VALUES (new.id, new.text);
END;

DROP TRIGGER IF EXISTS chunks_fts_au;
CREATE TRIGGER chunks_fts_au AFTER UPDATE OF text ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
    INSERT INTO chunks_fts (chunk_id, text) VALUES (new.id, new.text);
END;

DROP TRIGGER IF EXISTS chunks_fts_ad;
CREATE TRIGGER chunks_fts_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
END;
