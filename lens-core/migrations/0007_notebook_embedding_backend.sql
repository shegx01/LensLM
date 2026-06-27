-- M4 Phase 4b-B: backend dimension. ALL changes are O(1) ADD COLUMN + a
-- standalone partial-index SWAP. 0001's table-level UNIQUE(notebook_id,
-- model, dim) is ALREADY GONE — 0005 replaced it with the standalone
-- partial index `uq_embidx_active`, so we DROP that standalone index
-- (NOT the un-droppable sqlite_autoindex) and recreate it on the 4-col
-- tuple. No table rewrite, no row copy.
ALTER TABLE notebooks       ADD COLUMN embedding_backend TEXT NOT NULL DEFAULT 'fastembed';
ALTER TABLE embedding_index ADD COLUMN backend           TEXT NOT NULL DEFAULT 'fastembed';
DROP INDEX uq_embidx_active;
CREATE UNIQUE INDEX uq_embidx_active
    ON embedding_index (notebook_id, backend, model, dim) WHERE status='active';
