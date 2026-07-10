-- M13 #158a: entity-graph eval harness + opt-in gate. Adds the per-notebook
-- graph-retrieval override, the synthetic-QA store, and the ablation log. The
-- eval only writes evidence here; it never flips the flag (that is the user, in
-- #158b). Gold chunk ids are generation-provenance (LLM-emitted from the fed corpus)
-- or hand-authored fixture markers — independent of both retrievers.

-- Per-notebook override of the app-wide RetrievalConfig.graph_retrieval_enabled.
-- NULL = inherit the app default. Not consumed by live retrieval until #21.
ALTER TABLE notebooks ADD COLUMN graph_retrieval_enabled INTEGER;

-- Synthetic QA generated from a notebook's RAW source text. `kind` is validated
-- by the Rust QuestionKind enum (no SQL CHECK, per convention); `seed_entities`
-- and `gold_chunk_ids` are JSON arrays. `prompt_version` gives question→prompt
-- lineage. Cascades on notebook delete.
CREATE TABLE IF NOT EXISTS eval_questions (
    id             TEXT PRIMARY KEY,
    notebook_id    TEXT NOT NULL,
    kind           TEXT NOT NULL,
    question       TEXT NOT NULL,
    seed_entities  TEXT NOT NULL,
    gold_chunk_ids TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_questions_nb ON eval_questions (notebook_id);

-- One ablation row per eval run. `graph_enabled` is the effective flag value at
-- run time recorded purely as context for #158b (the eval always runs both arms
-- regardless of the flag). `delta_pp` is the graph−hybrid recall@5 gap on the
-- bridging+rollup subset; `dropped_n` counts empty-gold questions excluded from
-- the scored sample. `passed` = delta_pp >= 5 AND p95_ms < 500. Cascades on
-- notebook delete.
CREATE TABLE IF NOT EXISTS notebook_eval_log (
    id             TEXT PRIMARY KEY,
    notebook_id    TEXT NOT NULL,
    ran_at         TEXT NOT NULL,
    graph_recall   REAL NOT NULL,
    hybrid_recall  REAL NOT NULL,
    delta_pp       REAL NOT NULL,
    p95_ms         REAL NOT NULL,
    passed         INTEGER NOT NULL,
    sample_n       INTEGER NOT NULL,
    dropped_n      INTEGER NOT NULL,
    graph_enabled  INTEGER NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (notebook_id) REFERENCES notebooks (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notebook_eval_log_nb ON notebook_eval_log (notebook_id);
