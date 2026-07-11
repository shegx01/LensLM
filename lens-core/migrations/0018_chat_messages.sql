-- M5 #22: chat persistence. The M0 `chat_messages` skeleton (0001) had only
-- id/notebook_id/role/content/created_at; #22 adds turn grouping, the citation
-- payload, feedback, and token accounting. A "turn" is one user row plus its
-- assistant version(s) sharing `turn_id`. Only committed truth is stored — a user
-- row on send, an assistant row on stream `Done`; cancelled/errored turns write
-- nothing. `citations` is the raw JSON payload (Vec<Citation>) owned by the engine;
-- `feedback` is validated by the Rust ChatFeedback enum (no SQL CHECK, per
-- convention). The notebook-delete cascade is already declared on the 0001 FK.
--
-- `turn_id NOT NULL DEFAULT ''`: SQLite requires a default to add a NOT NULL column;
-- the table has no pre-existing rows (chat was never persisted before #22), so the
-- default is never actually applied to real data.
ALTER TABLE chat_messages ADD COLUMN turn_id TEXT NOT NULL DEFAULT '';
ALTER TABLE chat_messages ADD COLUMN citations TEXT;
ALTER TABLE chat_messages ADD COLUMN feedback TEXT;
ALTER TABLE chat_messages ADD COLUMN tokens_used INTEGER;

CREATE INDEX IF NOT EXISTS idx_chat_messages_nb_created ON chat_messages (notebook_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_turn ON chat_messages (turn_id);
