-- M5 #22: chat persistence (see chat.rs for turn/citations/feedback semantics).
-- `turn_id NOT NULL DEFAULT ''`: SQLite requires a default to add a NOT NULL column;
-- the table has no pre-existing rows, so the default is never applied to real data.
ALTER TABLE chat_messages ADD COLUMN turn_id TEXT NOT NULL DEFAULT '';
ALTER TABLE chat_messages ADD COLUMN citations TEXT;
ALTER TABLE chat_messages ADD COLUMN feedback TEXT;
ALTER TABLE chat_messages ADD COLUMN tokens_used INTEGER;

CREATE INDEX IF NOT EXISTS idx_chat_messages_nb_created ON chat_messages (notebook_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_turn ON chat_messages (turn_id);
