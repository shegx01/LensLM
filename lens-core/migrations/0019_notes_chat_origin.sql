-- M6 #24: extend the pre-existing `notes` table (0001:75-83) for origin=chat snapshots.
-- No FK on `source_message_id`: SQLite ALTER TABLE cannot add one; it is a soft
-- toggle-linkage key back to the originating chat_messages row.
ALTER TABLE notes ADD COLUMN citations         TEXT;
ALTER TABLE notes ADD COLUMN source_title      TEXT;
ALTER TABLE notes ADD COLUMN source_message_id TEXT;

CREATE INDEX IF NOT EXISTS idx_notes_src_msg ON notes (source_message_id);
