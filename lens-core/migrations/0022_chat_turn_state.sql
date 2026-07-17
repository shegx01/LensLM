-- Plan 2 (context management): terminal-state markers for chat turns.
-- A cancelled/errored turn now persists an assistant marker row (see chat.rs) so a
-- reload renders a "Stopped"/"Couldn't complete" line instead of a bare, dangling
-- question. `state` is NULL for normal (Done) assistant rows and all user rows;
-- `error_kind` carries the sanitized LensError kind on errored markers only.
ALTER TABLE chat_messages ADD COLUMN state TEXT;
ALTER TABLE chat_messages ADD COLUMN error_kind TEXT;
