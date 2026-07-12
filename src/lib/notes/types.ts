// SYNC-CHECK: must match lens-core/src/notes.rs Note / NoteOrigin — update together.

/** `notes.origin`. Mirrors `lens-core/src/notes.rs` `NoteOrigin`. `manual` is #25's domain. */
export type NoteOrigin = 'chat' | 'manual';

/**
 * Wire shape from the notes commands. ASYMMETRY (matches chat): `citations` is a
 * raw JSON string here (JSON.parse on read); `save_chat_note` TAKES a typed
 * `Citation[] | null`.
 */
export interface Note {
  id: string;
  notebook_id: string;
  origin: NoteOrigin;
  content: string;
  /** Raw JSON `Citation[]`; `null` for uncited chat notes / manual notes. */
  citations: string | null;
  /** Frozen title of the ordinal-1 cited source; survives source rename/delete. */
  source_title: string | null;
  /** Soft toggle-linkage key to the originating `chat_messages.id` (no DB FK). */
  source_message_id: string | null;
  created_at: string;
  updated_at: string;
}
