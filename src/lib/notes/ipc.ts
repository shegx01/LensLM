// Typed IPC wrappers for the notes Tauri commands (#24). Guarded with `isTauri()`.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { Citation } from '$lib/chat/types.js';
import type { Note } from './types.js';

/** Saves a completed assistant answer as an `origin=chat` note snapshot. */
export async function saveChatNote(
  notebookId: string,
  content: string,
  citations: Citation[] | null,
  sourceMessageId: string
): Promise<Note> {
  if (!isTauri()) throw new Error('saveChatNote: not running under Tauri');
  return invoke<Note>('save_chat_note', {
    notebookId,
    content,
    citations,
    sourceMessageId
  });
}

/** Saves a user-authored `origin=manual` note. */
export async function saveManualNote(notebookId: string, content: string): Promise<Note> {
  if (!isTauri()) throw new Error('saveManualNote: not running under Tauri');
  return invoke<Note>('save_manual_note', { notebookId, content });
}

/** Lists a notebook's notes, newest first. Returns `[]` outside Tauri. */
export async function listNotes(notebookId: string): Promise<Note[]> {
  if (!isTauri()) return [];
  return invoke<Note[]>('list_notes', { notebookId });
}

/** Deletes a note by id (drives chat toggle-unsave). */
export async function deleteNote(noteId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('delete_note', { noteId });
}
