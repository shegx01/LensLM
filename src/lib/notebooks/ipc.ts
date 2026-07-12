// Typed IPC wrappers for the notebooks Tauri commands. Guarded with `isTauri()`.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { Notebook, NotebookSummary } from './types.js';

/**
 * List all non-trashed notebooks with `source_count` (JOIN+COUNT query).
 * Returns `[]` outside a Tauri host (test isolation).
 */
export async function listNotebooks(): Promise<NotebookSummary[]> {
  if (!isTauri()) return [];
  return invoke<NotebookSummary[]>('list_notebooks');
}

/** Create a new notebook. Returns the created `Notebook` row. */
export async function createNotebook(
  title: string,
  description?: string | null,
  focusMode?: string | null
): Promise<Notebook> {
  if (!isTauri()) throw new Error('createNotebook: not running under Tauri');
  return invoke<Notebook>('create_notebook', {
    title,
    description: description ?? null,
    focusMode: focusMode ?? null
  });
}

/** Rename an existing notebook. */
export async function renameNotebook(id: string, title: string): Promise<void> {
  if (!isTauri()) throw new Error('renameNotebook: not running under Tauri');
  return invoke<void>('rename_notebook', { id, title });
}

/** Soft-delete a notebook (sets `trashed_at`). Recoverable via `restoreNotebook`. */
export async function trashNotebook(id: string): Promise<void> {
  if (!isTauri()) throw new Error('trashNotebook: not running under Tauri');
  return invoke<void>('trash_notebook', { id });
}

/** Restore a trashed notebook (clears `trashed_at`). */
export async function restoreNotebook(id: string): Promise<void> {
  if (!isTauri()) throw new Error('restoreNotebook: not running under Tauri');
  return invoke<void>('restore_notebook', { id });
}

/**
 * List all trashed notebooks with `source_count`, newest-trashed first.
 * Returns `[]` outside a Tauri host (test isolation).
 */
export async function listTrashed(): Promise<NotebookSummary[]> {
  if (!isTauri()) return [];
  return invoke<NotebookSummary[]>('list_trashed');
}

/** Permanently delete a notebook and cascade its sources ("Delete forever"). */
export async function purgeNotebook(id: string): Promise<void> {
  if (!isTauri()) throw new Error('purgeNotebook: not running under Tauri');
  return invoke<void>('purge_notebook', { id });
}

/**
 * Record that the user opened/interacted with a notebook. Fire-and-forget:
 * a failed DB write must not block selection.
 */
export async function touchNotebookActivity(notebookId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('touch_notebook_activity', { notebookId });
}
