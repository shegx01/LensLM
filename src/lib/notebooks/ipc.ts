// Typed IPC wrappers for the notebooks Tauri commands.
//
// Every function is guarded with `isTauri()` so callers work identically in
// vitest (no native backend) and the real Tauri host. Pattern mirrors the
// existing invoke<T> usage in `src/lib/config.ts` and `src/lib/onboarding/system-check.ts`.
//
// Command name convention: Tauri maps Rust snake_case fn names to camelCase JS
// automatically for `#[tauri::command]`; args are also camelCase from the TS side
// and Tauri deserialises them into snake_case Rust params.

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

/**
 * Create a new notebook. Returns the created `Notebook` row (source_count is
 * implicitly 0 at creation — no join needed).
 */
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

/**
 * Rename an existing notebook. Bumps `updated_at` on the backend.
 */
export async function renameNotebook(id: string, title: string): Promise<void> {
  if (!isTauri()) throw new Error('renameNotebook: not running under Tauri');
  return invoke<void>('rename_notebook', { id, title });
}

/**
 * Soft-delete a notebook (sets `trashed_at`). The notebook disappears from
 * `listNotebooks()` and appears in `listTrashed()`. Recoverable via `restoreNotebook`.
 */
export async function trashNotebook(id: string): Promise<void> {
  if (!isTauri()) throw new Error('trashNotebook: not running under Tauri');
  return invoke<void>('trash_notebook', { id });
}

/**
 * Restore a trashed notebook (clears `trashed_at`). The notebook returns to
 * `listNotebooks()`.
 */
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

/**
 * Permanently delete a notebook and cascade its sources. This is the ONLY hard
 * delete path — `trashNotebook` is the soft-delete. Used by "Delete forever".
 */
export async function purgeNotebook(id: string): Promise<void> {
  if (!isTauri()) throw new Error('purgeNotebook: not running under Tauri');
  return invoke<void>('purge_notebook', { id });
}

/**
 * Record that the user opened/interacted with a notebook. Fire-and-forget: a
 * failed DB write must not block the selection. Returns silently outside a
 * Tauri host (test isolation).
 */
export async function touchNotebookActivity(notebookId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('touch_notebook_activity', { notebook_id: notebookId });
}
