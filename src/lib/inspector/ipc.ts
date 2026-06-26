// Typed IPC wrapper for the dev/QA Embeddings Inspector Tauri command.
//
// Guarded with `isTauri()` so callers work identically in vitest (no native
// backend) and the real Tauri host. Pattern mirrors `src/lib/sources/ipc.ts`.
//
// Command name convention: Tauri maps the Rust snake_case fn name to camelCase
// JS automatically for `#[tauri::command]`; args are passed camelCase from the
// TS side and Tauri deserialises them into the snake_case Rust params
// (`source_id`, `notebook_id`).
//
// The command itself is `#[cfg(debug_assertions)]`-gated on the Rust side, so a
// dev frontend talking to a RELEASE backend will see `isTauri()` return true but
// the command be absent. The try/catch covers that mismatch: warn + return an
// empty response rather than surfacing an unhandled rejection.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { InspectorResponse } from './types.js';

/**
 * List the chunks + embedding stats for one source in a notebook.
 * Returns an empty response outside a Tauri host (test isolation) or when the
 * command is unavailable (dev frontend + release backend mismatch).
 */
export async function listSourceChunks(
  sourceId: string,
  notebookId: string
): Promise<InspectorResponse> {
  if (!isTauri()) return { chunks: [], stats: [] };
  try {
    return await invoke<InspectorResponse>('list_source_chunks', { sourceId, notebookId });
  } catch {
    console.warn('Inspector command unavailable — release backend?');
    return { chunks: [], stats: [] };
  }
}
