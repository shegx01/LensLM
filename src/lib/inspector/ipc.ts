// Typed IPC wrapper for the dev/QA Embeddings Inspector Tauri command.
//
// The command is `#[cfg(debug_assertions)]`-gated on the Rust side, so a dev
// frontend against a release backend gets `isTauri()=true` but no command.
// The try/catch covers that mismatch instead of surfacing an unhandled rejection.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { InspectorResponse } from './types.js';

/** Returns an empty response outside Tauri or when the command is unavailable (dev+release mismatch). */
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
