// Typed IPC wrappers for the sources Tauri commands.
//
// Every function is guarded with `isTauri()` so callers work identically in
// vitest (no native backend) and the real Tauri host. Pattern mirrors
// `src/lib/notebooks/ipc.ts`.
//
// Command name convention: Tauri maps Rust snake_case fn names to camelCase JS
// automatically for `#[tauri::command]`; args are also camelCase from the TS side
// and Tauri deserialises them into snake_case Rust params.

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { Source, IngestProgress, StreamEvent } from './types.js';

/**
 * List all sources for a notebook.
 * Returns `[]` outside a Tauri host (test isolation).
 */
export async function listSources(notebookId: string): Promise<Source[]> {
  if (!isTauri()) return [];
  return invoke<Source[]>('list_sources', { notebookId });
}

/**
 * Add a pasted-text or Markdown source to a notebook.
 * Returns the created `Source` row.
 */
export async function addTextSource(
  notebookId: string,
  title: string,
  text: string,
  kind: string
): Promise<Source> {
  if (!isTauri()) throw new Error('addTextSource: not running under Tauri');
  return invoke<Source>('add_text_source', { notebookId, title, text, kind });
}

/**
 * Add a file-backed source to a notebook.
 * Returns the created `Source` row.
 */
export async function addFileSource(
  notebookId: string,
  title: string,
  path: string
): Promise<Source> {
  if (!isTauri()) throw new Error('addFileSource: not running under Tauri');
  return invoke<Source>('add_source', { notebookId, title, locator: path });
}

/**
 * Ingest a source, streaming progress events via a Channel.
 * `onProgress` receives each `StreamEvent<IngestProgress>` as it arrives.
 *
 * The channel is created internally; `onProgress` is called for every message.
 * The returned Promise resolves when the command completes (after `done` or `failed`).
 */
export async function ingestSource(
  sourceId: string,
  onProgress: (e: StreamEvent<IngestProgress>) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<StreamEvent<IngestProgress>>();
  channel.onmessage = onProgress;
  await invoke<void>('ingest_source', { sourceId, onProgress: channel });
}

/**
 * Toggle source selection. `selected` is a boolean (Rust serde expects JSON bool).
 * `Source.selected` stays as `number` (INTEGER column mirror) — only this IPC arg is bool.
 * Persisted to the backend so the state survives session restarts.
 */
export async function setSourceSelected(sourceId: string, selected: boolean): Promise<void> {
  if (!isTauri()) throw new Error('setSourceSelected: not running under Tauri');
  return invoke<void>('set_source_selected', { sourceId, selected });
}
