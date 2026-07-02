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
import type {
  Source,
  TrashedSource,
  IngestProgress,
  StreamEvent,
  AddSourceOutcome
} from './types.js';

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
 *
 * Returns an `AddSourceOutcome` (`{ source, wasExisting }`): on a content-dedup
 * hit (#100) the existing live source is returned with `wasExisting = true` and
 * no new row is written. Mirrors the Rust `AddSourceOutcome` (serde camelCase).
 */
export async function addTextSource(
  notebookId: string,
  title: string,
  text: string,
  kind: string
): Promise<AddSourceOutcome> {
  if (!isTauri()) throw new Error('addTextSource: not running under Tauri');
  return invoke<AddSourceOutcome>('add_text_source', {
    notebookId,
    title,
    text,
    kind
  });
}

/**
 * Add a file-backed source to a notebook.
 *
 * Routes to `add_file_source`, which copies the file into managed storage and
 * detects the source `kind` from the file EXTENSION (.md/.txt/.pdf/.docx/json/
 * jsonl/yaml/xml); an unsupported extension is rejected. (The older `add_source`
 * command recorded a generic `kind="file"` that the ingest pipeline rejects.)
 *
 * Returns an `AddSourceOutcome` (`{ source, wasExisting }`): on a content-dedup
 * hit (#96) the existing live source is returned with `wasExisting = true` and
 * no new row is written. Mirrors the Rust `AddSourceOutcome` (serde camelCase).
 */
export async function addFileSource(
  notebookId: string,
  title: string,
  path: string
): Promise<AddSourceOutcome> {
  if (!isTauri()) throw new Error('addFileSource: not running under Tauri');
  return invoke<AddSourceOutcome>('add_file_source', {
    notebookId,
    path,
    title
  });
}

/**
 * Add a URL-backed source to a notebook.
 *
 * Routes to `add_url_source`, which inserts a `queued` row whose `locator` is
 * the verbatim URL (no fetch happens here — `ingestSource` does that later).
 *
 * Returns an `AddSourceOutcome` (`{ source, wasExisting }`): on a content-dedup
 * hit (#100, keyed on the moderately-normalized URL) the existing live source is
 * returned with `wasExisting = true` and no new row is written. Mirrors the Rust
 * `AddSourceOutcome` (serde camelCase).
 *
 * `forceJsRender` (#78) persists the per-source "SPA / render this page" opt-in;
 * when `true`, ingest ALWAYS routes the source through the JS-render path instead
 * of relying on static-extraction auto-detection. Sent camelCase — Tauri maps it
 * to the snake_case `force_js_render` command arg.
 */
export async function addUrlSource(
  notebookId: string,
  title: string,
  url: string,
  forceJsRender: boolean = false
): Promise<AddSourceOutcome> {
  if (!isTauri()) throw new Error('addUrlSource: not running under Tauri');
  return invoke<AddSourceOutcome>('add_url_source', {
    notebookId,
    title,
    url,
    forceJsRender
  });
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

/**
 * Move a source to the trash (soft-delete).
 * Returns outside a Tauri host (test isolation — callers rely on the store's optimistic remove).
 */
export async function trashSource(sourceId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('trash_source', { sourceId });
}

/**
 * Restore a previously-trashed source by id.
 * Returns outside a Tauri host (test isolation).
 */
export async function restoreSource(sourceId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('restore_source', { sourceId });
}

/**
 * List all individually-trashed sources (whose parent notebook is live).
 * Returns `[]` outside a Tauri host (test isolation).
 */
export async function listTrashedSources(): Promise<TrashedSource[]> {
  if (!isTauri()) return [];
  // Coerce a null/undefined response to [] so a misbehaving host can never
  // poison `trashedSources` (a null there makes `trashCount`'s `.length` throw).
  return (await invoke<TrashedSource[] | null>('list_trashed_sources')) ?? [];
}

/**
 * Permanently delete a trashed source and its Lance vectors.
 * Returns outside a Tauri host (test isolation).
 */
export async function purgeSource(sourceId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('purge_source', { sourceId });
}
