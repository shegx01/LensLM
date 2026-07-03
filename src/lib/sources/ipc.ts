// Typed IPC wrappers for the sources Tauri commands. All guards with `isTauri()`.

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

/** Add a pasted-text or Markdown source. Returns `{ source, wasExisting }` — dedup hits return `wasExisting = true`. */
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
 * Add a file-backed source. Detects `kind` from extension; unsupported extensions are rejected.
 * Returns `{ source, wasExisting }` — dedup hits return `wasExisting = true`.
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
 * Add a URL-backed source (inserts a `queued` row; fetch happens later in `ingestSource`).
 * Returns `{ source, wasExisting }`. `forceJsRender` always routes through the JS-render path.
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

/** Ingest a source, streaming `StreamEvent<IngestProgress>`. Resolves after `done` or `failed`. */
export async function ingestSource(
  sourceId: string,
  onProgress: (e: StreamEvent<IngestProgress>) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<StreamEvent<IngestProgress>>();
  channel.onmessage = onProgress;
  await invoke<void>('ingest_source', { sourceId, onProgress: channel });
}

/** Toggle source selection. IPC arg is bool; `Source.selected` stays as `number` (INTEGER column mirror). */
export async function setSourceSelected(sourceId: string, selected: boolean): Promise<void> {
  if (!isTauri()) throw new Error('setSourceSelected: not running under Tauri');
  return invoke<void>('set_source_selected', { sourceId, selected });
}

/** Soft-delete a source (move to trash). No-op outside Tauri; callers rely on optimistic store removal. */
export async function trashSource(sourceId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('trash_source', { sourceId });
}

/** Restore a previously-trashed source. No-op outside Tauri. */
export async function restoreSource(sourceId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('restore_source', { sourceId });
}

/** List individually-trashed sources (parent notebook is live). Returns `[]` outside Tauri. */
export async function listTrashedSources(): Promise<TrashedSource[]> {
  if (!isTauri()) return [];
  // Coerce null to [] — a null `trashedSources` makes `trashCount`'s `.length` throw.
  return (await invoke<TrashedSource[] | null>('list_trashed_sources')) ?? [];
}

/** Permanently delete a trashed source and its Lance vectors. No-op outside Tauri. */
export async function purgeSource(sourceId: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>('purge_source', { sourceId });
}

/** Retry a single errored source (must be non-trashed `error` status). Resolves after `done` or `failed`. */
export async function retryIngestSource(
  sourceId: string,
  onProgress: (e: StreamEvent<IngestProgress>) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<StreamEvent<IngestProgress>>();
  channel.onmessage = onProgress;
  await invoke<void>('retry_ingest_source', { sourceId, onProgress: channel });
}

/** Retry all errored non-trashed sources in a notebook. Continues on failure; each failure updates `error_meta`. */
export async function retryAllFailedSources(
  notebookId: string,
  onProgress: (e: StreamEvent<IngestProgress>) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<StreamEvent<IngestProgress>>();
  channel.onmessage = onProgress;
  await invoke<void>('retry_all_failed_sources', { notebookId, onProgress: channel });
}
