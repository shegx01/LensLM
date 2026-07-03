// Typed IPC wrappers for the M4 Phase 4b-B embedding-backend surfaces.
//
// Every function is guarded with `isTauri()` so callers (and component tests
// without a native backend) share one code path. Pattern mirrors
// `src/lib/notebooks/ipc.ts` and `src/lib/onboarding/system-check.ts`.
//
// Command-name convention: Tauri maps Rust snake_case fn names to camelCase JS
// for `#[tauri::command]`; args are passed camelCase from TS and Tauri
// deserialises them into the snake_case Rust params.

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { EmbeddingBackend } from './models.js';

// SYNC-CHECK: must match src-tauri/src/commands/notebooks.rs ReembedProgress.
//
// One progress event streamed by `set_notebook_embedding_model` while re-embedding
// a notebook's chunks under the new (notebook, backend, model, dim) coordinate.
export interface ReembedProgress {
  /** Chunks processed so far. */
  done: number;
  /** Total chunks to process. */
  total: number;
}

// SYNC-CHECK: must match src-tauri/src/stream.rs StreamEvent<T>.
//
// Adjacently-tagged (`{ type, data }`) streaming envelope. Data-carrying
// variants (`chunk`, `failed`) carry `data`; unit variants (`started`,
// `progress`, `done`) may omit it (progress carries its own fields under data).
export type StreamEvent<T> =
  | { type: 'started' }
  | { type: 'chunk'; data: T }
  | { type: 'progress'; data: { done: number; total: number | null } }
  | { type: 'done' }
  | { type: 'failed'; data: { kind: string; message: string } };

// SYNC-CHECK: must match src-tauri/src/commands/notebooks.rs EmbeddingModelInfo.
//
// The notebook's current embedding coordinate. `status` is backend-scoped:
// `"active"` when a live `embedding_index` row exists for the FULL
// (notebook, backend, model, dim) coordinate, `"none"` when not yet indexed.
export interface EmbeddingModelInfo {
  /** Canonical model id (e.g. `"nomic-embed-text-v1.5"`). */
  model_id: string;
  /** Output vector dimension (e.g. `768`). */
  dim: number;
  /** Embedding backend serving this coordinate (`"fastembed"` | `"ollama"`). */
  backend: EmbeddingBackend;
  /** `"active"` when the coordinate is live, `"none"` when not yet indexed. */
  status: 'active' | 'none';
}

/**
 * Read a notebook's current embedding model + backend + index status.
 *
 * Outside Tauri (component tests / `vite dev`) returns a benign default
 * (`status: "none"`) so the picker renders without a native backend.
 */
export async function getNotebookEmbeddingModel(notebookId: string): Promise<EmbeddingModelInfo> {
  if (!isTauri()) {
    return { model_id: '', dim: 0, backend: 'fastembed', status: 'none' };
  }
  return invoke<EmbeddingModelInfo>('get_notebook_embedding_model', { notebookId });
}

/**
 * Set a notebook's embedding model + backend and re-embed all of its sources
 * under the new coordinate, streaming {@link ReembedProgress} over a Channel.
 *
 * `onProgress(done, total)` is fed each batch's chunk counters; resolves when
 * the re-embed completes (the backend flips the active coordinate and retires
 * the old one). A `failed` event rejects the promise with its message.
 *
 * Outside Tauri this is a no-op that resolves immediately.
 */
export async function setNotebookEmbeddingModel(
  notebookId: string,
  modelId: string,
  backend: EmbeddingBackend,
  onProgress: (done: number, total: number) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<StreamEvent<ReembedProgress>>();
  return new Promise<void>((resolve, reject) => {
    channel.onmessage = (ev) => {
      if (ev.type === 'chunk') onProgress(ev.data.done, ev.data.total);
      else if (ev.type === 'done') resolve();
      else if (ev.type === 'failed') reject(new Error(ev.data.message));
    };
    invoke<void>('set_notebook_embedding_model', {
      notebookId,
      modelId,
      backend,
      onProgress: channel
    }).catch(reject);
  });
}

/**
 * The set of registry embedding-model ids whose fastembed weights are already
 * cached on disk (the fastembed-side counterpart to {@link listOllamaModels}).
 *
 * Outside Tauri returns `[]` (no local cache to probe).
 */
export async function fastembedModelsCached(): Promise<string[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<string[]>('fastembed_models_cached');
  } catch {
    return [];
  }
}

/**
 * Warm (download + cache) a fastembed model's weights so a fastembed selection
 * can pass the readiness gate up-front. There is no byte progress (fastembed
 * init is opaque); callers show an indeterminate phase spinner and await this.
 *
 * Outside Tauri this is a no-op that resolves immediately.
 */
export async function warmFastembedModel(model: string): Promise<void> {
  if (!isTauri()) return;
  await invoke<void>('warm_fastembed_model', { model });
}

/**
 * Whether the GPU (candle + Apple Metal) embedding path is active on this build —
 * `true` on Apple Silicon, `false` elsewhere (issue #91). Drives the on-device
 * provider label ("On-device · Apple GPU") and the "fastest" hint.
 *
 * Outside Tauri (component tests / `vite dev`) returns `false` — the neutral
 * "On-device" label, no native backend to query.
 */
export async function embeddingGpuActive(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    return await invoke<boolean>('embedding_gpu_active');
  } catch {
    return false;
  }
}

/**
 * The locally-pulled Ollama models at `baseUrl` via the live `/api/tags` probe.
 * Graceful by contract: an unreachable runtime yields `[]`, never an error.
 *
 * Outside Tauri returns `[]`.
 */
export async function listOllamaModels(baseUrl: string): Promise<string[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<string[]>('list_ollama_models', { base_url: baseUrl });
  } catch {
    return [];
  }
}
