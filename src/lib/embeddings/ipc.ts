// Typed IPC wrappers for the embedding-backend surfaces. Guarded with `isTauri()`.

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { EmbeddingBackend } from './models.js';

// SYNC-CHECK: must match src-tauri/src/commands/notebooks.rs ReembedProgress.
export interface ReembedProgress {
  /** Chunks processed so far. */
  done: number;
  /** Total chunks to process. */
  total: number;
}

// SYNC-CHECK: must match src-tauri/src/stream.rs StreamEvent<T>.
export type StreamEvent<T> =
  | { type: 'started' }
  | { type: 'chunk'; data: T }
  | { type: 'progress'; data: { done: number; total: number | null } }
  | { type: 'done' }
  | { type: 'failed'; data: { kind: string; message: string } };

// SYNC-CHECK: must match src-tauri/src/commands/notebooks.rs EmbeddingModelInfo.
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

// SYNC-CHECK: must match src-tauri/src/commands/notebooks.rs EvalReportDto.
export interface EvalReportDto {
  graph_recall: number;
  hybrid_recall: number;
  delta_pp: number;
  p95_ms: number;
  passed: boolean;
  sample_n: number;
  dropped_n: number;
  graph_enabled: boolean;
  prompt_version: string;
  ran_at: string;
}

// SYNC-CHECK: must match src-tauri/src/commands/notebooks.rs EvalOutcomeDto.
export type EvalOutcomeDto =
  | { status: 'skipped'; reason: string }
  | { status: 'ran'; report: EvalReportDto };

// SYNC-CHECK: must match src-tauri/src/commands/notebooks.rs EvalPhaseDto.
export type EvalPhaseDto = 'generating_qa' | 'done';

/** Outside Tauri returns `{ status: "none" }` so the picker renders without a native backend. */
export async function getNotebookEmbeddingModel(notebookId: string): Promise<EmbeddingModelInfo> {
  if (!isTauri()) {
    return { model_id: '', dim: 0, backend: 'fastembed', status: 'none' };
  }
  return invoke<EmbeddingModelInfo>('get_notebook_embedding_model', { notebookId });
}

/**
 * Re-embed all sources under the new (model, backend) coordinate, streaming
 * {@link ReembedProgress}. A `failed` event rejects with its message.
 * No-op outside Tauri.
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

/** Fastembed model ids already cached on disk. Returns `[]` outside Tauri. */
export async function fastembedModelsCached(): Promise<string[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<string[]>('fastembed_models_cached');
  } catch {
    return [];
  }
}

/**
 * Download + cache a fastembed model's weights. No byte progress (fastembed init
 * is opaque); callers show an indeterminate spinner. No-op outside Tauri.
 */
export async function warmFastembedModel(model: string): Promise<void> {
  if (!isTauri()) return;
  await invoke<void>('warm_fastembed_model', { model });
}

/**
 * Model ids running on Apple GPU (candle + Metal) — `["nomic-embed-text-v1.5"]`
 * on Apple Silicon, `[]` elsewhere (issue #91). Returns `[]` outside Tauri.
 */
export async function gpuAcceleratedModels(): Promise<string[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<string[]>('gpu_accelerated_models');
  } catch {
    return [];
  }
}

/** Locally-pulled Ollama models at `baseUrl`. Unreachable runtime yields `[]`. */
export async function listOllamaModels(baseUrl: string): Promise<string[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<string[]>('list_ollama_models', { base_url: baseUrl });
  } catch {
    return [];
  }
}

/** Effective per-notebook graph-retrieval setting (override, else global default).
 *  Returns `false` outside Tauri. */
export async function getNotebookGraphRetrievalEnabled(notebookId: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('get_notebook_graph_retrieval_enabled', { notebookId });
}

/** Persists the binary per-notebook graph-retrieval override. No-op outside Tauri. */
export async function setNotebookGraphRetrievalEnabled(
  notebookId: string,
  enabled: boolean
): Promise<void> {
  if (!isTauri()) return;
  await invoke<void>('set_notebook_graph_retrieval_enabled', { notebookId, enabled });
}

/** Latest eval verdict for a notebook, or `null` if it has never run (or outside Tauri). */
export async function latestNotebookEval(notebookId: string): Promise<EvalReportDto | null> {
  if (!isTauri()) return null;
  return invoke<EvalReportDto | null>('latest_notebook_eval', { notebookId });
}

/**
 * Runs the graph-retrieval eval on demand, streaming {@link EvalPhaseDto} phases and
 * resolving with the {@link EvalOutcomeDto}. A `failed` event rejects with its message
 * (missing/unreachable provider surfaces here). No-op resolve outside Tauri.
 */
export async function runNotebookGraphEval(
  notebookId: string,
  onPhase: (phase: EvalPhaseDto) => void
): Promise<EvalOutcomeDto> {
  if (!isTauri()) return { status: 'skipped', reason: 'not running in Tauri' };
  const channel = new Channel<StreamEvent<EvalPhaseDto>>();
  return new Promise<EvalOutcomeDto>((resolve, reject) => {
    channel.onmessage = (ev) => {
      if (ev.type === 'chunk') onPhase(ev.data);
      else if (ev.type === 'failed') {
        const e = new Error(ev.data.message);
        (e as Error & { kind?: string }).kind = ev.data.kind;
        reject(e);
      }
    };
    invoke<EvalOutcomeDto>('run_notebook_graph_eval', { notebookId, onEvent: channel })
      .then(resolve)
      .catch(reject);
  });
}
