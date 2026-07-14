// SYNC-CHECK: must match lens-core/src/system_check.rs — update both together.
// serde uses snake_case; `Option<CheckAction>` maps to `action: ... | null` (no None variant).

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import { updateConfig } from '$lib/config.js';
import type { TtsConfig } from '$lib/theme/types.js';

export type CheckId = 'llm_runtime' | 'embedding_model' | 'text_to_speech';

export type CheckStatus = 'pass' | 'fail';

export type CheckAction = 'configure' | 'choose';

/** One row in the system-check screen. Frozen IPC contract — see header. */
export interface CheckResult {
  id: CheckId;
  label: string;
  status: CheckStatus;
  detail: string;
  action: CheckAction | null;
}

// SYNC-CHECK: must match lens-core/src/system_check.rs LlmDetection
// `reachable` is the primary gate; `version`/`models` are best-effort (may be null/empty).
export interface LlmDetection {
  reachable: boolean;
  version: string | null;
  models: string[];
}

/** Probe an OpenAI-compatible local LLM endpoint. Returns empty result outside Tauri. */
export async function detectLlm(baseUrl: string): Promise<LlmDetection> {
  if (!isTauri()) return { reachable: false, version: null, models: [] };
  return invoke<LlmDetection>('detect_llm', { base_url: baseUrl });
}

/** Run all system probes. Returns `[]` outside a Tauri host. */
export async function runSystemCheck(): Promise<CheckResult[]> {
  if (!isTauri()) return [];
  return invoke<CheckResult[]>('run_system_check');
}

// SYNC-CHECK: contract — Rust side to implement invoke('list_tts_voices')
export interface TtsVoice {
  id: string;
  name: string;
  gender: 'male' | 'female';
}

// SYNC-CHECK: must match lens-core/src/embedding.rs InstallProgress
// `completed`/`total` are per-layer byte counters; absent on status-only events.
export interface InstallProgress {
  status: string;
  completed: number | null;
  total: number | null;
}

// SYNC-CHECK: must match lens-core/src/tts/mod.rs DownloadProgress
// `done` flips true on the final event (incl. already-present fast path).
export interface DownloadProgress {
  received: number;
  total: number | null;
  done: boolean;
}

// SYNC-CHECK: catalog lives in `$lib/embeddings/models` — re-exported here for backward compat.
// Import before re-export (import-before-export ordering required by the bundler).
import type { EmbeddingModelId } from '$lib/embeddings/models.js';

export type { EmbeddingModelId, EmbeddingModelSpec } from '$lib/embeddings/models.js';
export { EMBEDDING_MODELS } from '$lib/embeddings/models.js';

/** Clamp a 0..1 ratio to an integer 0..100 percentage. */
function toPct(completed: number | null, total: number | null): number | null {
  if (completed === null || total === null || total <= 0) return null;
  return Math.min(100, Math.max(0, Math.round((completed / total) * 100)));
}

/**
 * Install an embedding model via `install_embedding_model`, streaming 0–100% progress.
 * `pct` holds at the last known value on status-only events. No-op outside Tauri.
 */
export async function installEmbeddingModel(
  model: EmbeddingModelId,
  onProgress: (pct: number) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<InstallProgress>();
  let lastPct = 0;
  channel.onmessage = (p) => {
    const pct = toPct(p.completed, p.total);
    if (pct !== null) lastPct = pct;
    onProgress(lastPct);
  };
  await invoke<void>('install_embedding_model', { model, onProgress: channel });
}

/**
 * Download a TTS model artifact (registry id, e.g. `"orpheus"`/`"snac"`) for the given
 * engine, streaming 0–100% progress. When total is unknown, `done` is surfaced as 100%.
 * No-op outside Tauri. Mirrors `download_whisper_model`.
 */
export async function downloadTtsModel(
  engine: string,
  model: string,
  onProgress: (pct: number) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<DownloadProgress>();
  channel.onmessage = (p) => {
    const pct = toPct(p.received, p.total);
    if (pct !== null) onProgress(pct);
    else if (p.done) onProgress(100);
  };
  await invoke<void>('download_tts_model', { engine, model, onProgress: channel });
}

/** List the active backend's named-voice catalog. Adapter-driven — empty when no provider resolves. */
export async function listTtsVoices(): Promise<TtsVoice[]> {
  if (!isTauri()) return [];
  return invoke<TtsVoice[]>('list_tts_voices');
}

/** Whether the given TTS model artifact is already on disk (skip the download step). */
export async function ttsModelDownloaded(engine: string, model: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('tts_model_downloaded', { engine, model });
}

// SYNC-CHECK: a UI selector mapped to the wire `TtsBackend` (lens-core/src/tts/mod.rs) by
// `nextTtsConfig` — NOT the wire type itself: it collapses moss_local/moss_ttsd → 'moss'
// and every Cloud kind → 'cloud'.
export type TtsProvider = 'orpheus' | 'moss' | 'cloud';

/**
 * Persist a TTS backend/provider selection into the current `TtsConfig` shape
 * (read-modify-write). The panel's Cloud tab is ElevenLabs-only for now; full
 * multi-backend/provider selection is deferred to #194.
 */
export async function saveTtsProvider(input: {
  provider: TtsProvider;
  apiKey: string;
}): Promise<void> {
  await updateConfig((cfg) => ({
    ...cfg,
    tts: nextTtsConfig(cfg.tts, input)
  }));
}

function nextTtsConfig(
  prev: TtsConfig,
  input: { provider: TtsProvider; apiKey: string }
): TtsConfig {
  // Local backends clear any lingering cloud config; the full local picker is deferred to #194.
  if (input.provider === 'orpheus') {
    return { ...prev, version: 1, backend: 'orpheus', cloud: null };
  }
  if (input.provider === 'moss') {
    return { ...prev, version: 1, backend: 'moss_local', cloud: null };
  }
  return {
    version: 1,
    backend: { cloud: 'eleven_labs' },
    model: prev.model,
    cloud: { kind: 'eleven_labs', api_key: input.apiKey, base_url: '' }
  };
}
