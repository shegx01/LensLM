// SYNC-CHECK: must match lens-core/src/system_check.rs — update both together.
// serde uses snake_case; `Option<CheckAction>` maps to `action: ... | null` (no None variant).

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import { updateConfig } from '$lib/config.js';
import type { TtsConfig } from '$lib/theme/types.js';

export type CheckId = 'llm_runtime' | 'embedding_model';

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

/**
 * The raw backend row shape: still includes the legacy `text_to_speech` gate
 * (lens-core/src/system_check.rs unchanged) even though onboarding no longer
 * blocks on it — TTS setup moved to Settings (#194). Filtered out below.
 */
interface RawCheckResult extends Omit<CheckResult, 'id'> {
  id: CheckId | 'text_to_speech';
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

/**
 * Run all system probes, filtering out the legacy `text_to_speech` gate (see
 * [`RawCheckResult`]). Returns `[]` outside a Tauri host.
 */
export async function runSystemCheck(): Promise<CheckResult[]> {
  if (!isTauri()) return [];
  const results = await invoke<RawCheckResult[]>('run_system_check');
  return results.filter((r): r is CheckResult => r.id !== 'text_to_speech');
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

/** TTS engine identity. Mirrors lens-core `TtsEngineId` (serde snake_case). */
export type TtsEngineId = 'orpheus' | 'qwen3_local' | 'cloud';

// SYNC-CHECK: must match lens-core/src/tts/catalog.rs EngineCatalogEntry (serde snake_case).
/** One engine in the static capability catalog — the selector's single source of truth. */
export interface TtsEngineCatalogEntry {
  id: TtsEngineId;
  platform: 'cross_platform' | 'apple_silicon';
  needs_key: boolean;
  /** Selectable on this build with the current config (Qwen needs Apple Silicon; Cloud needs a key). */
  available: boolean;
  /** Why not, when `available` is false. */
  unavailable_reason: string | null;
  /** `true` for the Cloud reserved slot (provider-defined language set). */
  multilingual: boolean;
  /** Concrete supported languages (whatlang-comparable, snake_case); empty when `multilingual`. */
  supported_languages: string[];
  preset_voices: TtsVoice[];
  model_size_bytes: number | null;
  language_capability_label: string;
  // SYNC-CHECK: authority is lens-core `TtsBackend::required_model_ids` (tts/mod.rs).
  /** Registry model ids the engine needs on disk (empty when none, e.g. Qwen/Cloud). */
  required_model_ids: string[];
}

/** The static per-engine TTS capability catalog for the Settings selector. Returns `[]` outside Tauri. */
export async function ttsEngineCatalog(): Promise<TtsEngineCatalogEntry[]> {
  if (!isTauri()) return [];
  return invoke<TtsEngineCatalogEntry[]>('tts_engine_catalog');
}

/** Whether the given TTS model artifact is already on disk (skip the download step). */
export async function ttsModelDownloaded(engine: string, model: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('tts_model_downloaded', { engine, model });
}

// SYNC-CHECK: a UI selector mapped to the wire `TtsBackend` (lens-core/src/tts/mod.rs) by
// `nextTtsConfig` — NOT the wire type itself: it maps 'qwen3' → qwen3_local and every
// Cloud kind → 'cloud'.
export type TtsProvider = 'orpheus' | 'qwen3' | 'cloud';

/**
 * Persist a TTS backend/provider selection into the current `TtsConfig` shape
 * (read-modify-write). The panel's Cloud tab is ElevenLabs-only for now; the
 * full local engine selector ships in #194.
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

/**
 * Compute the next `TtsConfig` for a provider selection. A local backend
 * deactivates cloud (the active `backend` no longer points at it) but PRESERVES
 * the saved key so switching back to Cloud doesn't lose it.
 */
export function nextTtsConfig(
  prev: TtsConfig,
  input: { provider: TtsProvider; apiKey: string }
): TtsConfig {
  if (input.provider === 'orpheus') {
    return { ...prev, version: 1, backend: 'orpheus' };
  }
  if (input.provider === 'qwen3') {
    return { ...prev, version: 1, backend: 'qwen3_local' };
  }
  return {
    version: 1,
    backend: { cloud: 'eleven_labs' },
    model: prev.model,
    cloud: { kind: 'eleven_labs', api_key: input.apiKey, base_url: '' }
  };
}
