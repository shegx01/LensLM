// SYNC-CHECK: must match lens-core/src/system_check.rs — update both together.
// serde uses snake_case; `Option<CheckAction>` maps to `action: ... | null` (no None variant).

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import { updateConfig } from '$lib/config.js';
import type { AppConfig, TtsConfig } from '$lib/theme/types.js';

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

/** The live save handle a step's picker hands up to drive its footer's Save & continue. */
export type SaveApi = { save: () => Promise<void> };

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
 * Build the shared `DownloadProgress` channel for the download/prepare wrappers below.
 * `done` always reports 100 (even on a rounding-short determinate finish); otherwise
 * `toPct` reports the known percentage or `null` when the total is unknown — callers
 * render `null` as an indeterminate bar rather than holding at a stale value.
 */
function makeProgressChannel(onProgress: (pct: number | null) => void): Channel<DownloadProgress> {
  const channel = new Channel<DownloadProgress>();
  channel.onmessage = (p) => {
    if (p.done) {
      onProgress(100);
      return;
    }
    onProgress(toPct(p.received, p.total));
  };
  return channel;
}

/**
 * Download a TTS model artifact (registry id, e.g. `"orpheus"`/`"snac"`) for the given
 * engine, streaming 0–100% progress (`null` while the total is unknown). No-op outside
 * Tauri. Mirrors `download_whisper_model`.
 */
export async function downloadTtsModel(
  engine: string,
  model: string,
  onProgress: (pct: number | null) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = makeProgressChannel(onProgress);
  await invoke<void>('download_tts_model', { engine, model, onProgress: channel });
}

/**
 * Prepare (download) the Qwen3-TTS MLX model, streaming 0–100% progress (`null` while
 * the total is unknown). Apple-Silicon only — the backing command is absent elsewhere.
 * No-op outside Tauri. Mirrors `downloadTtsModel`.
 */
export async function prepareQwenModel(onProgress: (pct: number | null) => void): Promise<void> {
  if (!isTauri()) return;
  const channel = makeProgressChannel(onProgress);
  await invoke<void>('prepare_qwen_model', { onProgress: channel });
}

/**
 * Cancel an in-flight Qwen prepare/download (`cancel_prepare`, macOS-aarch64-only —
 * cfg-gated out on other targets). Swallows an unregistered-command/invoke error so
 * callers (e.g. an unmount handler) can call this unconditionally as a defensive no-op.
 */
export async function cancelPrepare(): Promise<void> {
  if (!isTauri()) return;
  try {
    await invoke<boolean>('cancel_prepare');
  } catch {
    // Command absent on this platform, or nothing was in flight — both are no-ops.
  }
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

// SYNC-CHECK: must match src-tauri/src/commands/system.rs TtsModelStatus (serde snake_case).
/** Tri-state download status of a TTS model artifact. */
export type TtsModelStatus = 'complete' | 'partial' | 'absent';

/** Download status of the given TTS model artifact — drives the download/re-download affordance. */
export async function ttsModelStatus(engine: string, model: string): Promise<TtsModelStatus> {
  if (!isTauri()) return 'absent';
  return invoke<TtsModelStatus>('tts_model_status', { engine, model });
}

/**
 * Whether the configured TTS backend can synthesize — reused to gate #29's Generate,
 * folding in "Cloud needs a key" / "Qwen needs Apple Silicon" (catalog `available`).
 */
export async function isTtsReady(): Promise<boolean> {
  if (!isTauri()) return false;
  const [cfg, catalog] = await Promise.all([invoke<AppConfig>('get_config'), ttsEngineCatalog()]);
  const id: TtsEngineId = typeof cfg.tts.backend === 'object' ? 'cloud' : cfg.tts.backend;
  const entry = catalog.find((e) => e.id === id);
  if (!entry || !entry.available) return false;
  for (const model of entry.required_model_ids) {
    if ((await ttsModelStatus(id, model)) !== 'complete') return false;
  }
  return true;
}

// SYNC-CHECK: a UI selector mapped to the wire `TtsBackend` (lens-core/src/tts/mod.rs) by
// `nextTtsConfig` — NOT the wire type itself: it maps 'qwen3' → qwen3_local and every
// Cloud kind → 'cloud'.
export type TtsProvider = 'orpheus' | 'qwen3' | 'cloud';

/**
 * Persist a TTS backend/provider selection into the current `TtsConfig` shape
 * (read-modify-write). The Cloud tab (#195) is OpenAI-compatible-first; `hostVoice`/
 * `guestVoice`, when given, overwrite `AppConfig.voices` too — a Cloud save must
 * replace whatever voice ids a previously-active local engine left behind (a
 * stale id like "leo" would otherwise be sent verbatim to the cloud provider).
 */
export async function saveTtsProvider(input: {
  provider: TtsProvider;
  apiKey: string;
  baseUrl?: string;
  hostVoice?: string;
  guestVoice?: string;
}): Promise<void> {
  await updateConfig((cfg) => ({
    ...cfg,
    voices:
      input.hostVoice !== undefined || input.guestVoice !== undefined
        ? { host: input.hostVoice ?? cfg.voices.host, guest: input.guestVoice ?? cfg.voices.guest }
        : cfg.voices,
    tts: nextTtsConfig(cfg.tts, input)
  }));
}

/**
 * Compute the next `TtsConfig` for a provider selection. A local backend
 * deactivates cloud (the active `backend` no longer points at it) but PRESERVES
 * the saved key so switching back to Cloud doesn't lose it. Cloud kind defaults
 * to OpenAI-compatible (#195) — the only kind the backend adapter dispatches;
 * Deepgram/ElevenLabs are reserved but not user-selectable from this form.
 */
export function nextTtsConfig(
  prev: TtsConfig,
  input: { provider: TtsProvider; apiKey: string; baseUrl?: string }
): TtsConfig {
  if (input.provider === 'orpheus') {
    return { ...prev, version: 1, backend: 'orpheus' };
  }
  if (input.provider === 'qwen3') {
    return { ...prev, version: 1, backend: 'qwen3_local' };
  }
  return {
    version: 1,
    backend: { cloud: 'open_ai_compatible' },
    model: prev.model,
    cloud: {
      kind: 'open_ai_compatible',
      api_key: input.apiKey,
      base_url: input.baseUrl ?? ''
    }
  };
}
