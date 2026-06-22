// SYNC-CHECK: must match lens-core/src/system_check.rs
//
// TypeScript mirror of the FROZEN `CheckResult` IPC contract (plan §2.1/§2.5).
// serde on the Rust side uses verbatim snake_case field names and snake_case
// enum renames, so this shape must match exactly. The Rust `Option<CheckAction>`
// (NO `CheckAction::None` variant) maps to `action: ... | null` here — absence
// of an action is `null`, never a string.

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import { updateConfig } from '$lib/config.js';

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
//
// Result of probing a local LLM endpoint. The backend command is `detect_llm`
// (frozen contract, parallel agent adds the Rust impl). `reachable` is the
// primary gate; `version` and `models` are best-effort (may be null/empty even
// when reachable, depending on the runtime's /api/version + /api/tags support).
export interface LlmDetection {
  reachable: boolean;
  version: string | null;
  models: string[];
}

/**
 * Probe an OpenAI-compatible local LLM endpoint via `detect_llm`. Guarded for
 * non-Tauri contexts: returns `{reachable:false, version:null, models:[]}` so
 * callers can use the same code path in tests and the browser dev server.
 */
export async function detectLlm(baseUrl: string): Promise<LlmDetection> {
  if (!isTauri()) return { reachable: false, version: null, models: [] };
  return invoke<LlmDetection>('detect_llm', { base_url: baseUrl });
}

/**
 * Run all system probes via the aggregate `run_system_check` command. Guarded
 * for `ssr=false` / tests-without-Tauri: outside a Tauri host this returns `[]`
 * (the UI renders its empty/loading state — there is no native backend to probe).
 */
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
//
// One progress event from `install_embedding_model`. Mirrors the NDJSON status
// lines Ollama emits during `POST /api/pull`: `status` is its own string (e.g.
// "pulling manifest", "downloading", "success"); `completed`/`total` are the
// per-layer byte counters (absent on status-only lines).
export interface InstallProgress {
  status: string;
  completed: number | null;
  total: number | null;
}

// SYNC-CHECK: must match lens-core/src/tts.rs DownloadProgress
//
// One progress event from `download_tts_engine`. `received` is bytes written so
// far, `total` is the advertised Content-Length (or null), `done` flips true on
// the final event (including the idempotent already-present fast path).
export interface DownloadProgress {
  received: number;
  total: number | null;
  done: boolean;
}

export type EmbeddingModelId = 'nomic-embed-text' | 'mxbai-embed-large' | 'all-minilm' | 'bge-m3';

export interface EmbeddingModelSpec {
  id: EmbeddingModelId;
  name: string;
  dims: number;
  sizeMb: number;
  speed: 'Very fast' | 'Fast' | 'Medium';
  description: string;
}

// SYNC-CHECK: the `id`s here must stay in lockstep with the single source of
// truth `ALLOWED_EMBEDDING_MODELS` in lens-core/src/system_check.rs (the install
// allowlist). Adding/removing a model means editing the Rust slice too.
export const EMBEDDING_MODELS: EmbeddingModelSpec[] = [
  {
    id: 'nomic-embed-text',
    name: 'nomic-embed-text',
    dims: 768,
    sizeMb: 274,
    speed: 'Fast',
    description: 'Best general-purpose. Default recommendation.'
  },
  {
    id: 'mxbai-embed-large',
    name: 'mxbai-embed-large',
    dims: 1024,
    sizeMb: 670,
    speed: 'Medium',
    description: 'Higher accuracy, better semantic recall.'
  },
  {
    id: 'all-minilm',
    name: 'all-minilm',
    dims: 384,
    sizeMb: 46,
    speed: 'Very fast',
    description: 'Lightweight. Ideal for constrained environments.'
  },
  {
    id: 'bge-m3',
    name: 'bge-m3',
    dims: 1024,
    sizeMb: 1300,
    speed: 'Medium',
    description: 'Multilingual. Best for non-English content.'
  }
];

/** Clamp a 0..1 ratio to an integer 0..100 percentage. */
function toPct(completed: number | null, total: number | null): number | null {
  if (completed === null || total === null || total <= 0) return null;
  return Math.min(100, Math.max(0, Math.round((completed / total) * 100)));
}

/**
 * Install an embedding model via the real `install_embedding_model` command,
 * streaming Ollama pull progress over a {@link Channel}. The panel-facing
 * `onProgress(pct)` is fed a 0–100 percentage derived from the byte counters
 * (held at the last known value on status-only lines that carry no counters).
 *
 * Outside Tauri (component tests / `vite dev`) this is a no-op that resolves
 * immediately — there is no native backend and no fake progress to simulate.
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
 * Download the Kokoro TTS engine via the real `download_tts_engine` command,
 * streaming byte progress over a {@link Channel}. The panel-facing
 * `onProgress(pct)` is fed a 0–100 percentage from `received/total`; when the
 * total is unknown we surface the terminal `done` event as 100.
 *
 * Outside Tauri this is a no-op that resolves immediately.
 */
export async function downloadTtsEngine(onProgress: (pct: number) => void): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<DownloadProgress>();
  channel.onmessage = (p) => {
    const pct = toPct(p.received, p.total);
    if (pct !== null) onProgress(pct);
    else if (p.done) onProgress(100);
  };
  await invoke<void>('download_tts_engine', { onProgress: channel });
}

/** List available TTS voices (Kokoro). Contract — Rust invoke to be implemented. */
export async function listTtsVoices(): Promise<TtsVoice[]> {
  if (!isTauri()) return [];
  return invoke<TtsVoice[]>('list_tts_voices');
}

/** Whether the Kokoro engine is already downloaded on disk (skip the download step). */
export async function kokoroDownloaded(): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('kokoro_downloaded');
}

// SYNC-CHECK: must match lens-core/src/config.rs TtsConfig.provider
//
// The only cloud TTS provider wired today. Kept a string union (not a bare
// string) so the panel + the readiness gate agree on the exact provider id the
// Rust `has_cloud_tts` check matches (`"elevenlabs"`).
export type TtsProvider = 'elevenlabs';

/**
 * Persist the cloud text-to-speech provider config via the standard client-side
 * read-modify-write over `config.tts` (same `updateConfig` pattern as
 * `saveLlmProvider` and the embedding/voice saves). This only STORES the config
 * so the TTS readiness gate can pass — actual cloud synthesis is a later
 * milestone. Guarded for non-Tauri contexts (a no-op inside `updateConfig`).
 */
export async function saveTtsProvider(input: {
  provider: TtsProvider;
  apiKey: string;
}): Promise<void> {
  await updateConfig((cfg) => ({
    ...cfg,
    tts: { provider: input.provider, api_key: input.apiKey }
  }));
}
