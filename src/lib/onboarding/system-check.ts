// SYNC-CHECK: must match lens-core/src/system_check.rs
//
// TypeScript mirror of the FROZEN `CheckResult` IPC contract (plan §2.1/§2.5).
// serde on the Rust side uses verbatim snake_case field names and snake_case
// enum renames, so this shape must match exactly. The Rust `Option<CheckAction>`
// (NO `CheckAction::None` variant) maps to `action: ... | null` here — absence
// of an action is `null`, never a string.

import { invoke, isTauri } from '@tauri-apps/api/core';

export type CheckId =
  | 'local_backend'
  | 'llm_runtime'
  | 'embedding_model'
  | 'vector_database'
  | 'disk_permissions'
  | 'text_to_speech';

export type CheckStatus = 'pass' | 'fail' | 'pending';

export type CheckAction = 'configure' | 'choose' | 'retry';

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

// SYNC-CHECK: contract — Rust side to implement invoke('download_tts_engine')
// and invoke('install_embedding_model', { model: string })
// These use streaming progress channels — UI polls a progress state.

export type EmbeddingModelId = 'nomic-embed-text' | 'mxbai-embed-large' | 'all-minilm' | 'bge-m3';

export interface EmbeddingModelSpec {
  id: EmbeddingModelId;
  name: string;
  dims: number;
  sizeMb: number;
  speed: 'Very fast' | 'Fast' | 'Medium';
  description: string;
}

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

/** Install an embedding model. Contract — Rust invoke to be implemented. */
export async function installEmbeddingModel(
  model: EmbeddingModelId,
  onProgress: (pct: number) => void
): Promise<void> {
  if (!isTauri()) {
    // Simulate progress in non-Tauri context for dev/test
    for (let i = 0; i <= 100; i += 10) {
      onProgress(i);
      await new Promise((r) => setTimeout(r, 50));
    }
    return;
  }
  // Real: invoke('install_embedding_model', { model }) with Channel progress
  // For now stub — replace with Channel when Rust side is ready
  await invoke<void>('install_embedding_model', { model });
}

/** Download TTS engine (Kokoro). Contract — Rust invoke to be implemented. */
export async function downloadTtsEngine(onProgress: (pct: number) => void): Promise<void> {
  if (!isTauri()) {
    for (let i = 0; i <= 100; i += 10) {
      onProgress(i);
      await new Promise((r) => setTimeout(r, 50));
    }
    return;
  }
  await invoke<void>('download_tts_engine');
}

/** List available TTS voices (Kokoro). Contract — Rust invoke to be implemented. */
export async function listTtsVoices(): Promise<TtsVoice[]> {
  if (!isTauri()) return [];
  return invoke<TtsVoice[]>('list_tts_voices');
}
