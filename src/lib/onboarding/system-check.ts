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
  | 'disk_permissions';

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

// SYNC-CHECK: must match lens-core/src/llm_detect.rs LlmDetection
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
