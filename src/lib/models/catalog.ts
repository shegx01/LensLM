// Catalog data-fetch helpers (M4 Phase 3, Stage 3 — capability-aware pickers).
//
// Thin, Tauri-guarded wrappers over the Stage-1/3 model-catalog commands:
//   - `list_provider_models` → the typed models.dev catalog for a CLOUD provider
//     (id → ModelInfo: reasoning flag, context limit, cost, …).
//   - `list_ollama_models`   → the LIVE list of LOCALLY-pulled Ollama models at a
//     base_url (models.dev only catalogs cloud providers).
//
// Both are guarded for non-Tauri contexts (tests / `vite dev`): they resolve to
// an empty result rather than throwing, so the onboarding picker renders an
// empty/"none found" state instead of an error — onboarding stays NON-BLOCKING.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { ModelInfo } from './types.js';

/** One option in a model picker, paired with its capability info so the UI can
 * gate the thinking toggle / surface context+cost hints from a single source. */
export interface ModelOption {
  /** The model id sent to the backend (the catalog/Ollama key). */
  id: string;
  /** Display label (the catalog `name`, falling back to the id for Ollama). */
  label: string;
  /** Capability info from the catalog. `null` for a live Ollama model (not in
   * models.dev) — the UI then hides capability-keyed controls (e.g. thinking). */
  info: ModelInfo | null;
}

/**
 * Lists the CLOUD catalog models for one provider as picker options, sorted by
 * label. Returns `[]` outside Tauri or for an unknown provider (the command
 * already yields an empty map for an unknown provider).
 *
 * `provider` is the catalog key (`anthropic`, `openai`, `google`, `zai`).
 */
export async function listCloudModelOptions(provider: string): Promise<ModelOption[]> {
  if (!isTauri()) return [];
  const models = await invoke<Record<string, ModelInfo>>('list_provider_models', { provider });
  return Object.entries(models)
    .map(([id, info]) => ({ id, label: info.name || id, info }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

/**
 * Lists the LOCALLY-pulled Ollama models at `base_url` as picker options. Each
 * option carries `info: null` (a live Ollama model isn't in models.dev), so the
 * UI hides capability-keyed controls for it.
 *
 * Graceful by contract: the command returns an empty list (never an error) when
 * Ollama is unreachable, so this resolves to `[]` and the picker shows a
 * not-reachable / no-models state — never an error toast. A no-op outside Tauri.
 */
export async function listOllamaModelOptions(baseUrl: string): Promise<ModelOption[]> {
  if (!isTauri()) return [];
  const ids = await invoke<string[]>('list_ollama_models', { base_url: baseUrl });
  return ids.map((id) => ({ id, label: id, info: null }));
}
