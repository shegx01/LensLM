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
 * Lists the CLOUD catalog models for one provider as picker options for the LLM
 * (chat) picker. Only TEXT-capable models are kept: an entry survives iff its
 * `modalities.input` includes `'text'` AND `modalities.output` includes
 * `'text'`. This drops embedding-only, image-generation, audio/TTS-only, and
 * other non-chat models. The filter is intentionally STRICT: an entry with
 * missing/empty modalities is treated as NOT text-capable and excluded — the
 * catalog reliably populates modalities, so an entry without them isn't a usable
 * chat model. (A future TTS picker filters the same catalog by audio output;
 * the text gate lives here in the presentation layer, NOT in the Rust command.)
 *
 * Survivors are ordered by catalog recency: `last_updated` DESC, then
 * `release_date` DESC as a tiebreak, then `label` ASC. Models with NO
 * `last_updated`/`release_date` sort LAST (after all dated ones), alphabetically
 * among themselves. Dates are ISO `YYYY-MM-DD` strings, so a string compare
 * gives the correct chronological order. The ordering is total + stable. Returns
 * `[]` outside Tauri or for an unknown provider (the command already yields an
 * empty map for an unknown provider).
 *
 * `provider` is the catalog key (`anthropic`, `openai`, `google`, `zai`).
 */
export async function listCloudModelOptions(provider: string): Promise<ModelOption[]> {
  if (!isTauri()) return [];
  const models = await invoke<Record<string, ModelInfo>>('list_provider_models', { provider });
  return Object.entries(models)
    .filter(([, info]) => isTextCapable(info))
    .map(([id, info]) => ({ id, label: info.name || id, info }))
    .sort(compareCloudOptions);
}

/** True iff a model can both take text in AND produce text out — the minimum to
 * be a usable chat/LLM model. Tolerant of missing modality data by EXCLUDING it
 * (returns false): the catalog reliably populates modalities, so an entry that
 * lacks text-in/text-out isn't a chat model the LLM picker should offer. */
function isTextCapable(info: ModelInfo): boolean {
  return (
    Array.isArray(info.modalities?.input) &&
    info.modalities.input.includes('text') &&
    Array.isArray(info.modalities?.output) &&
    info.modalities.output.includes('text')
  );
}

/**
 * Total, stable ordering for cloud picker options: newest catalog entry first.
 *
 * A dated option (one with a `last_updated` OR `release_date`) always sorts ahead
 * of an undated one. Among dated options, compare `last_updated` then
 * `release_date` (both ISO `YYYY-MM-DD`, so lexicographic === chronological),
 * newest first; a missing date on one side sorts that side after the present one.
 * Final tiebreak (and the sole key for undated options) is `label` ascending.
 */
function compareCloudOptions(a: ModelOption, b: ModelOption): number {
  const aDated = Boolean(a.info?.last_updated || a.info?.release_date);
  const bDated = Boolean(b.info?.last_updated || b.info?.release_date);
  if (aDated !== bDated) return aDated ? -1 : 1; // dated before undated

  if (aDated && bDated) {
    const byUpdated = compareDateDesc(a.info?.last_updated, b.info?.last_updated);
    if (byUpdated !== 0) return byUpdated;
    const byRelease = compareDateDesc(a.info?.release_date, b.info?.release_date);
    if (byRelease !== 0) return byRelease;
  }

  return a.label.localeCompare(b.label);
}

/** Compares two optional ISO date strings DESC (newer first); a present date
 * sorts ahead of a missing one. ISO `YYYY-MM-DD` compares lexicographically. */
function compareDateDesc(a: string | null | undefined, b: string | null | undefined): number {
  if (a === b) return 0;
  if (!a) return 1; // missing sorts after present
  if (!b) return -1;
  return b.localeCompare(a); // descending
}

/**
 * Triggers a live catalog refresh from models.dev (the `refresh_models` command),
 * so the picker converges to the CURRENT full catalog — new models appear, removed
 * ones disappear — with no code change. The backend gates this on a staleness
 * check, so a recently-refreshed cache is left untouched (no refetch storm on
 * repeated picker opens).
 *
 * Graceful by contract: resolves `false` (never throws) outside Tauri or on any
 * failure (offline, HTTP error), so the caller keeps the already-loaded list and
 * onboarding stays NON-BLOCKING. Resolves `true` when the cache was refreshed.
 */
export async function refreshCatalog(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    return await invoke<boolean>('refresh_models');
  } catch {
    return false;
  }
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
