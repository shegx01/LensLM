// Reactive whole-AppConfig snapshot (Svelte 5 runes, module singleton). Getters below are
// authoritative ONLY for models/enrichment/asr/audioCloudConsent — every other AppConfig field
// is written directly by its own section via updateConfig() and is not reflected here.

import { invoke, isTauri } from '@tauri-apps/api/core';
import { updateConfig } from '$lib/config.js';
import { toLensError } from '$lib/sources/lens-error.js';
import type { AppConfig, AsrConfig, EnrichmentConfig, ModelConfig } from '$lib/theme/types.js';

const DEFAULT_ENRICHMENT: EnrichmentConfig = {
  enabled: false,
  coref_strategy: 'llm_inline',
  cloud_consent: false
};

let cfg = $state<AppConfig | null>(null);
let loadError = $state<string | null>(null);
let persistError = $state<string | null>(null);
let loadPromise: Promise<void> | null = null;

function toErrorMessage(err: unknown): string {
  return toLensError(err).message;
}

async function load(): Promise<void> {
  if (!isTauri()) {
    cfg = null;
    loadError = null;
    persistError = null;
    return;
  }
  try {
    cfg = await invoke<AppConfig>('get_config');
    loadError = null;
    persistError = null;
  } catch (err) {
    loadError = toErrorMessage(err);
  }
}

export const appConfigStore = {
  get models(): ModelConfig[] {
    return cfg?.models ?? [];
  },
  get enrichment(): EnrichmentConfig {
    return cfg?.enrichment ?? DEFAULT_ENRICHMENT;
  },
  get asr(): AsrConfig | null {
    return cfg?.asr ?? null;
  },
  get audioCloudConsent(): boolean {
    return cfg?.audio_cloud_consent ?? false;
  },
  /** Non-null only while `cfg` itself is untrustworthy (still unloaded) — a `persist()`
   *  re-read failure does not count, since `cfg` already holds a good optimistic value then. */
  get loadError(): string | null {
    return cfg === null ? loadError : null;
  },
  /** Non-null when a `persist()` write landed but its confirmation re-read failed. Distinct
   *  from `loadError`: the caller's own write should surface this, not every other panel. */
  get persistError(): string | null {
    return persistError;
  }
};

/** Load-once: concurrent callers share one in-flight fetch; a no-op once `cfg` is populated. */
export async function ensureLoaded(): Promise<void> {
  if (cfg !== null) return;
  if (!loadPromise) {
    loadPromise = load().finally(() => {
      loadPromise = null;
    });
  }
  await loadPromise;
}

/** Forced reload, distinct from `ensureLoaded()` — call after a write with engine-side effects.
 *  On failure `cfg` is left as-is (never wiped to defaults): a stale-but-real snapshot beats
 *  nuking every consumer's picker back to "nothing configured" over one transient re-fetch. */
export async function refreshConfig(): Promise<void> {
  await load();
}

/**
 * The single writer for the fields this store owns: delegates to `updateConfig` (already
 * serialized via `writeQueue`) then re-reads. If the write succeeds but the re-read fails,
 * keep the optimistically-mutated value rather than falling back to a default — that would
 * flip a persisted `audio_cloud_consent: true` to a rendered `off`.
 */
export async function persist(mutate: (cfg: AppConfig) => AppConfig): Promise<void> {
  if (!isTauri()) return;
  const optimistic = cfg ? mutate(cfg) : null;
  await updateConfig(mutate);
  try {
    cfg = await invoke<AppConfig>('get_config');
    persistError = null;
  } catch (err) {
    if (optimistic) cfg = optimistic;
    persistError = toErrorMessage(err);
  }
}

/** Test hook: clears the snapshot so the next `ensureLoaded()` reloads it. */
export function resetConfig(): void {
  cfg = null;
  loadError = null;
  persistError = null;
  loadPromise = null;
}
