// Reactive whole-AppConfig snapshot (Svelte 5 runes, module singleton). Getters below are
// authoritative ONLY for models/enrichment/asr/audioCloudConsent — every other AppConfig field
// is written directly by its own section via updateConfig() and is not reflected here (R14).

import { invoke, isTauri } from '@tauri-apps/api/core';
import { updateConfig } from '$lib/config.js';
import type { AppConfig, AsrConfig, EnrichmentConfig, ModelConfig } from '$lib/theme/types.js';

const DEFAULT_ENRICHMENT: EnrichmentConfig = {
  enabled: false,
  coref_strategy: 'llm_inline',
  cloud_consent: false
};

let cfg = $state<AppConfig | null>(null);
let loadError = $state<string | null>(null);
let loadPromise: Promise<void> | null = null;

function toErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : 'Could not load settings.';
}

async function load(): Promise<void> {
  if (!isTauri()) {
    cfg = null;
    loadError = null;
    return;
  }
  try {
    cfg = await invoke<AppConfig>('get_config');
    loadError = null;
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
  /** Set when the last load or persist re-read failed — surface this instead of a confident default. */
  get loadError(): string | null {
    return loadError;
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

/** Forced reload, distinct from `ensureLoaded()` — call after a write with engine-side effects. */
export async function refreshConfig(): Promise<void> {
  await load();
}

/**
 * The single writer for the fields this store owns: delegates to `updateConfig` (already
 * serialized via `writeQueue`) then re-reads. If the write succeeds but the re-read fails,
 * keep the optimistically-mutated value rather than falling back to a default — that would
 * flip a persisted `audio_cloud_consent: true` to a rendered `off` (R12).
 */
export async function persist(mutate: (cfg: AppConfig) => AppConfig): Promise<void> {
  if (!isTauri()) return;
  const optimistic = cfg ? mutate(cfg) : null;
  await updateConfig(mutate);
  try {
    cfg = await invoke<AppConfig>('get_config');
    loadError = null;
  } catch (err) {
    if (optimistic) cfg = optimistic;
    loadError = toErrorMessage(err);
  }
}

/** Test hook: clears the snapshot so the next `ensureLoaded()` reloads it. */
export function resetConfig(): void {
  cfg = null;
  loadError = null;
  loadPromise = null;
}
