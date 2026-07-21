// Reactive AppConfig snapshot (models[] + enrichment) shared across the AI Model
// settings sections (Svelte 5 runes, module singleton) — mirrors
// active-model.svelte.ts's shape. Both sections DERIVE from this store and call
// refreshConfig() after every persist, so a credential saved under Providers surfaces
// as a usable provider under Active model within the same pane open.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { AppConfig, EnrichmentConfig, ModelConfig } from '$lib/theme/types.js';

const DEFAULT_ENRICHMENT: EnrichmentConfig = {
  enabled: false,
  coref_strategy: 'llm_inline',
  cloud_consent: false
};

let models = $state<ModelConfig[]>([]);
let enrichment = $state<EnrichmentConfig>(DEFAULT_ENRICHMENT);

export const appConfigStore = {
  get models() {
    return models;
  },
  get enrichment() {
    return enrichment;
  }
};

/** Re-query the engine config. Never throws (resolves to safe defaults on failure). */
export async function refreshConfig(): Promise<void> {
  if (!isTauri()) {
    models = [];
    enrichment = DEFAULT_ENRICHMENT;
    return;
  }
  try {
    const cfg = await invoke<AppConfig>('get_config');
    models = cfg.models ?? [];
    enrichment = cfg.enrichment ?? DEFAULT_ENRICHMENT;
  } catch {
    models = [];
    enrichment = DEFAULT_ENRICHMENT;
  }
}

/** Reset to defaults. Call in `afterEach` to prevent cross-test bleed. */
export function resetConfig(): void {
  models = [];
  enrichment = DEFAULT_ENRICHMENT;
}
