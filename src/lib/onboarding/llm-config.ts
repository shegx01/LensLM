// LLM provider persistence helpers — read-modify-write over get_config/set_config.
// Provider id is the models.dev catalog key; 'openai-compatible' is the custom endpoint escape hatch.

import type {
  ModelConfig,
  CorefStrategy,
  EnrichmentConfig,
  LlmRouting,
  TaskModel
} from '$lib/theme/types.js';
import { updateConfig } from '$lib/config.js';

export type LlmProviderTab = 'local' | 'cloud';

export interface LlmProviderInput {
  /** Canonical provider id (= models.dev catalog key). `'ollama'` for local; cloud provider id or `'openai-compatible'` for custom. */
  provider: string;
  base_url: string;
  model: string;
  api_key: string;
  /** Context window (tokens). Persisted to ModelConfig.context. */
  context: number;
}

/** Upserts `models[]` by provider, preserving all other AppConfig fields. No-op outside Tauri. */
export async function saveLlmProvider(input: LlmProviderInput): Promise<void> {
  const entry: ModelConfig = {
    provider: input.provider,
    base_url: input.base_url,
    model: input.model,
    context: input.context,
    temperature: 0.7,
    api_key: input.api_key
  };

  await updateConfig((cfg) => {
    const existing = cfg.models ?? [];
    const idx = existing.findIndex((m) => m.provider === input.provider);
    const models: ModelConfig[] =
      idx >= 0 ? existing.map((m, i) => (i === idx ? entry : m)) : [...existing, entry];
    return { ...cfg, models };
  });
}

/** The enrichment preferences captured by the onboarding LLM step. */
export interface EnrichmentPrefsInput {
  enabled: boolean;
  coref_strategy: CorefStrategy;
  /** Cloud-LLM consent. Ignored (and forced false) for local-only setups. */
  cloud_consent: boolean;
  /** `undefined` ⇒ keep prior routing; Rust defaults to `cloud_first`. */
  routing?: LlmRouting;
  /** `null` clears to routing default; `undefined` leaves prior value. */
  coref_model?: TaskModel | null;
  /** Same null/undefined semantics as `coref_model`. */
  map_model?: TaskModel | null;
  /** Studio & Chat model; `null` clears, `undefined` leaves prior value. Non-blocking — never gates save. */
  chat_model?: TaskModel | null;
}

/**
 * Merges `enrichment` onto the existing config section; `undefined` fields are left untouched.
 * No-op outside Tauri — a skipped onboarding step never writes, Rust defaults apply.
 */
export async function saveEnrichmentPrefs(input: EnrichmentPrefsInput): Promise<void> {
  await updateConfig((cfg) => {
    const prior = cfg.enrichment;
    const enrichment: EnrichmentConfig = {
      ...prior,
      enabled: input.enabled,
      coref_strategy: input.coref_strategy,
      cloud_consent: input.cloud_consent
    };
    if (input.routing !== undefined) enrichment.routing = input.routing;
    if (input.coref_model !== undefined) enrichment.coref_model = input.coref_model;
    if (input.map_model !== undefined) enrichment.map_model = input.map_model;
    if (input.chat_model !== undefined) enrichment.chat_model = input.chat_model;
    return { ...cfg, enrichment };
  });
}
