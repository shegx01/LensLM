// LLM provider persistence helper (M1 onboarding — Configure panel).
//
// Read-modify-write over the EXISTING M0 IPC (get_config / set_config).
// set_config replaces the WHOLE AppConfig struct, so we re-fetch the current
// config, upsert the matching ModelConfig entry in `models[]`, and write the
// rest back verbatim — mirroring the RMW pattern in completeOnboarding.ts.
//
// Provider mapping (per spec):
//   Local tab  → provider: 'ollama'
//   Cloud API tab → the REAL provider id matching the models.dev catalog key
//     ('openai' | 'anthropic' | 'google'), so a claude-*/gemini-* model validates
//     against its OWN catalog namespace on the Rust side. 'openai-compatible' is
//     reserved for a genuinely custom/self-hosted endpoint (exempt from catalog
//     validation) — the cloud cards are all first-class providers.
//
// No new Rust command, no main.rs touch.

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
  /** The canonical provider id. `'ollama'` for the Local tab; the real cloud
   * provider id (`'openai' | 'anthropic' | 'google'`) for the Cloud API cards;
   * `'openai-compatible'` only for a genuinely custom/self-hosted endpoint. */
  provider: 'ollama' | 'openai' | 'anthropic' | 'google' | 'openai-compatible';
  base_url: string;
  model: string;
  api_key: string;
  /** Context window (tokens). Persisted to ModelConfig.context. */
  context: number;
}

/**
 * Read-modify-write `models[]` while preserving every other AppConfig field.
 * Upserts the first entry whose `provider` matches; appends if none exists.
 * Guarded for non-Tauri contexts: a no-op outside Tauri (so tests can call it
 * without the IPC stub if they don't need the write assertion).
 */
export async function saveLlmProvider(input: LlmProviderInput): Promise<void> {
  // Build the upserted ModelConfig entry. The context window comes from the UI
  // picker; temperature carries a sensible default (Settings owns it in M2+).
  const entry: ModelConfig = {
    provider: input.provider,
    base_url: input.base_url,
    model: input.model,
    context: input.context,
    temperature: 0.7,
    api_key: input.api_key
  };

  await updateConfig((cfg) => {
    // Upsert: replace the first model with matching provider, or append.
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
  /** Typed routing policy (Stage 2). Optional — omit to leave the existing value
   * (or the Rust `cloud_first` default) untouched. */
  routing?: LlmRouting;
  /** Per-task coref model override (Stage 3). `null` clears it (use the routing
   * default); `undefined` leaves the existing value untouched. */
  coref_model?: TaskModel | null;
  /** Per-task structural-map model override (Stage 3). Same null/undefined
   * semantics as {@link coref_model}. */
  map_model?: TaskModel | null;
}

/**
 * Read-modify-write `enrichment` while preserving every other AppConfig field
 * (mirrors {@link saveLlmProvider}). MERGES onto the existing `enrichment` section
 * rather than replacing it wholesale: the Stage-2 `routing` and Stage-3 per-task
 * overrides (`coref_model`/`map_model`/`chat_model`) co-exist with the three core
 * fields, so a partial save must never drop a field the caller didn't set.
 *
 * `routing`/`coref_model`/`map_model` are applied ONLY when the caller provides
 * them (`undefined` ⇒ keep the prior value); passing `null` for a per-task model
 * explicitly clears the override (back to the routing default). `chat_model` is
 * never touched here (M5's concern) — it is round-tripped from the prior config.
 *
 * A no-op outside Tauri (the `updateConfig` guard), so onboarding stays
 * non-blocking: a skipped step simply never calls this and the Rust-side
 * `#[serde(default)]` keeps the conservative defaults.
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
    return { ...cfg, enrichment };
  });
}
