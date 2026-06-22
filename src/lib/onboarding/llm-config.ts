// LLM provider persistence helper (M1 onboarding — Configure panel).
//
// Read-modify-write over the EXISTING M0 IPC (get_config / set_config).
// set_config replaces the WHOLE AppConfig struct, so we re-fetch the current
// config, upsert the matching ModelConfig entry in `models[]`, and write the
// rest back verbatim — mirroring the RMW pattern in completeOnboarding.ts.
//
// Provider mapping (per spec):
//   Local tab  → provider: 'ollama'
//   Cloud API tab → provider: 'openai-compatible'
//
// No new Rust command, no main.rs touch.

import type { ModelConfig } from '$lib/theme/types.js';
import { updateConfig } from '$lib/config.js';

export type LlmProviderTab = 'local' | 'cloud';

export interface LlmProviderInput {
  /** 'ollama' for Local tab, 'openai-compatible' for Cloud API tab. */
  provider: 'ollama' | 'openai-compatible';
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
