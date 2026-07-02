// Model validation — interactive pre-save probe (role-neutral).
//
// Wraps the `validate_model_interactive` Tauri command, which constructs a
// temporary provider from the user-entered (not-yet-persisted) form values and
// validates the model BEFORE config is saved. Role-neutral: the SAME probe backs
// both the Enrichment model (blocking) and the Studio & Chat model (informational)
// selectors — the caller decides whether an `invalid` result blocks or not.
//
// IPC contract (snake_case over IPC):
//   command:  validate_model_interactive
//   params:   { provider, model, base_url, api_key }
//   returns:  { status: "valid" | "invalid", reason?: string }

import { invoke, isTauri } from '@tauri-apps/api/core';

export interface ModelValidationResult {
  status: 'valid' | 'invalid';
  reason?: string;
}

/**
 * Validate a model interactively using the user's not-yet-persisted form values.
 * Returns `{ status: 'valid' }` when the model is reachable/present, or
 * `{ status: 'invalid', reason }` with an actionable message on failure.
 *
 * Role-neutral: used for both the blocking enrichment model and the non-blocking
 * studio/chat model. The caller decides how to treat an `invalid` result.
 *
 * @param provider - The provider id: `'ollama'` for the local tab; the cloud
 *   provider id (e.g. `'openai'`, `'anthropic'`) for the cloud tab.
 * @param model    - The model id to validate (user-entered, not yet persisted).
 * @param baseUrl  - The endpoint base URL (for Ollama: `'http://localhost:11434'`;
 *   for native cloud providers: `''`; for custom: the user-supplied URL).
 * @param apiKey   - The API key (for cloud providers; `''` for local Ollama).
 */
export async function validateModelInteractive(
  provider: string,
  model: string,
  baseUrl: string,
  apiKey: string
): Promise<ModelValidationResult> {
  if (!isTauri()) return { status: 'valid' };
  const result = await invoke<{ status: 'valid' | 'invalid'; reason?: string }>(
    'validate_model_interactive',
    {
      provider,
      model,
      base_url: baseUrl,
      api_key: apiKey
    }
  );
  return { status: result.status, reason: result.reason };
}
