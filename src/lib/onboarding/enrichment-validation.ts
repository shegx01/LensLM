// Interactive pre-save model validation (role-neutral).
// Wraps `validate_model_interactive`; caller decides whether `invalid` result blocks.
// IPC params: { provider, model, base_url, api_key } → { status: "valid"|"invalid", reason? }

import { invoke, isTauri } from '@tauri-apps/api/core';

export interface ModelValidationResult {
  status: 'valid' | 'invalid';
  reason?: string;
}

/**
 * Validate a not-yet-persisted (provider, model) pair via the backend probe.
 * Returns `{ status: 'valid' }` or `{ status: 'invalid', reason }` on failure.
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
