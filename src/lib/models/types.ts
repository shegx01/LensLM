// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs — update both together.
// serde uses verbatim snake_case field names.

// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs ReasoningOption enum.
/** `{ type: 'other' }` covers unrecognized catalog variants from the Rust side. */
export type ReasoningOption =
  | { type: 'effort'; values: string[] }
  | { type: 'budget_tokens'; min: number | null; max: number | null }
  | { type: 'toggle' }
  | { type: 'other' };

// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs Modalities struct.
export interface Modalities {
  input: string[];
  output: string[];
}

// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs Cost struct.
/** Per-token cost in USD per 1M tokens. Fields are optional — providers report different subsets. */
export interface Cost {
  input?: number | null;
  output?: number | null;
  cache_read?: number | null;
  cache_write?: number | null;
}

// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs ModelInfo struct.
// `context_limit`/`output_limit` are flattened from the catalog's nested `limit` object.
export interface ModelInfo {
  id: string;
  name: string;
  family?: string | null;
  reasoning: boolean;
  reasoning_options: ReasoningOption[];
  tool_call: boolean;
  temperature: boolean;
  modalities: Modalities;
  context_limit: number | null;
  output_limit: number | null;
  open_weights: boolean;
  cost?: Cost | null;
  /** ISO `YYYY-MM-DD`, or null. Cloud picker sorts by this (newest first). */
  last_updated: string | null;
  /** ISO `YYYY-MM-DD`, or null. Tiebreaker after `last_updated`. */
  release_date: string | null;
}

// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs ProviderEntry struct.
export interface ProviderEntry {
  id: string;
  name: string;
  env: string[];
  doc?: string | null;
  models: Record<string, ModelInfo>;
}

/** Full catalog keyed by provider id. (`list_provider_models` returns the inner `Record<string, ModelInfo>`.) */
export type ModelCatalog = Record<string, ProviderEntry>;

// SYNC-CHECK: must match lens-core/src/llm.rs ActiveModelCandidate. `reason` is `null`
// exactly when `available` is true.
export interface ActiveModelCandidate {
  provider: string;
  model: string;
  /** Display label, e.g. "Ollama · llama3.2:3b". */
  label: string;
  available: boolean;
  reason: string | null;
}

// SYNC-CHECK: must match src-tauri/src/commands/models.rs ActiveModelSelection.
export interface ActiveModelSelection {
  active: { provider: string; model: string } | null;
  candidates: ActiveModelCandidate[];
}
