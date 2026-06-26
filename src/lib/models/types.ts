// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs — update both together.
//
// TypeScript mirrors of the Rust model-catalog structs. serde on the Rust side
// uses verbatim snake_case field names, so these shapes must match exactly.
// Stage 1 of the LLM-interface overhaul: a TYPED model catalog sourced from
// models.dev so the app never stores model ids as unvalidated free strings.

/** One reasoning control on a model — the typed mirror of the Rust
 * `ReasoningOption` enum (serde `#[serde(tag = "type", rename_all =
 * "snake_case")]`). An unrecognized `type` from the catalog deserializes to
 * `{ type: 'other' }` on the Rust side, so it is included here. */
// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs ReasoningOption enum
export type ReasoningOption =
  | { type: 'effort'; values: string[] }
  | { type: 'budget_tokens'; min: number | null; max: number | null }
  | { type: 'toggle' }
  | { type: 'other' };

/** Input/output modalities a model accepts/produces (e.g. 'text', 'image',
 * 'pdf'). Mirrors the Rust `Modalities` struct. */
// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs Modalities struct
export interface Modalities {
  input: string[];
  output: string[];
}

/** Per-token cost in USD per 1M tokens. Every field is optional — providers
 * report different subsets. Mirrors the Rust `Cost` struct. */
// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs Cost struct
export interface Cost {
  input?: number | null;
  output?: number | null;
  cache_read?: number | null;
  cache_write?: number | null;
}

/** One model's capabilities + economics — the picker reads id, name, reasoning
 * flag, context limit, and cost from here. Mirrors the Rust `ModelInfo` struct.
 * `context_limit`/`output_limit` are flattened from the catalog's nested
 * `limit: { context, output }` object on the Rust side. */
// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs ModelInfo struct
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
  // SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs ModelInfo struct
  /** Catalog `last_updated` date (ISO `YYYY-MM-DD`), or null. The cloud picker
   * sorts options by this (newest first). */
  last_updated: string | null;
  /** Catalog `release_date` (ISO `YYYY-MM-DD`), or null. Tiebreaker for the sort. */
  release_date: string | null;
}

/** One provider's entry (Anthropic, OpenAI, …). `models` is keyed by model id.
 * Mirrors the Rust `ProviderEntry` struct. */
// SYNC-CHECK: must match lens-core/src/model_catalog/mod.rs ProviderEntry struct
export interface ProviderEntry {
  id: string;
  name: string;
  env: string[];
  doc?: string | null;
  models: Record<string, ModelInfo>;
}

/** The full catalog as returned by the `list_models` command: provider key →
 * entry. (`list_provider_models` returns the inner `Record<string, ModelInfo>`
 * for a single provider.) */
export type ModelCatalog = Record<string, ProviderEntry>;
