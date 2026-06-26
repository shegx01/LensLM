// The single source of truth for the cloud-LLM provider set surfaced in the
// onboarding combobox (M4 Phase 3).
//
// This list MIRRORS the backend provider table in `lens-core/src/llm.rs`
// (`adapter_for` / `catalog_key_for`): every entry here has a genai native
// adapter AND a models.dev catalog namespace, so the Rust factory validates the
// chosen `(provider, model)` against the right catalog. Providers WITHOUT a genai
// adapter (e.g. Mistral) are intentionally absent — they are unservable.
//
// `id` is the canonical provider id persisted to `ModelConfig.provider` AND the
// models.dev catalog key (`catalogKey === id` for every first-class provider).
// Local Ollama is its OWN separate "local" tab and is deliberately NOT in this
// cloud list.

/** Display grouping for the combobox: a small curated set first, the rest after. */
export type CloudProviderGroup = 'popular' | 'all';

/** One selectable cloud provider in the onboarding combobox. */
export interface CloudProvider {
  /** Canonical provider id = models.dev catalog key (persisted to config). */
  id: string;
  /** Human-readable label shown in the combobox. */
  name: string;
  /** The models.dev catalog key used to load this provider's model picker. For
   * every first-class provider this equals `id`; kept explicit so the custom
   * OpenAI-compatible entry (no catalog) can opt out. `null` ⇒ no catalog. */
  catalogKey: string | null;
  /** Combobox grouping bucket. */
  group: CloudProviderGroup;
  /** Default base URL. Empty string ⇒ rely on genai's built-in native endpoint
   * (the backend resolves it); only the custom OpenAI-compatible entry needs a
   * user-supplied URL. */
  baseUrl: string;
  /** Whether selecting this entry reveals the custom base-URL field. */
  custom?: boolean;
  /** The OFFLINE seed model id for this provider: a sensible recent, catalog-valid
   * model used as a floor when the live catalog is empty (offline / non-Tauri /
   * no-catalog provider). When the catalog loads, the smart default picks the
   * newest catalog model instead. Catalog-valid ids only — an offline save must
   * pass the backend `catalog.validate(provider, model)` gate (fix #3). Omitted
   * for catalog-less providers (custom endpoint), where the model is free-text. */
  defaultModel?: string;
}

/**
 * The surfaced cloud providers. Order within each group is the display order.
 * `baseUrl` is empty for native adapters (genai resolves the canonical endpoint);
 * only the custom OpenAI-compatible entry carries a placeholder URL.
 */
export const CLOUD_PROVIDERS: readonly CloudProvider[] = [
  // --- Popular ---
  {
    id: 'openai',
    name: 'OpenAI',
    catalogKey: 'openai',
    group: 'popular',
    baseUrl: '',
    defaultModel: 'gpt-4o'
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    catalogKey: 'anthropic',
    group: 'popular',
    baseUrl: '',
    defaultModel: 'claude-sonnet-4-5'
  },
  {
    id: 'google',
    name: 'Google (Gemini)',
    catalogKey: 'google',
    group: 'popular',
    baseUrl: '',
    defaultModel: 'gemini-2.5-pro'
  },
  // --- All ---
  {
    id: 'groq',
    name: 'Groq',
    catalogKey: 'groq',
    group: 'all',
    baseUrl: '',
    defaultModel: 'llama-3.3-70b-versatile'
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    catalogKey: 'deepseek',
    group: 'all',
    baseUrl: '',
    defaultModel: 'deepseek-chat'
  },
  {
    id: 'xai',
    name: 'xAI (Grok)',
    catalogKey: 'xai',
    group: 'all',
    baseUrl: '',
    defaultModel: 'grok-4.3'
  },
  {
    id: 'cohere',
    name: 'Cohere',
    catalogKey: 'cohere',
    group: 'all',
    baseUrl: '',
    defaultModel: 'command-a-03-2025'
  },
  {
    id: 'zai',
    name: 'Z.ai (GLM)',
    catalogKey: 'zai',
    group: 'all',
    baseUrl: '',
    defaultModel: 'glm-4.6'
  },
  {
    id: 'ollama-cloud',
    name: 'Ollama Cloud',
    // Ollama Cloud HAS a models.dev namespace (43 hosted models), and the backend
    // validates `(ollama-cloud, model)` against that catalog (`catalog_key_for`
    // returns "ollama-cloud"). Load the catalog PICKER so the frontend selection
    // matches backend validation — a free-text input would let the user type a
    // model the backend rejects, silently dropping enrichment (fix #1).
    catalogKey: 'ollama-cloud',
    group: 'all',
    baseUrl: ''
  },
  // The escape hatch: a genuinely custom/self-hosted OpenAI-protocol endpoint.
  // Reveals a base-URL field and skips catalog validation (arbitrary models).
  // No catalog ⇒ free-text model, so no `defaultModel` seed.
  {
    id: 'openai-compatible',
    name: 'Custom (OpenAI-compatible)',
    catalogKey: null,
    group: 'all',
    baseUrl: 'https://api.openai.com/v1',
    custom: true
  }
] as const;

/** Look up a provider by its canonical id. */
export function findCloudProvider(id: string): CloudProvider | undefined {
  return CLOUD_PROVIDERS.find((p) => p.id === id);
}

/**
 * The OFFLINE seed model id for a provider — the catalog-valid floor used when the
 * live catalog is empty (offline / non-Tauri / no-catalog provider). Returns the
 * provider's `defaultModel` when defined; otherwise falls back to the provider id
 * (a harmless placeholder for the catalog-less custom endpoint, where the model is
 * free-text and the user edits it). Single source of truth for the panel's offline
 * floor (fix #3) — the SMART default still overrides this with the newest catalog
 * model once the catalog loads.
 */
export function defaultModelFor(id: string): string {
  return findCloudProvider(id)?.defaultModel ?? id;
}

/** All provider ids (used to match a saved config entry to a combobox entry). */
export const CLOUD_PROVIDER_IDS: readonly string[] = CLOUD_PROVIDERS.map((p) => p.id);
