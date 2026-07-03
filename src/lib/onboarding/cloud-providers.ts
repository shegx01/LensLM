// Cloud LLM providers for the onboarding combobox.
// Mirrors backend `adapter_for`/`catalog_key_for` in lens-core/src/llm.rs — every entry has a genai
// native adapter AND a models.dev namespace. Providers without a genai adapter are absent (unservable).

/** Display grouping for the combobox: a small curated set first, the rest after. */
export type CloudProviderGroup = 'popular' | 'all';

/** One selectable cloud provider in the onboarding combobox. */
export interface CloudProvider {
  /** Canonical provider id = models.dev catalog key (persisted to config). */
  id: string;
  /** Human-readable label shown in the combobox. */
  name: string;
  /** models.dev catalog key for the model picker. `null` ⇒ no catalog (free-text model). */
  catalogKey: string | null;
  /** Combobox grouping bucket. */
  group: CloudProviderGroup;
  /** Empty string ⇒ genai resolves the native endpoint; only custom entries supply a URL. */
  baseUrl: string;
  /** Whether selecting this entry reveals the custom base-URL field. */
  custom?: boolean;
  /** Catalog-valid floor model used when the live catalog is empty. Omitted for catalog-less providers. */
  defaultModel?: string;
}

/** Surfaced cloud providers. Order within each group is the display order. */
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
    // Has a models.dev namespace; use the catalog picker so backend validation matches.
    catalogKey: 'ollama-cloud',
    group: 'all',
    baseUrl: ''
  },
  // Custom/self-hosted OpenAI-protocol endpoint: reveals a base-URL field, no catalog, free-text model.
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

/** Offline floor model for a provider: `defaultModel` if defined, else the provider id as placeholder. */
export function defaultModelFor(id: string): string {
  return findCloudProvider(id)?.defaultModel ?? id;
}

/** All provider ids (used to match a saved config entry to a combobox entry). */
export const CLOUD_PROVIDER_IDS: readonly string[] = CLOUD_PROVIDERS.map((p) => p.id);
