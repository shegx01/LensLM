// Single source of truth for the provider descriptor list + usability predicates shared
// by the AI Model settings sections, so the "is this provider usable" invariant can't
// drift between the Providers editor and the Active-model picker.

import type { ModelConfig } from '$lib/theme/types.js';
import { CLOUD_PROVIDERS } from '$lib/onboarding/cloud-providers.js';

export type ProviderKind = 'local' | 'cloud' | 'custom';

/** Local Ollama endpoint default. */
export const LOCAL_DEFAULT_ENDPOINT = 'http://localhost:11434';

/** Fallback context window (tokens) when neither config nor catalog supplies one. */
export const DEFAULT_CONTEXT = 8192;

/** One provider descriptor: Ollama plus every cloud entry. */
export interface ProviderDescriptor {
  id: string;
  name: string;
  kind: ProviderKind;
  /** Catalog key for cloud model lookups; `null` for local/custom (no catalog). */
  catalogKey: string | null;
  /** Default base URL: local endpoint, custom's configured URL, or '' for native cloud. */
  baseUrl: string;
}

/** Ollama (local) followed by every cloud provider, in display order. */
export function providerDescriptors(): ProviderDescriptor[] {
  return [
    {
      id: 'ollama',
      name: 'Ollama',
      kind: 'local',
      catalogKey: null,
      baseUrl: LOCAL_DEFAULT_ENDPOINT
    },
    ...CLOUD_PROVIDERS.map(
      (p): ProviderDescriptor => ({
        id: p.id,
        name: p.name,
        kind: p.custom ? 'custom' : 'cloud',
        catalogKey: p.catalogKey,
        baseUrl: p.baseUrl
      })
    )
  ];
}

/** A cloud/custom provider is keyed when its config entry has a non-empty API key. */
export function isKeyed(entry: ModelConfig | undefined): boolean {
  return (entry?.api_key.trim() ?? '') !== '';
}

/** Local Ollama is reachable when it reports at least one pulled model. */
export function isReachable(ollamaCount: number | null): boolean {
  return ollamaCount != null && ollamaCount > 0;
}

/** Usable = local reachable, or cloud/custom keyed. */
export function isUsable(
  descriptor: ProviderDescriptor,
  entry: ModelConfig | undefined,
  ollamaCount: number | null
): boolean {
  return descriptor.kind === 'local' ? isReachable(ollamaCount) : isKeyed(entry);
}
