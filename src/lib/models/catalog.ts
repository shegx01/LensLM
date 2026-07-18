// Tauri-guarded catalog helpers. Both resolve to empty outside Tauri so the
// onboarding picker shows a "none found" state rather than an error.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { ModelInfo } from './types.js';

/** One option in a model picker with its capability info. */
export interface ModelOption {
  /** Model id sent to the backend. */
  id: string;
  /** Display label (catalog `name`, falls back to id for Ollama). */
  label: string;
  /** `null` for live Ollama models (not in models.dev) — UI hides capability-keyed controls. */
  info: ModelInfo | null;
}

/**
 * Cloud catalog models for one provider, filtered to text-in/text-out only.
 * Ordered: `last_updated` DESC, `release_date` DESC, `label` ASC; undated last.
 * Returns `[]` outside Tauri or for an unknown provider.
 */
export async function listCloudModelOptions(provider: string): Promise<ModelOption[]> {
  if (!isTauri()) return [];
  const models = await invoke<Record<string, ModelInfo>>('list_provider_models', { provider });
  return Object.entries(models)
    .filter(([, info]) => isTextCapable(info))
    .map(([id, info]) => ({ id, label: info.name || id, info }))
    .sort(compareCloudOptions);
}

/** True iff a model has both text input AND text output — the minimum for a chat model. */
function isTextCapable(info: ModelInfo): boolean {
  return (
    Array.isArray(info.modalities?.input) &&
    info.modalities.input.includes('text') &&
    Array.isArray(info.modalities?.output) &&
    info.modalities.output.includes('text')
  );
}

/**
 * Stable ordering: dated before undated; `last_updated` DESC, `release_date` DESC,
 * `label` ASC. ISO `YYYY-MM-DD` compares lexicographically.
 */
function compareCloudOptions(a: ModelOption, b: ModelOption): number {
  const aDated = Boolean(a.info?.last_updated || a.info?.release_date);
  const bDated = Boolean(b.info?.last_updated || b.info?.release_date);
  if (aDated !== bDated) return aDated ? -1 : 1;

  if (aDated && bDated) {
    const byUpdated = compareDateDesc(a.info?.last_updated, b.info?.last_updated);
    if (byUpdated !== 0) return byUpdated;
    const byRelease = compareDateDesc(a.info?.release_date, b.info?.release_date);
    if (byRelease !== 0) return byRelease;
  }

  return a.label.localeCompare(b.label);
}

/** ISO date strings DESC; present before absent. */
function compareDateDesc(a: string | null | undefined, b: string | null | undefined): number {
  if (a === b) return 0;
  if (!a) return 1;
  if (!b) return -1;
  return b.localeCompare(a);
}

/**
 * Wraps `has_chat_provider`; resolves `false` outside Tauri.
 * See the chat-provider store for the usable-gate rationale.
 */
export async function hasChatProvider(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    return await invoke<boolean>('has_chat_provider');
  } catch {
    return false;
  }
}

/**
 * Live catalog refresh from models.dev. Backend gates on staleness — no refetch
 * storm on repeated opens. Resolves `false` outside Tauri or on failure.
 */
export async function refreshCatalog(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    return await invoke<boolean>('refresh_models');
  } catch {
    return false;
  }
}

/**
 * Locally-pulled Ollama models as picker options (`info: null` — not in models.dev).
 * Returns `[]` when Ollama is unreachable. No-op outside Tauri.
 */
export async function listOllamaModelOptions(baseUrl: string): Promise<ModelOption[]> {
  if (!isTauri()) return [];
  const ids = await invoke<string[]>('list_ollama_models', { base_url: baseUrl });
  return ids.map((id) => ({ id, label: id, info: null }));
}

/**
 * Compact K/M/B suffix: `8K`, `200K`, `1.05M`, `1B`. Up to 2 decimals, trailing
 * zeros trimmed. Values below 1000 render as-is.
 */
export function formatCompact(n: number): string {
  const units = [
    { value: 1_000_000_000, suffix: 'B' },
    { value: 1_000_000, suffix: 'M' },
    { value: 1_000, suffix: 'K' }
  ] as const;
  for (const { value, suffix } of units) {
    if (Math.abs(n) >= value) {
      const scaled = (n / value).toFixed(2).replace(/\.?0+$/, '');
      return `${scaled}${suffix}`;
    }
  }
  return String(n);
}

/** Per-1M-token USD: `5` → "5", `0.5` → "0.5". */
export function formatUsd(n: number): string {
  return String(n);
}
