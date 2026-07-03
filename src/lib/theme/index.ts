// Theme persistence. Dataflow is ONE-DIRECTIONAL: toggle → setMode + localStorage → debounced RMW to AppConfig.
// persistTheme NEVER calls setMode; loadThemeFromConfig() is the only place config drives mode (boot only).

import { invoke, isTauri } from '@tauri-apps/api/core';
import { setMode } from 'mode-watcher';
import type { AppConfig } from './types.js';
import { updateConfig } from '$lib/config.js';

// mode-watcher's `Mode` union is not re-exported from the package root; mirrored locally.
export type Mode = 'light' | 'dark' | 'system';

/** Trailing debounce window for the durable write, in milliseconds. */
export const PERSIST_DEBOUNCE_MS = 300;

/** Guards all `as Mode` casts; hand-edited bad config returns false. */
function isValidMode(v: string): v is Mode {
  return v === 'light' || v === 'dark' || v === 'system';
}

/**
 * Boot-time reconciliation: drives mode-watcher to match AppConfig (config wins).
 * Pass `cfg` to skip an internal `get_config` when the caller already holds a fresh config.
 * No-op outside Tauri. Invalid stored values fall back to `"system"`.
 */
export async function loadThemeFromConfig(cfg?: AppConfig): Promise<void> {
  if (!isTauri()) return;
  try {
    const config = cfg ?? (await invoke<AppConfig>('get_config'));
    const stored = config.theme;
    const mode: Mode =
      stored === '' || stored === 'system' ? 'system' : isValidMode(stored) ? stored : 'system';
    setMode(mode);
  } catch (err) {
    console.error('loadThemeFromConfig: failed to read AppConfig.theme', err);
  }
}

// --- Trailing-debounced durable persistence -------------------------------

let pendingTheme: string | null = null;
let timer: ReturnType<typeof setTimeout> | null = null;

/** Optional sink for surfacing persist failures (no silent divergence). */
export type PersistErrorHandler = (err: unknown) => void;
let onError: PersistErrorHandler | null = null;

/** Registers a handler invoked when a durable write fails. */
export function setPersistErrorHandler(handler: PersistErrorHandler | null): void {
  onError = handler;
}

/** RMW `.theme` at flush time (avoids clobbering concurrent changes); surfaces errors via handler. */
async function flush(theme: string): Promise<void> {
  try {
    await updateConfig((current) => ({ ...current, theme }));
  } catch (err) {
    console.error('persistTheme: failed to write AppConfig.theme', err);
    onError?.(err);
  }
}

/** Queue a trailing-debounced durable write; coalesces rapid toggles. Never touches mode-watcher. */
export function persistTheme(mode: Mode): void {
  pendingTheme = mode;
  if (timer !== null) clearTimeout(timer);
  timer = setTimeout(() => {
    timer = null;
    const theme = pendingTheme;
    pendingTheme = null;
    if (theme !== null) void flush(theme);
  }, PERSIST_DEBOUNCE_MS);
}

/** Test-only: synchronously flush any pending debounced write. */
export async function __flushNow(): Promise<void> {
  if (timer !== null) {
    clearTimeout(timer);
    timer = null;
  }
  const theme = pendingTheme;
  pendingTheme = null;
  if (theme !== null) await flush(theme);
}
