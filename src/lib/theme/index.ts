// Theme persistence + reconciliation (M1-0, #51).
//
// Dataflow is intentionally ONE-DIRECTIONAL:
//   user toggles → ThemeSwitcher → setMode() + localStorage (immediate, live UI)
//                                 → persistTheme(mode) (debounced durable write)
//
// The durable store is `AppConfig.theme` reached through the EXISTING M0 IPC
// (get_config / set_config). set_config replaces the WHOLE struct, so we do a
// trailing-debounced READ-MODIFY-WRITE *at flush time*: re-fetch the current
// config, mutate only `.theme`, write it back. Reading at flush (not at call
// time) avoids clobbering concurrent changes to models/api_key/etc.
//
// persistTheme NEVER calls setMode — there is no setConfig→setMode feedback loop.
// loadThemeFromConfig() (boot/reconcile) is the ONLY place config drives the mode,
// and it runs once on mount with config winning on disagreement.

import { invoke, isTauri } from '@tauri-apps/api/core';
import { setMode } from 'mode-watcher';
import type { AppConfig } from './types.js';

// mode-watcher's `Mode` union is not re-exported from the package root, so we
// mirror it locally (matches `modes = ["dark", "light", "system"]` upstream).
export type Mode = 'light' | 'dark' | 'system';

/** Trailing debounce window for the durable write, in milliseconds. */
export const PERSIST_DEBOUNCE_MS = 300;

/**
 * Validates that a stored string is a known Mode value.
 * Any invalid or unknown value (e.g. hand-edited bad config.theme) returns false.
 * Used to guard all `as Mode` casts and prevent unsafe coercions.
 */
function isValidMode(v: string): v is Mode {
  return v === 'light' || v === 'dark' || v === 'system';
}

/**
 * Boot-time reconciliation: pull the durable theme from AppConfig and drive
 * mode-watcher to match (config wins on disagreement). One-shot, on mount.
 *
 * `""` and `"system"` → set "system" so mode-watcher tracks the OS.
 * Any invalid stored value (e.g. a hand-edited bad config.theme) falls back to
 * "system" — a safe default that lets the OS preference win.
 *
 * Guarded for `ssr=false` and tests-without-Tauri: if not running under Tauri,
 * this is a no-op and the localStorage/pre-paint hint remains the live state.
 */
export async function loadThemeFromConfig(): Promise<void> {
  if (!isTauri()) return;
  try {
    const config = await invoke<AppConfig>('get_config');
    const stored = config.theme;
    // `""`/`"system"` → let mode-watcher track the OS by setting "system".
    // Any unrecognised stored value (bad config) → fall back to "system".
    const mode: Mode =
      stored === '' || stored === 'system' ? 'system' : isValidMode(stored) ? stored : 'system';
    setMode(mode);
  } catch (err) {
    // Read failure is non-fatal: keep the pre-paint/localStorage live state.
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

/**
 * READ-MODIFY-WRITE the durable theme. Runs at flush time so the re-read picks
 * up any concurrent config changes; only `.theme` is mutated, the rest of the
 * struct is written back verbatim. On failure the error is surfaced (handler +
 * console) — the live UI state (mode-watcher/localStorage) is deliberately NOT
 * reverted, so we never silently diverge the durable store from the UI.
 */
async function flush(theme: string): Promise<void> {
  if (!isTauri()) return;
  try {
    const current = await invoke<AppConfig>('get_config');
    const next: AppConfig = { ...current, theme };
    await invoke<void>('set_config', { config: next });
  } catch (err) {
    console.error('persistTheme: failed to write AppConfig.theme', err);
    onError?.(err);
  }
}

/**
 * Queue a durable write of the given mode, coalescing rapid toggles into a
 * single trailing write after {@link PERSIST_DEBOUNCE_MS}. The config is read at
 * flush time, not now. One-directional: this never touches mode-watcher.
 *
 * `"system"` is persisted literally (per the toggle-cycle contract).
 */
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
