// AppConfig read-modify-write helper (M1 onboarding).
//
// set_config replaces the WHOLE AppConfig struct, so every durable write must be
// a read-modify-write: re-fetch the current config at write time, mutate only the
// fields it owns, and write the rest back verbatim. This helper centralizes that
// pattern so the four consumers (completeOnboarding, saveLlmProvider, persistTheme,
// TtsConfigPanel voice-save) don't each hand-roll the get_config → set_config dance.
//
// Reading at write time (not at call time) avoids clobbering concurrent changes to
// other fields. No new Rust command, no main.rs touch.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { AppConfig } from '$lib/theme/types.js';

/**
 * Read-modify-write the durable AppConfig. Fetches the current config, applies
 * `mutate` to derive the next config, and writes it back. The mutator should
 * return a NEW config object (spread + override) and preserve every field it
 * does not own.
 *
 * Guarded for `ssr=false` / tests-without-Tauri: a no-op when not under Tauri.
 */
export async function updateConfig(mutate: (cfg: AppConfig) => AppConfig): Promise<void> {
  if (!isTauri()) return;
  const cfg = await invoke<AppConfig>('get_config');
  await invoke<void>('set_config', { config: mutate(cfg) });
}
