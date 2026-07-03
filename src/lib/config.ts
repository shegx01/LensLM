// AppConfig read-modify-write helper.
//
// set_config replaces the WHOLE AppConfig struct, so every durable write must
// re-fetch the current config at write time, mutate only its own fields, and
// write the rest back verbatim. Reading at write time avoids clobbering
// concurrent changes to other fields.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { AppConfig } from '$lib/theme/types.js';

/**
 * Read-modify-write the durable AppConfig. Applies `mutate` to the current
 * config and writes it back. The mutator must preserve every field it doesn't own.
 * No-op outside Tauri.
 */
export async function updateConfig(mutate: (cfg: AppConfig) => AppConfig): Promise<void> {
  if (!isTauri()) return;
  const cfg = await invoke<AppConfig>('get_config');
  await invoke<void>('set_config', { config: mutate(cfg) });
}
