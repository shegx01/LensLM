// AppConfig read-modify-write helper.
//
// set_config replaces the WHOLE AppConfig struct, so every durable write must
// re-fetch the current config at write time, mutate only its own fields, and
// write the rest back verbatim. Reading at write time avoids clobbering
// concurrent changes to other fields.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { AppConfig } from '$lib/theme/types.js';

// Serializes writes: each read-modify-write chains off the previous so
// fast-firing fields (e.g. the AI Model panel) can't interleave a stale
// get_config between another write's get and set, losing the earlier update.
let writeQueue: Promise<void> = Promise.resolve();

/**
 * Read-modify-write the durable AppConfig. Applies `mutate` to the current
 * config and writes it back. The mutator must preserve every field it doesn't own.
 * Writes are serialized to avoid lost updates under concurrent callers.
 * No-op outside Tauri.
 */
export async function updateConfig(mutate: (cfg: AppConfig) => AppConfig): Promise<void> {
  if (!isTauri()) return;
  const run = writeQueue.then(async () => {
    const cfg = await invoke<AppConfig>('get_config');
    await invoke<void>('set_config', { config: mutate(cfg) });
  });
  // Keep the chain alive even if this write rejects, so a failure doesn't wedge the queue.
  writeQueue = run.catch(() => {});
  return run;
}
