// Onboarding completion / reset helpers (Commit 3a, #9).
//
// Both are UI-side READ-MODIFY-WRITE over the EXISTING M0 IPC (get_config /
// set_config). set_config replaces the WHOLE AppConfig struct, so we re-fetch
// the current config, flip ONLY `onboarding_complete`, and write the rest back
// verbatim — mirroring the theme persistence pattern in src/lib/theme/index.ts.
// No new Rust command, no main.rs touch (plan change #9).
//
// FROZEN SIGNATURE (Commit 1, plan §2.5): `completeOnboarding(): Promise<void>`
// is locked so Commit 4 can import + call it without re-implementing persistence.
// Navigation (goto('/')) is owned by the caller (Commit 4's Continue button), so
// these helpers stay pure persistence — testable without a router and reusable
// from a settings/showcase reset entry.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { AppConfig } from '$lib/theme/types.js';

/**
 * Read-modify-write `onboarding_complete` while preserving every other field.
 * Guarded for `ssr=false` / tests-without-Tauri: a no-op when not under Tauri.
 */
async function setOnboardingComplete(complete: boolean): Promise<void> {
  if (!isTauri()) return;
  const cfg = await invoke<AppConfig>('get_config');
  await invoke<void>('set_config', { config: { ...cfg, onboarding_complete: complete } });
}

/**
 * Marks first-run onboarding as complete (persists `onboarding_complete = true`).
 *
 * FROZEN SIGNATURE — the body is the RMW persistence; navigation is the caller's
 * responsibility. Errors propagate so the caller can surface them.
 */
export async function completeOnboarding(): Promise<void> {
  await setOnboardingComplete(true);
}

/**
 * Re-arms first-run onboarding (persists `onboarding_complete = false`). Next
 * boot the routing gate will route to `/onboarding`.
 */
export async function resetOnboarding(): Promise<void> {
  await setOnboardingComplete(false);
}
