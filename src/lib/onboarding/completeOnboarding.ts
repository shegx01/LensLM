// Onboarding completion / reset helpers.
//
// Both are UI-side READ-MODIFY-WRITE over the EXISTING M0 IPC (get_config /
// set_config). set_config replaces the WHOLE AppConfig struct, so we re-fetch
// the current config, flip ONLY `onboarding_complete`, and write the rest back
// verbatim — mirroring the theme persistence pattern in src/lib/theme/index.ts.
// No new Rust command, no main.rs touch.
//
// `completeOnboarding(): Promise<void>` is pure persistence. The caller
// (SystemCheck's Continue handler) awaits it and, on success, signals the layout
// to swap from the onboarding screen to the app via a reactive flag — there is
// NO navigation. Keeping these helpers persistence-only makes them testable
// without a router and reusable from a settings/showcase reset entry.

import { updateConfig } from '$lib/config.js';

/**
 * Read-modify-write `onboarding_complete` while preserving every other field.
 * Guarded for `ssr=false` / tests-without-Tauri: a no-op when not under Tauri.
 */
async function setOnboardingComplete(complete: boolean): Promise<void> {
  await updateConfig((cfg) => ({ ...cfg, onboarding_complete: complete }));
}

/**
 * Marks first-run onboarding as complete (persists `onboarding_complete = true`).
 *
 * Pure persistence — there is NO navigation. The caller (SystemCheck's Continue
 * handler) awaits this and, on success, signals the layout to swap screens via a
 * reactive flag. Errors propagate so the caller can surface them.
 */
export async function completeOnboarding(): Promise<void> {
  await setOnboardingComplete(true);
}

/**
 * Re-arms first-run onboarding (persists `onboarding_complete = false`). Next
 * boot the layout gate will render the SystemCheck screen instead of the app.
 */
export async function resetOnboarding(): Promise<void> {
  await setOnboardingComplete(false);
}
