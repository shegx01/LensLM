// Onboarding completion / reset helpers — read-modify-write over get_config/set_config.
// Pure persistence only; callers swap screens via reactive flags, not navigation.

import { updateConfig } from '$lib/config.js';

/**
 * Read-modify-write `onboarding_complete` while preserving every other field.
 * Guarded for `ssr=false` / tests-without-Tauri: a no-op when not under Tauri.
 */
async function setOnboardingComplete(complete: boolean): Promise<void> {
  await updateConfig((cfg) => ({ ...cfg, onboarding_complete: complete }));
}

/** Persists `onboarding_complete = true`. Errors propagate to the caller. */
export async function completeOnboarding(): Promise<void> {
  await setOnboardingComplete(true);
}

/** Persists `onboarding_complete = false`; next boot renders the SystemCheck screen. */
export async function resetOnboarding(): Promise<void> {
  await setOnboardingComplete(false);
}
