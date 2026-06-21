// FROZEN CONTRACT (Commit 1, plan §2.5): the onboarding-completion helper
// signature is locked here so Commits 3a and 4 can be coded in parallel against
// a fixed interface.
//
//   export async function completeOnboarding(): Promise<void>
//
// Commit 3a (#9) implements the body: a UI-side read-modify-write that reads the
// current config via `get_config`, sets `onboarding_complete = true`, persists
// via `set_config` (preserving theme and other fields), and navigates home —
// all guarded by `isTauri()`. Commit 4 (#7) imports and CALLS this helper from
// the Continue button; it does NOT re-implement persistence.
//
// This Commit-1 body is an intentional no-op stub. Do not add persistence logic
// here — that belongs to Commit 3a.

/**
 * Marks first-run onboarding as complete and navigates to the main app.
 *
 * FROZEN SIGNATURE — see the file header. Body is implemented by Commit 3a.
 */
export async function completeOnboarding(): Promise<void> {
  // Stub: Commit 3a implements RMW persistence (get_config → set
  // onboarding_complete = true → set_config) and navigation, isTauri()-guarded.
}
