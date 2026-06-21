// Pure routing-gate decision (Commit 3a, #9).
//
// Extracted as a side-effect-free function so the redirect logic is unit-testable
// without a router, a Tauri host, or the DOM. The `+layout` boot reads config +
// pathname once, calls this, and performs the `goto(..., { replaceState: true })`
// only when a target path is returned.

/**
 * Decide where the boot gate should redirect, given onboarding state + the
 * current pathname.
 *
 * - First-run (incomplete) and NOT already on onboarding → '/onboarding'.
 * - Completed but still on an onboarding route → '/' (kick back to the app).
 * - Otherwise → null (no redirect; stay put).
 *
 * @returns the target path, or null when no navigation is needed.
 */
export function decideOnboardingRoute(
  onboardingComplete: boolean,
  pathname: string
): string | null {
  if (!onboardingComplete && pathname !== '/onboarding') return '/onboarding';
  if (onboardingComplete && pathname.startsWith('/onboarding')) return '/';
  return null;
}
