import { expect, test, type Page } from '@playwright/test';

// First-run onboarding e2e (M1 Commit 3b, plan §6).
//
// These run against the plain SvelteKit dev server (NO native Tauri backend),
// so we inject a fake Tauri runtime BEFORE app boot via page.addInitScript().
// This is net-new e2e infrastructure (the vitest `mockIPC` is in-process /
// happy-dom only, and showcase-theme.e2e.ts does NOT mock IPC — there isTauri()
// is false and IPC is a no-op).
//
// Two pieces are required for the gate to take the real (non-fail-open) path:
//   1. `window.isTauri = true`  — @tauri-apps/api's isTauri() returns
//      `!!(globalThis || window).isTauri` (it does NOT inspect __TAURI_INTERNALS__).
//   2. `window.__TAURI_INTERNALS__.invoke(cmd, args, options)` — invoke() in
//      @tauri-apps/api/core delegates straight to this. Our stub dispatches on
//      `cmd` and returns a Promise, recording set_config calls on the window so
//      the test can read them back via page.evaluate().

type Voices = { host: string; guest: string };
type AppConfig = {
  onboarding_complete: boolean;
  theme: string;
  models: unknown[];
  endpoints: Record<string, unknown>;
  voices: Voices;
  paths: { data_dir: string };
  tier_thresholds: Record<string, unknown>;
};

function makeConfig(onboardingComplete: boolean): AppConfig {
  return {
    onboarding_complete: onboardingComplete,
    theme: 'dark',
    models: [],
    endpoints: {},
    voices: { host: '', guest: '' },
    paths: { data_dir: '' },
    tier_thresholds: {}
  };
}

// Five rows mirroring the frozen CheckResult contract (snake_case ids, lowercase
// statuses). A deliberate mix of pass / fail / pending exercises icon + action
// rendering. local_backend + disk_permissions stay `pass` so Continue is NOT
// blocked (gating predicate, plan change #12).
const SYSTEM_CHECK = [
  { id: 'local_backend', label: 'Local backend', status: 'pass', detail: 'In-process engine ready', action: null },
  { id: 'llm_runtime', label: 'LLM runtime', status: 'fail', detail: 'No local LLM runtime detected', action: 'configure' },
  { id: 'embedding_model', label: 'Embedding model', status: 'pending', detail: 'Set up when you add your first source', action: 'choose' },
  { id: 'vector_database', label: 'Vector database', status: 'pending', detail: 'Built-in storage, set up automatically', action: null },
  { id: 'disk_permissions', label: 'Disk permissions', status: 'pass', detail: '/tmp/lens', action: null }
];

/**
 * Inject a fake Tauri runtime that makes isTauri() true and stubs invoke().
 * `onboardingComplete` chooses what get_config returns. set_config calls are
 * recorded on `window.__SET_CONFIG_CALLS__` for the test to assert against.
 */
async function installTauriStub(page: Page, onboardingComplete: boolean): Promise<void> {
  await page.addInitScript(
    ({ cfg, checks }) => {
      const w = window as unknown as {
        isTauri?: boolean;
        __TAURI_INTERNALS__?: Record<string, unknown>;
        __SET_CONFIG_CALLS__?: unknown[];
      };
      // isTauri() reads (globalThis || window).isTauri — set it on both so the
      // gate takes the real get_config path instead of failing open.
      w.isTauri = true;
      (globalThis as unknown as { isTauri?: boolean }).isTauri = true;

      w.__SET_CONFIG_CALLS__ = [];

      let currentCfg = cfg;

      w.__TAURI_INTERNALS__ = {
        // invoke(cmd, args, options) — the only entrypoint the app uses.
        invoke(cmd: string, args?: Record<string, unknown>): Promise<unknown> {
          switch (cmd) {
            case 'get_config':
              return Promise.resolve(currentCfg);
            case 'run_system_check':
              return Promise.resolve(checks);
            case 'set_config': {
              const next = args?.config as typeof cfg | undefined;
              w.__SET_CONFIG_CALLS__!.push(next);
              if (next) currentCfg = next; // reflect the write for any later read
              return Promise.resolve(null);
            }
            default:
              return Promise.resolve(null);
          }
        },
        // No-op shims for the rest of the internals surface the API may touch.
        transformCallback(callback: unknown) {
          return callback;
        },
        unregisterCallback() {},
        convertFileSrc(path: string) {
          return path;
        }
      };
    },
    { cfg: makeConfig(onboardingComplete), checks: SYSTEM_CHECK }
  );
}

test('first run redirects to /onboarding and renders the system check', async ({ page }) => {
  await installTauriStub(page, /* onboardingComplete */ false);

  await page.goto('/');

  // The boot gate reads get_config (onboarding_complete=false) and redirects.
  await expect(page).toHaveURL(/\/onboarding$/);

  // Heading + the rows from run_system_check render.
  await expect(page.getByText('System check', { exact: true })).toBeVisible();
  await expect(page.getByText('Local backend', { exact: true })).toBeVisible();
  await expect(page.getByText('LLM runtime', { exact: true })).toBeVisible();
  await expect(page.getByText('Embedding model', { exact: true })).toBeVisible();
  await expect(page.getByText('Vector database', { exact: true })).toBeVisible();
  await expect(page.getByText('Disk permissions', { exact: true })).toBeVisible();
});

test('Continue completes onboarding and returns to /', async ({ page }) => {
  await installTauriStub(page, /* onboardingComplete */ false);

  await page.goto('/');
  await expect(page).toHaveURL(/\/onboarding$/);

  // Wait for the check to finish so Continue is enabled (not in loading state).
  const continueButton = page.getByRole('button', { name: 'Continue' });
  await expect(continueButton).toBeEnabled();

  await continueButton.click();

  // completeOnboarding() does a read-modify-write set_config with the flag flipped.
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (window as unknown as { __SET_CONFIG_CALLS__?: { onboarding_complete?: boolean }[] })
            .__SET_CONFIG_CALLS__ ?? []
      )
    )
    .toContainEqual(expect.objectContaining({ onboarding_complete: true }));

  // ...then goto('/'). The gate now sees onboarding_complete=true and stays.
  await expect(page).toHaveURL(/\/$/);
  await expect(page).not.toHaveURL(/\/onboarding$/);
});

test('returning user stays on / (no onboarding redirect)', async ({ page }) => {
  await installTauriStub(page, /* onboardingComplete */ true);

  await page.goto('/');

  // Give the async boot gate a chance to (incorrectly) redirect, then assert it didn't.
  await expect(page).toHaveURL(/\/$/);
  await expect(page).not.toHaveURL(/\/onboarding$/);
});
