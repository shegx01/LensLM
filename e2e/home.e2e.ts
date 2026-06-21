import { expect, test } from '@playwright/test';

// The M1 routing gate redirects first-run users (onboarding_complete === false,
// or no Tauri runtime → fail-open) to /onboarding. To exercise the main route we
// stub the Tauri IPC as a RETURNING user (onboarding_complete: true) so the gate
// keeps us on '/', where the current M0 scaffold still renders.
test('home page renders the Hello World heading for a returning user', async ({ page }) => {
  await page.addInitScript(() => {
    const config = {
      theme: 'dark',
      models: [],
      endpoints: {},
      voices: { host: '', guest: '' },
      paths: { data_dir: '' },
      tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
      onboarding_complete: true,
    };
    (window as unknown as { isTauri: boolean }).isTauri = true;
    (globalThis as unknown as { isTauri: boolean }).isTauri = true;
    (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = {
      invoke: (cmd: string) => {
        if (cmd === 'get_config') return Promise.resolve(config);
        return Promise.resolve(undefined);
      },
    };
  });

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Hello World' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Invoke core action' })).toBeVisible();
});
