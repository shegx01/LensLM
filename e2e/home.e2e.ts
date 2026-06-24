import { expect, test } from '@playwright/test';
import { installTauriStub } from './helpers/tauri-stub.js';

// The layout renders the SystemCheck screen for first-run users
// (onboarding_complete === false, or no Tauri runtime → fail-open). To exercise
// the main route we stub the Tauri IPC as a RETURNING user so the layout renders
// the app shell.
test('home page renders the app shell for a returning user', async ({ page }) => {
  await installTauriStub(page, { onboardingComplete: true });

  await page.goto('/');
  // The app shell replaced the old Hello World scaffold: three structural regions.
  await expect(page.getByText('Your workspace')).toBeVisible();
  await expect(page.getByText('Notebooks', { exact: true })).toBeVisible();
  // M4 replaced the static "Sources & Studio" placeholder with the SourcesRail,
  // whose header is "Sources".
  await expect(page.getByText('Sources', { exact: true })).toBeVisible();
});
