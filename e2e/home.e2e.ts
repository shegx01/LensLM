import { expect, test } from '@playwright/test';
import { installTauriStub } from './helpers/tauri-stub.js';

// The layout renders the SystemCheck screen for first-run users
// (onboarding_complete === false, or no Tauri runtime → fail-open). To exercise
// the main route we stub the Tauri IPC as a RETURNING user so the layout renders
// the app content, where the current M0 scaffold still lives.
test('home page renders the Hello World heading for a returning user', async ({ page }) => {
  await installTauriStub(page, { onboardingComplete: true });

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Hello World' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Invoke core action' })).toBeVisible();
});
