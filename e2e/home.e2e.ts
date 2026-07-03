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

// reopen_last_notebook defaults on: a returning user with ≥1 notebook lands
// straight in the most-recently-active notebook's workspace (auto-open), not the
// empty "Your workspace" state. In the stub, list_notebooks preserves array order,
// so notebooks[0] is the auto-selected one.
test('cold launch auto-opens the most-recent notebook (reopen default on)', async ({ page }) => {
  await installTauriStub(page, {
    onboardingComplete: true,
    notebooks: [{ id: 'nb-1', title: 'Quarterly Review' }]
  });

  await page.goto('/');

  // Shell is up…
  await expect(page.getByText('Notebooks', { exact: true })).toBeVisible();
  // …and the notebook auto-opened, so the empty state is suppressed.
  await expect(page.getByText('Your workspace')).not.toBeVisible();
  await expect(page.getByText('Quarterly Review').first()).toBeVisible();
});

// The toggle off (Settings → General) suppresses auto-open: even with a notebook
// present, the app lands on the empty workspace until the user picks one.
test('cold launch shows the empty state when reopen is off', async ({ page }) => {
  await installTauriStub(page, {
    onboardingComplete: true,
    notebooks: [{ id: 'nb-1', title: 'Quarterly Review' }],
    reopenLastNotebook: false
  });

  await page.goto('/');

  await expect(page.getByText('Your workspace')).toBeVisible();
  await expect(page.getByText('Quarterly Review').first()).toBeVisible();
});
