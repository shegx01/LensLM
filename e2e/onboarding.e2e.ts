import { expect, test } from '@playwright/test';
import { installTauriStub, readSetConfigCalls } from './helpers/tauri-stub.js';

// First-run onboarding e2e (M1, plan §6) — STATE-BASED, race-free.
//
// Onboarding is a first-run STATE, not a route: the layout conditionally renders
// the SystemCheck screen vs. the app, with NO navigation. These tests therefore
// assert RENDERED STATE at '/', never URLs — there is no goto()/redirect to race.

test('first run renders the System check screen and all six rows at /', async ({ page }) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');

  // The layout reads get_config (onboarding_complete=false) and renders the
  // SystemCheck component in place — same URL, no redirect.
  await expect(page.getByText('System check', { exact: true })).toBeVisible();
  await expect(page.getByText('Local backend', { exact: true })).toBeVisible();
  await expect(page.getByText('LLM runtime', { exact: true })).toBeVisible();
  await expect(page.getByText('Embedding model', { exact: true })).toBeVisible();
  await expect(page.getByText('Vector database', { exact: true })).toBeVisible();
  await expect(page.getByText('Disk permissions', { exact: true })).toBeVisible();
  // Synthetic 6th row added by the UI layer
  await expect(page.getByText('Text-to-speech', { exact: true })).toBeVisible();
});

test('Continue persists onboarding_complete and swaps to the app (no URL change)', async ({
  page
}) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');
  await expect(page.getByText('System check', { exact: true })).toBeVisible();

  const continueButton = page.getByRole('button', { name: 'Continue' });
  await expect(continueButton).toBeEnabled();
  await continueButton.click();

  // completeOnboarding() does a read-modify-write set_config with the flag flipped.
  await expect
    .poll(() => readSetConfigCalls(page))
    .toContainEqual(expect.objectContaining({ onboarding_complete: true }));

  // The onboarding screen disappears and the main app content renders in place.
  await expect(page.getByText('System check', { exact: true })).toBeHidden();
  await expect(page.getByRole('heading', { name: 'Hello World' })).toBeVisible();
});

test('returning user sees the app immediately, never the System check', async ({ page }) => {
  await installTauriStub(page, { onboardingComplete: true });

  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Hello World' })).toBeVisible();
  await expect(page.getByText('System check', { exact: true })).toBeHidden();
});
