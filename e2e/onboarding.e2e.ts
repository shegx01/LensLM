import { expect, test } from '@playwright/test';
import { installTauriStub, readSetConfigCalls } from './helpers/tauri-stub.js';

// First-run onboarding e2e — STATE-BASED, race-free.
//
// Onboarding is a first-run STATE, not a route: the layout conditionally renders
// the SystemCheck screen vs. the app, with NO navigation. These tests therefore
// assert RENDERED STATE at '/', never URLs — there is no goto()/redirect to race.

test('first run renders the System check screen with the LLM + embedding rows at /', async ({
  page
}) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');

  // The layout reads get_config (onboarding_complete=false) and renders the
  // SystemCheck component in place — same URL, no redirect.
  await expect(page.getByText('System check', { exact: true })).toBeVisible();
  await expect(page.getByText('LLM runtime', { exact: true })).toBeVisible();
  await expect(page.getByText('Embedding model', { exact: true })).toBeVisible();
  // TTS setup moved to Settings (#194): onboarding gates on LLM + embeddings only.
  await expect(page.getByText('Text-to-speech', { exact: true })).toHaveCount(0);
});

test('Continue to setup advances to Make it yours WITHOUT completing onboarding', async ({
  page
}) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');
  await expect(page.getByText('System check', { exact: true })).toBeVisible();

  // "Continue to setup" now advances the step machine — it no longer persists
  // onboarding_complete (that moves to the final step).
  await page.getByRole('button', { name: 'Continue to setup' }).click();

  await expect(page.getByRole('heading', { name: 'Make it yours' })).toBeVisible();
  await expect(page.getByText('System check', { exact: true })).toBeHidden();

  // No completion write has happened yet.
  const calls = await readSetConfigCalls(page);
  expect(calls).not.toContainEqual(expect.objectContaining({ onboarding_complete: true }));
});

test('completes the full onboarding walk and swaps to the app (no URL change)', async ({
  page
}) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');
  await expect(page.getByText('System check', { exact: true })).toBeVisible();

  // Step 1 → Make it yours
  await page.getByRole('button', { name: 'Continue to setup' }).click();
  await expect(page.getByRole('heading', { name: 'Make it yours' })).toBeVisible();
  await page.getByPlaceholder('e.g. Jamie or jdoe').fill('Jamie'); // name is required
  await page.getByRole('button', { name: 'Continue', exact: true }).click();

  // Step 2 → Create notebook (accent defaults to purple, no selection needed)
  await expect(page.getByRole('heading', { name: 'Create your first notebook' })).toBeVisible();
  await page.getByPlaceholder('e.g. Q3 Earnings Research').fill('My Notebook');
  await page.getByRole('button', { name: /next/i }).click();

  // Step 3 → Add sources → Skip for now completes onboarding
  await expect(page.getByRole('heading', { name: 'Add sources' })).toBeVisible();
  await page.getByRole('button', { name: 'Skip for now' }).click();

  // Completion persists onboarding_complete and the app renders in place.
  await expect
    .poll(() => readSetConfigCalls(page))
    .toContainEqual(expect.objectContaining({ onboarding_complete: true }));
  await expect(page.getByText('System check', { exact: true })).toBeHidden();
  await expect(page.getByText('Your workspace')).toBeVisible();
});

test('returning user sees the app immediately, never the System check', async ({ page }) => {
  await installTauriStub(page, { onboardingComplete: true });

  await page.goto('/');

  await expect(page.getByText('Your workspace')).toBeVisible();
  await expect(page.getByText('System check', { exact: true })).toBeHidden();
});
