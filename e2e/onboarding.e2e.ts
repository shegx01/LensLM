import { expect, test } from '@playwright/test';
import { installTauriStub, readSetConfigCalls } from './helpers/tauri-stub.js';

// First-run onboarding e2e — STATE-BASED, race-free.
//
// Onboarding is a first-run STATE, not a route: the layout conditionally renders
// the onboarding screens vs. the app, with NO navigation. These tests therefore
// assert RENDERED STATE at '/', never URLs — there is no goto()/redirect to race.
//
// The system check is split into two sequential steps (#251):
//   Local AI → Embedding model → Make it yours → Create notebook → Add sources.
// Both gate steps share the "System check" card title.

test('first run renders the Local AI step at / (embedding is a later step)', async ({ page }) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');

  // The layout reads get_config (onboarding_complete=false) and renders the
  // Local AI step in place — same URL, no redirect.
  await expect(page.getByText('System check', { exact: true })).toBeVisible();
  await expect(page.getByText('Local AI', { exact: true })).toBeVisible();
  // The embedding gate lives on the NEXT step, not this one.
  await expect(page.getByText('Embedding model', { exact: true })).toHaveCount(0);
  // TTS setup moved to Settings (#194): onboarding gates on LLM + embeddings only.
  await expect(page.getByText('Text-to-speech', { exact: true })).toHaveCount(0);
});

test('Skip for now on Local AI advances to the Embedding step WITHOUT persisting', async ({
  page
}) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');
  await expect(page.getByText('Local AI', { exact: true })).toBeVisible();

  // "Skip for now" advances the step machine without persisting anything —
  // onboarding_complete is set only by the final Add Sources step.
  await page.getByRole('button', { name: 'Skip for now' }).click();

  await expect(page.getByText('Embedding model', { exact: true })).toBeVisible();
  await expect(page.getByText('Local AI', { exact: true })).toBeHidden();

  // No completion write has happened yet.
  const calls = await readSetConfigCalls(page);
  expect(calls).not.toContainEqual(expect.objectContaining({ onboarding_complete: true }));
});

test('completes the full onboarding walk and swaps to the app (no URL change)', async ({
  page
}) => {
  await installTauriStub(page, { onboardingComplete: false });

  await page.goto('/');
  await expect(page.getByText('Local AI', { exact: true })).toBeVisible();

  // Step 1 (Local AI) → Embedding model
  await page.getByRole('button', { name: 'Skip for now' }).click();

  // Step 2 (Embedding model) → Make it yours (DEFAULT_CHECKS reports embedding ready)
  await expect(page.getByText('Embedding model', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Continue', exact: true }).click();

  // Step 3 → Create notebook
  await expect(page.getByRole('heading', { name: 'Make it yours' })).toBeVisible();
  await page.getByPlaceholder('e.g. Jamie or jdoe').fill('Jamie'); // name is required
  await page.getByRole('button', { name: 'Continue', exact: true }).click();

  // Step 4 → Add sources (accent defaults to purple, no selection needed)
  await expect(page.getByRole('heading', { name: 'Create your first notebook' })).toBeVisible();
  await page.getByPlaceholder('e.g. Q3 Earnings Research').fill('My Notebook');
  await page.getByRole('button', { name: /next/i }).click();

  // Step 5 → Add sources → Skip for now completes onboarding
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
