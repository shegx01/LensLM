import { expect, test } from '@playwright/test';
import {
  installTauriStub,
  readSetConfigCalls,
  readReembedCalls,
  DEFAULT_CHECKS,
  type CheckResult
} from './helpers/tauri-stub.js';

// E2E for the M4 Phase 4b-B Embeddings surfaces (plan Steps 9–11). These run
// against the SvelteKit dev server with the fake Tauri runtime (no native
// backend), asserting RENDERED STATE + recorded IPC writes.

// ── Step 9: fresh fastembed-only install passes the onboarding gate ──────────
//
// The D2 showstopper guard: a brand-new machine with fastembed weights cached
// and Ollama UNREACHABLE must still pass the embedding readiness gate so
// onboarding can complete. We model the gate as already `pass` (the backend's
// per-backend OR-gate computes this from fastembed_weights_cached), with Ollama
// down (detect_llm/list_ollama_models empty), and assert the embedding row is
// ready + Continue is not blocked.
test('fresh_install_fastembed_only_passes_gate', async ({ page }) => {
  await installTauriStub(page, {
    onboardingComplete: false,
    // fastembed default model cached; Ollama unreachable.
    fastembedCached: ['nomic-embed-text-v1.5'],
    ollamaModels: []
  });

  await page.goto('/');

  await expect(page.getByText('System check', { exact: true })).toBeVisible();
  await expect(page.getByText('Embedding model', { exact: true })).toBeVisible();

  // The Continue gate is not blocked (all rows pass; fastembed satisfied the
  // embedding arm without Ollama). Advancing to "Make it yours" proves the gate
  // did not dead-end a fastembed-only machine.
  await page.getByRole('button', { name: 'Continue to setup' }).click();
  await expect(page.getByRole('heading', { name: 'Make it yours' })).toBeVisible();
});

// ── Step 9 (panel): the onboarding embed panel sets the GLOBAL default ────────
//
// Expanding the embedding row exposes the provider selector + model cards; a
// fastembed Install warms the model and persists embedding_model + backend.
test('onboarding embed panel sets the global default (model + backend)', async ({ page }) => {
  // Make the embedding row FAIL so its "Choose" affordance expands the panel.
  const checks: CheckResult[] = DEFAULT_CHECKS.map((c) =>
    c.id === 'embedding_model'
      ? { ...c, status: 'fail', detail: 'No embedding model installed' }
      : c
  );
  await installTauriStub(page, {
    onboardingComplete: false,
    checks,
    fastembedCached: [],
    ollamaModels: []
  });

  await page.goto('/');
  await expect(page.getByText('System check', { exact: true })).toBeVisible();

  // Expand the embedding row via its action button. A failing row's action is
  // "Choose"; the button is accessibly named "{action} {row label}".
  await page.getByRole('button', { name: 'Choose Embedding model' }).click();

  // The reworked panel shows the provider selector + the verbatim warning.
  await expect(page.getByText('Select your local embeddings provider')).toBeVisible();
  await expect(page.getByText(/permanently linked/)).toBeVisible();

  // Install the default fastembed model → warm + persist as the global default.
  await page.getByRole('button', { name: /Install nomic-embed-text-v1\.5/i }).click();

  await expect
    .poll(() => readSetConfigCalls(page))
    .toContainEqual(
      expect.objectContaining({
        embedding_model: 'nomic-embed-text-v1.5',
        embedding_backend: 'fastembed'
      })
    );
});

// ── Step 10: global Settings>Embeddings changes the default for new notebooks ─
test('global Settings>Embeddings sets the default a new notebook adopts', async ({ page }) => {
  await installTauriStub(page, {
    onboardingComplete: true,
    fastembedCached: ['nomic-embed-text-v1.5', 'all-minilm'],
    ollamaModels: []
  });

  await page.goto('/');
  await expect(page.getByText('Your workspace')).toBeVisible();

  // Open the account menu → Settings → the Preferences shell (Embeddings live).
  await page.getByRole('button', { name: /account menu/i }).click();
  await page.getByRole('menuitem', { name: /settings/i }).click();

  // The Preferences shell shows the Embeddings section.
  await expect(page.getByRole('heading', { name: 'Embeddings' })).toBeVisible();
  await expect(page.getByText(/local only — all vectors computed on-device/i)).toBeVisible();

  // Pick a different cached model and set it as the default.
  await page.getByRole('radio', { name: /all-minilm/i }).click();
  await page.getByRole('button', { name: /apply selected model/i }).click();

  await expect
    .poll(() => readSetConfigCalls(page))
    .toContainEqual(
      expect.objectContaining({ embedding_model: 'all-minilm', embedding_backend: 'fastembed' })
    );
});

// ── Step 11: per-notebook model-change happy path (re-embed) ──────────────────
test('per-notebook settings change → confirm → re-embed streams progress', async ({ page }) => {
  await installTauriStub(page, {
    onboardingComplete: true,
    notebooks: [{ id: 'nb-1', title: 'Q3 Earnings' }],
    // The notebook is indexed on nomic/fastembed; switching to all-minilm needs
    // a re-embed (the confirm dialog).
    notebookEmbedding: {
      model_id: 'nomic-embed-text-v1.5',
      dim: 768,
      backend: 'fastembed',
      status: 'active'
    },
    fastembedCached: ['nomic-embed-text-v1.5', 'all-minilm']
  });

  await page.goto('/');
  await expect(page.getByText('Your workspace')).toBeVisible();

  // Select the notebook so it becomes active (the gear targets the active one).
  await page.getByText('Q3 Earnings').click();

  // Open the per-notebook settings sheet via the pill's gear. Target the gear
  // by its exact aria-label — `getByRole('button', { name: /notebook settings/i })`
  // is ambiguous (it also matches the bits-ui tooltip-trigger wrapper button).
  await page.getByLabel('Notebook settings', { exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Notebook settings' })).toBeVisible();

  // Change the model → the re-embed confirm dialog (with the warning) appears.
  await page.getByRole('radio', { name: /all-minilm/i }).click();
  await page.getByRole('button', { name: /apply selected model/i }).click();
  await expect(page.getByText(/re-embed this notebook/i)).toBeVisible();
  await expect(page.getByText(/permanently linked/)).toBeVisible();

  // Confirm → set_notebook_embedding_model runs with the new coordinate.
  await page.getByRole('button', { name: /confirm re-embed/i }).click();

  await expect
    .poll(() => readReembedCalls(page))
    .toContainEqual(
      expect.objectContaining({ notebookId: 'nb-1', modelId: 'all-minilm', backend: 'fastembed' })
    );
});
