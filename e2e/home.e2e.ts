import { expect, test } from '@playwright/test';

// Runs against the plain SvelteKit dev server (no Tauri runtime). The page's
// invoke() call fails gracefully (caught + logged), so the UI still renders.
test('home page renders the Hello World heading', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Hello World' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Invoke core action' })).toBeVisible();
});
