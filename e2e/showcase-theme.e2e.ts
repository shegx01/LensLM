import { expect, test } from '@playwright/test';

// Runs against the plain SvelteKit dev server (no Tauri runtime), so the durable
// AppConfig persistence path is a no-op (isTauri() is false) — that path is
// covered by the theme unit tests via mockIPC. Here we assert the live UI layer:
// toggling persists to localStorage (mode-watcher) and survives a reload, which
// is what the pre-paint app.html script reads to stay FOUC-free.
test('theme toggle on /showcase persists across reload', async ({ page }) => {
  await page.goto('/showcase');

  // Force a deterministic starting point, then pick dark.
  await page.getByRole('button', { name: 'Dark' }).click();
  await expect(page.locator('html')).toHaveClass(/dark/);
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem('mode-watcher-mode')))
    .toBe('dark');

  // Reload: the pre-paint script should re-apply dark from localStorage.
  await page.reload();
  await expect(page.locator('html')).toHaveClass(/dark/);

  // Switch to light and confirm it persists too.
  await page.getByRole('button', { name: 'Light' }).click();
  await expect(page.locator('html')).not.toHaveClass(/dark/);
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem('mode-watcher-mode')))
    .toBe('light');

  await page.reload();
  await expect(page.locator('html')).not.toHaveClass(/dark/);
});
