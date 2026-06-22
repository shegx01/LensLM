import { expect, test } from '@playwright/test';
import { installTauriStub } from './helpers/tauri-stub';

// Asserts the live UI theme layer: toggling persists to localStorage (mode-watcher)
// and survives a reload, which is what the pre-paint app.html script reads to stay
// FOUC-free. We stub a RETURNING user (onboarding_complete: true) so the root gate
// renders the routed children (/showcase) instead of failing open to the onboarding
// SystemCheck screen.
test('theme toggle on /showcase persists across reload', async ({ page }) => {
  await installTauriStub(page, { onboardingComplete: true });
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
