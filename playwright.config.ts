import { defineConfig } from '@playwright/test';

// E2E against the SvelteKit dev server. Full native-Tauri E2E is not viable on
// macOS (no WKWebView driver), so we test the UI/routing in a real browser here.
export default defineConfig({
  testDir: 'e2e',
  testMatch: '**/*.e2e.{js,ts}',
  use: {
    baseURL: 'http://localhost:1420'
  },
  webServer: {
    command: 'bun run dev',
    url: 'http://localhost:1420',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000
  }
});
