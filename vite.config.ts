/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { svelteTesting } from '@testing-library/svelte/vite';
import { defineConfig } from 'vitest/config';

const host = process.env.TAURI_DEV_HOST;

// https://v2.tauri.app/start/frontend/sveltekit/
export default defineConfig({
  // tailwindcss() = Tailwind v4 CSS-first plugin; svelteTesting() adds auto-cleanup
  // + browser resolution conditions for tests.
  plugins: [tailwindcss(), sveltekit(), svelteTesting()],
  // Prevent Vite from obscuring Rust compiler errors.
  clearScreen: false,
  server: {
    // Tauri expects a fixed port; fail if it is unavailable.
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: 'ws',
          host,
          port: 1421
        }
      : undefined,
    watch: {
      // Don't watch the Rust backend from the frontend dev server.
      ignored: ['**/src-tauri/**']
    }
  },
  test: {
    // Simulated DOM for component tests (faster than jsdom; fixes Svelte transition RAF).
    environment: 'happy-dom',
    setupFiles: ['./vitest-setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts}', 'src/**/*.svelte.{test,spec}.{js,ts}']
  }
});
