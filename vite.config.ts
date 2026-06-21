import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// @ts-expect-error process is a node global available under the Vite/Bun runtime.
const host = process.env.TAURI_DEV_HOST;

// https://v2.tauri.app/start/frontend/sveltekit/
export default defineConfig({
  plugins: [sveltekit()],
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
  }
});
