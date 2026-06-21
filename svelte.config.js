import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    // Tauri serves a prerendered static SPA — emit a single-page fallback.
    adapter: adapter({
      fallback: 'index.html'
    })
  }
};

export default config;
