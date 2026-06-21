// adapter-static: disable SSR; Tauri loads the build output as a client-side SPA.
// We rely on `fallback: 'index.html'` (svelte.config.js) for routing rather than
// prerendering, so future dynamic routes (e.g. /chat/[id]) won't break the build.
export const ssr = false;
