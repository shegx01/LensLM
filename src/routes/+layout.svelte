<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { ModeWatcher } from 'mode-watcher';
  import { loadThemeFromConfig } from '$lib/theme/index.js';

  let { children } = $props();

  // First paint is handled by the pre-paint script in app.html (FOUC-free under
  // ssr=false). ModeWatcher owns runtime toggling only. On mount, reconcile the
  // durable AppConfig.theme into the live mode (config wins); guarded so it is a
  // no-op under SSR / in tests without a Tauri backend.
  onMount(() => {
    void loadThemeFromConfig();
  });
</script>

<ModeWatcher />

{@render children?.()}
