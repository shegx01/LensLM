<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { ModeWatcher } from 'mode-watcher';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { loadThemeFromConfig } from '$lib/theme/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { decideOnboardingRoute } from '$lib/onboarding/route-gate.js';

  let { children } = $props();

  // Anti-FOUC boot gate: hold the main content render until the single config
  // read resolves and the routing decision is made, so the user never sees the
  // app flash before a first-run redirect (plan §3, pre-mortem #3).
  let booting = $state(true);

  // First paint is handled by the pre-paint script in app.html (FOUC-free under
  // ssr=false). ModeWatcher owns runtime toggling only. On mount we do ONE
  // `get_config` read (plan change #14) that drives BOTH theme reconciliation
  // AND the onboarding routing gate — one IPC round-trip.
  onMount(() => {
    void boot();
  });

  async function boot(): Promise<void> {
    // Outside Tauri (e2e / plain `vite dev`) isTauri() is false and there is no
    // config to read: fail OPEN to /onboarding — the safe first-run default
    // (plan §3, change #4).
    if (!isTauri()) {
      const target = decideOnboardingRoute(false, page.url.pathname);
      if (target) await goto(target, { replaceState: true });
      booting = false;
      return;
    }

    try {
      const cfg = await invoke<AppConfig>('get_config');
      // Single read drives theme reconciliation...
      await loadThemeFromConfig(cfg);
      // ...and the routing gate.
      const target = decideOnboardingRoute(cfg.onboarding_complete, page.url.pathname);
      if (target) await goto(target, { replaceState: true });
    } catch (err) {
      // Fail OPEN: a config read error must not trap the user; route to the
      // first-run onboarding screen (safe default) and log non-fatally.
      console.error('+layout boot: get_config failed, failing open to /onboarding', err);
      const target = decideOnboardingRoute(false, page.url.pathname);
      if (target) await goto(target, { replaceState: true });
    } finally {
      booting = false;
    }
  }
</script>

<!-- disableHeadScriptInjection: under ssr=false the ModeWatcher head script is
     never executed; the pre-paint script in app.html owns first paint (FOUC-free).
     Runtime onMount reconciliation via loadThemeFromConfig still runs normally. -->
<ModeWatcher disableHeadScriptInjection />

{#if !booting}
  {@render children?.()}
{/if}
