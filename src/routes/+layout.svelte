<script lang="ts">
  import '../app.css';
  import { onMount, tick } from 'svelte';
  import { ModeWatcher } from 'mode-watcher';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { page } from '$app/state';
  import { goto, afterNavigate } from '$app/navigation';
  import { loadThemeFromConfig } from '$lib/theme/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { decideOnboardingRoute } from '$lib/onboarding/route-gate.js';

  let { children } = $props();

  // Anti-FOUC boot gate: hold the main content render until the single config
  // read resolves and the routing decision is made, so the user never sees the
  // app flash before a first-run redirect (plan §3, pre-mortem #3).
  let booting = $state(true);

  // Router-ready latch (root-cause fix, see boot() below). SvelteKit only marks
  // the client router as `started` AFTER its initial `enter` navigation/hydration
  // settles; a `goto()` fired before that point races that flag and can be
  // silently dropped — which is exactly the first-run redirect bug (navigating to
  // '/' would NOT redirect to '/onboarding' on slower/cold-start hydration).
  // afterNavigate() fires once the router has navigated (initial enter included),
  // at which point goto() is guaranteed to take effect. We await this latch before
  // performing any redirect so the gate decision can never be lost to the race.
  let resolveRouterReady: () => void;
  const routerReady = new Promise<void>((resolve) => {
    resolveRouterReady = resolve;
  });
  afterNavigate(() => resolveRouterReady());

  // First paint is handled by the pre-paint script in app.html (FOUC-free under
  // ssr=false). ModeWatcher owns runtime toggling only. On mount we do ONE
  // `get_config` read (plan change #14) that drives BOTH theme reconciliation
  // AND the onboarding routing gate — one IPC round-trip.
  onMount(() => {
    void boot();
  });

  /**
   * Redirect via the router, but only once it is ready. Awaiting `routerReady`
   * (resolved by afterNavigate) guarantees the initial hydration navigation has
   * settled and `goto()` will actually navigate instead of no-opping. A `tick()`
   * fallback covers the (rare) case where the initial `enter` fired before our
   * afterNavigate callback registered, so the latch still resolves promptly.
   */
  async function redirect(target: string | null): Promise<void> {
    if (!target) return;
    await Promise.race([routerReady, tick()]);
    await goto(target, { replaceState: true });
  }

  async function boot(): Promise<void> {
    // Outside Tauri (e2e / plain `vite dev`) isTauri() is false and there is no
    // config to read: fail OPEN to /onboarding — the safe first-run default
    // (plan §3, change #4).
    if (!isTauri()) {
      await redirect(decideOnboardingRoute(false, page.url.pathname));
      booting = false;
      return;
    }

    try {
      const cfg = await invoke<AppConfig>('get_config');
      // Single read drives theme reconciliation...
      await loadThemeFromConfig(cfg);
      // ...and the routing gate.
      await redirect(decideOnboardingRoute(cfg.onboarding_complete, page.url.pathname));
    } catch (err) {
      // Fail OPEN: a config read error must not trap the user; route to the
      // first-run onboarding screen (safe default) and log non-fatally.
      console.error('+layout boot: get_config failed, failing open to /onboarding', err);
      await redirect(decideOnboardingRoute(false, page.url.pathname));
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
