<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { ModeWatcher } from 'mode-watcher';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { loadThemeFromConfig } from '$lib/theme/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import SystemCheck from '$lib/components/onboarding/SystemCheck.svelte';

  let { children } = $props();

  // Onboarding is a first-run STATE, not a route. This Tauri window has no
  // address bar, so a route-based gate has no user value AND races SvelteKit's
  // router (a boot-time goto() is dropped before the initial navigation
  // settles). Instead we hold a single boot flag and conditionally render the
  // SystemCheck screen vs. the app — no navigation, no router race.
  let booting = $state(true);
  let onboardingComplete = $state(false);

  // First paint is handled by the pre-paint script in app.html (FOUC-free under
  // ssr=false). ModeWatcher owns runtime toggling only. On mount we do ONE
  // `get_config` read that drives BOTH theme reconciliation AND the onboarding
  // gate — one IPC round-trip.
  onMount(() => {
    void boot();
  });

  async function boot(): Promise<void> {
    // Outside Tauri (e2e / plain `vite dev`) isTauri() is false and there is no
    // config to read: fail OPEN to onboarding — the safe first-run default.
    if (!isTauri()) {
      booting = false;
      return;
    }

    try {
      const cfg = await invoke<AppConfig>('get_config');
      // Single read drives theme reconciliation...
      await loadThemeFromConfig(cfg);
      // ...the persisted accent (drives the [data-accent] token layer; the
      // picker UI lands in a later milestone, so we only apply here)...
      document.documentElement.dataset.accent = cfg.accent || 'purple';
      // ...and the onboarding gate.
      onboardingComplete = cfg.onboarding_complete;
    } catch (err) {
      // Fail OPEN: a config read error must not trap the user past onboarding.
      // `onboardingComplete` stays false → the SystemCheck screen renders.
      console.error('+layout boot: get_config failed, failing open to onboarding', err);
    } finally {
      booting = false;
    }
  }

  // SystemCheck has already durably persisted onboarding_complete (RMW) by the
  // time it fires `oncomplete`; flipping the reactive flag swaps the render to
  // the app. On a persistence failure SystemCheck surfaces an inline error and
  // does NOT call this, so we stay on the onboarding screen.
  function handleOnboardingComplete(): void {
    onboardingComplete = true;
  }
</script>

<!-- disableHeadScriptInjection: under ssr=false the ModeWatcher head script is
     never executed; the pre-paint script in app.html owns first paint (FOUC-free).
     Runtime onMount reconciliation via loadThemeFromConfig still runs normally. -->
<ModeWatcher disableHeadScriptInjection />

{#if booting}
  <!-- Hold render until the single config read resolves so the app never
       flashes before the first-run onboarding decision (anti-FOUC boot gate). -->
{:else if !onboardingComplete}
  <SystemCheck oncomplete={handleOnboardingComplete} />
{:else}
  {@render children?.()}
{/if}
