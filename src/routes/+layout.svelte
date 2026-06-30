<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { ModeWatcher } from 'mode-watcher';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { loadThemeFromConfig } from '$lib/theme/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { ACCENT_IDS } from '$lib/theme/accents.js';
  import SystemCheck from '$lib/components/onboarding/SystemCheck.svelte';
  import MakeItYours from '$lib/components/onboarding/MakeItYours.svelte';
  import CreateNotebook from '$lib/components/onboarding/CreateNotebook.svelte';
  import AddSources from '$lib/components/onboarding/AddSources.svelte';
  import { draft, resetDraft } from '$lib/components/onboarding/onboarding-state.svelte.js';
  import ToastContainer from '$lib/components/ui/ToastContainer.svelte';

  let { children } = $props();

  // Onboarding is a first-run STATE, not a route. This Tauri window has no
  // address bar, so a route-based gate has no user value AND races SvelteKit's
  // router (a boot-time goto() is dropped before the initial navigation
  // settles). Instead we hold a single boot flag and conditionally render the
  // SystemCheck screen vs. the app — no navigation, no router race.
  let booting = $state(true);
  let onboardingComplete = $state(false);

  // WITHIN the !onboardingComplete branch, this drives which onboarding screen
  // renders. It deliberately has NO 'app'/'complete' value: `onboardingComplete`
  // (the boolean, driven by cfg.onboarding_complete) remains the SOLE gate
  // between onboarding and the app, preserving the FOUC-free hold-render.
  let onboardingStep = $state<'system-check' | 'make-it-yours' | 'create-notebook' | 'add-sources'>(
    'system-check'
  );

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
      // picker UI lands in a later milestone, so we only apply here). Validate
      // against the known accents so a hand-edited/unknown value can't drop us
      // into an undefined token state — fall back to 'purple'.
      const accent = (ACCENT_IDS as readonly string[]).includes(cfg.accent) ? cfg.accent : 'purple';
      document.documentElement.dataset.accent = accent;
      // ...the onboarding gate...
      onboardingComplete = cfg.onboarding_complete;
      // ...and seed the draft store from persisted values so the personalize
      // screens start from what's already saved (validated accent, not raw cfg).
      draft.userName = cfg.user_name ?? '';
      draft.accent = accent;
    } catch (err) {
      // Fail OPEN: a config read error must not trap the user past onboarding.
      // `onboardingComplete` stays false → the SystemCheck screen renders.
      console.error('+layout boot: get_config failed, failing open to onboarding', err);
    } finally {
      booting = false;
    }
  }

  // The final onboarding screen (AddSources) has already durably persisted
  // onboarding_complete (RMW) by the time it fires `oncomplete`; flipping the
  // reactive flag swaps the render to the app, then we clear the draft singleton
  // so a future re-arm starts clean. On a persistence failure the final screen
  // surfaces an inline error and does NOT call this, so we stay in onboarding.
  function handleOnboardingComplete(): void {
    onboardingComplete = true;
    resetDraft();
  }
</script>

<!-- disableHeadScriptInjection: under ssr=false the ModeWatcher head script is
     never executed; the pre-paint script in app.html owns first paint (FOUC-free).
     Runtime onMount reconciliation via loadThemeFromConfig still runs normally. -->
<ModeWatcher disableHeadScriptInjection />

<!-- Window chrome: the macOS title bar uses titleBarStyle "Overlay" (tauri.conf),
     so the NATIVE traffic lights float over full-bleed content at the top-left and
     handle close/minimize/zoom on every surface. No custom window-control buttons
     are needed; draggability is provided by per-region drag bars: the AppShell rails
     and, during first-run, each onboarding screen's <main data-tauri-drag-region>
     (the empty canvas around the card), mirroring the SourcesRail pattern. -->

<!-- ToastContainer is mounted OUTSIDE the booting/onboarding conditional so toasts
     are always visible regardless of which surface is active. z-[100] places it
     above the modal overlay (z-50). -->
<ToastContainer />

{#if booting}
  <!-- Hold render until the single config read resolves so the app never
       flashes before the first-run onboarding decision (anti-FOUC boot gate). -->
{:else if !onboardingComplete}
  {#if onboardingStep === 'system-check'}
    <SystemCheck onadvance={() => (onboardingStep = 'make-it-yours')} />
  {:else if onboardingStep === 'make-it-yours'}
    <MakeItYours
      onadvance={() => (onboardingStep = 'create-notebook')}
      onback={() => (onboardingStep = 'system-check')}
    />
  {:else if onboardingStep === 'create-notebook'}
    <CreateNotebook
      onadvance={() => (onboardingStep = 'add-sources')}
      onback={() => (onboardingStep = 'make-it-yours')}
    />
  {:else if onboardingStep === 'add-sources'}
    <AddSources
      oncomplete={handleOnboardingComplete}
      onback={() => (onboardingStep = 'create-notebook')}
    />
  {/if}
{:else}
  {@render children?.()}
{/if}
