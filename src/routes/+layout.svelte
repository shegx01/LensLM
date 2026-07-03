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

  // Onboarding is a first-run STATE, not a route: a boot-time goto() races SvelteKit's
  // router. A boot flag conditionally renders onboarding vs. app — no navigation.
  let booting = $state(true);
  let onboardingComplete = $state(false);

  // `onboardingComplete` is the sole gate between onboarding and the app (FOUC-free).
  // `onboardingStep` drives which screen renders within the !onboardingComplete branch.
  let onboardingStep = $state<'system-check' | 'make-it-yours' | 'create-notebook' | 'add-sources'>(
    'system-check'
  );

  // One `get_config` read on mount drives both theme reconciliation and the onboarding gate.
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
      await loadThemeFromConfig(cfg);
      // Validate accent against known IDs; hand-edited or unknown values fall back to 'purple'.
      const accent = (ACCENT_IDS as readonly string[]).includes(cfg.accent) ? cfg.accent : 'purple';
      document.documentElement.dataset.accent = accent;
      onboardingComplete = cfg.onboarding_complete;
      // Seed the draft from persisted values so personalize screens start pre-filled.
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

  // AddSources persists onboarding_complete before firing `oncomplete`; on failure
  // it surfaces an inline error instead, so this is only called on success.
  function handleOnboardingComplete(): void {
    onboardingComplete = true;
    resetDraft();
  }
</script>

<!-- disableHeadScriptInjection: pre-paint script in app.html owns first paint; ModeWatcher
     head script is unused under ssr=false. Runtime reconciliation via loadThemeFromConfig still runs. -->
<ModeWatcher disableHeadScriptInjection />

<!-- titleBarStyle "Overlay": native macOS traffic lights float over content; no custom buttons
     needed. Drag bars are per-region (AppShell rails and onboarding <main data-tauri-drag-region>). -->
<ToastContainer />

{#if booting}{:else if !onboardingComplete}
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
