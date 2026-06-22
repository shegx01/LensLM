<script lang="ts">
  import { onMount } from 'svelte';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import Aperture from '@lucide/svelte/icons/aperture';
  import Sun from '@lucide/svelte/icons/sun';
  import Moon from '@lucide/svelte/icons/moon';
  import Monitor from '@lucide/svelte/icons/monitor';
  import { Button } from '$lib/components/ui/button/index.js';
  import SystemCheckRow from '$lib/components/onboarding/SystemCheckRow.svelte';
  import {
    runSystemCheck,
    type CheckResult,
    type CheckAction
  } from '$lib/onboarding/system-check.js';
  import { completeOnboarding } from '$lib/onboarding/completeOnboarding.js';
  import { setMode, userPrefersMode } from 'mode-watcher';
  import { persistTheme, type Mode } from '$lib/theme/index.js';

  let { oncomplete }: { oncomplete: () => void } = $props();

  let results = $state<CheckResult[]>([]);
  let loading = $state(true);
  let finishing = $state(false);
  let checkError = $state<string | null>(null);
  let continueError = $state<string | null>(null);

  // Continue is blocked ONLY when local_backend or disk_permissions FAIL.
  const BLOCKING_IDS = ['local_backend', 'disk_permissions'] as const;
  const blocked = $derived(
    checkError !== null ||
      results.some((r) => (BLOCKING_IDS as readonly string[]).includes(r.id) && r.status === 'fail')
  );

  const readyCount = $derived(results.filter((r) => r.status === 'pass').length);
  const totalCount = $derived(results.length);

  async function check(): Promise<void> {
    loading = true;
    checkError = null;
    try {
      // run_system_check now returns all six rows (including text_to_speech) from
      // the backend; render them directly. Re-checks flip the TTS row pass↔pending.
      results = await runSystemCheck();
    } catch (err) {
      console.error('SystemCheck: runSystemCheck failed', err);
      results = [];
      checkError = 'Could not run the system check. Please retry.';
    } finally {
      loading = false;
    }
  }

  async function handleContinue(): Promise<void> {
    finishing = true;
    continueError = null;
    try {
      await completeOnboarding();
      oncomplete();
    } catch (err) {
      console.error('SystemCheck: completeOnboarding failed', err);
      continueError = 'Could not save your setup. Please try again.';
    } finally {
      finishing = false;
    }
  }

  function handleAction(action: CheckAction): void {
    if (action === 'retry') void check();
  }

  const CYCLE: Mode[] = ['light', 'dark', 'system'];
  const CYCLE_ICON = { light: Sun, dark: Moon, system: Monitor } as const;
  const CYCLE_LABEL = { light: 'Light', dark: 'Dark', system: 'System' } as const;

  const currentMode = $derived(userPrefersMode.current ?? 'system');
  const NextIcon = $derived(CYCLE_ICON[currentMode]);
  const nextLabel = $derived(CYCLE_LABEL[currentMode]);

  function cycleTheme(): void {
    const idx = CYCLE.indexOf(currentMode);
    const next = CYCLE[(idx + 1) % CYCLE.length];
    setMode(next);
    persistTheme(next);
  }

  onMount(() => {
    void check();
  });
</script>

<div class="absolute top-4 right-4 z-10">
  <Button
    variant="outline"
    size="icon"
    aria-label={`Theme: ${nextLabel}`}
    onclick={cycleTheme}
    class="size-9 rounded-lg"
  >
    <NextIcon class="size-4" />
  </Button>
</div>

<main class="flex min-h-svh items-center justify-center p-6">
  <div class="w-full max-w-lg flex flex-col gap-4">
    <!-- Header -->
    <div class="flex flex-col items-center text-center gap-3 pb-2">
      <div
        class="bg-primary flex size-14 items-center justify-center rounded-2xl text-primary-foreground shadow-lg"
        aria-hidden="true"
      >
        <Aperture class="size-7" />
      </div>
      <div>
        <h1 class="text-2xl font-bold text-foreground">System check</h1>
        <p class="text-muted-foreground text-sm mt-1">
          Verifying your local intelligence engine before launch
        </p>
      </div>
    </div>

    <!-- Check rows -->
    <div class="flex flex-col gap-2">
      {#if loading}
        <div
          class="text-muted-foreground flex items-center justify-center gap-2 py-12 text-sm"
          aria-live="polite"
        >
          <LoaderCircle class="size-4 animate-spin" />
          Checking your system…
        </div>
      {:else if checkError}
        <div
          class="text-destructive flex items-center justify-center gap-2 py-12 text-center text-sm"
          role="alert"
        >
          <TriangleAlert class="size-4 shrink-0" />
          {checkError}
        </div>
      {:else}
        {#each results as result (result.id)}
          <SystemCheckRow {result} onaction={handleAction} oncheck={check} />
        {/each}
      {/if}
    </div>

    <!-- Footer: summary + Continue, NOT in a card (plain layout) -->
    <div class="flex flex-col gap-3 pt-1">
      {#if continueError}
        <p class="text-destructive w-full text-center text-sm" role="alert">{continueError}</p>
      {/if}
      {#if !loading && !checkError}
        <p class="text-muted-foreground w-full text-center text-[0.6875rem]">
          {readyCount} of {totalCount} checks passed
        </p>
      {/if}
      <Button
        class="h-11 w-full"
        onclick={handleContinue}
        disabled={loading || finishing || blocked}
      >
        Continue to setup
        <ArrowRight />
      </Button>
    </div>
  </div>
</main>
