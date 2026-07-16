<script lang="ts">
  import { onMount } from 'svelte';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import Aperture from '@lucide/svelte/icons/aperture';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card } from '$lib/components/ui/card/index.js';
  import SystemCheckRow from '$lib/components/onboarding/SystemCheckRow.svelte';
  import { runSystemCheck, type CheckResult } from '$lib/onboarding/system-check.js';
  import ThemeCycleButton from '$lib/components/ThemeCycleButton.svelte';

  let { onadvance }: { onadvance: () => void } = $props();

  let results = $state<CheckResult[]>([]);
  let loading = $state(true);
  let finishing = $state(false);
  let checkError = $state<string | null>(null);
  let continueError = $state<string | null>(null);

  // Every check is now a real readiness gate: Continue is blocked unless ALL
  // three rows pass. An empty result set (still loading / nothing returned) or a
  // check error also keeps it blocked.
  const blocked = $derived(
    checkError !== null || results.length === 0 || !results.every((r) => r.status === 'pass')
  );

  const readyCount = $derived(results.filter((r) => r.status === 'pass').length);
  const totalCount = $derived(results.length);

  async function check(): Promise<void> {
    loading = true;
    checkError = null;
    try {
      // runSystemCheck returns the two onboarding readiness gates (llm_runtime,
      // embedding_model); the legacy text_to_speech gate is filtered out (TTS
      // setup moved to Settings). Re-checks flip a row's pass↔fail status.
      results = await runSystemCheck();
    } catch (err) {
      console.error('SystemCheck: runSystemCheck failed', err);
      results = [];
      checkError = 'Could not run the system check. Please retry.';
    } finally {
      loading = false;
    }
  }

  // This screen no longer persists anything — it is the first of four steps and
  // simply advances the layout's step machine. The finishing/continueError
  // inline-error scaffolding is kept intact for any future per-step failure.
  async function handleContinue(): Promise<void> {
    finishing = true;
    continueError = null;
    try {
      onadvance();
    } catch (err) {
      console.error('SystemCheck: advance failed', err);
      continueError = 'Could not save your setup. Please try again.';
    } finally {
      finishing = false;
    }
  }

  onMount(() => {
    void check();
  });
</script>

<!-- macOS drag region (titleBarStyle Overlay): the empty canvas is the window
     drag handle; every interactive block carries -webkit-app-region: no-drag so
     clicks/drags-within still work (mirrors SourcesRail.svelte). -->
<div class="absolute top-4 right-4 z-10" style="-webkit-app-region: no-drag;">
  <ThemeCycleButton class="size-9 rounded-lg" />
</div>

<main data-tauri-drag-region class="flex min-h-svh items-center justify-center p-6">
  <!-- Per design: outer card 540 px, 14 px radius, 36/40/32 padding; check rows
       are inner surface cards; footer is plain (not its own card). Width 592 px
       so rows land at the previous lg (512 px) without description truncation. -->
  <div class="w-full max-w-[592px]" style="-webkit-app-region: no-drag;">
    <Card class="w-full gap-4 rounded-[14px] px-10 pt-9 pb-8 shadow-2xl ring-0">
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
            <SystemCheckRow {result} oncheck={check} />
          {/each}
        {/if}
      </div>

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
    </Card>
  </div>
</main>
