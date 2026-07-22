<script lang="ts">
  import { onMount } from 'svelte';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import Aperture from '@lucide/svelte/icons/aperture';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card } from '$lib/components/ui/card/index.js';
  import SystemCheckRow from '$lib/components/onboarding/SystemCheckRow.svelte';
  import { runSystemCheck, type CheckResult, type SaveApi } from '$lib/onboarding/system-check.js';
  import ThemeCycleButton from '$lib/components/ThemeCycleButton.svelte';

  let { onadvance }: { onadvance: () => void } = $props();

  let results = $state<CheckResult[]>([]);
  let loading = $state(true);
  let loaded = $state(false);
  let finishing = $state(false);
  let checkError = $state<string | null>(null);
  let continueError = $state<string | null>(null);

  // The LLM picker hands up a live { save } so this footer can drive Save &
  // continue; Skip never touches it.
  let llmApi = $state<SaveApi | null>(null);

  // Embedding is the retrieval floor, so it stays REQUIRED — both footer buttons
  // are blocked until it passes. The LLM row is skippable and never contributes
  // to the gate. A load-in / check error, or an absent passing embedding row,
  // also keeps the gate blocked (fail-closed).
  const blocked = $derived(
    checkError !== null ||
      results.length === 0 ||
      !results.some((r) => r.id === 'embedding_model' && r.status === 'pass')
  );

  const readyCount = $derived(results.filter((r) => r.status === 'pass').length);
  const totalCount = $derived(results.length);

  async function check(): Promise<void> {
    // Only the first run gates the whole screen; a re-check triggered by a picker
    // persist/refresh updates `results` in place, leaving the pickers mounted (no
    // flash, no lost in-component state).
    if (!loaded) loading = true;
    checkError = null;
    try {
      // runSystemCheck returns the two onboarding readiness gates (llm_runtime,
      // embedding_model); the legacy text_to_speech gate is filtered out (TTS
      // setup moved to Settings).
      results = await runSystemCheck();
      loaded = true;
    } catch (err) {
      console.error('SystemCheck: runSystemCheck failed', err);
      // Only the first load dead-ends into the error screen. A failed re-check
      // (a post-persist gate refresh) keeps the last-good results and the mounted
      // pickers rather than tearing the screen down mid-interaction.
      if (!loaded) {
        results = [];
        checkError = 'Could not run the system check. Please retry.';
      }
    } finally {
      loading = false;
    }
  }

  // Skip writes nothing and always advances (onboarding stays non-blocking).
  async function handleSkip(): Promise<void> {
    finishing = true;
    continueError = null;
    try {
      onadvance();
    } catch (err) {
      console.error('SystemCheck: advance failed', err);
      continueError = 'Could not continue. Please try again.';
    } finally {
      finishing = false;
    }
  }

  // Save persists the local LLM (Variant-B chat_model pin) via the picker, then
  // advances. A failed save surfaces its own inline error in the tile, so we hold
  // on the step instead of advancing.
  async function handleSave(): Promise<void> {
    finishing = true;
    continueError = null;
    try {
      await llmApi?.save();
      onadvance();
    } catch (err) {
      console.error('SystemCheck: save failed', err);
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
    <Card
      class="w-full gap-4 rounded-[14px] border border-[color-mix(in_oklch,var(--primary)_50%,transparent)] px-10 pt-9 pb-8 shadow-2xl ring-0"
    >
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
            <SystemCheckRow {result} oncheck={check} onready={(api) => (llmApi = api)} />
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
        <div class="flex items-center gap-3">
          <Button
            variant="ghost"
            class="h-11 flex-1 active:scale-[0.97] disabled:active:scale-100 motion-reduce:transition-none"
            onclick={handleSkip}
            disabled={loading || finishing || blocked}
          >
            Skip for now
          </Button>
          <Button
            class="group h-11 flex-1 shadow-[inset_0_1px_0_color-mix(in_oklch,var(--primary-foreground)_25%,transparent),0_8px_24px_-8px_color-mix(in_oklch,var(--primary)_55%,transparent)] active:scale-[0.97] disabled:active:scale-100 motion-reduce:transition-none"
            onclick={handleSave}
            disabled={loading || finishing || blocked}
          >
            Save &amp; continue
            <ArrowRight
              class="transition-transform group-hover:translate-x-0.5 motion-reduce:transition-none"
            />
          </Button>
        </div>
      </div>
    </Card>
  </div>
</main>
