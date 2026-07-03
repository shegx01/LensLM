<!-- PROP CONTRACT (do not change without updating +layout.svelte):
     onadvance → 'create-notebook', onback → 'system-check'. Reads/writes draft.userName + draft.accent. -->
<script lang="ts">
  import { onMount } from 'svelte';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Search from '@lucide/svelte/icons/search';
  import Check from '@lucide/svelte/icons/check';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Card } from '$lib/components/ui/card/index.js';
  import { draft } from '$lib/components/onboarding/onboarding-state.svelte.js';
  import ProgressDots from '$lib/components/onboarding/ProgressDots.svelte';
  import OnboardingBackButton from '$lib/components/onboarding/OnboardingBackButton.svelte';
  import { ACCENTS } from '$lib/theme/accents.js';
  import { updateConfig } from '$lib/config.js';

  let { onadvance, onback }: { onadvance: () => void; onback: () => void } = $props();

  let saving = $state(false);
  let continueError = $state<string | null>(null);

  const canContinue = $derived(draft.userName.trim().length > 0);

  /** Apply the accent immediately to the DOM so the live preview retints. */
  function selectAccent(id: string): void {
    draft.accent = id;
    document.documentElement.dataset.accent = id;
  }

  /** Persist user_name + accent via one read-modify-write, then advance. */
  async function handleContinue(): Promise<void> {
    if (!canContinue) return;
    saving = true;
    continueError = null;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        user_name: draft.userName.trim(),
        accent: draft.accent
      }));
      onadvance();
    } catch (err) {
      console.error('MakeItYours: failed to persist personalisation', err);
      continueError = 'Could not save your preferences. Please try again.';
    } finally {
      saving = false;
    }
  }

  onMount(() => {
    document.documentElement.dataset.accent = draft.accent;
  });
</script>

<!-- macOS drag region (titleBarStyle Overlay): the empty canvas drags the window;
     the Card carries -webkit-app-region: no-drag so every inner control (Back,
     name input, swatches, Continue) stays clickable (mirrors SourcesRail.svelte). -->
<main data-tauri-drag-region class="flex min-h-svh items-center justify-center p-6">
  <div class="w-full max-w-[520px]" style="-webkit-app-region: no-drag;">
    <Card class="w-full rounded-[14px] px-10 pt-9 pb-8 shadow-2xl ring-0">
      <div class="mb-7 flex items-center justify-between">
        <OnboardingBackButton {onback} />
        <ProgressDots current={1} total={3} />
      </div>

      <h1 class="text-foreground mb-1.5 text-[20px] font-bold tracking-tight">Make it yours</h1>
      <p class="text-muted-foreground mb-[22px] text-[13px]">
        Tell us your name and pick an accent — both editable later in Settings.
      </p>

      <div class="mb-[22px]">
        <div
          class="text-muted-foreground mb-2 text-[10px] font-bold tracking-[0.08em] uppercase"
          id="name-label"
        >
          Your name
        </div>
        <Input
          type="text"
          aria-labelledby="name-label"
          placeholder="e.g. Jamie or jdoe"
          class="h-11 rounded-[10px] px-[14px] text-sm"
          bind:value={draft.userName}
        />
      </div>

      <div
        class="bg-muted/40 border-border mb-6 flex items-center gap-3 rounded-[10px] border p-4"
        aria-label="Accent colour preview"
        aria-live="polite"
      >
        <div
          class="bg-primary flex size-[38px] shrink-0 items-center justify-center rounded-[10px]"
          aria-hidden="true"
        >
          <Search class="text-primary-foreground size-[18px]" />
        </div>

        <div class="min-w-0 flex-1">
          <span
            class="text-primary bg-primary/10 mb-[5px] inline-block rounded px-2 py-0.5 text-[10px] font-bold"
          >
            Q3 Insights
          </span>
          <div class="text-muted-foreground text-[12px]">
            This is how highlights and links will look
          </div>
        </div>

        <button
          type="button"
          tabindex="-1"
          aria-hidden="true"
          class="bg-primary text-primary-foreground h-[30px] shrink-0 cursor-default rounded-[7px] px-3.5 text-[12px] font-semibold"
        >
          Ask
        </button>
      </div>

      <div
        class="text-muted-foreground mb-3.5 text-[10px] font-bold tracking-[0.08em] uppercase"
        id="accent-label"
      >
        Accent
      </div>

      <div
        class="mb-[30px] grid grid-cols-6 gap-[10px]"
        role="radiogroup"
        aria-labelledby="accent-label"
      >
        {#each ACCENTS as sw (sw.id)}
          {@const selected = draft.accent === sw.id}
          <button
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={sw.label}
            onclick={() => selectAccent(sw.id)}
            class="flex cursor-pointer flex-col items-center gap-2 border-0 bg-transparent p-0 hover:opacity-85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
            style="focus-visible:ring-color: {sw.solid}"
          >
            <div
              class="flex size-[38px] items-center justify-center rounded-full transition-shadow duration-[140ms]"
              style:background={sw.solid}
              style:box-shadow={selected ? `0 0 0 2px var(--card), 0 0 0 4px ${sw.solid}` : 'none'}
              aria-hidden="true"
            >
              {#if selected}
                <Check class="size-[15px] text-white" stroke-width={3} />
              {/if}
            </div>
            <span
              class="text-[10px] font-semibold transition-colors"
              class:text-foreground={selected}
              class:text-muted-foreground={!selected}
            >
              {sw.label}
            </span>
          </button>
        {/each}
      </div>

      {#if continueError}
        <p class="text-destructive mb-3 flex items-center gap-1.5 text-center text-sm" role="alert">
          <TriangleAlert class="size-4 shrink-0" />
          {continueError}
        </p>
      {/if}

      <Button
        class="h-[46px] w-full rounded-[10px] text-[15px] font-semibold"
        onclick={handleContinue}
        disabled={!canContinue || saving}
        aria-disabled={!canContinue || saving}
      >
        Continue
        <ArrowRight class="size-[15px]" />
      </Button>
    </Card>
  </div>
</main>
