<!-- PROP CONTRACT (do not change without updating +layout.svelte):
     onadvance: () => void  — advance to 'add-sources' (after create_notebook, storing draft.notebookId)
     onback:    () => void  — return to 'make-it-yours'
     Reads/writes the shared draft via $lib/components/onboarding/onboarding-state.svelte.ts
     (draft.nbName, draft.nbDesc, draft.focusMode, draft.notebookId). -->
<script lang="ts">
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Search from '@lucide/svelte/icons/search';
  import Code from '@lucide/svelte/icons/code';
  import Pencil from '@lucide/svelte/icons/pencil';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { Card } from '$lib/components/ui/card/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { draft } from '$lib/components/onboarding/onboarding-state.svelte.js';
  import ProgressDots from '$lib/components/onboarding/ProgressDots.svelte';
  import OnboardingBackButton from '$lib/components/onboarding/OnboardingBackButton.svelte';

  let { onadvance, onback }: { onadvance: () => void; onback: () => void } = $props();

  let submitting = $state(false);
  let createError = $state<string | null>(null);

  interface FocusMode {
    id: string;
    label: string;
    desc: string;
    Icon: typeof Search;
  }

  const MODES: FocusMode[] = [
    { id: 'research', label: 'Research', desc: 'Docs & papers', Icon: Search },
    { id: 'coding', label: 'Coding', desc: 'Code & specs', Icon: Code },
    { id: 'notes', label: 'Notes', desc: 'Free thinking', Icon: Pencil }
  ];

  const canAdvance = $derived(draft.nbName.trim().length > 0);

  async function handleNext(): Promise<void> {
    if (!canAdvance || submitting) return;
    submitting = true;
    createError = null;
    try {
      // Within-session reuse: if notebookId already set the user went Back then
      // Forward — skip creation to avoid a duplicate, just advance.
      if (draft.notebookId === null) {
        if (isTauri()) {
          const notebook = await invoke<{ id: string }>('create_notebook', {
            title: draft.nbName.trim(),
            description: draft.nbDesc.trim() || null,
            focusMode: draft.focusMode
          });
          draft.notebookId = notebook.id;
        }
      }
      onadvance();
    } catch (err) {
      console.error('CreateNotebook: create_notebook failed', err);
      createError = 'Could not create your notebook. Please try again.';
    } finally {
      submitting = false;
    }
  }
</script>

<!-- macOS drag region (titleBarStyle Overlay): the empty canvas drags the window;
     the Card carries -webkit-app-region: no-drag so every inner control (Back,
     name/desc inputs, focus modes, CTA) stays clickable (mirrors SourcesRail.svelte). -->
<main data-tauri-drag-region class="flex min-h-svh items-center justify-center p-6">
  <div class="w-full max-w-[520px]" style="-webkit-app-region: no-drag;">
    <Card class="w-full rounded-[14px] px-10 pt-9 pb-8 shadow-2xl ring-0 gap-0">
      <!-- Header: Back + progress dots -->
      <div class="flex items-center justify-between mb-7">
        <OnboardingBackButton {onback} />
        <ProgressDots current={2} total={3} />
      </div>

      <!-- Title + subtitle -->
      <h1
        class="m-0 mb-[6px] text-[20px] font-bold text-foreground leading-tight tracking-[-0.35px]"
      >
        Create your first notebook
      </h1>
      <p class="m-0 mb-6 text-[13px] text-muted-foreground">
        Name your knowledge space and choose a focus mode
      </p>

      <!-- Name field -->
      <div class="mb-3">
        <div
          class="text-[10px] font-bold text-muted-foreground tracking-[0.08em] uppercase mb-[7px]"
        >
          Name
        </div>
        <Input
          type="text"
          id="nb-name"
          aria-label="Notebook name"
          placeholder="e.g. Q3 Earnings Research"
          bind:value={draft.nbName}
          class="h-[46px] rounded-[10px] px-4 text-[14px]"
        />
      </div>

      <!-- Description field (optional) -->
      <div class="mb-[22px]">
        <div
          class="text-[10px] font-bold text-muted-foreground tracking-[0.08em] uppercase mb-[7px]"
        >
          Description
          <span class="font-normal normal-case opacity-70"> — optional</span>
        </div>
        <textarea
          id="nb-desc"
          aria-label="Notebook description"
          placeholder="What will this notebook focus on?"
          rows={2}
          value={draft.nbDesc}
          oninput={(e) => {
            draft.nbDesc = (e.currentTarget as HTMLTextAreaElement).value;
          }}
          class="w-full rounded-[10px] bg-input/30 border border-input outline-none px-4 py-3 text-[13px] text-foreground font-[inherit] leading-[1.55] resize-none transition-colors focus:border-ring focus:ring-2 focus:ring-ring/30"
        ></textarea>
      </div>

      <!-- Focus mode selector -->
      <div class="mb-7">
        <div
          class="text-[10px] font-bold text-muted-foreground tracking-[0.08em] uppercase mb-[10px]"
        >
          Focus mode
        </div>
        <div class="flex gap-2" role="radiogroup" aria-label="Focus mode">
          {#each MODES as mode (mode.id)}
            {@const selected = draft.focusMode === mode.id}
            <button
              role="radio"
              aria-checked={selected}
              aria-label={mode.label}
              onclick={() => {
                draft.focusMode = mode.id;
              }}
              class={[
                'flex-1 px-[10px] py-[14px] rounded-[10px] border cursor-pointer text-center transition-all duration-[140ms] outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
                selected
                  ? 'border-primary bg-primary/5'
                  : 'border-border bg-card hover:border-muted-foreground/40 hover:bg-accent/30'
              ].join(' ')}
            >
              <!-- Icon tile -->
              <div
                class={[
                  'w-[34px] h-[34px] rounded-[9px] flex items-center justify-center mx-auto mb-[9px]',
                  selected ? 'bg-primary/15' : 'bg-muted'
                ].join(' ')}
              >
                <mode.Icon
                  class={['size-4', selected ? 'text-primary' : 'text-muted-foreground'].join(' ')}
                />
              </div>
              <!-- Label -->
              <div
                class={[
                  'text-[13px] font-bold mb-[3px]',
                  selected ? 'text-foreground' : 'text-muted-foreground'
                ].join(' ')}
              >
                {mode.label}
              </div>
              <!-- Description -->
              <div class="text-[10px] text-muted-foreground">{mode.desc}</div>
            </button>
          {/each}
        </div>
      </div>

      <!-- Inline error -->
      {#if createError}
        <p
          class="flex items-center gap-1.5 text-destructive text-sm text-center justify-center mb-3"
          role="alert"
        >
          <TriangleAlert class="size-4 shrink-0" />
          {createError}
        </p>
      {/if}

      <!-- CTA button -->
      <Button
        class="h-[46px] w-full rounded-[10px] text-[15px] font-semibold"
        onclick={handleNext}
        disabled={!canAdvance || submitting}
        aria-label="Next — add sources"
      >
        Next — add sources
        <ArrowRight class="size-[15px]" />
      </Button>
    </Card>
  </div>
</main>
