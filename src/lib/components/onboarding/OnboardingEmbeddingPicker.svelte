<!--
  OnboardingEmbeddingPicker — inline, always-visible embedding gate for onboarding.
  fastembed downloads in place; Ollama is detect-only. Persists the global default
  reactively (no Save button); full config lives in Settings › Embeddings. State +
  IPC live in the shared EmbeddingPickerState controller.
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Boxes from '@lucide/svelte/icons/boxes';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import { prefersReducedMotion } from '$lib/motion/index.js';
  import {
    formatSize,
    type EmbeddingBackend,
    type EmbeddingModelId
  } from '$lib/embeddings/models.js';
  import { EmbeddingPickerState, CHIP_CLASS } from '$lib/embeddings/pickerState.svelte.js';
  import type { CheckResult } from '$lib/onboarding/system-check.js';
  import SystemCheckTile from './SystemCheckTile.svelte';
  import InstallRingButton from '$lib/components/embeddings/InstallRingButton.svelte';

  let {
    result,
    oncheck
  }: {
    /** Authoritative embedding gate row from run_system_check — drives the header pill. */
    result: CheckResult;
    /** Re-run the parent system check after a persist / detection refresh. */
    oncheck?: () => Promise<void>;
  } = $props();

  // onchange read via a thunk so a changed `oncheck` handler is always seen.
  const picker = new EmbeddingPickerState({ mode: 'global', onchange: () => oncheck?.() });

  onMount(() => {
    void picker.init();
  });
  onDestroy(() => picker.dispose());

  const reduceMotion = $derived(prefersReducedMotion());
  // Header reflects the authoritative gate, not the local probe, so it can never
  // contradict the footer's Continue button.
  const ready = $derived(result.status === 'pass');

  const providers: { id: EmbeddingBackend; label: string }[] = [
    { id: 'fastembed', label: 'On-device' },
    { id: 'ollama', label: 'Ollama' }
  ];

  // Reactive persist: selecting an installed model makes it the default at once.
  async function onPickModel(id: EmbeddingModelId): Promise<void> {
    picker.pickModel(id);
    if (picker.installed.has(id)) await picker.commit();
  }

  // Re-detect Ollama AND re-run the gate, so a freshly-started daemon flips the
  // footer, not just the local dots (otherwise the header could read Ready while
  // Continue stays disabled).
  async function onRefresh(): Promise<void> {
    await picker.refreshOllama();
    await oncheck?.();
  }
</script>

<SystemCheckTile icon={Boxes} title="Embedding model" subtitle="Powers search over your sources">
  {#snippet status()}
    <span
      class={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.7rem] font-medium ${
        ready ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
      }`}
    >
      <span class="size-1.5 rounded-full bg-current" aria-hidden="true"></span>
      {ready ? 'Ready' : 'Needs a model'}
    </span>
  {/snippet}

  <div class="mt-3 flex flex-col">
    <!-- Provider tabs: the backend switch is one slim row, reclaiming the space a
         left rail would cost in this narrow tile. -->
    <div
      class="grid grid-cols-2 gap-1.5 rounded-[11px] bg-muted p-1"
      role="radiogroup"
      aria-label="Embeddings provider"
    >
      {#each providers as p (p.id)}
        {@const isSel = picker.backend === p.id}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          disabled={picker.busy}
          onclick={() => picker.pickBackend(p.id)}
          class={cn(
            'flex items-center justify-center gap-2 rounded-[8px] px-3 py-2 text-[0.78rem] font-bold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel
              ? 'bg-card text-foreground shadow-[var(--shadow-tile)]'
              : 'text-muted-foreground hover:text-foreground',
            picker.busy && 'cursor-not-allowed opacity-60'
          )}
        >
          <span
            class={cn(
              'size-1.5 rounded-full',
              picker.providerReady(p.id) ? 'bg-primary' : 'bg-muted-foreground/50'
            )}
            aria-hidden="true"
          ></span>
          {p.label}
        </button>
      {/each}
    </div>

    <!-- Focused model: the selected model in full, with its primary action. -->
    <div
      class="mt-3 flex items-center gap-3 rounded-[12px] p-3 ring-[1.5px] ring-inset ring-primary bg-[color-mix(in_oklch,var(--primary)_6%,transparent)] dark:bg-[color-mix(in_oklch,var(--primary)_12%,transparent)]"
    >
      <div class="min-w-0 flex-1">
        <div class="truncate font-mono text-[0.9rem] font-bold text-foreground">
          {picker.focusedModel.label}
        </div>
        <div class="mt-0.5 text-[0.72rem] text-muted-foreground">{picker.focusedModel.desc}</div>
        <div class="mt-2 flex flex-wrap items-center gap-1.5">
          <span class={CHIP_CLASS}>{picker.focusedModel.dims}d</span>
          <span class={CHIP_CLASS}>{formatSize(picker.focusedModel.sizeMb)}</span>
          <span class={CHIP_CLASS}>{picker.focusedModel.speed}</span>
          {#if picker.gpuModels.has(picker.focusedModel.id)}
            <span
              class="flex items-center gap-1 text-[0.66rem] font-semibold text-primary"
              aria-label={`${picker.focusedModel.label} runs on the Apple GPU`}
            >
              <span aria-hidden="true">⚡</span>
              Apple GPU
            </span>
          {/if}
        </div>
      </div>

      <div class="shrink-0">
        {#if picker.selectedReady}
          <span
            class="flex items-center gap-1 text-[0.72rem] font-bold text-primary"
            aria-label={`${picker.focusedModel.label} ${picker.backend === 'ollama' ? 'detected' : 'ready'}`}
          >
            <CircleCheck class="size-3.5" />
            {picker.backend === 'ollama' ? 'Detected' : 'Ready'}
          </span>
        {:else if picker.backend === 'fastembed'}
          <InstallRingButton
            installing={picker.installing}
            label={`Install ${picker.focusedModel.label}`}
            onclick={() => picker.install()}
          />
        {/if}
      </div>
    </div>

    <!-- Quick-switch: every model of this backend as a pill; the dot encodes
         install/detect state, so all options and their status show at a glance. -->
    <p
      class="mt-3.5 mb-2 text-[0.6rem] font-bold uppercase tracking-[0.07em] text-muted-foreground"
    >
      {picker.backend === 'fastembed' ? 'On-device models' : 'Ollama models'}
    </p>
    <div class="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Embedding model">
      {#each picker.models as m (m.id)}
        {@const isSel = picker.selectedModel === m.id}
        {@const isReady = picker.installed.has(m.id)}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          disabled={picker.busy}
          onclick={() => onPickModel(m.id)}
          class={cn(
            'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[0.72rem] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel
              ? 'border-primary/45 bg-primary/10 text-foreground'
              : 'border-input bg-card text-muted-foreground hover:text-foreground',
            picker.busy && 'cursor-not-allowed opacity-60'
          )}
        >
          <span
            class={cn(
              'size-1.5 rounded-full',
              isReady ? 'bg-primary' : 'ring-[1.5px] ring-inset ring-muted-foreground'
            )}
            aria-hidden="true"
          ></span>
          {m.short}
        </button>
      {/each}
    </div>

    <!-- Reserved footer: sized to the tallest state (Ollama hint + note) so
         switching backend / install-state never changes the tile height. -->
    <div class="mt-3 flex min-h-[4.75rem] flex-col gap-2">
      {#if picker.backend === 'fastembed' && picker.installing}
        <p class="text-[0.72rem] text-muted-foreground" aria-live="polite">
          {reduceMotion ? 'Installing…' : picker.installPhase}
        </p>
      {/if}

      {#if picker.backend === 'ollama' && !picker.selectedReady}
        <div class="flex items-center gap-2.5">
          <Button
            variant="outline"
            size="icon"
            onclick={onRefresh}
            disabled={picker.refreshing}
            aria-label="Refresh Ollama models"
            class="size-8 rounded-lg"
          >
            {#if picker.refreshing}
              <LoaderCircle class="size-4 animate-spin" />
            {:else}
              <RefreshCw class="size-4" />
            {/if}
          </Button>
          <span class="text-[0.72rem] text-muted-foreground">
            Pull with
            <code class="rounded bg-muted px-1.5 py-0.5 text-[0.7rem] text-foreground"
              >ollama pull {picker.focusedModel.ollamaName}</code
            >
          </span>
        </div>
      {/if}

      {#if picker.backend === 'ollama'}
        <p class="text-[0.68rem] leading-relaxed text-muted-foreground">
          Lens detects your local Ollama models but never downloads them.
        </p>
      {/if}

      {#if picker.actionError}
        <p class="text-[0.72rem] text-destructive" role="alert">{picker.actionError}</p>
      {/if}
    </div>
  </div>
</SystemCheckTile>
