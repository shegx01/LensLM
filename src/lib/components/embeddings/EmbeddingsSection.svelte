<script lang="ts">
  // Shared Embeddings section for global Settings and per-notebook sheets.
  // mode="global" sets the app-wide default; mode="notebook" changes one notebook
  // and streams re-embed progress when the coordinate is already indexed. State +
  // IPC live in the shared EmbeddingPickerState controller.

  import { onMount, onDestroy } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter
  } from '$lib/components/ui/dialog/index.js';
  import { cn } from '$lib/utils.js';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import { formatSize, REEMBED_WARNING, type EmbeddingBackend } from '$lib/embeddings/models.js';
  import {
    EmbeddingPickerState,
    CHIP_CLASS,
    PROVIDER_LABELS
  } from '$lib/embeddings/pickerState.svelte.js';
  import InstallRingButton from './InstallRingButton.svelte';

  let {
    mode,
    notebookId = null,
    showHeader = true,
    onchange
  }: {
    mode: 'global' | 'notebook';
    /** Required in notebook mode — the notebook whose coordinate this edits. */
    notebookId?: string | null;
    showHeader?: boolean;
    onchange?: () => void | Promise<void>;
  } = $props();

  // mode/notebookId are fixed for a component instance (snapshot at construction);
  // onchange is read via a thunk so a changed handler is always seen.
  // svelte-ignore state_referenced_locally
  const picker = new EmbeddingPickerState({ mode, notebookId, onchange: () => onchange?.() });

  onMount(() => {
    void picker.init();
  });
  onDestroy(() => picker.dispose());

  const providers: { id: EmbeddingBackend; label: string; desc: string }[] = (
    ['fastembed', 'ollama'] as const
  ).map((id) => ({ id, ...PROVIDER_LABELS[id] }));
  const selectedProvider = $derived(providers.find((p) => p.id === picker.backend) ?? providers[0]);
</script>

{#snippet modelSection()}
  <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
    Embedding model
  </p>
  <div class="mt-3 flex flex-col gap-1.5" role="radiogroup" aria-label="Embedding model">
    {#each picker.models as model (model.id)}
      {@const isSelected = picker.selectedModel === model.id}
      {@const isReady = picker.installed.has(model.id)}
      {@const isGpu = picker.gpuModels.has(model.id)}
      <div
        class={cn(
          'w-full rounded-[10px] border px-3 py-3 transition-[border-color,background-color] duration-150',
          isSelected
            ? 'border-primary bg-primary/10 ring-1 ring-primary'
            : 'border-border bg-card hover:border-primary/40'
        )}
      >
        <button
          type="button"
          role="radio"
          aria-checked={isSelected}
          aria-disabled={picker.busy}
          disabled={picker.busy}
          onclick={() => picker.pickModel(model.id)}
          class={cn(
            'flex w-full items-center gap-2.5 text-left transition-transform duration-150',
            picker.busy ? 'cursor-not-allowed' : 'active:scale-[0.99]'
          )}
        >
          <span class="min-w-0 flex-1">
            <span class="block text-[0.78rem] font-bold text-foreground">{model.label}</span>
            <span class="mt-1 flex flex-wrap items-center gap-1">
              {#each [`${model.dims}d`, formatSize(model.sizeMb), model.speed] as chip (chip)}
                <span class={CHIP_CLASS}>{chip}</span>
              {/each}
            </span>
            <span class="mt-1 block text-[0.68rem] text-muted-foreground">{model.desc}</span>
          </span>
          {#if isGpu}
            <span
              class="flex shrink-0 items-center gap-1 text-[0.66rem] font-semibold text-primary"
              aria-label={`${model.label} runs on the Apple GPU`}
            >
              <span aria-hidden="true">⚡</span>
              Apple GPU
            </span>
          {/if}
          {#if isReady}
            <span
              class="flex shrink-0 items-center gap-1 text-[0.66rem] font-semibold text-green-primary"
              aria-label={`${model.label} ready`}
            >
              <CircleCheck class="size-3.5" />
              Ready
            </span>
          {/if}
          <span
            class={cn(
              'flex size-4 shrink-0 items-center justify-center rounded-full border-[1.5px]',
              isSelected ? 'border-primary' : 'border-muted-foreground/40'
            )}
          >
            {#if isSelected}
              <span class="size-2 rounded-full bg-primary"></span>
            {/if}
          </span>
        </button>
      </div>
    {/each}
  </div>
  {#if picker.gpuModels.has(picker.selectedModel)}
    <p class="mt-2 flex items-center gap-1.5 text-[0.7rem] text-muted-foreground">
      <span aria-hidden="true">⚡</span>
      Best performance — embeds on your Apple GPU.
    </p>
  {/if}
{/snippet}

{#snippet actionsBlock()}
  {#if picker.actionError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{picker.actionError}</p>
  {/if}

  {#if picker.reembedding}
    <div class="mt-4 flex items-center gap-2.5" aria-live="polite">
      <span class="size-2 shrink-0 rounded-full bg-amber-500 animate-pulse" aria-hidden="true"
      ></span>
      <span class="text-[0.78rem] tabular-nums text-foreground">
        Re-embedding sources… {picker.reembedTotal > 0
          ? `${picker.reembedDone}/${picker.reembedTotal} (${picker.reembedPct}%)`
          : ''}
      </span>
    </div>
  {:else if picker.backend === 'ollama'}
    <div class="mt-4 flex flex-col gap-2.5">
      <div class="flex items-center justify-between gap-2">
        <Button
          variant="outline"
          size="sm"
          onclick={() => picker.refreshOllama()}
          disabled={picker.refreshing}
          aria-label="Refresh Ollama models"
        >
          {#if picker.refreshing}
            <LoaderCircle class="size-4 animate-spin" />
            Refreshing…
          {:else}
            <RefreshCw class="size-4" />
            Refresh
          {/if}
        </Button>
        {#if picker.isDirty && picker.selectedReady}
          <Button size="sm" onclick={() => picker.commit()} aria-label="Apply selected model">
            {mode === 'notebook' ? 'Switch model' : 'Set as default'}
          </Button>
        {/if}
      </div>
      {#if !picker.selectedReady}
        <p class="text-[0.72rem] text-muted-foreground">
          Not detected on your Ollama runtime. Pull it with
          <code class="rounded bg-muted px-1 py-0.5 text-[0.7rem] text-foreground"
            >ollama pull {picker.focusedModel.ollamaName}</code
          >, then Refresh.
        </p>
      {/if}
    </div>
  {:else if picker.needsInstall}
    <div class="mt-4 flex items-center gap-3">
      <InstallRingButton
        installing={picker.installing}
        label={`Install ${picker.focusedModel.label}`}
        onclick={() => picker.install()}
      />
      {#if picker.installing}
        <span class="text-[0.72rem] text-muted-foreground" aria-live="polite">
          {picker.installLabel}
        </span>
      {/if}
    </div>
  {:else if picker.isDirty}
    <Button
      class="mt-4 h-10 w-full"
      onclick={() => picker.commit()}
      aria-label="Apply selected model"
    >
      {mode === 'notebook' ? 'Switch model' : 'Set as default'}
    </Button>
  {/if}

  <p class="mt-5 text-[0.7rem] leading-relaxed text-muted-foreground">
    Ollama must be installed if chosen — Lens detects your local models but never downloads them.
  </p>
{/snippet}

<section class="flex flex-col" aria-label="Embeddings settings">
  {#if showHeader}
    <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Embeddings</h2>
    <p class="mt-1 text-[0.8rem] text-muted-foreground">
      Local only — all vectors computed on-device.
    </p>
  {/if}

  <!-- Master-detail two-column (mirrors the AI Model Providers pane). -->
  <div class="mt-6 grid grid-cols-1 items-start gap-3.5 md:grid-cols-[minmax(200px,0.85fr)_1.15fr]">
    <!-- Left rail: provider. Selection drives the detail; the persisted default gets Active. -->
    <div class="flex flex-col gap-1.5" role="radiogroup" aria-label="Embeddings provider">
      {#each providers as p (p.id)}
        {@const isSel = picker.backend === p.id}
        {@const isActive = picker.activeBackend === p.id}
        {@const isReady = picker.providerReady(p.id)}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          aria-disabled={picker.busy}
          disabled={picker.busy}
          onclick={() => picker.pickBackend(p.id)}
          class={cn(
            'flex w-full items-center gap-2.5 rounded-[10px] border px-3 py-2.5 text-left transition-[background-color,border-color,transform] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel ? 'border-primary/40 bg-primary/10' : 'border-transparent hover:bg-muted',
            picker.busy ? 'cursor-not-allowed opacity-60' : 'active:scale-[0.98]'
          )}
        >
          <span class="min-w-0 flex-1">
            <span class="flex items-center gap-2 text-[0.8rem] font-bold text-foreground">
              <span
                class={cn(
                  'size-[7px] shrink-0 rounded-full',
                  isReady ? 'bg-primary' : 'bg-muted-foreground/50'
                )}
                aria-hidden="true"
              ></span>
              <span class="truncate">{p.label}</span>
              {#if isActive}
                <span
                  class="shrink-0 rounded-full bg-primary px-1.5 py-px text-[0.58rem] font-bold uppercase tracking-[0.05em] text-primary-foreground"
                >
                  Active
                </span>
              {/if}
            </span>
            <span class="mt-px block truncate text-[0.68rem] text-muted-foreground">{p.desc}</span>
          </span>
        </button>
      {/each}
    </div>

    <!-- Right detail panel: the selected provider's models + install/apply. -->
    <div class="rounded-xl border border-border bg-card p-[18px]">
      <div class="min-w-0">
        <div class="truncate text-[0.95rem] font-extrabold text-foreground">
          {selectedProvider.label}
        </div>
        <div class="text-[0.7rem] text-muted-foreground">{selectedProvider.desc}</div>
      </div>
      <div class="mt-4">
        {@render modelSection()}
        {@render actionsBlock()}
      </div>
    </div>
  </div>
</section>

<Dialog bind:open={picker.confirmOpen}>
  <DialogContent class="max-w-md">
    <DialogHeader>
      <DialogTitle class="flex items-center gap-2">
        <TriangleAlert class="size-5 text-amber-500" />
        Re-embed this notebook?
      </DialogTitle>
      <DialogDescription class="leading-relaxed">
        {REEMBED_WARNING}
      </DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button variant="outline" onclick={() => picker.cancelReembed()}>Cancel</Button>
      <Button onclick={() => picker.runReembed()} aria-label="Confirm re-embed"
        >Re-embed from scratch</Button
      >
    </DialogFooter>
  </DialogContent>
</Dialog>
