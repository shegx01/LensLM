<script lang="ts">
  // Shared Embeddings section used in both global Settings and per-notebook sheets.
  // mode="global" sets the app-wide default; mode="notebook" changes one notebook
  // and streams re-embed progress when the coordinate is already indexed.

  import { onMount, onDestroy } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
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
  import Download from '@lucide/svelte/icons/download';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';
  import { prefersReducedMotion } from '$lib/motion/index.js';
  import {
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_BACKEND,
    resolveModel,
    resolveBackend,
    formatSize,
    ollamaMatches,
    REEMBED_WARNING,
    type EmbeddingBackend,
    type EmbeddingModelId
  } from '$lib/embeddings/models.js';
  import {
    fastembedModelsCached,
    listOllamaModels,
    warmFastembedModel,
    getNotebookEmbeddingModel,
    setNotebookEmbeddingModel,
    gpuAcceleratedModels
  } from '$lib/embeddings/ipc.js';

  let {
    mode,
    notebookId = null,
    compact = false,
    showHeader = true,
    onchange
  }: {
    mode: 'global' | 'notebook';
    /** Required in notebook mode — the notebook whose coordinate this edits. */
    notebookId?: string | null;
    compact?: boolean;
    showHeader?: boolean;
    onchange?: () => void | Promise<void>;
  } = $props();

  let backend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);
  let selectedModel = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);

  let activeModel = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);
  let activeBackend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);
  let coordinateIndexed = $state(false);

  let fastembedCached = $state<Set<EmbeddingModelId>>(new Set());
  let ollamaInstalled = $state<Set<EmbeddingModelId>>(new Set());
  let ollamaEndpoint = $state('http://localhost:11434');
  let refreshing = $state(false);
  // GPU acceleration is per-model (candle+Metal for nomic; CPU for others),
  // so the badge is keyed on this set, not the provider (issue #91).
  let gpuModels = $state<Set<string>>(new Set());

  let installProgress = $state<number | null>(null);
  let installPhase = $state<string>('');
  let actionError = $state<string | null>(null);

  let confirmOpen = $state(false);
  let reembedDone = $state(0);
  let reembedTotal = $state(0);
  let reembedding = $state(false);

  const installed = $derived(backend === 'fastembed' ? fastembedCached : ollamaInstalled);
  const selectedReady = $derived(installed.has(selectedModel));
  const pickerDisabled = $derived(reembedding);

  const filteredModels = $derived(EMBEDDING_MODELS.filter((m) => m.backends.includes(backend)));

  const providers = [
    { id: 'fastembed', label: 'On-device' },
    { id: 'ollama', label: 'Ollama' }
  ];

  async function refreshOllama(): Promise<void> {
    refreshing = true;
    try {
      const names = await listOllamaModels(ollamaEndpoint);
      const found = new Set<EmbeddingModelId>();
      for (const m of EMBEDDING_MODELS.filter((m) => m.backends.includes('ollama'))) {
        if (names.some((d) => ollamaMatches(d, m))) found.add(m.id);
      }
      ollamaInstalled = found;
    } catch {
      ollamaInstalled = new Set();
    } finally {
      refreshing = false;
    }
  }

  async function refreshFastembed(): Promise<void> {
    try {
      const ids = await fastembedModelsCached();
      fastembedCached = new Set(ids.map((id) => resolveModel(id).id));
    } catch {
      fastembedCached = new Set();
    }
  }

  onMount(async () => {
    void gpuAcceleratedModels().then((ids) => {
      gpuModels = new Set(ids);
    });
    if (mode === 'global') {
      if (isTauri()) {
        try {
          const cfg = await invoke<AppConfig>('get_config');
          activeBackend = resolveBackend(cfg.embedding_backend);
          activeModel = resolveModel(cfg.embedding_model).id;
          const ep = cfg.endpoints?.ollama;
          if (ep) ollamaEndpoint = ep;
        } catch {
          // fall back to defaults
        }
      }
    } else if (notebookId) {
      const info = await getNotebookEmbeddingModel(notebookId);
      activeBackend = resolveBackend(info.backend);
      activeModel = resolveModel(info.model_id).id;
      coordinateIndexed = info.status === 'active';
    }
    backend = activeBackend;
    selectedModel = activeModel;
    await Promise.all([refreshFastembed(), refreshOllama()]);
  });

  function pickBackend(b: EmbeddingBackend): void {
    if (pickerDisabled) return;
    backend = b;
    actionError = null;
    const newFiltered = EMBEDDING_MODELS.filter((m) => m.backends.includes(b));
    if (!newFiltered.some((m) => m.id === selectedModel)) {
      selectedModel = newFiltered[0]?.id ?? DEFAULT_EMBEDDING_MODEL;
    }
  }

  function pickModel(id: EmbeddingModelId): void {
    if (pickerDisabled) return;
    selectedModel = id;
    actionError = null;
  }

  // fastembed has no byte-level progress, so phase copy advances on a timer.
  let phaseTimer: ReturnType<typeof setInterval> | null = null;

  onDestroy(() => {
    if (phaseTimer) clearInterval(phaseTimer);
  });

  async function handleInstall(): Promise<void> {
    actionError = null;
    installProgress = 0; // non-null marks "installing" (indeterminate)
    const phases = ['Downloading…', 'Extracting…', 'Configuring…', 'Almost ready…'];
    let i = 0;
    installPhase = phases[0];
    // Reduced motion renders a hard-coded "Installing…" label, so the phase
    // ticker is dead work in that branch — skip scheduling it entirely.
    if (!reduceMotion) {
      phaseTimer = setInterval(() => {
        i = Math.min(i + 1, phases.length - 1);
        installPhase = phases[i];
      }, 1200);
    }
    try {
      await warmFastembedModel(selectedModel);
      await refreshFastembed();
      await commitSelection();
    } catch (err) {
      actionError = err instanceof Error ? err.message : 'Installation failed.';
    } finally {
      if (phaseTimer) clearInterval(phaseTimer);
      phaseTimer = null;
      installProgress = null;
    }
  }

  async function commitSelection(): Promise<void> {
    if (mode === 'global') {
      await persistGlobal();
    } else {
      await maybeReembed();
    }
  }

  async function persistGlobal(): Promise<void> {
    actionError = null;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        embedding_model: selectedModel,
        embedding_backend: backend
      }));
      activeModel = selectedModel;
      activeBackend = backend;
      await onchange?.();
    } catch (err) {
      actionError = err instanceof Error ? err.message : 'Could not save the default.';
    }
  }

  // An indexed coordinate change opens the confirm dialog; unindexed applies immediately.
  async function maybeReembed(): Promise<void> {
    if (coordinateIndexed && (selectedModel !== activeModel || backend !== activeBackend)) {
      confirmOpen = true;
      return;
    }
    await runReembed();
  }

  async function runReembed(): Promise<void> {
    if (!notebookId) return;
    confirmOpen = false;
    actionError = null;
    reembedding = true;
    reembedDone = 0;
    reembedTotal = 0;
    try {
      await setNotebookEmbeddingModel(notebookId, selectedModel, backend, (done, total) => {
        reembedDone = done;
        reembedTotal = total;
      });
      activeModel = selectedModel;
      activeBackend = backend;
      coordinateIndexed = true;
      await onchange?.();
    } catch (err) {
      actionError = err instanceof Error ? err.message : 'Re-embedding failed.';
    } finally {
      reembedding = false;
    }
  }

  const isDirty = $derived(selectedModel !== activeModel || backend !== activeBackend);
  const needsInstall = $derived(backend === 'fastembed' && !selectedReady);

  const reduceMotion = $derived(prefersReducedMotion());

  const reembedPct = $derived(
    reembedTotal > 0 ? Math.min(100, Math.round((reembedDone / reembedTotal) * 100)) : 0
  );

  const CHIP =
    'rounded-full bg-muted px-1.5 py-0.5 text-[0.62rem] font-semibold text-muted-foreground';
</script>

<section class="flex flex-col" aria-label="Embeddings settings">
  {#if showHeader}
    <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Embeddings</h2>
    <p class="mt-1 text-[0.8rem] text-muted-foreground">
      Local only — all vectors computed on-device.
    </p>
  {/if}

  <div class={compact ? '' : 'mt-6'}>
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Select your local embeddings provider
    </p>
    <div class="mt-2.5 grid grid-cols-2 gap-2" role="radiogroup" aria-label="Embeddings provider">
      {#each providers as p (p.id)}
        {@const isSel = backend === p.id}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          aria-disabled={pickerDisabled}
          disabled={pickerDisabled}
          onclick={() => pickBackend(p.id as EmbeddingBackend)}
          class={cn(
            'rounded-lg border px-3 text-sm font-semibold transition-[color,background-color,border-color,transform] duration-150',
            compact ? 'h-9' : 'h-10',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel
              ? 'border-primary bg-primary/10 text-foreground ring-1 ring-primary'
              : 'border-border bg-card text-muted-foreground hover:text-foreground',
            pickerDisabled ? 'cursor-not-allowed opacity-60' : 'active:scale-[0.97]'
          )}
        >
          {p.label}
        </button>
      {/each}
    </div>
    {#if gpuModels.has(selectedModel)}
      <p class="mt-2 flex items-center gap-1.5 text-[0.7rem] text-muted-foreground">
        <span aria-hidden="true">⚡</span>
        Best performance — embeds on your Apple GPU.
      </p>
    {/if}
  </div>

  <p class="mt-6 text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
    Embedding model
  </p>
  <div class="mt-2.5 flex flex-col gap-1.5" role="radiogroup" aria-label="Embedding model">
    {#each filteredModels as model (model.id)}
      {@const isSelected = selectedModel === model.id}
      {@const isReady = installed.has(model.id)}
      {@const isGpu = gpuModels.has(model.id)}
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
          aria-disabled={pickerDisabled}
          disabled={pickerDisabled}
          onclick={() => pickModel(model.id)}
          class={cn(
            'flex w-full items-center gap-2.5 text-left transition-transform duration-150',
            pickerDisabled ? 'cursor-not-allowed' : 'active:scale-[0.99]'
          )}
        >
          <span class="min-w-0 flex-1">
            <span class="block text-[0.78rem] font-bold text-foreground">{model.label}</span>
            <span class="mt-1 flex flex-wrap items-center gap-1">
              {#each [`${model.dims}d`, formatSize(model.sizeMb), model.speed] as chip (chip)}
                <span class={CHIP}>{chip}</span>
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

  {#if actionError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{actionError}</p>
  {/if}

  {#if reembedding}
    <div class="mt-4 flex items-center gap-2.5" aria-live="polite">
      <span class="size-2 shrink-0 rounded-full bg-amber-500 animate-pulse" aria-hidden="true"
      ></span>
      <span class="text-[0.78rem] tabular-nums text-foreground">
        Re-embedding sources… {reembedTotal > 0
          ? `${reembedDone}/${reembedTotal} (${reembedPct}%)`
          : ''}
      </span>
    </div>
  {:else if backend === 'ollama'}
    <div class="mt-4 flex flex-col gap-2.5">
      <div class="flex items-center justify-between gap-2">
        <Button
          variant="outline"
          size="sm"
          onclick={refreshOllama}
          disabled={refreshing}
          aria-label="Refresh Ollama models"
        >
          {#if refreshing}
            <LoaderCircle class="size-4 animate-spin" />
            Refreshing…
          {:else}
            <RefreshCw class="size-4" />
            Refresh
          {/if}
        </Button>
        {#if isDirty && selectedReady}
          <Button size="sm" onclick={commitSelection} aria-label="Apply selected model">
            {mode === 'notebook' ? 'Switch model' : 'Set as default'}
          </Button>
        {/if}
      </div>
      {#if !selectedReady}
        <p class="text-[0.72rem] text-muted-foreground">
          Not detected on your Ollama runtime. Pull it with
          <code class="rounded bg-muted px-1 py-0.5 text-[0.7rem] text-foreground"
            >ollama pull {resolveModel(selectedModel).ollamaName}</code
          >, then Refresh.
        </p>
      {/if}
    </div>
  {:else if needsInstall}
    <div class="mt-4 flex items-center gap-3">
      <span class="relative inline-flex size-11 shrink-0 items-center justify-center">
        {#if installProgress !== null}
          <span
            class={cn('absolute inset-0 rounded-full', !reduceMotion && 'animate-install-ring')}
            style={`border: 2.5px solid color-mix(in oklch, var(--primary) ${reduceMotion ? '45' : '20'}%, transparent); border-top-color: ${reduceMotion ? 'color-mix(in oklch, var(--primary) 45%, transparent)' : 'var(--primary)'};`}
            aria-hidden="true"
          ></span>
        {/if}
        <Button
          type="button"
          size="icon"
          onclick={handleInstall}
          disabled={installProgress !== null}
          aria-label={`Install ${resolveModel(selectedModel).label}`}
          class="relative size-9 rounded-full transition-transform duration-150 active:scale-[0.97]"
        >
          <Download class="size-4" />
        </Button>
      </span>
      {#if installProgress !== null}
        <span class="text-[0.72rem] text-muted-foreground" aria-live="polite">
          {reduceMotion ? 'Installing…' : installPhase}
        </span>
      {/if}
    </div>
  {:else if isDirty}
    <Button class="mt-4 h-10 w-full" onclick={commitSelection} aria-label="Apply selected model">
      {mode === 'notebook' ? 'Switch model' : 'Set as default'}
    </Button>
  {/if}

  <p class="mt-5 text-[0.7rem] leading-relaxed text-muted-foreground">
    Ollama must be installed if chosen — Lens detects your local models but never downloads them.
  </p>
</section>

<Dialog bind:open={confirmOpen}>
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
      <Button variant="outline" onclick={() => (confirmOpen = false)}>Cancel</Button>
      <Button onclick={runReembed} aria-label="Confirm re-embed">Re-embed from scratch</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>

<style>
  /* Indeterminate install ring: gated by --rail-motion so a runtime "reduce
     motion" toggle also stalls it, in addition to the reduceMotion JS check
     that skips this class entirely. */
  @keyframes install-ring-spin {
    to {
      transform: rotate(360deg);
    }
  }
  .animate-install-ring {
    animation: install-ring-spin calc(0.9s / max(var(--rail-motion, 1), 0.0001)) linear infinite;
  }
</style>
