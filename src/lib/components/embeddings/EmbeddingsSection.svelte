<script lang="ts">
  // Shared Embeddings section — the single component behind BOTH the global
  // Settings>Embeddings panel and the per-notebook "{notebook} settings" sheet
  // (plan Steps 10 + 11, ADR decision B1). Two modes:
  //
  //   mode="global"   — sets the APP-WIDE default (new notebooks). Persists
  //                     config.embedding_model + config.embedding_backend via the
  //                     standard read-modify-write. NO re-embed warning (changing
  //                     the default never touches an existing notebook's vectors).
  //
  //   mode="notebook" — changes ONE notebook + re-indexes it. Reads the current
  //                     coordinate via get_notebook_embedding_model; a change with
  //                     an indexed coordinate opens the re-embed confirm dialog,
  //                     then streams set_notebook_embedding_model progress (the
  //                     picker is disabled while a re-embed is in flight).
  //
  // Layout matches the design's Embeddings panel verbatim (Lens.dc.html
  // `stIsEmbed`): title "Embeddings" + subtitle "Local only — all vectors
  // computed on-device.", a provider selector (the user-specified extension),
  // then the model radio-list of cards. Tokens only — light + dark + every
  // accent ([[theming-light-dark-accent]]).

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
  import {
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_BACKEND,
    resolveModel,
    resolveBackend,
    modelMeta,
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
    setNotebookEmbeddingModel
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
    /**
     * Compact layout (onboarding inline expansion): tighter spacing + `h-9`
     * provider buttons. The default (`false`) is the roomy Settings/sheet layout
     * with `h-10` buttons. Behavior is identical; only the chrome differs so the
     * onboarding panel can reuse this component instead of forking it.
     */
    compact?: boolean;
    /**
     * Render the "Embeddings" title + "Local only…" subtitle. The onboarding
     * panel supplies its own row label, so it sets this `false`.
     */
    showHeader?: boolean;
    /**
     * Callback after a successful persist/re-embed (e.g. close the sheet, or
     * re-run the onboarding system-check and collapse the panel). May be async.
     */
    onchange?: () => void | Promise<void>;
  } = $props();

  // ── Selection state ────────────────────────────────────────────────────────
  let backend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);
  let selectedModel = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);

  // The currently-persisted coordinate (global: config; notebook: the index row).
  let activeModel = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);
  let activeBackend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);
  // Notebook mode: whether the current coordinate is already indexed ("active").
  let coordinateIndexed = $state(false);

  // ── Availability state (per backend) ────────────────────────────────────────
  let fastembedCached = $state<Set<EmbeddingModelId>>(new Set());
  let ollamaInstalled = $state<Set<EmbeddingModelId>>(new Set());
  let ollamaEndpoint = $state('http://localhost:11434');
  let refreshing = $state(false);

  // ── fastembed install state ─────────────────────────────────────────────────
  let installProgress = $state<number | null>(null);
  let installPhase = $state<string>('');
  let actionError = $state<string | null>(null);

  // ── notebook re-embed state ──────────────────────────────────────────────────
  let confirmOpen = $state(false);
  let reembedDone = $state(0);
  let reembedTotal = $state(0);
  let reembedding = $state(false);

  const installed = $derived(backend === 'fastembed' ? fastembedCached : ollamaInstalled);
  const selectedReady = $derived(installed.has(selectedModel));
  // Disable the picker while a notebook re-embed is in flight (status-driven).
  const pickerDisabled = $derived(reembedding);

  async function refreshOllama(): Promise<void> {
    refreshing = true;
    try {
      const names = await listOllamaModels(ollamaEndpoint);
      const found = new Set<EmbeddingModelId>();
      for (const m of EMBEDDING_MODELS) {
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
    if (mode === 'global') {
      if (isTauri()) {
        try {
          const cfg = await invoke<AppConfig>('get_config');
          activeBackend = resolveBackend(cfg.embedding_backend);
          activeModel = resolveModel(cfg.embedding_model).id;
          const ep = cfg.endpoints?.ollama;
          if (ep) ollamaEndpoint = ep;
        } catch {
          // Non-fatal: fall back to defaults.
        }
      }
    } else if (notebookId) {
      const info = await getNotebookEmbeddingModel(notebookId);
      activeBackend = resolveBackend(info.backend);
      activeModel = resolveModel(info.model_id).id;
      coordinateIndexed = info.status === 'active';
      // If a re-embed is detected in flight (active but mid-stream), the picker
      // stays disabled until it completes; the status is the gate.
    }
    backend = activeBackend;
    selectedModel = activeModel;
    await Promise.all([refreshFastembed(), refreshOllama()]);
  });

  function pickBackend(b: EmbeddingBackend): void {
    if (pickerDisabled) return;
    backend = b;
    actionError = null;
  }

  function pickModel(id: EmbeddingModelId): void {
    if (pickerDisabled) return;
    selectedModel = id;
    actionError = null;
  }

  // ── fastembed install (warm: download weights to disk) ───────────────────────
  // fastembed has no byte-level progress (init is opaque), so we advance the
  // design's phase copy (Downloading → Extracting → Configuring → Almost ready)
  // on a timer for feedback, and resolve when the warm command completes.
  let phaseTimer: ReturnType<typeof setInterval> | null = null;

  // Clear the phase ticker if the surface is unmounted mid-install (e.g. the
  // dialog/sheet is closed) so the interval never fires against a destroyed graph.
  onDestroy(() => {
    if (phaseTimer) clearInterval(phaseTimer);
  });

  async function handleInstall(): Promise<void> {
    actionError = null;
    installProgress = 0; // non-null marks "installing" (indeterminate)
    const phases = ['Downloading…', 'Extracting…', 'Configuring…', 'Almost ready…'];
    let i = 0;
    installPhase = phases[0];
    phaseTimer = setInterval(() => {
      i = Math.min(i + 1, phases.length - 1);
      installPhase = phases[i];
    }, 1200);
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

  // ── commit the selection (the mode-specific persist) ──────────────────────────
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

  // Notebook mode: a change to an INDEXED coordinate needs a re-embed (confirm
  // first). An unindexed coordinate (no vectors yet) is applied without a warning.
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

  // The primary action button label depends on backend + readiness + dirty state.
  const isDirty = $derived(selectedModel !== activeModel || backend !== activeBackend);
  const needsInstall = $derived(backend === 'fastembed' && !selectedReady);

  const reembedPct = $derived(
    reembedTotal > 0 ? Math.min(100, Math.round((reembedDone / reembedTotal) * 100)) : 0
  );
</script>

<section class="flex flex-col" aria-label="Embeddings settings">
  {#if showHeader}
    <!-- Title + subtitle (verbatim design copy) -->
    <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Embeddings</h2>
    <p class="mt-1 text-[0.8rem] text-muted-foreground">
      Local only — all vectors computed on-device.
    </p>
  {/if}

  <!-- ── Provider selector (the design extension the user specified) ── -->
  <div class={compact ? '' : 'mt-6'}>
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Select your local embeddings provider
    </p>
    <div class="mt-2.5 grid grid-cols-2 gap-2" role="radiogroup" aria-label="Embeddings provider">
      {#each [{ id: 'fastembed', label: 'fastembed' }, { id: 'ollama', label: 'Ollama' }] as p (p.id)}
        {@const isSel = backend === p.id}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          aria-disabled={pickerDisabled}
          disabled={pickerDisabled}
          onclick={() => pickBackend(p.id as EmbeddingBackend)}
          class={cn(
            'rounded-lg border px-3 text-sm font-semibold transition-colors',
            compact ? 'h-9' : 'h-10',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel
              ? 'border-primary bg-primary/10 text-foreground ring-1 ring-primary'
              : 'border-border bg-card text-muted-foreground hover:text-foreground',
            pickerDisabled && 'cursor-not-allowed opacity-60'
          )}
        >
          {p.label}
        </button>
      {/each}
    </div>
  </div>

  <!-- ── Embedding model radio-list ── -->
  <p class="mt-6 text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
    Embedding model
  </p>
  <div class="mt-2.5 flex flex-col gap-1.5" role="radiogroup" aria-label="Embedding model">
    {#each EMBEDDING_MODELS as model (model.id)}
      {@const isSelected = selectedModel === model.id}
      {@const isReady = installed.has(model.id)}
      <div
        class={cn(
          'w-full rounded-[10px] border px-3 py-3 transition-colors',
          isSelected ? 'border-primary bg-primary/10 ring-1 ring-primary' : 'border-border bg-card'
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
            'flex w-full items-center gap-2.5 text-left',
            pickerDisabled && 'cursor-not-allowed'
          )}
        >
          <span class="min-w-0 flex-1">
            <span class="block text-[0.78rem] font-bold text-foreground">{model.label}</span>
            <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">{modelMeta(model)}</span
            >
          </span>
          {#if isReady}
            <span
              class="flex shrink-0 items-center gap-1 text-[0.66rem] font-semibold text-green-primary"
              aria-label={`${model.label} ready`}
            >
              <CircleCheck class="size-3.5" />
              Ready
            </span>
          {/if}
          <!-- Radio dot -->
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

  <!-- ── Action + status row ── -->
  {#if actionError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{actionError}</p>
  {/if}

  {#if reembedding}
    <!-- Re-embed in progress — amber-pulse dot (shared sources/status idiom). -->
    <div class="mt-4 flex items-center gap-2.5" aria-live="polite">
      <span class="size-2 shrink-0 rounded-full bg-amber-500 animate-pulse" aria-hidden="true"
      ></span>
      <span class="text-[0.78rem] text-foreground">
        Re-embedding sources… {reembedTotal > 0
          ? `${reembedDone}/${reembedTotal} (${reembedPct}%)`
          : ''}
      </span>
    </div>
  {:else if backend === 'ollama'}
    <!-- Ollama: detect-only + Refresh + install hint (the app never pulls). -->
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
    <!-- fastembed: Install + download-progress. -->
    <Button
      class="mt-4 h-10 w-full"
      onclick={handleInstall}
      disabled={installProgress !== null}
      aria-label={`Install ${resolveModel(selectedModel).label}`}
    >
      {#if installProgress !== null}
        <LoaderCircle class="size-4 animate-spin" />
        {installPhase}
      {:else}
        <Download class="size-4" />
        Install {resolveModel(selectedModel).label}
      {/if}
    </Button>
  {:else if isDirty}
    <Button class="mt-4 h-10 w-full" onclick={commitSelection} aria-label="Apply selected model">
      {mode === 'notebook' ? 'Switch model' : 'Set as default'}
    </Button>
  {/if}

  <!-- Provider helper note (anchored at the bottom of the panel). -->
  <p class="mt-5 text-[0.7rem] leading-relaxed text-muted-foreground">
    Ollama must be installed if chosen — Lens detects your local models but never downloads them.
  </p>
</section>

<!-- ── Re-embed confirm dialog (notebook mode only) ── -->
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
