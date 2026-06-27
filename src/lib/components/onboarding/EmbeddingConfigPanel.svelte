<script lang="ts">
  // Onboarding embedding-config panel (plan Step 9). The inline expansion under
  // the system-check "Embedding model" row.
  //
  // Reworked for M4 Phase 4b-B (backend dimension):
  //   - Per-notebook re-embed warning (verbatim design copy) — embedding models
  //     are permanently linked to the notebook's vector index.
  //   - Provider selector ("Select your local embeddings provider": fastembed |
  //     Ollama) + a bottom note that Ollama must be installed if chosen.
  //   - fastembed: cards light up "Ready" from on-disk cache detection
  //     (fastembed_models_cached); Install warms (downloads) the weights with a
  //     phase spinner (Downloading → Extracting → Configuring → Almost ready).
  //   - Ollama: DETECT-ONLY via /api/tags + a Refresh action + a pull hint (the
  //     app NEVER pulls Ollama models).
  //   - Sets the GLOBAL default (config.embedding_model + config.embedding_backend),
  //     which new notebooks adopt.
  //
  // Tokens only — light + dark + every accent ([[theming-light-dark-accent]]).

  import { onMount, onDestroy } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import Download from '@lucide/svelte/icons/download';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';
  import {
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    resolveModel,
    resolveBackend,
    modelMeta,
    ollamaMatches,
    type EmbeddingBackend,
    type EmbeddingModelId
  } from '$lib/embeddings/models.js';
  import {
    fastembedModelsCached,
    listOllamaModels,
    warmFastembedModel
  } from '$lib/embeddings/ipc.js';

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  let backend = $state<EmbeddingBackend>('fastembed');
  let selectedModel = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);
  let activeModel = $state<EmbeddingModelId | ''>('');
  let activeBackend = $state<EmbeddingBackend>('fastembed');

  let fastembedCached = $state<Set<EmbeddingModelId>>(new Set());
  let ollamaInstalled = $state<Set<EmbeddingModelId>>(new Set());
  let endpoint = $state('http://localhost:11434');
  let refreshing = $state(false);

  let installing = $state(false);
  let installPhase = $state('');
  let actionError = $state<string | null>(null);
  let phaseTimer: ReturnType<typeof setInterval> | null = null;

  // Clear the phase ticker if the panel collapses/unmounts mid-install so the
  // interval never fires against a destroyed reactive graph.
  onDestroy(() => {
    if (phaseTimer) clearInterval(phaseTimer);
  });

  const installed = $derived(backend === 'fastembed' ? fastembedCached : ollamaInstalled);
  const selectedReady = $derived(installed.has(selectedModel));
  const isActive = $derived(
    activeModel === selectedModel && activeBackend === backend && selectedReady
  );

  async function refreshOllama(): Promise<void> {
    refreshing = true;
    try {
      const names = await listOllamaModels(endpoint);
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
    if (isTauri()) {
      try {
        const cfg = await invoke<AppConfig>('get_config');
        if (cfg.embedding_model) {
          activeModel = resolveModel(cfg.embedding_model).id;
          selectedModel = activeModel;
        }
        activeBackend = resolveBackend(cfg.embedding_backend);
        backend = activeBackend;
        const ep = cfg.endpoints?.ollama;
        if (ep) endpoint = ep;
      } catch {
        // Non-fatal: fall back to defaults.
      }
    }
    await Promise.all([refreshFastembed(), refreshOllama()]);
  });

  function pickBackend(b: EmbeddingBackend): void {
    backend = b;
    actionError = null;
  }

  function pickModel(id: EmbeddingModelId): void {
    selectedModel = id;
    actionError = null;
  }

  // Persist the GLOBAL default (model + backend), re-run the check, collapse.
  async function setGlobalDefault(): Promise<void> {
    actionError = null;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        embedding_model: selectedModel,
        embedding_backend: backend
      }));
      activeModel = selectedModel;
      activeBackend = backend;
      await oncheck();
      oncollapse();
    } catch (err) {
      actionError = err instanceof Error ? err.message : 'Could not set the default.';
    }
  }

  // fastembed Install = warm (download) the weights, then set the default.
  async function handleInstall(): Promise<void> {
    actionError = null;
    installing = true;
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
      await setGlobalDefault();
    } catch (err) {
      actionError = err instanceof Error ? err.message : 'Installation failed.';
    } finally {
      if (phaseTimer) clearInterval(phaseTimer);
      phaseTimer = null;
      installing = false;
    }
  }
</script>

<div class="flex flex-col gap-3 pt-3">
  <!-- Re-embed warning banner (verbatim design copy) -->
  <div
    class="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/15 px-3 py-2.5"
  >
    <TriangleAlert class="mt-0.5 size-4 shrink-0 text-amber-500" />
    <p class="text-[0.78rem] leading-relaxed text-amber-500">
      Embedding models are <strong>permanently linked</strong> to this notebook's vector index. Switching
      models later requires re-embedding all sources from scratch. Choose carefully.
    </p>
  </div>

  <!-- Provider selector -->
  <div>
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Select your local embeddings provider
    </p>
    <div class="mt-2 grid grid-cols-2 gap-2" role="radiogroup" aria-label="Embeddings provider">
      {#each [{ id: 'fastembed', label: 'fastembed' }, { id: 'ollama', label: 'Ollama' }] as p (p.id)}
        {@const isSel = backend === p.id}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          onclick={() => pickBackend(p.id as EmbeddingBackend)}
          class={cn(
            'h-9 rounded-lg border px-3 text-sm font-semibold transition-colors',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel
              ? 'border-primary bg-primary/10 text-foreground ring-1 ring-primary'
              : 'border-border bg-card text-muted-foreground hover:text-foreground'
          )}
        >
          {p.label}
        </button>
      {/each}
    </div>
  </div>

  <!-- Model radio-list -->
  <div class="flex flex-col gap-2" role="radiogroup" aria-label="Embedding model">
    {#each EMBEDDING_MODELS as model (model.id)}
      {@const isSelected = selectedModel === model.id}
      {@const isReady = installed.has(model.id)}
      <div
        class={cn(
          'w-full rounded-lg border px-3 py-2.5 transition-colors',
          isSelected ? 'border-primary bg-primary/10 ring-1 ring-primary' : 'border-border bg-card'
        )}
      >
        <button
          type="button"
          role="radio"
          aria-checked={isSelected}
          onclick={() => pickModel(model.id)}
          class="flex w-full items-center gap-2.5 text-left"
        >
          <span class="min-w-0 flex-1">
            <span class="block text-sm font-semibold text-foreground">{model.label}</span>
            <span class="mt-0.5 block text-[0.72rem] text-muted-foreground">{modelMeta(model)}</span
            >
          </span>
          {#if isReady}
            <span
              class="flex shrink-0 items-center gap-1 text-[0.7rem] font-semibold text-green-primary"
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
    <p class="text-[0.75rem] text-destructive" role="alert">{actionError}</p>
  {/if}

  <!-- Action row, by backend + readiness -->
  {#if backend === 'ollama'}
    <div class="flex flex-col gap-2">
      <div class="flex items-center justify-between gap-2">
        <Button
          variant="outline"
          size="sm"
          onclick={refreshOllama}
          disabled={refreshing}
          aria-label="Refresh Ollama models"
        >
          {#if refreshing}
            <LoaderCircle class="size-4 animate-spin" /> Refreshing…
          {:else}
            <RefreshCw class="size-4" /> Refresh
          {/if}
        </Button>
        {#if selectedReady && !isActive}
          <Button size="sm" onclick={setGlobalDefault} aria-label="Use selected model">
            Use {resolveModel(selectedModel).label}
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
  {:else if !selectedReady}
    <Button
      class="h-10 w-full"
      onclick={handleInstall}
      disabled={installing}
      aria-label={`Install ${resolveModel(selectedModel).label}`}
    >
      {#if installing}
        <LoaderCircle class="size-4 animate-spin" />
        {installPhase}
      {:else}
        <Download class="size-4" />
        Install {resolveModel(selectedModel).label}
      {/if}
    </Button>
  {:else if !isActive}
    <Button class="h-10 w-full" onclick={setGlobalDefault} aria-label="Use selected model">
      Use {resolveModel(selectedModel).label}
    </Button>
  {/if}

  <!-- Provider helper note -->
  <p class="text-[0.7rem] leading-relaxed text-muted-foreground">
    Ollama must be installed if chosen — Lens detects your local models but never downloads them.
  </p>
</div>
