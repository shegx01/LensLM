<!--
  OnboardingEmbeddingPicker — compact, always-visible embedding picker for the
  onboarding system check. Presents both local backends inline with no expand
  step: provider tabs, one focused model with its action, and quick-switch pills
  for the rest. On-device (fastembed) weights download in place; Ollama is
  detect-only. Persists the global default reactively (no Save button) and re-runs
  the parent system check via `oncheck`. Full config lives in Settings › Embeddings.
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import Boxes from '@lucide/svelte/icons/boxes';
  import Download from '@lucide/svelte/icons/download';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';
  import { prefersReducedMotion } from '$lib/motion/index.js';
  import SystemCheckTile from './SystemCheckTile.svelte';
  import {
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_BACKEND,
    resolveModel,
    resolveBackend,
    formatSize,
    ollamaMatches,
    type EmbeddingBackend,
    type EmbeddingModelId
  } from '$lib/embeddings/models.js';
  import {
    fastembedModelsCached,
    listOllamaModels,
    warmFastembedModel,
    gpuAcceleratedModels
  } from '$lib/embeddings/ipc.js';

  let { oncheck }: { oncheck?: () => Promise<void> } = $props();

  const feModels = EMBEDDING_MODELS.filter((m) => m.backends.includes('fastembed'));
  const olModels = EMBEDDING_MODELS.filter((m) => m.backends.includes('ollama'));
  const FIRST_OLLAMA = olModels[0]?.id ?? DEFAULT_EMBEDDING_MODEL;

  let backend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);
  let selectedFe = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);
  let selectedOl = $state<EmbeddingModelId>(FIRST_OLLAMA);
  let activeModel = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);
  let activeBackend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);

  let fastembedCached = $state<Set<EmbeddingModelId>>(new Set());
  let ollamaInstalled = $state<Set<EmbeddingModelId>>(new Set());
  let ollamaEndpoint = $state('http://localhost:11434');
  let gpuModels = $state<Set<string>>(new Set());
  let refreshing = $state(false);
  let installProgress = $state<number | null>(null); // non-null marks "installing" (indeterminate)
  let installPhase = $state('');
  let actionError = $state<string | null>(null);

  const models = $derived(backend === 'fastembed' ? feModels : olModels);
  const installed = $derived(backend === 'fastembed' ? fastembedCached : ollamaInstalled);
  const selectedId = $derived(backend === 'fastembed' ? selectedFe : selectedOl);
  const focused = $derived(models.find((m) => m.id === selectedId) ?? models[0]);
  const focusedReady = $derived(installed.has(focused.id));
  const reduceMotion = $derived(prefersReducedMotion());
  // Header pill tracks the PERSISTED default (the value the gate checks), not the
  // model being previewed — so browsing an uninstalled model never reads as Ready.
  const activeReady = $derived(
    activeBackend === 'fastembed'
      ? fastembedCached.has(activeModel)
      : ollamaInstalled.has(activeModel)
  );

  const CHIP =
    'rounded-full bg-muted px-1.5 py-0.5 text-[0.62rem] font-semibold text-muted-foreground';

  function providerReady(b: EmbeddingBackend): boolean {
    return b === 'fastembed' ? true : ollamaInstalled.size > 0;
  }

  // Compact pill label — trims the shared vocabulary that bloats the full ids.
  function shortLabel(id: string): string {
    return id
      .replace('-embed-text', '')
      .replace('-embed-large', '')
      .replace('-embedding', '')
      .replace('embed-', '');
  }

  async function refreshFastembed(): Promise<void> {
    try {
      const ids = await fastembedModelsCached();
      fastembedCached = new Set(ids.map((id) => resolveModel(id).id));
    } catch {
      fastembedCached = new Set();
    }
  }

  async function refreshOllama(): Promise<void> {
    refreshing = true;
    try {
      const names = await listOllamaModels(ollamaEndpoint);
      const found = new Set<EmbeddingModelId>();
      for (const m of olModels) {
        if (names.some((d) => ollamaMatches(d, m))) found.add(m.id);
      }
      ollamaInstalled = found;
    } catch {
      ollamaInstalled = new Set();
    } finally {
      refreshing = false;
    }
  }

  onMount(async () => {
    void gpuAcceleratedModels().then((ids) => (gpuModels = new Set(ids)));
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
    backend = activeBackend;
    if (activeBackend === 'fastembed') selectedFe = activeModel;
    else selectedOl = activeModel;
    await Promise.all([refreshFastembed(), refreshOllama()]);
  });

  let phaseTimer: ReturnType<typeof setInterval> | null = null;
  onDestroy(() => {
    if (phaseTimer) clearInterval(phaseTimer);
  });

  function pickBackend(b: EmbeddingBackend): void {
    if (installProgress !== null) return;
    backend = b;
    actionError = null;
  }

  async function pickModel(id: EmbeddingModelId): Promise<void> {
    if (installProgress !== null) return;
    actionError = null;
    if (backend === 'fastembed') selectedFe = id;
    else selectedOl = id;
    // Reactive persist: selecting a ready model makes it the default immediately
    // (no Save button). An uninstalled pick just previews — never persisted.
    if (installed.has(id)) await persist(id, backend);
  }

  async function persist(model: EmbeddingModelId, b: EmbeddingBackend): Promise<void> {
    try {
      await updateConfig((cfg) => ({ ...cfg, embedding_model: model, embedding_backend: b }));
      activeModel = model;
      activeBackend = b;
      await oncheck?.();
    } catch (err) {
      actionError = err instanceof Error ? err.message : 'Could not save the default.';
    }
  }

  async function handleInstall(): Promise<void> {
    actionError = null;
    installProgress = 0;
    const phases = ['Downloading…', 'Extracting…', 'Configuring…', 'Almost ready…'];
    let i = 0;
    installPhase = phases[0];
    // Reduced motion shows a static "Installing…", so skip the phase ticker there.
    if (!reduceMotion) {
      phaseTimer = setInterval(() => {
        i = Math.min(i + 1, phases.length - 1);
        installPhase = phases[i];
      }, 1200);
    }
    try {
      await warmFastembedModel(selectedFe);
      await refreshFastembed();
      await persist(selectedFe, 'fastembed');
    } catch (err) {
      actionError = err instanceof Error ? err.message : 'Installation failed.';
    } finally {
      if (phaseTimer) clearInterval(phaseTimer);
      phaseTimer = null;
      installProgress = null;
    }
  }

  const providers: { id: EmbeddingBackend; label: string }[] = [
    { id: 'fastembed', label: 'On-device' },
    { id: 'ollama', label: 'Ollama' }
  ];
</script>

<SystemCheckTile icon={Boxes} title="Embedding model" subtitle="Powers search over your sources">
  {#snippet status()}
    <span
      class={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.7rem] font-medium ${
        activeReady ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
      }`}
    >
      <span class="size-1.5 rounded-full bg-current" aria-hidden="true"></span>
      {activeReady ? 'Ready' : 'Needs a model'}
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
        {@const isSel = backend === p.id}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          disabled={installProgress !== null}
          onclick={() => pickBackend(p.id)}
          class={cn(
            'flex items-center justify-center gap-2 rounded-[8px] px-3 py-2 text-[0.78rem] font-bold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel
              ? 'bg-card text-foreground shadow-[var(--shadow-tile)]'
              : 'text-muted-foreground hover:text-foreground',
            installProgress !== null && 'cursor-not-allowed opacity-60'
          )}
        >
          <span
            class={cn(
              'size-1.5 rounded-full',
              providerReady(p.id) ? 'bg-primary' : 'bg-muted-foreground/50'
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
          {focused.label}
        </div>
        <div class="mt-0.5 text-[0.72rem] text-muted-foreground">{focused.desc}</div>
        <div class="mt-2 flex flex-wrap items-center gap-1.5">
          <span class={CHIP}>{focused.dims}d</span>
          <span class={CHIP}>{formatSize(focused.sizeMb)}</span>
          <span class={CHIP}>{focused.speed}</span>
          {#if gpuModels.has(focused.id)}
            <span
              class="flex items-center gap-1 text-[0.66rem] font-semibold text-primary"
              aria-label={`${focused.label} runs on the Apple GPU`}
            >
              <span aria-hidden="true">⚡</span>
              Apple GPU
            </span>
          {/if}
        </div>
      </div>

      <div class="shrink-0">
        {#if focusedReady}
          <span
            class="flex items-center gap-1 text-[0.72rem] font-bold text-primary"
            aria-label={`${focused.label} ${backend === 'ollama' ? 'detected' : 'ready'}`}
          >
            <CircleCheck class="size-3.5" />
            {backend === 'ollama' ? 'Detected' : 'Ready'}
          </span>
        {:else if backend === 'fastembed'}
          <span class="relative inline-flex size-11 items-center justify-center">
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
              aria-label={`Install ${focused.label}`}
              class="relative size-9 rounded-full transition-transform duration-150 active:scale-[0.97]"
            >
              <Download class="size-4" />
            </Button>
          </span>
        {/if}
      </div>
    </div>

    <!-- Quick-switch: every model of this backend as a pill; the dot encodes
         install/detect state, so all options and their status show at a glance. -->
    <p
      class="mt-3.5 mb-2 text-[0.6rem] font-bold uppercase tracking-[0.07em] text-muted-foreground"
    >
      {backend === 'fastembed' ? 'On-device models' : 'Ollama models'}
    </p>
    <div class="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Embedding model">
      {#each models as m (m.id)}
        {@const isSel = selectedId === m.id}
        {@const isReady = installed.has(m.id)}
        <button
          type="button"
          role="radio"
          aria-checked={isSel}
          disabled={installProgress !== null}
          onclick={() => pickModel(m.id)}
          class={cn(
            'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[0.72rem] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            isSel
              ? 'border-primary/45 bg-primary/10 text-foreground'
              : 'border-input bg-card text-muted-foreground hover:text-foreground',
            installProgress !== null && 'cursor-not-allowed opacity-60'
          )}
        >
          <span
            class={cn(
              'size-1.5 rounded-full',
              isReady ? 'bg-primary' : 'ring-[1.5px] ring-inset ring-muted-foreground'
            )}
            aria-hidden="true"
          ></span>
          {shortLabel(m.id)}
        </button>
      {/each}
    </div>

    <!-- Reserved footer: sized to the tallest state (Ollama hint + note) so
         switching backend / install-state never changes the tile height. -->
    <div class="mt-3 flex min-h-[4.75rem] flex-col gap-2">
      {#if backend === 'fastembed' && installProgress !== null}
        <p class="text-[0.72rem] text-muted-foreground" aria-live="polite">
          {reduceMotion ? 'Installing…' : installPhase}
        </p>
      {/if}

      {#if backend === 'ollama' && !focusedReady}
        <div class="flex items-center gap-2.5">
          <Button
            variant="outline"
            size="icon"
            onclick={refreshOllama}
            disabled={refreshing}
            aria-label="Refresh Ollama models"
            class="size-8 rounded-lg"
          >
            {#if refreshing}
              <LoaderCircle class="size-4 animate-spin" />
            {:else}
              <RefreshCw class="size-4" />
            {/if}
          </Button>
          <span class="text-[0.72rem] text-muted-foreground">
            Pull with
            <code class="rounded bg-muted px-1.5 py-0.5 text-[0.7rem] text-foreground"
              >ollama pull {focused.ollamaName}</code
            >
          </span>
        </div>
      {/if}

      {#if backend === 'ollama'}
        <p class="text-[0.68rem] leading-relaxed text-muted-foreground">
          Lens detects your local Ollama models but never downloads them.
        </p>
      {/if}

      {#if actionError}
        <p class="text-[0.72rem] text-destructive" role="alert">{actionError}</p>
      {/if}
    </div>
  </div>
</SystemCheckTile>

<style>
  /* Indeterminate install ring: gated by --rail-motion so a runtime "reduce
     motion" toggle also stalls it, in addition to the reduceMotion JS check
     that skips the class entirely. Mirrors EmbeddingsSection. */
  @keyframes install-ring-spin {
    to {
      transform: rotate(360deg);
    }
  }
  .animate-install-ring {
    animation: install-ring-spin calc(0.9s / max(var(--rail-motion, 1), 0.0001)) linear infinite;
  }
</style>
