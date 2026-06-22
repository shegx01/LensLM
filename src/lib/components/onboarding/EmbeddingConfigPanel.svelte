<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import {
    EMBEDDING_MODELS,
    installEmbeddingModel,
    detectLlm,
    type EmbeddingModelId
  } from '$lib/onboarding/system-check.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  let selectedModel = $state<EmbeddingModelId>('nomic-embed-text');
  let installProgress = $state<number | null>(null); // null = idle, 0-100 = installing
  let installed = $state(false);
  let installError = $state<string | null>(null);

  // The currently-active (persisted) embedding model id, read on mount. '' = none.
  let activeModel = $state<string>('');
  // The allowlisted models actually installed on the Ollama runtime (intersection
  // of the runtime's `models` list with EMBEDDING_MODELS), read on mount.
  let installedModels = $state<Set<EmbeddingModelId>>(new Set());
  // The Ollama endpoint to probe (configured or default).
  let endpoint = $state('http://localhost:11434');
  // Tracks which model's "Set as active" is in flight (disables that card's button).
  let settingActive = $state<EmbeddingModelId | null>(null);

  const selected = $derived(EMBEDDING_MODELS.find((m) => m.id === selectedModel)!);

  // Ollama reports model names with a tag (e.g. "nomic-embed-text:latest"); match
  // an allowlist id when a detected name equals it or starts with "<id>:".
  function isAllowlistMatch(detected: string, id: EmbeddingModelId): boolean {
    return detected === id || detected.startsWith(`${id}:`);
  }

  async function refreshInstalled(): Promise<void> {
    try {
      const result = await detectLlm(endpoint);
      const found = new Set<EmbeddingModelId>();
      for (const model of EMBEDDING_MODELS) {
        if (result.models.some((d) => isAllowlistMatch(d, model.id))) found.add(model.id);
      }
      installedModels = found;
    } catch {
      installedModels = new Set();
    }
  }

  // On mount: read the ACTIVE model from config and detect which allowlisted
  // models are installed on the runtime, so each card renders its correct state.
  onMount(async () => {
    if (isTauri()) {
      try {
        const cfg = await invoke<AppConfig>('get_config');
        if (cfg.embedding_model) activeModel = cfg.embedding_model;
        const configured = cfg.endpoints?.ollama;
        if (configured) endpoint = configured;
        // Seed the selection from the active model so the Install button (if the
        // active model somehow isn't installed) targets the right one.
        const match = EMBEDDING_MODELS.find((m) => m.id === cfg.embedding_model);
        if (match) selectedModel = match.id;
      } catch {
        // Non-fatal: fall back to defaults.
      }
    }
    await refreshInstalled();
  });

  async function handleInstall(): Promise<void> {
    installError = null;
    installProgress = 0;
    try {
      await installEmbeddingModel(selectedModel, (pct) => {
        installProgress = pct;
      });
      installProgress = 100;
      installed = true;
      // Persist the chosen embedding model so the backend TTS/embedding check and
      // later source-add flow know which model is bound to this notebook's store.
      // On success the installed model becomes active.
      await updateConfig((cfg) => ({ ...cfg, embedding_model: selectedModel }));
      activeModel = selectedModel;
      await oncheck();
      oncollapse();
    } catch (err) {
      installError = err instanceof Error ? err.message : 'Installation failed.';
      installProgress = null;
    }
  }

  // Switch the active model to an ALREADY-INSTALLED one: persist embedding_model
  // (no re-download), re-run the check, then collapse. The green border follows.
  async function handleSetActive(id: EmbeddingModelId): Promise<void> {
    installError = null;
    settingActive = id;
    try {
      await updateConfig((cfg) => ({ ...cfg, embedding_model: id }));
      activeModel = id;
      await oncheck();
      oncollapse();
    } catch (err) {
      installError = err instanceof Error ? err.message : 'Could not set the active model.';
    } finally {
      settingActive = null;
    }
  }
</script>

<div class="pt-3 flex flex-col gap-3">
  <!-- Amber warning banner -->
  <div
    class="bg-amber-500/15 border border-amber-500/30 rounded-lg px-3 py-2.5 flex gap-2 items-start"
  >
    <TriangleAlert class="size-4 shrink-0 mt-0.5 text-amber-500" />
    <p class="text-[0.78rem] leading-relaxed text-amber-500">
      Switching the active model applies to <strong>new sources only</strong>. Sources already
      embedded with another model must be re-embedded from scratch to use it.
    </p>
  </div>

  <!-- Model cards -->
  <div class="flex flex-col gap-2" role="radiogroup" aria-label="Embedding model">
    {#each EMBEDDING_MODELS as model (model.id)}
      {@const isActive = activeModel === model.id}
      {@const isInstalled = installedModels.has(model.id)}
      {@const isSelected = selectedModel === model.id}
      <div
        class={cn(
          'w-full rounded-lg border px-3 py-2.5 transition-colors',
          isActive
            ? // Active model: design-system GREEN (not the accent --primary).
              'border-green-primary bg-green-primary/10 ring-1 ring-green-primary'
            : isSelected && !isInstalled
              ? 'border-primary bg-primary/10 ring-1 ring-primary'
              : 'border-border bg-card'
        )}
      >
        <div class="flex items-start justify-between gap-2">
          <button
            type="button"
            role="radio"
            aria-checked={isSelected}
            aria-disabled={isActive || isInstalled}
            onclick={() => {
              if (!isActive && !isInstalled) selectedModel = model.id;
            }}
            class={cn(
              'min-w-0 flex-1 text-left',
              (isActive || isInstalled) && 'cursor-default'
            )}
          >
            <p class="text-sm font-semibold text-foreground">{model.name}</p>
            <p class="text-[0.75rem] text-muted-foreground mt-0.5">
              {model.dims} dims · {model.sizeMb >= 1000
                ? (model.sizeMb / 1000).toFixed(1) + ' GB'
                : model.sizeMb + ' MB'} · {model.speed}
            </p>
            <p class="text-[0.75rem] text-muted-foreground mt-0.5">{model.description}</p>
          </button>

          <div class="flex shrink-0 items-center gap-2">
            {#if isActive}
              <!-- Active model: green badge, no action. -->
              <span
                class="flex items-center gap-1 text-[0.72rem] font-semibold text-green-primary"
                aria-label="Active embedding model"
              >
                <CircleCheck class="size-4" />
                Active
              </span>
            {:else if isInstalled}
              <!-- Installed but not active: switch without re-downloading. -->
              <Button
                variant="outline"
                size="sm"
                onclick={() => handleSetActive(model.id)}
                disabled={settingActive !== null}
                aria-label={`Set ${model.name} as active`}
              >
                {settingActive === model.id ? 'Setting…' : 'Set as active'}
              </Button>
            {:else if isSelected}
              <CircleCheck class="size-4 mt-0.5 text-primary" />
            {/if}
          </div>
        </div>
      </div>
    {/each}
  </div>

  <!-- Install error -->
  {#if installError}
    <p class="text-destructive text-[0.75rem]" role="alert">{installError}</p>
  {/if}

  <!-- Install button with progress — installs the SELECTED not-installed model. -->
  {#if !installedModels.has(selectedModel) && activeModel !== selectedModel}
    <Button
      class="h-10 w-full"
      onclick={handleInstall}
      disabled={installProgress !== null && !installed}
    >
      {#if installProgress !== null && installProgress < 100}
        <LoaderCircle class="size-4 animate-spin" />
        Installing… {installProgress}%
      {:else}
        Install {selected.name}
      {/if}
    </Button>
  {/if}
</div>
