<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import {
    EMBEDDING_MODELS,
    installEmbeddingModel,
    type EmbeddingModelId
  } from '$lib/onboarding/system-check.js';
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

  const selected = $derived(EMBEDDING_MODELS.find((m) => m.id === selectedModel)!);

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
      await updateConfig((cfg) => ({ ...cfg, embedding_model: selectedModel }));
      await oncheck();
      oncollapse();
    } catch (err) {
      installError = err instanceof Error ? err.message : 'Installation failed.';
      installProgress = null;
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
      Embedding models are <strong>permanently linked</strong> to this notebook's vector store. Switching
      models later requires re-embedding all sources from scratch. Choose carefully.
    </p>
  </div>

  <!-- Model cards -->
  <div class="flex flex-col gap-2" role="radiogroup" aria-label="Embedding model">
    {#each EMBEDDING_MODELS as model (model.id)}
      {@const isSelected = selectedModel === model.id}
      <button
        role="radio"
        aria-checked={isSelected}
        onclick={() => (selectedModel = model.id)}
        class={cn(
          'w-full text-left rounded-lg border px-3 py-2.5 transition-colors',
          isSelected
            ? 'border-primary bg-primary/10 ring-1 ring-primary'
            : 'border-border bg-card hover:bg-muted/50'
        )}
      >
        <div class="flex items-start justify-between gap-2">
          <div class="min-w-0">
            <p class="text-sm font-semibold text-foreground">{model.name}</p>
            <p class="text-[0.75rem] text-muted-foreground mt-0.5">
              {model.dims} dims · {model.sizeMb >= 1000
                ? (model.sizeMb / 1000).toFixed(1) + ' GB'
                : model.sizeMb + ' MB'} · {model.speed}
            </p>
            <p class="text-[0.75rem] text-muted-foreground mt-0.5">{model.description}</p>
          </div>
          {#if isSelected}
            <CircleCheck class="size-4 shrink-0 mt-0.5 text-primary" />
          {/if}
        </div>
      </button>
    {/each}
  </div>

  <!-- Install error -->
  {#if installError}
    <p class="text-destructive text-[0.75rem]" role="alert">{installError}</p>
  {/if}

  <!-- Install button with progress -->
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
</div>
