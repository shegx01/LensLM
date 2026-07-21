<!--
  ActiveModelSection — the "Active model" half of the AI Model panel. Provider-first:
  an active-model banner (ActiveModelPicker) on top, then a provider dropdown restricted
  to USABLE providers (cloud with a saved key, or Ollama reachable) whose selection reveals
  that provider's models as a radio list with capability chips. A custom (OpenAI-compatible)
  provider has no catalog, so it exposes a free-text model input instead. Picking/typing a
  model pins it via saveActiveModel + the consent-flipping saveEnrichmentPrefs write, then
  refreshes the config, chat-provider, and active-model stores. Reactive persist — no Save
  button. Temperature, context, and the enrichment override are model-level, shown only once
  a model is pinned.

  Context is not a control for cloud models (fixed catalog property, shown only as the
  "ctx" chip and persisted silently); local/Ollama keeps the manual ContextWindowField.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import type { AppConfig, ModelConfig, TaskModel } from '$lib/theme/types.js';
  import {
    listCloudModelOptions,
    listOllamaModelOptions,
    formatCompact,
    formatUsd,
    type ModelOption
  } from '$lib/models/catalog.js';
  import { saveActiveModel, saveEnrichmentPrefs } from '$lib/onboarding/llm-config.js';
  import { refreshChatProvider } from '$lib/models/chat-provider.svelte.js';
  import { refreshActiveModel } from '$lib/models/active-model.svelte.js';
  import { appConfigStore, refreshConfig } from '$lib/models/app-config.svelte.js';
  import {
    providerDescriptors,
    isReachable,
    isUsable,
    LOCAL_DEFAULT_ENDPOINT,
    DEFAULT_CONTEXT,
    type ProviderDescriptor
  } from '$lib/models/providers.js';
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import ActiveModelPicker from './ActiveModelPicker.svelte';
  import ContextWindowField from './ContextWindowField.svelte';
  import EnrichmentOverride from './EnrichmentOverride.svelte';

  let loaded = $state(false);
  let ollamaCount = $state<number | null>(null);
  let ollamaBaseUrl = $state(LOCAL_DEFAULT_ENDPOINT);

  let selectedId = $state('');
  let modelOptions = $state<ModelOption[]>([]);
  let customModel = $state('');
  let temperature = $state(0.7);
  let contextTokens = $state(DEFAULT_CONTEXT);
  let enrichmentModel = $state<TaskModel | null>(null);
  let saveError = $state<string | null>(null);

  function entryFor(id: string): ModelConfig | undefined {
    return appConfigStore.models.find((m) => m.provider === id);
  }

  // Only providers the user can actually use: Ollama when reachable, cloud when keyed.
  const usableProviders = $derived.by<ProviderDescriptor[]>(() =>
    providerDescriptors()
      .filter((d) => isUsable(d, entryFor(d.id), ollamaCount))
      .map((d) => {
        if (d.kind === 'local') return { ...d, baseUrl: ollamaBaseUrl };
        return { ...d, baseUrl: entryFor(d.id)?.base_url || d.baseUrl };
      })
  );

  const selectedProvider = $derived(usableProviders.find((p) => p.id === selectedId));
  const selectedEntry = $derived(entryFor(selectedId));
  const isCloud = $derived(selectedProvider ? selectedProvider.kind !== 'local' : false);
  const isCustom = $derived(selectedProvider?.kind === 'custom');
  const pinnedModel = $derived(selectedEntry?.model ?? '');

  // Context that a temperature/enrichment re-write should persist: cloud follows the
  // pinned model's fixed catalog limit; local uses the manual ContextWindowField value.
  const currentContext = $derived.by(() => {
    if (!selectedProvider || selectedProvider.kind === 'local') return contextTokens;
    const limit = modelOptions.find((o) => o.id === pinnedModel)?.info?.context_limit;
    return limit ?? selectedEntry?.context ?? DEFAULT_CONTEXT;
  });

  async function loadOllama(): Promise<void> {
    ollamaBaseUrl = entryFor('ollama')?.base_url || LOCAL_DEFAULT_ENDPOINT;
    try {
      const ids = await invoke<string[]>('list_ollama_models', { base_url: ollamaBaseUrl });
      ollamaCount = ids.length;
    } catch {
      ollamaCount = null;
    }
  }

  async function loadModelOptions(): Promise<void> {
    const p = selectedProvider;
    if (!p) {
      modelOptions = [];
      return;
    }
    try {
      if (p.kind === 'local') modelOptions = await listOllamaModelOptions(p.baseUrl);
      else if (p.catalogKey) modelOptions = await listCloudModelOptions(p.catalogKey);
      else modelOptions = [];
    } catch {
      modelOptions = [];
    }
  }

  async function selectProvider(id: string): Promise<void> {
    selectedId = id;
    const entry = entryFor(id);
    temperature = entry?.temperature ?? 0.7;
    contextTokens = entry?.context || DEFAULT_CONTEXT;
    customModel = entry?.model ?? '';
    // Drop an enrichment override that belongs to a different provider (no orphan pins).
    const coref = appConfigStore.enrichment.coref_model;
    enrichmentModel =
      coref && coref.provider === id ? { provider: coref.provider, model: coref.model } : null;
    await loadModelOptions();
  }

  onMount(async () => {
    if (!isTauri()) {
      loaded = true;
      return;
    }
    await refreshConfig();
    await loadOllama();
    const pin = appConfigStore.enrichment.chat_model;
    const usable = usableProviders;
    const initial =
      pin && usable.some((p) => p.id === pin.provider) ? pin.provider : (usable[0]?.id ?? '');
    if (initial) await selectProvider(initial);
    loaded = true;
  });

  // The consent-flipping pin write: persist the model entry (preserving credentials), then
  // pin enrichment.chat_model and flip cloud_consent → true for cloud (never false), sourcing
  // enabled/coref_strategy from prior config so a pin never resets enrichment.
  async function pinModel(opts: { model: string; context: number }): Promise<void> {
    if (!isTauri() || !selectedProvider) return;
    saveError = null;
    const provider = selectedProvider.id;
    const cloud = selectedProvider.kind !== 'local';
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const prior = cfg.enrichment;
      await saveActiveModel({ provider, model: opts.model, context: opts.context, temperature });
      await saveEnrichmentPrefs({
        enabled: prior.enabled,
        coref_strategy: prior.coref_strategy,
        cloud_consent: cloud ? true : prior.cloud_consent,
        chat_model: { provider, model: opts.model },
        coref_model: enrichmentModel,
        map_model: enrichmentModel
      });
      await refreshConfig();
      await refreshChatProvider();
      await refreshActiveModel();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not pin the model.';
    }
  }

  async function onPickModel(opt: ModelOption): Promise<void> {
    const ctx = isCloud ? (opt.info?.context_limit ?? contextTokens) : contextTokens;
    await pinModel({ model: opt.id, context: ctx });
  }

  function commitCustomModel(): void {
    const model = customModel.trim();
    if (model && model !== pinnedModel) void pinModel({ model, context: contextTokens });
  }

  function onTemperatureChange(): void {
    if (pinnedModel) void pinModel({ model: pinnedModel, context: currentContext });
  }

  function onContextChange(tokens: number): void {
    contextTokens = tokens;
    if (pinnedModel) void pinModel({ model: pinnedModel, context: tokens });
  }

  function onEnrichmentChange(next: TaskModel | null): void {
    enrichmentModel = next;
    if (pinnedModel) void pinModel({ model: pinnedModel, context: currentContext });
  }

  /** Capability/metadata chips for a model row; empty for Ollama (no catalog info). */
  function chipsFor(opt: ModelOption): { text: string; cap: boolean }[] {
    const info = opt.info;
    if (!info) return [];
    const chips: { text: string; cap: boolean }[] = [];
    if (info.context_limit != null)
      chips.push({ text: `${formatCompact(info.context_limit)} ctx`, cap: false });
    const input = info.cost?.input;
    const output = info.cost?.output;
    if (input != null && output != null) {
      chips.push({ text: `$${formatUsd(input)} / $${formatUsd(output)} per 1M`, cap: false });
    }
    if (info.reasoning) chips.push({ text: 'Reasoning', cap: true });
    if (info.tool_call) chips.push({ text: 'Tools', cap: true });
    return chips;
  }
</script>

<section class="flex flex-col" aria-label="Active model settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Active model</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Choose which model powers chat, notes, and the audio overview.
  </p>

  {#if loaded}
    <div class="mt-6 flex flex-col gap-4">
      <ActiveModelPicker />

      {#if usableProviders.length === 0}
        <p
          class="rounded-[10px] bg-muted px-3.5 py-3 text-[0.75rem] text-muted-foreground"
          role="status"
        >
          Set up a provider under Providers first — add an API key, or run Ollama locally — then
          pick a model here.
        </p>
      {:else}
        <div class="rounded-[10px] border border-border bg-card p-4">
          <div class="flex flex-col gap-1.5">
            <label for="active-provider" class="text-[0.72rem] font-bold text-foreground">
              Provider
            </label>
            <p class="text-[0.68rem] text-muted-foreground">
              Only providers you've set up (key saved, or Ollama reachable) appear here.
            </p>
            <Select
              type="single"
              value={selectedId}
              onValueChange={(v) => {
                if (v) void selectProvider(v);
              }}
              items={usableProviders.map((p) => ({
                value: p.id,
                label: p.kind === 'local' ? `${p.name} (local)` : p.name
              }))}
            >
              <SelectTrigger id="active-provider" class="w-full">
                <SelectValue placeholder="Select a provider" />
              </SelectTrigger>
              <SelectContent
                class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
              >
                {#each usableProviders as p (p.id)}
                  {@const label = p.kind === 'local' ? `${p.name} (local)` : p.name}
                  <SelectItem value={p.id} {label}>{label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
          </div>

          {#if isCustom}
            <div class="mt-3 flex flex-col gap-1.5">
              <label for="active-custom-model" class="text-[0.72rem] font-bold text-foreground">
                Model
              </label>
              <p class="text-[0.68rem] text-muted-foreground">
                Enter the model id your endpoint serves.
              </p>
              <Input
                id="active-custom-model"
                type="text"
                bind:value={customModel}
                onblur={commitCustomModel}
                placeholder="model id"
                autocomplete="off"
                spellcheck={false}
              />
            </div>
          {:else if modelOptions.length === 0}
            <p class="mt-3 text-[0.72rem] text-muted-foreground" role="status">
              No models found for this provider.
            </p>
          {:else}
            <div
              class="no-scrollbar mt-3 flex max-h-[286px] flex-col gap-1.5 overflow-y-auto"
              role="radiogroup"
              aria-label="Model"
            >
              {#each modelOptions as opt (opt.id)}
                {@const checked = opt.id === pinnedModel}
                {@const chips = chipsFor(opt)}
                <button
                  type="button"
                  role="radio"
                  aria-checked={checked}
                  onclick={() => void onPickModel(opt)}
                  class="flex items-center gap-2.5 rounded-[9px] border px-3 py-2.5 text-left transition-[background-color,border-color,transform] duration-150 active:scale-[0.985] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring {checked
                    ? 'border-primary/55 bg-primary/8'
                    : 'border-border bg-card hover:bg-muted'}"
                >
                  <span
                    class="grid size-[15px] shrink-0 place-items-center rounded-full border-2 {checked
                      ? 'border-primary'
                      : 'border-muted-foreground'}"
                    aria-hidden="true"
                  >
                    {#if checked}
                      <span class="size-[7px] rounded-full bg-primary"></span>
                    {/if}
                  </span>
                  <span class="min-w-0 flex-1">
                    <span class="block truncate text-[0.8rem] font-semibold text-foreground"
                      >{opt.label}</span
                    >
                    {#if chips.length > 0}
                      <span class="mt-1 flex flex-wrap items-center gap-1.5">
                        {#each chips as chip, i (i)}
                          <span
                            class="rounded-full px-1.5 py-px text-[0.58rem] font-bold tracking-[0.02em] {chip.cap
                              ? 'bg-primary/15 text-primary'
                              : 'bg-muted text-muted-foreground'}"
                          >
                            {chip.text}
                          </span>
                        {/each}
                      </span>
                    {/if}
                  </span>
                </button>
              {/each}
            </div>
          {/if}
        </div>

        {#if pinnedModel}
          <div class="rounded-[10px] border border-border bg-card p-4">
            <div class="flex flex-col gap-1.5">
              <div class="flex items-baseline justify-between">
                <label for="active-temperature" class="text-[0.72rem] font-bold text-foreground">
                  Temperature
                </label>
                <span class="text-[0.72rem] tabular-nums text-foreground"
                  >{temperature.toFixed(1)}</span
                >
              </div>
              <p class="text-[0.68rem] text-muted-foreground">
                Lower is more focused; higher is more varied.
              </p>
              <input
                id="active-temperature"
                type="range"
                min="0"
                max="2"
                step="0.1"
                bind:value={temperature}
                onchange={onTemperatureChange}
                aria-label="Temperature"
                class="mt-1 w-full accent-primary"
              />
            </div>
          </div>

          {#if !isCloud}
            <div class="rounded-[10px] border border-border bg-card p-4">
              <ContextWindowField bind:value={contextTokens} onchange={onContextChange} />
            </div>
          {/if}

          <EnrichmentOverride
            value={enrichmentModel}
            options={modelOptions}
            providerId={selectedId}
            onchange={onEnrichmentChange}
          />
        {/if}
      {/if}

      {#if saveError}
        <p class="text-[0.75rem] text-destructive" role="alert">{saveError}</p>
      {/if}
    </div>
  {/if}
</section>
