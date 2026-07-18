<!--
  AiModelSection — the "AI Model" panel inside the global Preferences view. Chat-first:
  one chat model, with a collapsed Advanced disclosure to pin a separate enrichment
  model. Reactive persist via the shared saveLlmProvider/saveEnrichmentPrefs helpers —
  no Save button.

  Two backend-coupled invariants:
  • The chosen chat entry is written as enrichment.chat_model = {provider,model} — the
    purpose-built chat pin the engine's chat_provider resolves FIRST (Variant B). Routing
    is left untouched; it stays the enrichment-only policy.
  • saveEnrichmentPrefs overwrites enabled/coref_strategy/cloud_consent unconditionally,
    so we source those from the persisted config and only ever flip cloud_consent → true
    for a cloud provider (never false) to avoid re-enqueuing or killing enrichment.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import type { AppConfig, ModelConfig, TaskModel } from '$lib/theme/types.js';
  import { saveLlmProvider, saveEnrichmentPrefs } from '$lib/onboarding/llm-config.js';
  import { findCloudProvider } from '$lib/onboarding/cloud-providers.js';
  import {
    listCloudModelOptions,
    listOllamaModelOptions,
    formatCompact,
    type ModelOption
  } from '$lib/models/catalog.js';
  import { refreshChatProvider } from '$lib/models/chat-provider.svelte.js';
  import LlmModelField from '$lib/components/llm/LlmModelField.svelte';
  import ApiKeyField from '$lib/components/llm/ApiKeyField.svelte';
  import LlmProviderField from './LlmProviderField.svelte';
  import ContextWindowField from './ContextWindowField.svelte';
  import EnrichmentOverride from './EnrichmentOverride.svelte';
  import { Input } from '$lib/components/ui/input/index.js';

  const LOCAL_DEFAULT_ENDPOINT = 'http://localhost:11434';

  let loaded = $state(false);
  let kind = $state<'local' | 'cloud'>('local');
  let providerId = $state('ollama');
  let baseUrl = $state(LOCAL_DEFAULT_ENDPOINT);
  let model = $state('');
  let context = $state(8192);
  let temperature = $state(0.7);
  let apiKeyValue = $state('');
  let editingKey = $state(false);
  let hasSavedKey = $state(false);
  let enrichmentModel = $state<TaskModel | null>(null);
  let saveError = $state<string | null>(null);

  let overrideOptions = $state<ModelOption[]>([]);

  const isCloud = $derived(kind === 'cloud');
  const selectedProvider = $derived(isCloud ? findCloudProvider(providerId) : undefined);
  const isCustom = $derived(selectedProvider?.custom === true);
  const catalogKey = $derived(isCloud ? (selectedProvider?.catalogKey ?? null) : null);
  const modelFieldKind = $derived<'local' | 'cloud' | 'custom'>(
    kind === 'local' ? 'local' : isCustom ? 'custom' : 'cloud'
  );

  const contextHint = $derived.by(() => {
    if (kind === 'local') return null;
    const limit = overrideOptions.find((o) => o.id === model)?.info?.context_limit;
    return limit != null ? `Catalog limit: ${formatCompact(limit)} tokens (advisory)` : null;
  });

  // chat_model is an explicit pin, so init just reads it back — no need to replicate the
  // engine's routing (CloudFirst/LocalFirst) semantics (a key Variant B simplification).
  // Legacy configs with no chat_model fall back to the first entry with a real model.
  function resolveInitialEntry(cfg: AppConfig): ModelConfig | null {
    const models = cfg.models ?? [];
    const pin = cfg.enrichment?.chat_model;
    if (pin) {
      const m = models.find((e) => e.provider === pin.provider && e.model === pin.model);
      if (m) return m;
    }
    return models.find((m) => m.model.trim() !== '') ?? null;
  }

  onMount(async () => {
    if (!isTauri()) {
      loaded = true;
      return;
    }
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const entry = resolveInitialEntry(cfg);
      if (entry) {
        providerId = entry.provider;
        kind = entry.provider === 'ollama' ? 'local' : 'cloud';
        baseUrl = entry.base_url || (kind === 'local' ? LOCAL_DEFAULT_ENDPOINT : '');
        model = entry.model;
        context = entry.context || 8192;
        temperature = entry.temperature ?? 0.7;
        hasSavedKey = kind === 'cloud' && entry.api_key.trim() !== '';
      }
      const coref = cfg.enrichment?.coref_model;
      if (coref) enrichmentModel = { provider: coref.provider, model: coref.model };
    } catch {
      // Non-fatal: fall back to the empty local default.
    } finally {
      loaded = true;
    }
  });

  // Reload the Advanced-override model list when the provider identity changes.
  $effect(() => {
    void kind;
    void catalogKey;
    void baseUrl;
    void loadOverrideOptions();
  });

  async function loadOverrideOptions(): Promise<void> {
    if (kind === 'local') {
      try {
        overrideOptions = await listOllamaModelOptions(baseUrl);
      } catch {
        overrideOptions = [];
      }
    } else if (catalogKey) {
      try {
        overrideOptions = await listCloudModelOptions(catalogKey);
      } catch {
        overrideOptions = [];
      }
    } else {
      overrideOptions = [];
    }
  }

  async function onProviderChange(sel: {
    kind: 'local' | 'cloud';
    providerId: string;
  }): Promise<void> {
    kind = sel.kind;
    providerId = sel.providerId;
    editingKey = false;
    apiKeyValue = '';

    // Restore the target provider's previously-saved entry so a provider round-trip
    // doesn't overwrite its models[] entry with an empty model. Recomputes hasSavedKey
    // for the newly-selected provider from the same fetch (no separate round-trip).
    let existing: ModelConfig | undefined;
    if (isTauri()) {
      try {
        const cfg = await invoke<AppConfig>('get_config');
        existing = (cfg.models ?? []).find((m) => m.provider === providerId);
      } catch {
        existing = undefined;
      }
    }

    if (existing) {
      model = existing.model;
      context = existing.context || 8192;
      temperature = existing.temperature ?? 0.7;
    } else {
      context = 8192;
      temperature = 0.7;
      // No saved entry: blank for local/catalog-less providers, else the catalog floor model.
      model = kind === 'cloud' ? (findCloudProvider(providerId)?.defaultModel ?? '') : '';
    }

    if (kind === 'local') {
      baseUrl = existing?.base_url || LOCAL_DEFAULT_ENDPOINT;
      hasSavedKey = false;
    } else {
      const p = findCloudProvider(providerId);
      baseUrl = existing?.base_url || (p?.custom ? p.baseUrl : '');
      hasSavedKey = (existing?.api_key.trim() ?? '') !== '';
    }

    // Never persist a chat_model pin with an empty model; defer until a model is chosen.
    if (model.trim() !== '') void persistChat();
  }

  async function persistChat(): Promise<void> {
    if (!isTauri()) return;
    saveError = null;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const existing = (cfg.models ?? []).find((m) => m.provider === providerId);
      // Focusing the masked field flips editingKey before any keystroke; an empty
      // buffer still means "keep the saved key", so guard on the buffer too (no wipe).
      const keyMasked = isCloud && hasSavedKey && (!editingKey || apiKeyValue.trim() === '');
      const api_key = isCloud ? (keyMasked ? (existing?.api_key ?? '') : apiKeyValue) : '';

      await saveLlmProvider({
        provider: providerId,
        base_url: baseUrl,
        model,
        context,
        temperature,
        api_key
      });

      const prior = cfg.enrichment;
      await saveEnrichmentPrefs({
        enabled: prior.enabled,
        coref_strategy: prior.coref_strategy,
        cloud_consent: isCloud ? true : prior.cloud_consent,
        chat_model: { provider: providerId, model },
        coref_model: enrichmentModel,
        map_model: enrichmentModel
      });

      if (isCloud) hasSavedKey = api_key.trim() !== '';
      editingKey = false;
      apiKeyValue = '';
      await refreshChatProvider();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save the model.';
    }
  }

  function onEnrichmentChange(next: TaskModel | null): void {
    enrichmentModel = next;
    void persistChat();
  }
</script>

<section class="flex flex-col" aria-label="AI Model settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">AI Model</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    The model that powers chat. Enrichment reuses it unless you set a separate one below.
  </p>

  {#if loaded}
    <div class="mt-6 flex flex-col gap-4">
      <div class="flex flex-col gap-1.5">
        <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
          Chat model
        </p>
        <LlmProviderField {kind} {providerId} onchange={onProviderChange} />
      </div>

      {#if kind === 'local' || isCustom}
        <div class="flex flex-col gap-1.5">
          <label for="ai-base-url" class="text-[0.68rem] font-medium text-muted-foreground">
            Base URL
          </label>
          <Input
            id="ai-base-url"
            type="url"
            bind:value={baseUrl}
            onblur={() => void persistChat()}
            placeholder={kind === 'local' ? LOCAL_DEFAULT_ENDPOINT : 'https://api.openai.com/v1'}
            autocomplete="off"
            spellcheck={false}
          />
        </div>
      {/if}

      <LlmModelField
        id="ai-model"
        kind={modelFieldKind}
        {providerId}
        {catalogKey}
        {baseUrl}
        apiKey={apiKeyValue}
        options={overrideOptions}
        bind:value={model}
        onchange={() => void persistChat()}
      />

      <ContextWindowField
        bind:value={context}
        hint={contextHint}
        onchange={() => void persistChat()}
      />

      <div class="flex flex-col gap-1.5">
        <div class="flex items-baseline justify-between">
          <label for="ai-temperature" class="text-[0.68rem] font-medium text-muted-foreground">
            Temperature
          </label>
          <span class="text-[0.72rem] tabular-nums text-foreground">{temperature.toFixed(1)}</span>
        </div>
        <input
          id="ai-temperature"
          type="range"
          min="0"
          max="2"
          step="0.1"
          bind:value={temperature}
          onchange={() => void persistChat()}
          aria-label="Temperature"
          class="w-full accent-primary"
        />
      </div>

      {#if isCloud}
        <ApiKeyField
          id="ai-api-key"
          bind:value={apiKeyValue}
          bind:editing={editingKey}
          {hasSavedKey}
          oncommit={() => void persistChat()}
        />
      {/if}

      <EnrichmentOverride
        value={enrichmentModel}
        options={overrideOptions}
        {providerId}
        onchange={onEnrichmentChange}
      />

      {#if saveError}
        <p class="text-[0.75rem] text-destructive" role="alert">{saveError}</p>
      {/if}
    </div>
  {/if}
</section>
