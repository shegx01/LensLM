<!--
  AiModelSection — the "AI Model" panel inside the global Preferences view. Chat-first:
  one chat model, with a collapsed Advanced disclosure to pin a separate enrichment
  model. Reactive persist via the shared saveLlmProvider/saveEnrichmentPrefs helpers —
  no Save button.

  Two backend-coupled invariants:
  • The chosen chat entry is written as enrichment.routing = Explicit{provider,model} —
    chat resolution ignores chat_model, so this is what actually makes it authoritative
    (and enrichment defaults to it).
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

  // Never blind models[0]: derive the entry chat currently resolves to (Explicit pin
  // first, else CloudFirst — a consented cloud entry outranks local), so opening the
  // panel on a multi-entry config doesn't silently repin a different model on blur.
  function resolveInitialEntry(cfg: AppConfig): ModelConfig | null {
    const models = cfg.models ?? [];
    const routing = cfg.enrichment?.routing;
    if (routing?.kind === 'explicit') {
      const m = models.find((e) => e.provider === routing.provider && e.model === routing.model);
      if (m) return m;
    }
    if (cfg.enrichment?.cloud_consent) {
      const cloud = models.find((m) => m.provider !== 'ollama' && m.model.trim() !== '');
      if (cloud) return cloud;
    }
    const local = models.find((m) => m.provider === 'ollama' && m.model.trim() !== '');
    if (local) return local;
    return models[0] ?? null;
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

  async function refreshSavedKeyState(): Promise<void> {
    if (!isTauri() || kind !== 'cloud') {
      hasSavedKey = false;
      return;
    }
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const existing = (cfg.models ?? []).find((m) => m.provider === providerId);
      hasSavedKey = (existing?.api_key.trim() ?? '') !== '';
    } catch {
      hasSavedKey = false;
    }
  }

  function onProviderChange(sel: { kind: 'local' | 'cloud'; providerId: string }): void {
    kind = sel.kind;
    providerId = sel.providerId;
    editingKey = false;
    apiKeyValue = '';
    model = '';
    if (kind === 'local') {
      baseUrl = baseUrl || LOCAL_DEFAULT_ENDPOINT;
      hasSavedKey = false;
      void persistChat();
    } else {
      const p = findCloudProvider(providerId);
      baseUrl = p?.custom ? p.baseUrl : '';
      void refreshSavedKeyState().then(() => persistChat());
    }
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
        routing: { kind: 'explicit', provider: providerId, model },
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
