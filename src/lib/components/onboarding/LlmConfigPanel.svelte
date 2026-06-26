<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import { cn } from '$lib/utils.js';
  import { detectLlm } from '$lib/onboarding/system-check.js';
  import {
    saveLlmProvider,
    saveEnrichmentPrefs,
    type LlmProviderTab
  } from '$lib/onboarding/llm-config.js';
  import type { AppConfig, CorefStrategy, LlmRouting, TaskModel } from '$lib/theme/types.js';
  import {
    listCloudModelOptions,
    listOllamaModelOptions,
    refreshCatalog,
    type ModelOption
  } from '$lib/models/catalog.js';
  import { SELECT_CLASS } from './styles.js';

  // Cloud config has no Context Window picker (it's Local-only), so cloud saves
  // persist a sensible large default that covers modern hosted models.
  const CLOUD_DEFAULT_CONTEXT = 128000;

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  // --- Segmented tab state ---
  let activeTab = $state<LlmProviderTab>('local');

  // --- Local tab fields ---
  let localEndpoint = $state('http://localhost:11434');
  let localModel = $state('llama3.2:3b');
  let contextWindow = $state(8192);
  // Detected models list (populated by Auto-detect)
  let detectedModels = $state<string[]>([]);
  let detectStatus = $state<'idle' | 'detecting' | 'found' | 'not-found'>('idle');
  let detectVersion = $state<string | null>(null);
  let detectError = $state<string | null>(null);
  let testStatus = $state<'idle' | 'testing' | 'success' | 'fail'>('idle');
  let testMessage = $state<string | null>(null);

  // --- Cloud API tab fields ---
  let cloudProvider = $state<'openai' | 'anthropic' | 'google'>('openai');
  let cloudBaseUrl = $state('https://api.openai.com/v1');
  let cloudApiKey = $state('');
  // User-editable model id. Seeded from the provider default as an offline floor,
  // but the SMART DEFAULT (loadCloudModels) overrides it with the newest
  // text-capable catalog model when the user hasn't explicitly picked one.
  // 'gpt-4o' matches the initial 'openai' provider.
  let cloudModel = $state('gpt-4o');
  // Whether the user has EXPLICITLY chosen a cloud model (via the picker) or
  // restored a saved one. While false, the smart default re-resolves to the
  // newest text-capable model on every (re)load — so a background catalog
  // refresh converges the default to the live newest model. Once the user picks
  // (or a saved model is restored), their choice is preserved across reloads.
  let cloudModelPicked = $state(false);

  // --- Capability-aware model pickers (M4 Phase 3, Stage 3) ---
  // The CLOUD catalog options for the selected provider (models.dev). Loaded on
  // expand and whenever the provider card changes. Resilient by contract: any
  // throw / non-Tauri / empty map leaves this `[]`, and we fall back to a single
  // default-model option so the picker still works offline (and legacy tests pass).
  let cloudModelOptions = $state<ModelOption[]>([]);
  // The locally-pulled Ollama models at the current endpoint (info: null). Loaded
  // on expand; surfaces pulled models in the model picker without Auto-detect.
  let ollamaModelOptions = $state<ModelOption[]>([]);
  // Whether a live catalog refresh (models.dev) is in flight. Drives the subtle
  // "updating…" affordance under the cloud picker. NEVER blocks selection: the
  // already-loaded options stay rendered + selectable while this is true.
  let catalogUpdating = $state(false);

  // --- Save state (cloud) ---
  let saving = $state(false);
  let saveError = $state<string | null>(null);

  // --- Saved cloud key masking (per-provider) ---
  // When a cloud config was previously saved with a non-empty key, we DON'T load
  // the real key into the DOM. We mask it and disable Save until the user clicks
  // the key field to enter a fresh key. `editingKey` flips on focus/input.
  //
  // The saved state is PER-PROVIDER: `savedProviderId` records which card the
  // saved key belongs to (mapped by base_url on mount). `hasSavedKey` is then
  // true ONLY when the currently-selected card is that saved one, so the masked
  // /disabled treatment never bleeds onto other provider cards.
  let savedProviderId = $state<string | null>(null);
  let editingKey = $state(false);
  const hasSavedKey = $derived(cloudProvider === savedProviderId);

  // --- Enrichment preferences (M4 Phase 3) ---
  // Non-blocking: the toggle defaults to ON for the local tab (provider-driven
  // default — local Ollama → enrichment on + LlmInline) and OFF for the cloud tab
  // (cloud is explicit-enable + requires consent). Saving the provider above also
  // persists these prefs via `saveEnrichmentPrefs` (RMW), so a user who completes
  // (or skips) the step lands on the conservative Rust-side defaults regardless.
  let enrichmentEnabled = $state(true);
  let corefStrategy = $state<CorefStrategy>('llm_inline');
  // Explicit, separate consent for sending document text to a CLOUD LLM. Shown
  // only on the Cloud API tab; forced false on the local save path.
  let cloudConsent = $state(false);

  // --- Routing + per-task overrides (M4 Phase 3, Stage 3) ---
  // Typed routing policy. 'explicit' pins to the currently-selected provider+model.
  let routingKind = $state<'cloud_first' | 'local_first' | 'explicit'>('cloud_first');
  // Coreference-model override (a per-task TaskModel pin). '' = use the configured
  // model (no override → coref_model: null); a model id sets the pin for the active
  // tab's provider. (map_model override is out of scope — left undefined on save.)
  let corefModelId = $state('');

  const ROUTING_OPTIONS = [
    { value: 'cloud_first' as const, label: 'Cloud-first' },
    { value: 'local_first' as const, label: 'Local-first' },
    { value: 'explicit' as const, label: 'Explicit' }
  ] as const;

  // Coref strategy options. Only the two strategies that actually ship: inline
  // coref in the enrichment LLM pass, or off. (A `dedicated_model` stub was
  // removed — it only ever fell back to inline, so surfacing it was a UX lie.)
  const COREF_OPTIONS = [
    { value: 'llm_inline' as const, label: 'Inline (recommended)' },
    { value: 'none' as const, label: 'Off' }
  ] as const;

  const CONTEXT_OPTIONS = [
    { label: '4K', value: 4096, helper: '~3,000 words. For very small models only.' },
    {
      label: '8K',
      value: 8192,
      helper: '~6,000 words. Balanced for 7B–13B models. Recommended starting point.'
    },
    { label: '16K', value: 16384, helper: '~12,000 words. For larger models (13B+).' },
    { label: '32K', value: 32768, helper: '~24,000 words. High memory usage.' },
    {
      label: '128K',
      value: 131072,
      helper: '~96,000 words. Only for models with full context support.'
    }
  ] as const;

  // `models` is the card subtitle only — keep it a generic family name (not
  // specific versions like "GPT-4o") so we don't have to ship a UI update every
  // time a provider releases a new model. `defaultModel` is the actual id sent
  // to the API and can be refined independently.
  const CLOUD_PROVIDERS = [
    {
      id: 'openai' as const,
      name: 'OpenAI',
      models: 'GPT Models',
      defaultModel: 'gpt-4o',
      baseUrl: 'https://api.openai.com/v1'
    },
    {
      id: 'anthropic' as const,
      name: 'Anthropic',
      models: 'Claude Models',
      defaultModel: 'claude-3-5-sonnet-latest',
      baseUrl: 'https://api.anthropic.com/v1'
    },
    {
      id: 'google' as const,
      name: 'Google',
      models: 'Gemini Models',
      defaultModel: 'gemini-1.5-pro',
      baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai'
    }
  ] as const;

  const contextHelper = $derived(
    CONTEXT_OPTIONS.find((o) => o.value === contextWindow)?.helper ?? ''
  );

  const selectedProvider = $derived(CLOUD_PROVIDERS.find((p) => p.id === cloudProvider)!);

  // The cloud catalog key for the selected provider card (openai/anthropic/google
  // map 1:1 to the models.dev catalog keys).
  const cloudCatalogKey = $derived(cloudProvider);

  // The canonical backend provider id for the ACTIVE tab — used for routing pins
  // and per-task TaskModel overrides. Local → 'ollama'; cloud → the REAL provider
  // id of the selected card ('openai' | 'anthropic' | 'google'), matching the
  // models.dev catalog key so the Rust factory validates against the right
  // namespace (NOT a blanket 'openai-compatible', which broke claude-*/gemini-*).
  const canonicalProviderId = $derived(activeTab === 'local' ? 'ollama' : cloudProvider);

  // The options actually rendered in the cloud model <select>. When the catalog
  // is empty (offline / non-Tauri / mock returns nothing), fall back to a single
  // option for the provider default so the picker stays usable and the legacy
  // cloud Save tests (which expect 'gpt-4o') still pass.
  const cloudSelectOptions = $derived<ModelOption[]>(
    cloudModelOptions.length > 0
      ? cloudModelOptions
      : [{ id: selectedProvider.defaultModel, label: selectedProvider.defaultModel, info: null }]
  );

  // The capability info for the currently-selected cloud model (drives the
  // thinking toggle + context/cost helper). `null` when the catalog is empty
  // (offline fallback) or the id isn't in the catalog.
  const selectedCloudModel = $derived(cloudModelOptions.find((o) => o.id === cloudModel) ?? null);

  // Pretty context-window string for the helper text (e.g. "Context: 1,000,000
  // tokens"). Empty when the selected model has no context limit.
  const contextHint = $derived(
    selectedCloudModel?.info?.context_limit != null
      ? `Context: ${selectedCloudModel.info.context_limit.toLocaleString('en-US')} tokens`
      : ''
  );

  // Small per-1M input-token cost hint (e.g. "~$3/1M input tokens"). Empty when
  // the model reports no input cost.
  const costHint = $derived(
    selectedCloudModel?.info?.cost?.input != null
      ? `~$${selectedCloudModel.info.cost.input}/1M input tokens`
      : ''
  );

  // The model options offered for the coref-override picker on the active tab.
  const corefModelOptions = $derived(
    activeTab === 'local' ? ollamaModelOptions : cloudModelOptions
  );

  // Loads the cloud catalog (text-capable models only, newest first) for the
  // selected provider and resolves the selected model. Resilient: any throw /
  // empty map leaves the options empty so the picker falls back to the seeded
  // default option offline (and the legacy "gpt-4o" save tests stay green).
  //
  // Smart default: when the user hasn't explicitly picked (or restored) a model,
  // select the FIRST option — the newest text-capable model for the provider —
  // so the default reflects the live catalog and re-resolves after a background
  // refresh. When the user HAS picked one, preserve it (keep it if still valid;
  // only fall back to the seed if it vanished from the catalog). When the
  // filtered list is empty (offline), keep the seeded default so the field is
  // never blank.
  async function loadCloudModels(): Promise<void> {
    try {
      const opts = await listCloudModelOptions(cloudCatalogKey);
      cloudModelOptions = opts.length > 0 ? opts : [];
    } catch {
      cloudModelOptions = [];
    }
    if (cloudModelOptions.length === 0) {
      // Offline / empty catalog: fall back to the per-provider seed so the field
      // is never blank (the picker renders the single seeded fallback option).
      cloudModel = selectedProvider.defaultModel;
      return;
    }
    if (!cloudModelPicked) {
      // Smart default: newest text-capable model (the list is sorted desc).
      cloudModel = cloudModelOptions[0].id;
      return;
    }
    // User/saved choice: keep it if still in the catalog, else fall back to seed.
    if (!cloudModelOptions.some((o) => o.id === cloudModel)) {
      cloudModel = selectedProvider.defaultModel;
    }
  }

  // Triggers a LIVE catalog refresh from models.dev, then RE-READS the loaded
  // catalog so the cloud picker converges to the CURRENT full list (new models
  // appear, removed ones disappear) — the data-driven contract. Fire-and-forget
  // from the caller's view: the loaded list is already rendered (loadCloudModels
  // ran first), so this only ever ADDS freshness, never blocks selection.
  //
  // Graceful: refreshCatalog() swallows offline/HTTP errors (resolves false), so
  // we always re-read whatever the backend now serves — the existing list when
  // offline, the freshly-fetched one when online. The backend gates the fetch on
  // staleness, so repeated opens don't trigger a refetch storm. Never throws.
  async function refreshAndReloadCloud(): Promise<void> {
    catalogUpdating = true;
    try {
      await refreshCatalog();
      // Re-read regardless of the refresh result: on success the cache now holds
      // the fresh catalog; on failure we harmlessly re-read the unchanged one.
      await loadCloudModels();
    } finally {
      catalogUpdating = false;
    }
  }

  // Loads the live Ollama model list for the current endpoint. Resilient: any
  // throw leaves the list empty (the free-text fallback then renders).
  async function loadOllamaModels(): Promise<void> {
    try {
      ollamaModelOptions = await listOllamaModelOptions(localEndpoint);
    } catch {
      ollamaModelOptions = [];
    }
  }

  // The routing payload sent to `saveEnrichmentPrefs`. 'explicit' pins the
  // currently-selected provider+model for the active tab.
  function buildRouting(): LlmRouting {
    if (routingKind === 'explicit') {
      return {
        kind: 'explicit',
        provider: canonicalProviderId,
        model: activeTab === 'local' ? localModel : cloudModel
      };
    }
    return { kind: routingKind };
  }

  // The coref override payload. '' clears the override (null); a chosen id pins
  // it to the active tab's canonical provider.
  function buildCorefModel(): TaskModel | null {
    return corefModelId ? { provider: canonicalProviderId, model: corefModelId } : null;
  }

  // Auto-detect: probe the current endpoint and populate the model list.
  async function handleAutoDetect(): Promise<void> {
    detectStatus = 'detecting';
    detectError = null;
    detectVersion = null;
    try {
      const result = await detectLlm(localEndpoint);
      if (result.reachable) {
        detectStatus = 'found';
        detectVersion = result.version;
        detectedModels = result.models;
        if (result.models.length > 0 && !result.models.includes(localModel)) {
          localModel = result.models[0];
        }
      } else {
        detectStatus = 'not-found';
        detectedModels = [];
      }
    } catch (err) {
      detectStatus = 'not-found';
      detectError = err instanceof Error ? err.message : 'Auto-detect failed';
    }
  }

  // Test connection: save config then probe, then re-run system check.
  async function handleTestConnection(): Promise<void> {
    testStatus = 'testing';
    testMessage = null;
    try {
      await saveLlmProvider({
        provider: 'ollama',
        base_url: localEndpoint,
        model: localModel,
        api_key: '',
        context: contextWindow
      });
      // Local provider never sends text off-machine → consent is irrelevant (false).
      await saveEnrichmentPrefs({
        enabled: enrichmentEnabled,
        coref_strategy: corefStrategy,
        cloud_consent: false,
        routing: buildRouting(),
        coref_model: buildCorefModel()
      });
      const result = await detectLlm(localEndpoint);
      if (result.reachable) {
        testStatus = 'success';
        testMessage = result.version ? `Connected — ${result.version}` : 'Connected';
        await oncheck();
      } else {
        testStatus = 'fail';
        testMessage = 'Could not reach the local server. Is it running?';
      }
    } catch (err) {
      testStatus = 'fail';
      testMessage = err instanceof Error ? err.message : 'Connection test failed.';
    }
  }

  // Cloud save: persist provider config then re-run the system check and collapse.
  async function handleSave(): Promise<void> {
    saving = true;
    saveError = null;
    try {
      const provider = CLOUD_PROVIDERS.find((p) => p.id === cloudProvider)!;
      await saveLlmProvider({
        // Persist the REAL provider id (openai/anthropic/google) so the Rust
        // factory validates the model against its OWN catalog namespace — a
        // blanket 'openai-compatible' validated claude-*/gemini-* against the
        // OpenAI namespace and silently broke routing (fix #1).
        provider: cloudProvider,
        base_url: cloudBaseUrl || provider.baseUrl,
        // User's chosen model id; fall back to the provider default if cleared.
        model: cloudModel.trim() || provider.defaultModel,
        api_key: cloudApiKey,
        context: CLOUD_DEFAULT_CONTEXT
      });
      // Cloud enrichment is gated on explicit consent: without it, enrichment
      // stays OFF so document text is never sent to a cloud LLM (and the Rust
      // factory rejects a cloud provider without consent regardless).
      await saveEnrichmentPrefs({
        enabled: enrichmentEnabled && cloudConsent,
        coref_strategy: corefStrategy,
        cloud_consent: cloudConsent,
        routing: buildRouting(),
        coref_model: buildCorefModel()
      });
      await oncheck();
      oncollapse();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save configuration.';
    } finally {
      saving = false;
    }
  }

  async function selectProvider(id: typeof cloudProvider): Promise<void> {
    cloudProvider = id;
    const p = CLOUD_PROVIDERS.find((p) => p.id === id)!;
    cloudBaseUrl = p.baseUrl;
    cloudModel = p.defaultModel;
    // Switching cards starts the model selection clean: drop any prior explicit
    // pick so loadCloudModels re-applies the smart default (newest text model)
    // for the new provider.
    cloudModelPicked = false;
    // Switching cards starts clean: clear any typed/edited key. `hasSavedKey`
    // recomputes from the derived (true only when `id` is the saved provider).
    editingKey = false;
    cloudApiKey = '';
    // The catalog (and thus the coref-override list) is provider-specific, so a
    // stale override id may no longer exist — clear it and reload the catalog.
    corefModelId = '';
    // Render the loaded list for the new provider immediately, then refresh live
    // in the background so the switched-to provider also converges to the current
    // models.dev list. Graceful when offline.
    await loadCloudModels();
    void refreshAndReloadCloud();
  }

  // The local model picker draws from Auto-detect results first, then the live
  // Ollama list (so pulled models surface without Auto-detect), de-duplicated.
  // When neither yields anything, fall back to the current single id (free-text
  // Input renders below for the single-or-empty case).
  const modelOptions = $derived.by(() => {
    const listed = ollamaModelOptions.map((o) => o.id);
    const merged = Array.from(new Set([...detectedModels, ...listed]));
    return merged.length > 0 ? merged : localModel ? [localModel] : [];
  });

  // Behaviorally identical to TtsConfigPanel's gate. For the SAVED provider we
  // require re-entry (a fresh non-empty key) since we never load the real key
  // into the DOM; for any UNSAVED provider the button enables as soon as a
  // non-empty key is typed and is disabled only while the field is empty.
  const cloudSaveDisabled = $derived(
    saving || (hasSavedKey ? !editingKey || !cloudApiKey.trim() : !cloudApiKey.trim())
  );

  // Pre-fills the Cloud tab from a previously-saved openai-compatible config
  // (provider/model) but keeps the real api_key OUT of the DOM — we only record
  // that a key exists so the field renders masked and Save stays disabled until
  // the user re-enters one. Its early `return`s are SELF-CONTAINED (this is a
  // dedicated function, not inline in onMount) so they never short-circuit the
  // background catalog refresh that runs after it.
  async function restoreSavedCloud(): Promise<void> {
    try {
      const cfg = await invoke<AppConfig>('get_config');
      // A saved cloud entry now carries the REAL provider id (openai/anthropic/
      // google); a legacy install may still carry 'openai-compatible' — match
      // either to a card (by provider id first, then by base_url for the legacy
      // shape). Only the matched card gets the per-provider saved/masked treatment.
      const cloudIds = CLOUD_PROVIDERS.map((p) => p.id) as string[];
      const saved = cfg.models?.find(
        (m) =>
          (cloudIds.includes(m.provider) || m.provider === 'openai-compatible') &&
          m.api_key.trim() !== ''
      );
      if (!saved) return;
      const match =
        CLOUD_PROVIDERS.find((p) => p.id === saved.provider) ??
        CLOUD_PROVIDERS.find((p) => p.baseUrl === saved.base_url);
      if (!match) return;
      savedProviderId = match.id;
      cloudProvider = match.id;
      cloudBaseUrl = saved.base_url || match.baseUrl;
      // Preserve the user's prior model choice (don't override with the smart
      // default). A truthy saved model counts as an explicit pick.
      if (saved.model) {
        cloudModel = saved.model;
        cloudModelPicked = true;
      } else {
        cloudModel = match.defaultModel;
      }
      cloudApiKey = '';
      // Reload the catalog for the saved provider so the picker + capability
      // controls reflect the restored selection.
      await loadCloudModels();
    } catch {
      // Non-fatal: fall back to the default empty Cloud form.
    }
  }

  // On mount, populate the pickers from the loaded catalog, restore any saved
  // cloud config, then refresh the catalog live in the background.
  onMount(async () => {
    if (!isTauri()) return;
    // Populate the capability-aware pickers on expand from the ALREADY-LOADED
    // catalog (cache-or-bundled-floor), so the picker is never empty. Both are
    // resilient (any throw ⇒ empty options + fallback), so onboarding stays
    // non-blocking.
    await Promise.all([loadCloudModels(), loadOllamaModels()]);
    await restoreSavedCloud();
    // After the immediate (loaded) options are on screen and any saved provider
    // is restored, trigger a LIVE refresh + re-read in the background so an online
    // user converges to the CURRENT full models.dev list. Fire-and-forget: NOT
    // awaited so the loaded options render instantly; the picker updates in place
    // when the fetch completes. Graceful when offline (keeps the loaded list).
    void refreshAndReloadCloud();
  });

  // Entering "editing" mode clears the masked field so the user types a fresh key.
  function startEditingKey(): void {
    if (hasSavedKey && !editingKey) {
      editingKey = true;
      cloudApiKey = '';
    }
  }
</script>

<div class="pt-3">
  <!-- Segmented tabs: Local | Cloud API -->
  <div
    class="bg-muted flex w-full items-center rounded-lg p-0.5"
    role="tablist"
    aria-label="LLM provider type"
  >
    <button
      role="tab"
      aria-selected={activeTab === 'local'}
      aria-controls="llm-panel-local"
      id="llm-tab-local"
      class={cn(
        'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        activeTab === 'local'
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
      onclick={() => (activeTab = 'local')}
    >
      Local
    </button>
    <button
      role="tab"
      aria-selected={activeTab === 'cloud'}
      aria-controls="llm-panel-cloud"
      id="llm-tab-cloud"
      class={cn(
        'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        activeTab === 'cloud'
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
      onclick={() => (activeTab = 'cloud')}
    >
      Cloud API
    </button>
  </div>

  <!-- Local tab panel -->
  <div
    id="llm-panel-local"
    role="tabpanel"
    aria-labelledby="llm-tab-local"
    tabindex={activeTab === 'local' ? 0 : -1}
    class={cn('mt-3 flex flex-col gap-3', activeTab !== 'local' && 'hidden')}
  >
    <!-- Helper text -->
    <p class="text-muted-foreground text-[0.78rem] leading-relaxed">
      Works with Ollama, LM Studio, vLLM, Jan, llama.cpp — any OpenAI-compatible local server.
    </p>

    <!-- API ENDPOINT field -->
    <div class="flex flex-col gap-1.5">
      <label
        for="llm-endpoint"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        API Endpoint
      </label>
      <div class="flex gap-2">
        <Input
          id="llm-endpoint"
          type="url"
          bind:value={localEndpoint}
          placeholder="http://localhost:11434"
          class="flex-1"
          autocomplete="off"
          spellcheck={false}
        />
        <Button
          variant="outline"
          size="sm"
          onclick={handleAutoDetect}
          disabled={detectStatus === 'detecting'}
          aria-label="Auto-detect local LLM"
          class="shrink-0"
        >
          {detectStatus === 'detecting' ? 'Detecting…' : 'Auto-detect'}
        </Button>
      </div>

      {#if detectStatus === 'found' && detectVersion}
        <p class="text-primary text-[0.75rem]">{detectVersion} detected</p>
      {:else if detectStatus === 'not-found'}
        <p class="text-muted-foreground text-[0.75rem]">
          {detectError ?? 'Not detected — check that your local server is running.'}
        </p>
      {/if}
    </div>

    <!-- MODEL field -->
    <div class="flex flex-col gap-1.5">
      <label
        for="llm-model-local"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        Model
      </label>
      {#if modelOptions.length > 1}
        <select id="llm-model-local" bind:value={localModel} class={SELECT_CLASS}>
          {#each modelOptions as m (m)}
            <option value={m}>{m}</option>
          {/each}
        </select>
      {:else}
        <Input
          id="llm-model-local"
          type="text"
          bind:value={localModel}
          placeholder="llama3.2:3b"
          autocomplete="off"
          spellcheck={false}
        />
      {/if}
    </div>

    <!-- CONTEXT WINDOW field -->
    <div class="flex flex-col gap-1.5">
      <div class="flex items-baseline gap-1">
        <label
          for="llm-context-custom"
          class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
        >
          Context Window
        </label>
        <span class="text-muted-foreground text-[0.68rem]">— affects source bloat</span>
      </div>
      <div id="llm-context-window" class="flex gap-1" role="group" aria-label="Context window size">
        {#each CONTEXT_OPTIONS as opt (opt.value)}
          <button
            type="button"
            onclick={() => (contextWindow = opt.value)}
            aria-pressed={contextWindow === opt.value}
            class={cn(
              'flex-1 rounded-md border px-2 py-1.5 text-[0.75rem] font-medium transition-colors',
              contextWindow === opt.value
                ? 'bg-primary text-primary-foreground border-primary'
                : 'border-border bg-transparent text-muted-foreground hover:bg-muted hover:text-foreground'
            )}
          >
            {opt.label}
          </button>
        {/each}
      </div>
      <!-- Custom override: presets are shortcuts, but many models use sizes
           outside the list (e.g. 2K, 64K). A value typed here takes priority. -->
      <div class="mt-2 flex items-center gap-2">
        <input
          id="llm-context-custom"
          type="number"
          min="256"
          step="256"
          value={contextWindow}
          oninput={(e) => {
            const v = parseInt(e.currentTarget.value, 10);
            if (Number.isFinite(v) && v >= 256) contextWindow = v;
          }}
          aria-label="Custom context window in tokens"
          class="border-input focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 placeholder:text-muted-foreground h-8 w-full min-w-0 rounded-lg border bg-transparent px-2.5 py-1 text-base transition-colors outline-none focus-visible:ring-3 md:text-sm"
        />
        <span class="text-muted-foreground shrink-0 text-[0.72rem]">tokens (custom)</span>
      </div>
      {#if contextHelper}
        <p class="text-muted-foreground text-[0.72rem] leading-relaxed">{contextHelper}</p>
      {/if}
    </div>

    <!-- Test connection status -->
    {#if testStatus === 'success' && testMessage}
      <p class="text-primary text-[0.75rem]">{testMessage}</p>
    {:else if testStatus === 'fail' && testMessage}
      <p class="text-destructive text-[0.75rem]" role="alert">{testMessage}</p>
    {/if}

    <!-- Test connection button -->
    <Button class="h-10 w-full" onclick={handleTestConnection} disabled={testStatus === 'testing'}>
      {#if testStatus === 'testing'}
        <LoaderCircle class="size-4 animate-spin" />
        Testing…
      {:else}
        Test connection
      {/if}
    </Button>
  </div>

  <!-- Cloud API tab panel -->
  <div
    id="llm-panel-cloud"
    role="tabpanel"
    aria-labelledby="llm-tab-cloud"
    tabindex={activeTab === 'cloud' ? 0 : -1}
    class={cn('mt-3 flex flex-col gap-3', activeTab !== 'cloud' && 'hidden')}
  >
    <!-- Provider cards -->
    <div class="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Cloud LLM provider">
      {#each CLOUD_PROVIDERS as provider (provider.id)}
        {@const isSelected = cloudProvider === provider.id}
        <button
          role="radio"
          aria-checked={isSelected}
          onclick={() => selectProvider(provider.id)}
          class={cn(
            'rounded-lg border px-2.5 py-2 text-left transition-colors',
            isSelected
              ? 'border-primary bg-primary/10 ring-1 ring-primary'
              : 'border-border bg-card hover:bg-muted/50'
          )}
        >
          <p class="text-[0.8rem] font-semibold text-foreground">{provider.name}</p>
          <p class="text-[0.7rem] text-muted-foreground mt-0.5">{provider.models}</p>
        </button>
      {/each}
    </div>

    <!-- MODEL -->
    <div class="flex flex-col gap-1.5">
      <div class="flex items-baseline gap-1.5">
        <label
          for="llm-cloud-model"
          class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
        >
          Model
        </label>
        {#if catalogUpdating}
          <span class="text-muted-foreground text-[0.68rem]" aria-live="polite">updating…</span>
        {/if}
      </div>
      <select
        id="llm-cloud-model"
        value={cloudModel}
        onchange={(e) => {
          cloudModel = e.currentTarget.value;
          // An explicit pick: pin it so a background refresh won't re-default it.
          cloudModelPicked = true;
        }}
        class={SELECT_CLASS}
      >
        {#each cloudSelectOptions as opt (opt.id)}
          <option value={opt.id}>{opt.label}</option>
        {/each}
      </select>
      {#if contextHint || costHint}
        <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
          {contextHint}{#if contextHint && costHint}{' · '}{/if}{costHint}
        </p>
      {/if}
    </div>

    <!-- API KEY -->
    <div class="flex flex-col gap-1.5">
      <label
        for="llm-cloud-key"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        API Key
      </label>
      <Input
        id="llm-cloud-key"
        type="password"
        bind:value={cloudApiKey}
        placeholder={hasSavedKey && !editingKey
          ? '•••••••••• saved — click to replace'
          : 'Paste API key…'}
        autocomplete="new-password"
        onfocus={startEditingKey}
        oninput={startEditingKey}
      />
      {#if hasSavedKey && !editingKey}
        <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
          A key is already saved. Click the field to replace it.
        </p>
      {/if}
    </div>

    <!-- BASE URL -->
    <div class="flex flex-col gap-1.5">
      <div class="flex items-baseline gap-1">
        <label
          for="llm-cloud-url"
          class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
        >
          Base URL
        </label>
        <span class="text-muted-foreground text-[0.68rem]">— optional override</span>
      </div>
      <Input
        id="llm-cloud-url"
        type="url"
        bind:value={cloudBaseUrl}
        placeholder="https://api.openai.com/v1"
        autocomplete="off"
        spellcheck={false}
      />
      <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
        Leave empty to use the default. Override for Groq, Together AI, Azure OpenAI, Anyscale and
        other compatible providers.
      </p>
    </div>

    <!-- Save error -->
    {#if saveError}
      <p class="text-destructive text-[0.75rem]" role="alert">{saveError}</p>
    {/if}

    <!-- Save button -->
    <Button class="h-10 w-full" onclick={handleSave} disabled={cloudSaveDisabled}>
      {saving ? 'Saving…' : 'Save'}
    </Button>
  </div>

  <!-- ── Enrichment (M4 Phase 3) ──────────────────────────────────────────────
       Shared across both tabs (the chosen provider runs the optional, additive
       enrichment pass). Saved alongside the provider above; non-blocking. -->
  <div class="border-border mt-4 flex flex-col gap-3 border-t pt-4">
    <!-- ENABLE toggle row -->
    <div class="flex items-start justify-between gap-3">
      <div class="flex flex-col gap-0.5">
        <label for="enrichment-enabled" class="text-foreground text-[0.82rem] font-medium">
          Improve retrieval with enrichment
        </label>
        <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
          After a source is indexed, your LLM builds context (entities, sections, a summary) in the
          background to sharpen search. Canonical text is never changed; sources stay usable
          throughout.
        </p>
      </div>
      <button
        id="enrichment-enabled"
        type="button"
        role="switch"
        aria-checked={enrichmentEnabled}
        aria-label="Enable enrichment"
        onclick={() => (enrichmentEnabled = !enrichmentEnabled)}
        class={cn(
          'relative mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors',
          'focus-visible:ring-ring/50 outline-none focus-visible:ring-3',
          enrichmentEnabled ? 'bg-primary' : 'bg-muted'
        )}
      >
        <span
          class={cn(
            'bg-background inline-block size-4 transform rounded-full shadow-sm transition-transform',
            enrichmentEnabled ? 'translate-x-4' : 'translate-x-0.5'
          )}
        ></span>
      </button>
    </div>

    <!-- COREF STRATEGY select -->
    <div class={cn('flex flex-col gap-1.5', !enrichmentEnabled && 'opacity-50')}>
      <label
        for="enrichment-coref"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        Pronoun resolution
      </label>
      <select
        id="enrichment-coref"
        bind:value={corefStrategy}
        disabled={!enrichmentEnabled}
        class={SELECT_CLASS}
      >
        {#each COREF_OPTIONS as opt (opt.value)}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
      <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
        Resolves pronouns to their referents while building context, so a query like “what did she
        invent?” still matches the right passage.
      </p>
    </div>

    <!-- ROUTING select — how the enrichment LLM is chosen. -->
    <div class="flex flex-col gap-1.5">
      <label
        for="enrichment-routing"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        Routing
      </label>
      <select
        id="enrichment-routing"
        value={routingKind}
        onchange={(e) =>
          (routingKind = e.currentTarget.value as 'cloud_first' | 'local_first' | 'explicit')}
        class={SELECT_CLASS}
      >
        {#each ROUTING_OPTIONS as opt (opt.value)}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
      <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
        How the enrichment model is picked. Cloud-first prefers a consented cloud provider then
        local; Local-first is the inverse; Explicit pins the provider and model selected above.
      </p>
    </div>

    <!-- COREF-MODEL OVERRIDE — optional per-task model pin. Gated on enrichment. -->
    <div class={cn('flex flex-col gap-1.5', !enrichmentEnabled && 'opacity-50')}>
      <label
        for="enrichment-coref-model"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        Coreference model
      </label>
      <select
        id="enrichment-coref-model"
        value={corefModelId}
        onchange={(e) => (corefModelId = e.currentTarget.value)}
        disabled={!enrichmentEnabled}
        class={SELECT_CLASS}
      >
        <option value="">Use the configured model</option>
        {#each corefModelOptions as opt (opt.id)}
          <option value={opt.id}>{opt.label}</option>
        {/each}
      </select>
      <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
        Optionally pin a specific model for pronoun resolution. Leave on the default to reuse the
        configured enrichment model.
      </p>
    </div>

    <!-- CLOUD CONSENT — only when a cloud provider is selected. Honest privacy +
         cost disclosure (not a dark pattern): defaults OFF, explicit-enable. -->
    {#if activeTab === 'cloud'}
      <div class="border-border bg-muted/40 flex flex-col gap-2 rounded-lg border p-3">
        <label class="flex items-start gap-2.5">
          <input
            type="checkbox"
            bind:checked={cloudConsent}
            class="border-input text-primary focus-visible:ring-ring/50 mt-0.5 size-4 shrink-0 rounded border outline-none focus-visible:ring-3"
            aria-describedby="enrichment-cloud-note"
          />
          <span class="text-foreground text-[0.8rem] font-medium">
            Send document text to this cloud provider for enrichment
          </span>
        </label>
        <p
          id="enrichment-cloud-note"
          class="text-muted-foreground pl-[1.625rem] text-[0.72rem] leading-relaxed"
        >
          Enrichment sends the full text of your sources to {selectedProvider.name} so it can build context.
          That text leaves your machine and may incur API costs billed by the provider per token. Local-first
          is the default — leave this off to keep enrichment on-device only (no cloud enrichment runs
          without your consent).
        </p>
      </div>
    {/if}
  </div>
</div>
