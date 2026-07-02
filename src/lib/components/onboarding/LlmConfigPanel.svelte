<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Combobox } from 'bits-ui';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import ChevronsUpDown from '@lucide/svelte/icons/chevrons-up-down';
  import Check from '@lucide/svelte/icons/check';
  import { cn } from '$lib/utils.js';
  import { detectLlm } from '$lib/onboarding/system-check.js';
  import {
    saveLlmProvider,
    saveEnrichmentPrefs,
    type LlmProviderTab
  } from '$lib/onboarding/llm-config.js';
  import { validateModelInteractive } from '$lib/onboarding/enrichment-validation.js';
  import {
    CLOUD_PROVIDERS,
    CLOUD_PROVIDER_IDS,
    findCloudProvider,
    defaultModelFor,
    type CloudProvider
  } from '$lib/onboarding/cloud-providers.js';
  import type { AppConfig, CorefStrategy, LlmRouting, TaskModel } from '$lib/theme/types.js';
  import {
    listCloudModelOptions,
    listOllamaModelOptions,
    refreshCatalog,
    formatCompact,
    formatUsd,
    type ModelOption
  } from '$lib/models/catalog.js';
  import { SELECT_CLASS } from './styles.js';

  // Cloud config has no Context Window picker (it's Local-only), so cloud saves
  // persist a sensible large default that covers modern hosted models.
  const CLOUD_DEFAULT_CONTEXT = 128000;

  // The placeholder shown in the (empty) free-text model fields when no Ollama
  // model is detected — a SUGGESTION, not a persisted default (Rev 2). Matches the
  // design mockup (`Lens Onboarding.dc.html`).
  const MODEL_PLACEHOLDER = 'e.g. llama3.2:3b';

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
  let contextWindow = $state(8192);
  // Detected models list (populated by Auto-detect)
  let detectedModels = $state<string[]>([]);
  let detectStatus = $state<'idle' | 'detecting' | 'found' | 'not-found'>('idle');
  let detectError = $state<string | null>(null);
  let testStatus = $state<'idle' | 'testing' | 'success' | 'fail'>('idle');
  let testMessage = $state<string | null>(null);

  // --- Two model roles (Rev 2) ---
  // ENRICHMENT model (BLOCKING): drives the base `models[]` entry that
  // `probe_llm_runtime` validates/gates. STUDIO & CHAT model (NON-blocking):
  // persists to `enrichment.chat_model` for M5. Each role has a Local and a Cloud
  // value; the active tab decides which pair is live. All start EMPTY (placeholder,
  // not a hardcoded default) so the picker/pull-prompt drives selection.
  let enrichmentLocalModel = $state('');
  let studioChatLocalModel = $state('');
  let enrichmentCloudModel = $state('');
  let studioChatCloudModel = $state('');

  // Per-role interactive validation status (shown inline). Enrichment BLOCKS save
  // on 'invalid'; studio/chat is informational only (NEVER blocks).
  let enrichmentValidation = $state<'idle' | 'checking' | 'valid' | 'invalid'>('idle');
  let enrichmentValidationMessage = $state<string | null>(null);
  let studioChatValidation = $state<'idle' | 'checking' | 'valid' | 'invalid'>('idle');
  let studioChatValidationMessage = $state<string | null>(null);

  // --- Cloud API tab fields ---
  // The canonical provider id (= models.dev catalog key for first-class providers).
  let cloudProvider = $state<string>(CLOUD_PROVIDERS[0].id);
  let providerQuery = $state('');
  let cloudBaseUrl = $state('');
  let cloudApiKey = $state('');
  // Whether the user has EXPLICITLY chosen an ENRICHMENT cloud model (via the
  // picker) or restored a saved one. While false, the smart default re-resolves to
  // the newest text-capable model on every (re)load. Studio/chat has no smart
  // default (starts "Not set").
  let cloudModelPicked = $state(false);

  // --- Capability-aware model pickers (M4 Phase 3, Stage 3) ---
  let cloudModelOptions = $state<ModelOption[]>([]);
  let ollamaModelOptions = $state<ModelOption[]>([]);
  let catalogUpdating = $state(false);

  // --- Save state (cloud) ---
  let saving = $state(false);
  let saveError = $state<string | null>(null);

  // --- Saved cloud key masking (per-provider) ---
  let savedProviderId = $state<string | null>(null);
  let editingKey = $state(false);
  const hasSavedKey = $derived(cloudProvider === savedProviderId);

  // --- Enrichment preferences (M4 Phase 3) ---
  let enrichmentEnabled = $state(true);
  let corefStrategy = $state<CorefStrategy>('llm_inline');
  let cloudConsent = $state(false);

  // --- Routing + per-task overrides (M4 Phase 3, Stage 3) ---
  let routingKind = $state<'cloud_first' | 'local_first' | 'explicit'>('cloud_first');
  let corefModelId = $state('');

  const ROUTING_OPTIONS = [
    { value: 'cloud_first' as const, label: 'Cloud-first' },
    { value: 'local_first' as const, label: 'Local-first' },
    { value: 'explicit' as const, label: 'Explicit' }
  ] as const;

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

  const PROVIDER_GROUPS = [
    { key: 'popular' as const, label: 'Popular' },
    { key: 'all' as const, label: 'All' }
  ] as const;

  const filteredProviders = $derived.by(() => {
    const raw = providerQuery.trim();
    const q = raw.toLowerCase();
    const isSelectionEcho = raw !== '' && raw === selectedProvider.name;
    const matches = (p: CloudProvider) =>
      q === '' ||
      isSelectionEcho ||
      p.name.toLowerCase().includes(q) ||
      p.id.toLowerCase().includes(q);
    return PROVIDER_GROUPS.map((g) => ({
      ...g,
      items: CLOUD_PROVIDERS.filter((p) => p.group === g.key && matches(p))
    })).filter((g) => g.items.length > 0);
  });

  const contextHelper = $derived(
    CONTEXT_OPTIONS.find((o) => o.value === contextWindow)?.helper ?? ''
  );

  const selectedProvider = $derived(findCloudProvider(cloudProvider) ?? CLOUD_PROVIDERS[0]);

  const isCustomProvider = $derived(selectedProvider.custom === true);

  const cloudCatalogKey = $derived(selectedProvider.catalogKey);

  // The canonical backend provider id for the ACTIVE tab.
  const canonicalProviderId = $derived(activeTab === 'local' ? 'ollama' : cloudProvider);

  // The options actually rendered in the cloud model <select>s. When the catalog is
  // empty (offline / non-Tauri / mock returns nothing), fall back to a single option
  // for the provider default so the enrichment picker stays usable + legacy tests pass.
  const cloudSelectOptions = $derived<ModelOption[]>(
    cloudModelOptions.length > 0
      ? cloudModelOptions
      : [
          {
            id: defaultModelFor(cloudProvider),
            label: defaultModelFor(cloudProvider),
            info: null
          }
        ]
  );

  // Capability info for the ENRICHMENT cloud model (drives the context/cost helper).
  const selectedCloudModel = $derived(
    cloudModelOptions.find((o) => o.id === enrichmentCloudModel) ?? null
  );

  const modelHint = $derived.by(() => {
    const info = selectedCloudModel?.info;
    if (!info) return '';
    const clauses: string[] = [];
    if (info.context_limit != null) {
      clauses.push(`${formatCompact(info.context_limit)} Context`);
    }
    if (info.cost?.input != null) {
      clauses.push(`~$${formatUsd(info.cost.input)}/1M input`);
    }
    if (info.cost?.output != null) {
      clauses.push(`~$${formatUsd(info.cost.output)}/1M output`);
    }
    return clauses.join(' · ');
  });

  const corefModelOptions = $derived(
    activeTab === 'local' ? ollamaModelOptions : cloudModelOptions
  );

  // The local model picker draws from Auto-detect results first, then the live
  // Ollama list (so pulled models surface without Auto-detect), de-duplicated. NO
  // hardcoded-default fallback (Rev 2): when neither yields anything the free-text
  // pull-prompt renders (empty value + placeholder).
  const localModelOptions = $derived.by(() => {
    const listed = ollamaModelOptions.map((o) => o.id);
    return Array.from(new Set([...detectedModels, ...listed]));
  });

  // Loads the cloud catalog (text-capable, newest first) for the selected provider
  // and resolves the ENRICHMENT cloud model via the smart default. Studio/chat is
  // left untouched here (starts "Not set"; user picks explicitly).
  async function loadCloudModels(): Promise<void> {
    if (cloudCatalogKey === null) {
      cloudModelOptions = [];
      if (!cloudModelPicked && !enrichmentCloudModel.trim()) {
        enrichmentCloudModel = defaultModelFor(cloudProvider);
      }
      return;
    }
    try {
      const opts = await listCloudModelOptions(cloudCatalogKey);
      cloudModelOptions = opts.length > 0 ? opts : [];
    } catch {
      cloudModelOptions = [];
    }
    if (cloudModelOptions.length === 0) {
      enrichmentCloudModel = defaultModelFor(cloudProvider);
      return;
    }
    if (!cloudModelPicked) {
      enrichmentCloudModel = cloudModelOptions[0].id;
      return;
    }
    if (!cloudModelOptions.some((o) => o.id === enrichmentCloudModel)) {
      enrichmentCloudModel = defaultModelFor(cloudProvider);
    }
  }

  async function refreshAndReloadCloud(): Promise<void> {
    catalogUpdating = true;
    try {
      await refreshCatalog();
      await loadCloudModels();
    } finally {
      catalogUpdating = false;
    }
  }

  // Loads the live Ollama model list for the current endpoint. Resilient: any throw
  // leaves the list empty (the free-text pull-prompt then renders).
  async function loadOllamaModels(): Promise<void> {
    try {
      ollamaModelOptions = await listOllamaModelOptions(localEndpoint);
    } catch {
      ollamaModelOptions = [];
    }
  }

  // ENRICHMENT auto-selects the first detected model when its field is empty or no
  // longer in the list. Studio & Chat stays empty until the user picks it.
  $effect(() => {
    const ids = localModelOptions;
    if (ids.length === 0) return;
    if (!enrichmentLocalModel || !ids.includes(enrichmentLocalModel)) {
      enrichmentLocalModel = ids[0];
    }
  });

  // The routing payload sent to `saveEnrichmentPrefs`. 'explicit' pins the active
  // tab's ENRICHMENT provider+model.
  function buildRouting(): LlmRouting {
    if (routingKind === 'explicit') {
      return {
        kind: 'explicit',
        provider: canonicalProviderId,
        model: activeTab === 'local' ? enrichmentLocalModel : enrichmentCloudModel
      };
    }
    return { kind: routingKind };
  }

  function buildCorefModel(): TaskModel | null {
    return corefModelId ? { provider: canonicalProviderId, model: corefModelId } : null;
  }

  // The Studio & Chat model TaskModel for the ACTIVE tab, or null when unpicked
  // (empty/whitespace). Persisted into `enrichment.chat_model` (non-blocking).
  function buildChatModel(): TaskModel | null {
    const raw = activeTab === 'local' ? studioChatLocalModel : studioChatCloudModel;
    const model = raw.trim();
    return model ? { provider: canonicalProviderId, model } : null;
  }

  // Auto-detect: probe the current endpoint and populate the model list.
  async function handleAutoDetect(): Promise<void> {
    detectStatus = 'detecting';
    detectError = null;
    try {
      const result = await detectLlm(localEndpoint);
      if (result.reachable) {
        detectStatus = 'found';
        detectedModels = result.models;
      } else {
        detectStatus = 'not-found';
        detectedModels = [];
      }
    } catch (err) {
      detectStatus = 'not-found';
      detectError = err instanceof Error ? err.message : 'Auto-detect failed';
    }
  }

  // Shared per-role interactive validation. Sets the role's status to 'checking',
  // runs the (role-neutral) interactive probe, then records 'valid'/'invalid' +
  // message. Returns { ok, reason } so the caller decides blocking vs informational.
  async function runModelValidation(
    role: 'enrichment' | 'studioChat',
    provider: string,
    model: string,
    baseUrl: string,
    apiKey: string
  ): Promise<{ ok: boolean; reason?: string }> {
    if (role === 'enrichment') {
      enrichmentValidation = 'checking';
      enrichmentValidationMessage = null;
    } else {
      studioChatValidation = 'checking';
      studioChatValidationMessage = null;
    }
    const result = await validateModelInteractive(provider, model, baseUrl, apiKey);
    const ok = result.status === 'valid';
    if (role === 'enrichment') {
      enrichmentValidation = ok ? 'valid' : 'invalid';
      enrichmentValidationMessage = ok ? null : (result.reason ?? 'Model validation failed.');
    } else {
      studioChatValidation = ok ? 'valid' : 'invalid';
      studioChatValidationMessage = ok ? null : (result.reason ?? 'Model validation failed.');
    }
    return { ok, reason: result.reason };
  }

  // Test connection (LOCAL): validate BEFORE persist. Enrichment blocks on invalid;
  // studio/chat is informational only.
  //
  // Signal strategy (Fix 1 — no double-signal):
  // • When enrichmentEnabled: the inline per-role validationStatus cue is the SOLE
  //   model-validation signal. testMessage stays null on both success and enrichment
  //   failure (the cue conveys it). testMessage only appears for the opt-out
  //   reachability path or an unexpected thrown error.
  // • When enrichment opted out (enrichmentEnabled === false): detectLlm drives
  //   testStatus + testMessage ("Connected — <version>" / "Could not reach…").
  async function handleTestConnection(): Promise<void> {
    testStatus = 'testing';
    testMessage = null;
    enrichmentValidation = 'idle';
    enrichmentValidationMessage = null;
    studioChatValidation = 'idle';
    studioChatValidationMessage = null;
    try {
      // ENRICHMENT (blocking) — only when enrichment is enabled (opt-out skips it).
      if (enrichmentEnabled) {
        const { ok } = await runModelValidation(
          'enrichment',
          'ollama',
          enrichmentLocalModel,
          localEndpoint,
          ''
        );
        if (!ok) {
          // Inline cue (enrichmentValidation='invalid') conveys the reason.
          // DO NOT set testMessage — that would duplicate the inline signal.
          testStatus = 'fail';
          return; // DO NOT persist, DO NOT oncheck.
        }
      }
      // STUDIO & CHAT (non-blocking) — validate informationally when a model is set.
      if (studioChatLocalModel.trim()) {
        await runModelValidation('studioChat', 'ollama', studioChatLocalModel, localEndpoint, '');
      }
      // Persist: base enrichment model entry + enrichment prefs (with chat_model).
      await saveLlmProvider({
        provider: 'ollama',
        base_url: localEndpoint,
        model: enrichmentLocalModel,
        api_key: '',
        context: contextWindow
      });
      await saveEnrichmentPrefs({
        enabled: enrichmentEnabled,
        coref_strategy: corefStrategy,
        cloud_consent: false,
        routing: buildRouting(),
        coref_model: buildCorefModel(),
        chat_model: buildChatModel()
      });
      if (enrichmentEnabled) {
        // Enrichment on + validation passed: inline "Available" cue is the confirmation.
        // No separate testMessage needed.
        testStatus = 'success';
        await oncheck();
      } else {
        // Enrichment opted out: reachability check drives the testMessage signal.
        const result = await detectLlm(localEndpoint);
        if (result.reachable) {
          testStatus = 'success';
          testMessage = result.version ? `Connected — ${result.version}` : 'Connected';
          await oncheck();
        } else {
          testStatus = 'fail';
          testMessage = 'Could not reach the local server. Is it running?';
        }
      }
    } catch (err) {
      testStatus = 'fail';
      testMessage = err instanceof Error ? err.message : 'Connection test failed.';
    }
  }

  // Cloud save: validate BEFORE persist. Enrichment blocks (gated on consent);
  // studio/chat informational only. Persists chat_model regardless.
  async function handleSave(): Promise<void> {
    saving = true;
    saveError = null;
    enrichmentValidation = 'idle';
    enrichmentValidationMessage = null;
    studioChatValidation = 'idle';
    studioChatValidationMessage = null;
    try {
      const enrichmentActive = enrichmentEnabled && cloudConsent;
      // ENRICHMENT (blocking) — only when enrichment is actually active for cloud.
      // Inline cue (enrichmentValidation='invalid') conveys the reason; DO NOT
      // duplicate into saveError (that would create a double signal).
      if (enrichmentActive) {
        const { ok } = await runModelValidation(
          'enrichment',
          cloudProvider,
          enrichmentCloudModel,
          cloudBaseUrl,
          cloudApiKey
        );
        if (!ok) {
          return; // Inline cue shows reason. DO NOT persist, DO NOT oncheck.
        }
      }
      // STUDIO & CHAT (non-blocking) — informational when a model is set.
      if (studioChatCloudModel.trim()) {
        await runModelValidation(
          'studioChat',
          cloudProvider,
          studioChatCloudModel,
          cloudBaseUrl,
          cloudApiKey
        );
      }
      await saveLlmProvider({
        provider: cloudProvider,
        base_url: isCustomProvider ? cloudBaseUrl || selectedProvider.baseUrl : '',
        model: enrichmentCloudModel.trim() || defaultModelFor(cloudProvider),
        api_key: cloudApiKey,
        context: CLOUD_DEFAULT_CONTEXT
      });
      await saveEnrichmentPrefs({
        enabled: enrichmentEnabled && cloudConsent,
        coref_strategy: corefStrategy,
        cloud_consent: cloudConsent,
        routing: buildRouting(),
        coref_model: buildCorefModel(),
        chat_model: buildChatModel()
      });
      await oncheck();
      oncollapse();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save configuration.';
    } finally {
      saving = false;
    }
  }

  async function selectProvider(id: string): Promise<void> {
    cloudProvider = id;
    const p = findCloudProvider(id) ?? CLOUD_PROVIDERS[0];
    cloudBaseUrl = p.baseUrl;
    enrichmentCloudModel = defaultModelFor(id);
    studioChatCloudModel = '';
    cloudModelPicked = false;
    editingKey = false;
    cloudApiKey = '';
    corefModelId = '';
    await loadCloudModels();
    void refreshAndReloadCloud();
  }

  const cloudSaveDisabled = $derived(
    saving ||
      !enrichmentCloudModel.trim() ||
      (hasSavedKey ? !editingKey || !cloudApiKey.trim() : !cloudApiKey.trim())
  );

  // Test Connection is disabled while testing OR while the ENRICHMENT local model is
  // empty (studio/chat empty must NOT disable anything).
  const testConnectionDisabled = $derived(testStatus === 'testing' || !enrichmentLocalModel.trim());

  async function restoreSavedCloud(): Promise<void> {
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const saved = cfg.models?.find(
        (m) => CLOUD_PROVIDER_IDS.includes(m.provider) && m.api_key.trim() !== ''
      );
      if (!saved) return;
      const match = findCloudProvider(saved.provider);
      if (!match) return;
      savedProviderId = match.id;
      cloudProvider = match.id;
      cloudBaseUrl = match.custom ? saved.base_url || match.baseUrl : '';
      if (saved.model) {
        enrichmentCloudModel = saved.model;
        cloudModelPicked = true;
      } else {
        enrichmentCloudModel = defaultModelFor(match.id);
      }
      // Round-trip: restore the saved studio/chat model when it targets the same
      // cloud provider. NEVER clobber the enrichment model with it.
      const chatModel = cfg.enrichment?.chat_model;
      if (chatModel && chatModel.provider === match.id && chatModel.model) {
        studioChatCloudModel = chatModel.model;
      }
      cloudApiKey = '';
      await loadCloudModels();
    } catch {
      // Non-fatal: fall back to the default empty Cloud form.
    }
  }

  // Restores a saved local (Ollama) studio/chat model from a prior session. Only
  // sets it when present; never clobbers a user-typed value.
  async function restoreSavedLocal(): Promise<void> {
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const chatModel = cfg.enrichment?.chat_model;
      if (chatModel && chatModel.provider === 'ollama' && chatModel.model) {
        studioChatLocalModel = chatModel.model;
      }
    } catch {
      // Non-fatal: keep the empty studio/chat field.
    }
  }

  onMount(async () => {
    if (!isTauri()) return;
    await Promise.all([loadCloudModels(), loadOllamaModels()]);
    await restoreSavedLocal();
    await restoreSavedCloud();
    void refreshAndReloadCloud();
  });

  function startEditingKey(): void {
    if (hasSavedKey && !editingKey) {
      editingKey = true;
      cloudApiKey = '';
    }
  }
</script>

<!-- ── Per-role LOCAL model selector ─────────────────────────────────────────
     `role` distinguishes the two ids. `value`/`onModel` bind to the role's state.
     Picker when Ollama models exist; free-text pull-prompt (empty + placeholder)
     when none, with a copyable `ollama pull` command + a Re-check button. The
     `notSet` flag adds a "Not set" option (studio/chat only). -->
{#snippet localModelSelector(
  id: string,
  role: 'enrichment' | 'studioChat',
  value: string,
  onModel: (v: string) => void,
  notSet: boolean
)}
  {#if localModelOptions.length > 0}
    <select {id} {value} onchange={(e) => onModel(e.currentTarget.value)} class={SELECT_CLASS}>
      {#if notSet}
        <option value="">Not set</option>
      {/if}
      {#each localModelOptions as m (m)}
        <option value={m}>{m}</option>
      {/each}
    </select>
  {:else}
    <Input
      {id}
      type="text"
      {value}
      oninput={(e) => onModel(e.currentTarget.value)}
      placeholder={MODEL_PLACEHOLDER}
      autocomplete="off"
      spellcheck={false}
    />
    <div class="flex items-center gap-2">
      <code
        class="border-input bg-muted/40 text-foreground min-w-0 flex-1 truncate rounded-md border px-2 py-1 text-[0.72rem]"
      >
        ollama pull {value.trim() || 'llama3.2:3b'}
      </code>
      <Button
        variant="outline"
        size="sm"
        onclick={loadOllamaModels}
        aria-label={`Re-check Ollama models for ${role} model`}
        class="shrink-0"
      >
        Re-check
      </Button>
    </div>
  {/if}
{/snippet}

<!-- ── Per-role CLOUD model selector ─────────────────────────────────────────
     Reuses the cloud catalog picker (or free-text for catalog-less providers).
     Enrichment uses `cloudSelectOptions` (with the offline seed fallback) and pins
     `cloudModelPicked` on change; studio/chat starts "Not set". -->
{#snippet cloudModelSelector(
  id: string,
  role: 'enrichment' | 'studioChat',
  value: string,
  onModel: (v: string) => void
)}
  {#if cloudCatalogKey === null}
    <Input
      {id}
      type="text"
      {value}
      oninput={(e) => onModel(e.currentTarget.value)}
      placeholder={role === 'enrichment' ? 'model id (e.g. gpt-oss:20b)' : 'model id (optional)'}
      autocomplete="off"
      spellcheck={false}
    />
  {:else}
    <select {id} {value} onchange={(e) => onModel(e.currentTarget.value)} class={SELECT_CLASS}>
      {#if role === 'studioChat'}
        <option value="">Not set</option>
      {/if}
      {#each cloudSelectOptions as opt (opt.id)}
        <option value={opt.id}>{opt.label}</option>
      {/each}
    </select>
  {/if}
{/snippet}

<!-- ── Per-role validation status cue ─────────────────────────────────────────
     Compact inline cue: icon + short text, sits close to the field (gap-1.5).
     Kept lightweight so two cues inside the Models card don't dominate.
     role="alert" on invalid is preserved — tests assert it. -->
{#snippet validationStatus(
  status: 'idle' | 'checking' | 'valid' | 'invalid',
  message: string | null
)}
  {#if status === 'checking'}
    <p class="text-muted-foreground mt-1 flex items-center gap-1 text-[0.72rem]" aria-live="polite">
      <LoaderCircle class="size-3 animate-spin" />
      Checking…
    </p>
  {:else if status === 'valid'}
    <p class="text-primary mt-1 flex items-center gap-1 text-[0.72rem]">
      <Check class="size-3" aria-hidden="true" />
      Available
    </p>
  {:else if status === 'invalid' && message}
    <p class="text-destructive mt-1 text-[0.72rem]" role="alert">{message}</p>
  {/if}
{/snippet}

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
    class={cn('mt-3 flex flex-col gap-5', activeTab !== 'local' && 'hidden')}
  >
    <!-- Helper text -->
    <p class="text-muted-foreground text-[0.78rem] leading-relaxed">
      Works with Ollama, LM Studio, vLLM, Jan, llama.cpp — any OpenAI-compatible local server.
    </p>

    <!-- ── Group 1: Connection ──────────────────────────────────────────── -->
    <div class="flex flex-col gap-1.5">
      <p class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase">
        Connection
      </p>
      <div class="flex gap-2">
        <Input
          id="llm-endpoint"
          type="url"
          bind:value={localEndpoint}
          placeholder="http://localhost:11434"
          class="flex-1"
          autocomplete="off"
          spellcheck={false}
          aria-label="API Endpoint"
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

      {#if detectStatus === 'found'}
        <p class="text-primary text-[0.75rem]">Configure your preferred LLM</p>
      {:else if detectStatus === 'not-found'}
        <p class="text-muted-foreground text-[0.75rem]">
          {detectError ?? 'Not detected — check that your local server is running.'}
        </p>
      {/if}
    </div>

    <!-- ── Group 2: Models ─────────────────────────────────────────────── -->
    <div class="border-border flex flex-col gap-3 border-t pt-4">
      <p class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase">
        Models
      </p>
      <div class="border-border bg-muted/40 flex flex-col gap-4 rounded-lg border p-3">
        <!-- ENRICHMENT MODEL (local) — BLOCKING -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-baseline gap-2">
            <label for="llm-model-local" class="text-foreground text-[0.8rem] font-medium">
              Enrichment model
            </label>
            <span class="text-primary text-[0.68rem] font-medium">Required</span>
          </div>
          {@render localModelSelector(
            'llm-model-local',
            'enrichment',
            enrichmentLocalModel,
            (v) => (enrichmentLocalModel = v),
            false
          )}
          <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
            Used to enrich sources (coreference + structural mapping).
          </p>
          {@render validationStatus(enrichmentValidation, enrichmentValidationMessage)}
        </div>

        <!-- divider -->
        <div class="border-border border-t"></div>

        <!-- STUDIO & CHAT MODEL (local) — NON-blocking -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-baseline gap-2">
            <label for="studio-chat-model-local" class="text-foreground text-[0.8rem] font-medium">
              Studio &amp; Chat model
            </label>
            <span class="text-muted-foreground text-[0.68rem]">Optional</span>
          </div>
          {@render localModelSelector(
            'studio-chat-model-local',
            'studioChat',
            studioChatLocalModel,
            (v) => (studioChatLocalModel = v),
            true
          )}
          <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
            Used for chat and Studio generation. Configured now; used when chat/Studio ships.
          </p>
          {@render validationStatus(studioChatValidation, studioChatValidationMessage)}

          <!-- ── Context window — sub-setting of Studio & Chat model ──────────
               Context size is a chat/generation concern. Persisted via
               saveLlmProvider({ context: contextWindow }) as before.
               TODO(M5): attach contextWindow directly to the studio/chat
               TaskModel once the backend supports per-model context overrides. -->
          <div class="border-border mt-1 flex flex-col gap-1.5 border-t pt-3">
            <div class="flex items-baseline gap-1.5">
              <span class="text-muted-foreground text-[0.8rem] font-medium">Context window</span>
              <span class="text-muted-foreground text-[0.68rem]">— affects source bloat</span>
            </div>
            <div
              id="llm-context-window"
              class="flex gap-1"
              role="group"
              aria-label="Context window size"
            >
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
            <div class="mt-1 flex items-center gap-2">
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
        </div>
      </div>
    </div>

    <!-- Test connection status -->
    {#if testStatus === 'success' && testMessage}
      <p class="text-primary text-[0.75rem]">{testMessage}</p>
    {:else if testStatus === 'fail' && testMessage}
      <p class="text-destructive text-[0.75rem]" role="alert">{testMessage}</p>
    {/if}

    <!-- Test connection button (disabled while enrichment model empty) -->
    <Button class="h-10 w-full" onclick={handleTestConnection} disabled={testConnectionDisabled}>
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
    class={cn('mt-3 flex flex-col gap-5', activeTab !== 'cloud' && 'hidden')}
  >
    <!-- ── Group 1: Connection ──────────────────────────────────────────── -->
    <div class="flex flex-col gap-3">
      <p class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase">
        Connection
      </p>

      <!-- Cloud provider — searchable combobox (type-to-filter, grouped). -->
      <div class="flex flex-col gap-1.5">
        <Combobox.Root
          type="single"
          value={cloudProvider}
          onValueChange={(v) => {
            if (v) void selectProvider(v);
          }}
          onOpenChange={(open) => {
            if (!open) providerQuery = '';
          }}
        >
          <div class="relative">
            <Combobox.Input
              id="llm-cloud-provider"
              aria-label="Cloud provider"
              defaultValue={selectedProvider.name}
              oninput={(e) => (providerQuery = e.currentTarget.value)}
              placeholder="Search providers…"
              class="border-input bg-transparent dark:bg-input/30 focus-visible:border-ring focus-visible:ring-ring/50 text-foreground placeholder:text-muted-foreground h-9 w-full min-w-0 rounded-lg border px-2.5 py-1 pr-8 text-sm outline-none transition-colors focus-visible:ring-3"
            />
            <Combobox.Trigger
              aria-label="Show providers"
              class="text-muted-foreground absolute inset-y-0 right-0 flex items-center pr-2.5 outline-none"
            >
              <ChevronsUpDown class="size-4" aria-hidden="true" />
            </Combobox.Trigger>
          </div>
          <Combobox.Portal>
            <Combobox.Content
              class="border-border bg-popover text-popover-foreground z-[70] max-h-64 w-[var(--bits-combobox-anchor-width)] overflow-y-auto rounded-lg border p-1 shadow-lg"
              sideOffset={4}
            >
              {#each filteredProviders as group (group.key)}
                <Combobox.Group>
                  <Combobox.GroupHeading
                    class="text-muted-foreground px-2 pt-1.5 pb-1 text-[0.62rem] font-semibold tracking-widest uppercase"
                  >
                    {group.label}
                  </Combobox.GroupHeading>
                  {#each group.items as provider (provider.id)}
                    <Combobox.Item
                      value={provider.id}
                      label={provider.name}
                      class="data-highlighted:bg-primary/10 data-highlighted:text-foreground text-foreground flex cursor-pointer items-center justify-between rounded-md px-2 py-1.5 text-sm outline-none"
                    >
                      {#snippet children({ selected })}
                        <span>{provider.name}</span>
                        {#if selected}
                          <Check class="text-primary size-4" aria-hidden="true" />
                        {/if}
                      {/snippet}
                    </Combobox.Item>
                  {/each}
                </Combobox.Group>
              {/each}
              {#if filteredProviders.length === 0}
                <div class="text-muted-foreground px-2 py-3 text-center text-[0.78rem]">
                  No providers found
                </div>
              {/if}
            </Combobox.Content>
          </Combobox.Portal>
        </Combobox.Root>
      </div>

      <!-- API KEY -->
      <div class="flex flex-col gap-1.5">
        <label for="llm-cloud-key" class="text-muted-foreground text-[0.68rem] font-medium">
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

      <!-- BASE URL — only for the custom OpenAI-compatible endpoint. -->
      {#if isCustomProvider}
        <div class="flex flex-col gap-1.5">
          <label for="llm-cloud-url" class="text-muted-foreground text-[0.68rem] font-medium">
            Base URL
          </label>
          <Input
            id="llm-cloud-url"
            type="url"
            bind:value={cloudBaseUrl}
            placeholder="https://api.openai.com/v1"
            autocomplete="off"
            spellcheck={false}
          />
          <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
            The OpenAI-compatible endpoint to call — for LM Studio, vLLM, a proxy, or any other
            self-hosted/compatible server.
          </p>
        </div>
      {/if}
    </div>

    <!-- ── Group 2: Models ─────────────────────────────────────────────── -->
    <div class="border-border flex flex-col gap-3 border-t pt-4">
      <div class="flex items-baseline gap-2">
        <p class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase">
          Models
        </p>
        {#if catalogUpdating}
          <span class="text-muted-foreground text-[0.68rem]" aria-live="polite">updating…</span>
        {/if}
      </div>
      <div class="border-border bg-muted/40 flex flex-col gap-4 rounded-lg border p-3">
        <!-- ENRICHMENT MODEL (cloud) — BLOCKING -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-baseline gap-2">
            <label for="llm-cloud-model" class="text-foreground text-[0.8rem] font-medium">
              Enrichment model
            </label>
            <span class="text-primary text-[0.68rem] font-medium">Required</span>
          </div>
          {@render cloudModelSelector(
            'llm-cloud-model',
            'enrichment',
            enrichmentCloudModel,
            (v) => {
              enrichmentCloudModel = v;
              cloudModelPicked = true;
            }
          )}
          <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
            Used to enrich sources (coreference + structural mapping).
          </p>
          {#if modelHint}
            <p class="text-muted-foreground text-[0.72rem] leading-relaxed">{modelHint}</p>
          {/if}
          {@render validationStatus(enrichmentValidation, enrichmentValidationMessage)}
        </div>

        <!-- divider -->
        <div class="border-border border-t"></div>

        <!-- STUDIO & CHAT MODEL (cloud) — NON-blocking -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-baseline gap-2">
            <label for="studio-chat-model-cloud" class="text-foreground text-[0.8rem] font-medium">
              Studio &amp; Chat model
            </label>
            <span class="text-muted-foreground text-[0.68rem]">Optional</span>
          </div>
          {@render cloudModelSelector(
            'studio-chat-model-cloud',
            'studioChat',
            studioChatCloudModel,
            (v) => (studioChatCloudModel = v)
          )}
          <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
            Used for chat and Studio generation. Configured now; used when chat/Studio ships.
          </p>
          {@render validationStatus(studioChatValidation, studioChatValidationMessage)}
        </div>
      </div>
    </div>

    <!-- Save error -->
    {#if saveError}
      <p class="text-destructive text-[0.75rem]" role="alert">{saveError}</p>
    {/if}

    <!-- Save button (disabled while enrichment model or key empty) -->
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

    <!-- Opt-out tradeoff text -->
    {#if !enrichmentEnabled}
      <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
        Sources will still be searchable via embeddings, but without enrichment quality boosts
        (coreference resolution, structural mapping).
      </p>
    {/if}

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

    <!-- ROUTING select -->
    <div class={cn('flex flex-col gap-1.5', !enrichmentEnabled && 'opacity-50')}>
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
        disabled={!enrichmentEnabled}
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

    <!-- COREF-MODEL OVERRIDE -->
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

    <!-- CLOUD CONSENT — only when a cloud provider is selected. -->
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
