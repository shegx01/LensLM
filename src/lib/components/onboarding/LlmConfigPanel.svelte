<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import { cn } from '$lib/utils.js';
  import { detectLlm } from '$lib/onboarding/system-check.js';
  import { saveLlmProvider, type LlmProviderTab } from '$lib/onboarding/llm-config.js';
  import type { AppConfig } from '$lib/theme/types.js';
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
  // User-editable model id (seeded from the provider default, like the Local
  // tab's model field). 'gpt-4o' matches the initial 'openai' provider.
  let cloudModel = $state('gpt-4o');

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
        provider: 'openai-compatible',
        base_url: cloudBaseUrl || provider.baseUrl,
        // User's chosen model id; fall back to the provider default if cleared.
        model: cloudModel.trim() || provider.defaultModel,
        api_key: cloudApiKey,
        context: CLOUD_DEFAULT_CONTEXT
      });
      await oncheck();
      oncollapse();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save configuration.';
    } finally {
      saving = false;
    }
  }

  function selectProvider(id: typeof cloudProvider): void {
    cloudProvider = id;
    const p = CLOUD_PROVIDERS.find((p) => p.id === id)!;
    cloudBaseUrl = p.baseUrl;
    cloudModel = p.defaultModel;
    // Switching cards starts clean: clear any typed/edited key. `hasSavedKey`
    // recomputes from the derived (true only when `id` is the saved provider).
    editingKey = false;
    cloudApiKey = '';
  }

  const modelOptions = $derived(
    detectedModels.length > 0 ? detectedModels : localModel ? [localModel] : []
  );

  // Behaviorally identical to TtsConfigPanel's gate. For the SAVED provider we
  // require re-entry (a fresh non-empty key) since we never load the real key
  // into the DOM; for any UNSAVED provider the button enables as soon as a
  // non-empty key is typed and is disabled only while the field is empty.
  const cloudSaveDisabled = $derived(
    saving || (hasSavedKey ? !editingKey || !cloudApiKey.trim() : !cloudApiKey.trim())
  );

  // On mount, pre-fill the Cloud tab from a previously-saved openai-compatible
  // config (provider/model) but keep the real api_key OUT of the DOM — we only
  // record that a key exists so the field renders masked and Save stays disabled
  // until the user re-enters one.
  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const saved = cfg.models?.find(
        (m) => m.provider === 'openai-compatible' && m.api_key.trim() !== ''
      );
      if (!saved) return;
      // Map the saved entry to a provider card by base_url. Only a recognized
      // provider gets the per-provider saved/masked treatment; an unmatched
      // base_url leaves every card in the normal (unsaved) entry state.
      const match = CLOUD_PROVIDERS.find((p) => p.baseUrl === saved.base_url);
      if (!match) return;
      savedProviderId = match.id;
      cloudProvider = match.id;
      cloudBaseUrl = saved.base_url || match.baseUrl;
      cloudModel = saved.model || match.defaultModel;
      cloudApiKey = '';
    } catch {
      // Non-fatal: fall back to the default empty Cloud form.
    }
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
          for="llm-context-window"
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
            if (Number.isFinite(v) && v > 0) contextWindow = v;
          }}
          aria-label="Custom context window in tokens"
          class="border-input focus-visible:border-ring focus-visible:ring-ring/50 h-8 w-full rounded-lg border bg-transparent px-2.5 py-1 text-sm transition-colors outline-none focus-visible:ring-3"
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
      <label
        for="llm-cloud-model"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        Model
      </label>
      <Input
        id="llm-cloud-model"
        type="text"
        bind:value={cloudModel}
        placeholder={selectedProvider.defaultModel}
        autocomplete="off"
        spellcheck={false}
      />
      <p class="text-muted-foreground text-[0.72rem] leading-relaxed">
        The exact model id to use (e.g. {selectedProvider.defaultModel}). Enter any model your
        provider supports — no need to wait for an app update when a new one ships.
      </p>
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
</div>
