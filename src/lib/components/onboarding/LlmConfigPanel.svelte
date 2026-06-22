<script lang="ts">
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import { cn } from '$lib/utils.js';
  import { detectLlm } from '$lib/onboarding/system-check.js';
  import { saveLlmProvider, type LlmProviderTab } from '$lib/onboarding/llm-config.js';
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

  // --- Save state (cloud) ---
  let saving = $state(false);
  let saveError = $state<string | null>(null);

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

  const CLOUD_PROVIDERS = [
    {
      id: 'openai' as const,
      name: 'OpenAI',
      models: 'GPT-4o, GPT-4',
      defaultModel: 'gpt-4o',
      baseUrl: 'https://api.openai.com/v1'
    },
    {
      id: 'anthropic' as const,
      name: 'Anthropic',
      models: 'Claude 3.5',
      defaultModel: 'claude-3-5-sonnet-latest',
      baseUrl: 'https://api.anthropic.com/v1'
    },
    {
      id: 'google' as const,
      name: 'Google',
      models: 'Gemini 1.5',
      defaultModel: 'gemini-1.5-pro',
      baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai'
    }
  ] as const;

  const contextHelper = $derived(
    CONTEXT_OPTIONS.find((o) => o.value === contextWindow)?.helper ?? ''
  );

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
        model: provider.defaultModel,
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
  }

  const modelOptions = $derived(
    detectedModels.length > 0 ? detectedModels : localModel ? [localModel] : []
  );
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
        placeholder="Paste API key…"
        autocomplete="new-password"
      />
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
    <Button class="h-10 w-full" onclick={handleSave} disabled={saving}>
      {saving ? 'Saving…' : 'Save'}
    </Button>
  </div>
</div>
