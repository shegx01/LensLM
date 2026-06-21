<script lang="ts">
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import { detectLlm } from '$lib/onboarding/system-check.js';
  import { saveLlmProvider, type LlmProviderTab } from '$lib/onboarding/llm-config.js';

  // Panel receives a callback to re-run the parent system check after Save, and
  // a callback to collapse itself (the parent owns the open/closed state).
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
  // Detected models list (populated by Auto-detect)
  let detectedModels = $state<string[]>([]);
  let detectStatus = $state<'idle' | 'detecting' | 'found' | 'not-found'>('idle');
  let detectVersion = $state<string | null>(null);
  let detectError = $state<string | null>(null);

  // --- Cloud API tab fields ---
  let cloudBaseUrl = $state('https://api.openai.com/v1');
  let cloudApiKey = $state('');
  let cloudModel = $state('gpt-4o-mini');

  // --- Save state ---
  let saving = $state(false);
  let saveError = $state<string | null>(null);

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
        // Pick the first detected model if the current default isn't in the list.
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

  // Save: persist provider config then re-run the system check and collapse.
  async function handleSave(): Promise<void> {
    saving = true;
    saveError = null;
    try {
      if (activeTab === 'local') {
        await saveLlmProvider({
          provider: 'ollama',
          base_url: localEndpoint,
          model: localModel,
          api_key: ''
        });
      } else {
        await saveLlmProvider({
          provider: 'openai-compatible',
          base_url: cloudBaseUrl,
          model: cloudModel,
          api_key: cloudApiKey
        });
      }
      // Re-run system check so the LLM row status updates, THEN collapse.
      await oncheck();
      oncollapse();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save configuration.';
    } finally {
      saving = false;
    }
  }

  // Use the detected models list when available, else show the typed model as
  // the sole option. This keeps the select usable before auto-detect runs.
  const modelOptions = $derived(
    detectedModels.length > 0 ? detectedModels : localModel ? [localModel] : []
  );
</script>

<!--
  Inline expansion panel — rendered inside the LLM runtime row's card.
  Matches the 02-onb-dark.png / 03-onb-dark.png mock exactly:
    • Local | Cloud API segmented tabs
    • Helper text (Local tab)
    • API ENDPOINT label + input + Auto-detect button (inline)
    • MODEL label + select
    • Save button
-->
<div class="border-border mt-3 border-t pt-3">
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
          ? 'bg-card text-foreground shadow-sm'
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
          ? 'bg-card text-foreground shadow-sm'
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

      <!-- Detect feedback: version or not-found hint -->
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
        <!-- Native select when auto-detect returned multiple options -->
        <select
          id="llm-model-local"
          bind:value={localModel}
          class={cn(
            'border-input bg-transparent dark:bg-input/30 focus-visible:border-ring focus-visible:ring-ring/50 h-8 w-full min-w-0 rounded-lg border px-2.5 py-1 text-sm outline-none transition-colors focus-visible:ring-3',
            'text-foreground'
          )}
        >
          {#each modelOptions as m (m)}
            <option value={m}>{m}</option>
          {/each}
        </select>
      {:else}
        <!-- Plain input when no auto-detect run yet or single option -->
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
  </div>

  <!-- Cloud API tab panel -->
  <div
    id="llm-panel-cloud"
    role="tabpanel"
    aria-labelledby="llm-tab-cloud"
    class={cn('mt-3 flex flex-col gap-3', activeTab !== 'cloud' && 'hidden')}
  >
    <!-- BASE URL -->
    <div class="flex flex-col gap-1.5">
      <label
        for="llm-cloud-url"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
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
        placeholder="sk-…"
        autocomplete="new-password"
      />
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
        placeholder="gpt-4o-mini"
        autocomplete="off"
        spellcheck={false}
      />
    </div>
  </div>

  <!-- Save error -->
  {#if saveError}
    <p class="text-destructive mt-2 text-[0.75rem]" role="alert">{saveError}</p>
  {/if}

  <!-- Save button -->
  <div class="mt-3 flex justify-end">
    <Button size="sm" onclick={handleSave} disabled={saving}>
      {saving ? 'Saving…' : 'Save'}
    </Button>
  </div>
</div>
