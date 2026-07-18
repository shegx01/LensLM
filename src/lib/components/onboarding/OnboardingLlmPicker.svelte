<!--
  OnboardingLlmPicker — compact, always-visible, LOCAL-ONLY LLM picker for the
  onboarding system check; cloud/other providers live in Settings instead.
  Persistence follows the #31 Variant-B chat_model seam — see save() below.
-->
<script lang="ts">
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import Cpu from '@lucide/svelte/icons/cpu';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import type { AppConfig } from '$lib/theme/types.js';
  import type { SaveApi } from '$lib/onboarding/system-check.js';
  import { saveLlmProvider, saveEnrichmentPrefs } from '$lib/onboarding/llm-config.js';
  import { refreshChatProvider } from '$lib/models/chat-provider.svelte.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import LlmModelField from '$lib/components/llm/LlmModelField.svelte';
  import SystemCheckTile from './SystemCheckTile.svelte';

  let {
    onready
  }: {
    /** Hands the step footer a live { save } so it can drive Save & continue. */
    onready?: (api: SaveApi) => void;
  } = $props();

  const LOCAL_DEFAULT_ENDPOINT = 'http://localhost:11434';

  let endpoint = $state(LOCAL_DEFAULT_ENDPOINT);
  let model = $state('');
  let saveError = $state<string | null>(null);

  const canSave = $derived(model.trim() !== '');

  // Mirror AiModelSection.persistChat, local-only: upsert the ollama models[] entry
  // and pin enrichment.chat_model, preserving prior enabled/coref_strategy/cloud_consent.
  // OMIT routing/coref_model/map_model so prior enrichment policy stays untouched.
  async function save(): Promise<void> {
    if (!isTauri()) return;
    // Never pin an empty chat_model (matches AiModelSection's guard).
    if (model.trim() === '') return;
    saveError = null;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      const prior = cfg.enrichment;

      await saveLlmProvider({
        provider: 'ollama',
        base_url: endpoint,
        model,
        context: 8192,
        temperature: 0.7,
        api_key: ''
      });

      await saveEnrichmentPrefs({
        enabled: prior.enabled,
        coref_strategy: prior.coref_strategy,
        cloud_consent: prior.cloud_consent,
        chat_model: { provider: 'ollama', model }
      });

      await refreshChatProvider();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save the model.';
      throw err;
    }
  }

  // Keep the footer's api live; canSave stays local (drives only the status pill).
  $effect(() => {
    onready?.({ save });
  });
</script>

<SystemCheckTile icon={Cpu} title="Local AI" subtitle="Runs privately on your machine">
  {#snippet status()}
    <span
      class={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.7rem] font-medium ${
        canSave ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
      }`}
    >
      <span class="size-1.5 rounded-full bg-current" aria-hidden="true"></span>
      {canSave ? 'Ready' : 'Action needed'}
    </span>
  {/snippet}

  <div class="mt-3 flex flex-col gap-3">
    <div
      class="flex flex-col gap-3 rounded-[10px] p-3 ring-1 ring-inset ring-[color-mix(in_oklch,var(--primary)_20%,transparent)] bg-[color-mix(in_oklch,var(--primary)_7%,transparent)] dark:bg-[color-mix(in_oklch,var(--primary)_13%,transparent)]"
    >
      <div class="flex flex-col gap-1.5">
        <label
          for="onboarding-llm-endpoint"
          class="text-muted-foreground text-[0.68rem] font-medium"
        >
          Endpoint
        </label>
        <Input
          id="onboarding-llm-endpoint"
          type="url"
          bind:value={endpoint}
          placeholder={LOCAL_DEFAULT_ENDPOINT}
          class="rounded-[8px] bg-background font-mono"
          autocomplete="off"
          spellcheck={false}
        />
      </div>

      <LlmModelField
        id="onboarding-llm-model"
        kind="local"
        providerId="ollama"
        baseUrl={endpoint}
        bind:value={model}
      />
    </div>

    <div class="border-border border-t pt-3">
      <p class="text-muted-foreground text-[0.75rem] leading-relaxed">
        <Settings2 class="mr-1 inline size-3.5 align-[-2px]" aria-hidden="true" />
        Prefer OpenAI, Anthropic, or another provider? Set it up in
        <span class="text-foreground font-semibold">Settings › AI Model</span>.
      </p>
    </div>

    {#if saveError}
      <p class="text-destructive text-[0.75rem]" role="alert">{saveError}</p>
    {/if}
  </div>
</SystemCheckTile>
