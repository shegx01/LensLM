<!--
  LlmModelField — model picker shared by the Settings AI Model panel and (issue #217)
  onboarding's curated 3-preset picker. Local providers get Auto-detect + pulled
  models; cloud gets the models.dev catalog; custom endpoints fall back to free text.
-->
<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import Check from '@lucide/svelte/icons/check';
  import { SELECT_CLASS } from '$lib/components/onboarding/styles.js';
  import { detectLlm } from '$lib/onboarding/system-check.js';
  import { validateModelInteractive } from '$lib/onboarding/enrichment-validation.js';
  import {
    listCloudModelOptions,
    listOllamaModelOptions,
    type ModelOption
  } from '$lib/models/catalog.js';

  let {
    value = $bindable(''),
    kind,
    providerId,
    catalogKey = null,
    baseUrl = '',
    apiKey = '',
    id,
    onchange
  }: {
    /** Selected model id. */
    value?: string;
    /** Provider shape: local Ollama-style, native cloud, or custom OpenAI-compatible. */
    kind: 'local' | 'cloud' | 'custom';
    /** Canonical provider id sent to the backend (`'ollama'` or a cloud id). */
    providerId: string;
    /** models.dev catalog key for cloud lookups; `null` ⇒ free-text model. */
    catalogKey?: string | null;
    baseUrl?: string;
    apiKey?: string;
    id: string;
    /** Fires when the user picks or edits the model. */
    onchange?: (model: string) => void;
  } = $props();

  let cloudOptions = $state<ModelOption[]>([]);
  let ollamaOptions = $state<ModelOption[]>([]);
  let detected = $state<string[]>([]);
  let detecting = $state(false);
  let detectMessage = $state<string | null>(null);

  let validation = $state<'idle' | 'checking' | 'valid' | 'invalid'>('idle');
  let validationMessage = $state<string | null>(null);

  const localIds = $derived(
    kind === 'local' ? Array.from(new Set([...detected, ...ollamaOptions.map((o) => o.id)])) : []
  );

  const options = $derived<ModelOption[]>(
    kind === 'local' ? localIds.map((mid) => ({ id: mid, label: mid, info: null })) : cloudOptions
  );

  // Custom endpoints have no catalog; local/cloud fall back to free text when nothing resolves.
  const freeText = $derived(kind === 'custom' || options.length === 0);

  async function loadOptions(): Promise<void> {
    if (kind === 'local') {
      try {
        ollamaOptions = await listOllamaModelOptions(baseUrl);
      } catch {
        ollamaOptions = [];
      }
    } else if (kind === 'cloud' && catalogKey) {
      try {
        cloudOptions = await listCloudModelOptions(catalogKey);
      } catch {
        cloudOptions = [];
      }
    } else {
      cloudOptions = [];
    }
  }

  // Reload whenever the provider identity changes (guarded so it doesn't re-run on
  // unrelated state writes — it reads only the keys it depends on).
  $effect(() => {
    void kind;
    void catalogKey;
    void baseUrl;
    void loadOptions();
  });

  function emit(model: string): void {
    value = model;
    validation = 'idle';
    validationMessage = null;
    onchange?.(model);
  }

  async function autoDetect(): Promise<void> {
    detecting = true;
    detectMessage = null;
    try {
      const result = await detectLlm(baseUrl);
      if (result.reachable) {
        detected = result.models;
        detectMessage = result.version ? `Connected — ${result.version}` : 'Connected';
      } else {
        detected = [];
        detectMessage = 'Not detected — check that your local server is running.';
      }
    } catch (err) {
      detected = [];
      detectMessage = err instanceof Error ? err.message : 'Auto-detect failed.';
    } finally {
      detecting = false;
    }
  }

  async function validate(): Promise<void> {
    if (!value.trim()) return;
    validation = 'checking';
    validationMessage = null;
    try {
      const result = await validateModelInteractive(providerId, value, baseUrl, apiKey);
      validation = result.status === 'valid' ? 'valid' : 'invalid';
      validationMessage =
        result.status === 'valid' ? null : (result.reason ?? 'Model validation failed.');
    } catch (err) {
      validation = 'invalid';
      validationMessage = err instanceof Error ? err.message : 'Model validation failed.';
    }
  }
</script>

<div class="flex flex-col gap-1.5">
  <label for={id} class="text-[0.68rem] font-medium text-muted-foreground">Model</label>

  {#if kind === 'local'}
    <div class="flex gap-2">
      {#if freeText}
        <Input
          {id}
          type="text"
          {value}
          oninput={(e) => emit(e.currentTarget.value)}
          placeholder="e.g. llama3.2:3b"
          class="flex-1"
          autocomplete="off"
          spellcheck={false}
        />
      {:else}
        <select
          {id}
          {value}
          onchange={(e) => emit(e.currentTarget.value)}
          class={`${SELECT_CLASS} flex-1`}
        >
          {#each options as opt (opt.id)}
            <option value={opt.id}>{opt.label}</option>
          {/each}
        </select>
      {/if}
      <Button
        variant="outline"
        size="sm"
        onclick={autoDetect}
        disabled={detecting}
        aria-label="Auto-detect local models"
        class="shrink-0"
      >
        {detecting ? 'Detecting…' : 'Auto-detect'}
      </Button>
    </div>
    {#if detectMessage}
      <p class="text-[0.72rem] text-muted-foreground">{detectMessage}</p>
    {/if}
  {:else if freeText}
    <Input
      {id}
      type="text"
      {value}
      oninput={(e) => emit(e.currentTarget.value)}
      placeholder="model id"
      autocomplete="off"
      spellcheck={false}
    />
  {:else}
    <select {id} {value} onchange={(e) => emit(e.currentTarget.value)} class={SELECT_CLASS}>
      {#each options as opt (opt.id)}
        <option value={opt.id}>{opt.label}</option>
      {/each}
    </select>
  {/if}

  <div class="flex items-center gap-2">
    <Button
      variant="ghost"
      size="sm"
      onclick={validate}
      disabled={validation === 'checking' || !value.trim()}
      aria-label="Validate model"
    >
      {validation === 'checking' ? 'Checking…' : 'Validate'}
    </Button>
    {#if validation === 'checking'}
      <span class="flex items-center gap-1 text-[0.72rem] text-muted-foreground" aria-live="polite">
        <LoaderCircle class="size-3 animate-spin" />
        Checking…
      </span>
    {:else if validation === 'valid'}
      <span class="flex items-center gap-1 text-[0.72rem] text-primary">
        <Check class="size-3" aria-hidden="true" />
        Available
      </span>
    {:else if validation === 'invalid' && validationMessage}
      <span class="text-[0.72rem] text-destructive" role="alert">{validationMessage}</span>
    {/if}
  </div>
</div>
