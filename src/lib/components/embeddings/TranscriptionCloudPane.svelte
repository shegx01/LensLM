<!--
  TranscriptionCloudPane — Cloud ASR detail pane. No props: every field it reads/writes
  lives on the shared `appConfigStore` snapshot (`asr` + `audioCloudConsent`), so
  PrivacySection's consent toggle and this pane can never disagree.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Input } from '$lib/components/ui/input/index.js';
  import ApiKeyField from '$lib/components/llm/ApiKeyField.svelte';
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import { CLOUD_ASR_PRESETS } from '$lib/asr/catalog.js';
  import type { CloudAsrProvider } from '$lib/theme/types.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import { toLensError } from '$lib/sources/lens-error.js';

  const PROVIDER_LABELS: Record<CloudAsrProvider, string> = {
    open_ai_compatible: 'OpenAI-compatible',
    deepgram: 'Deepgram'
  };
  const PROVIDER_IDS = Object.keys(CLOUD_ASR_PRESETS) as CloudAsrProvider[];
  const DEFAULT_PROVIDER: CloudAsrProvider = 'open_ai_compatible';

  let provider = $state<CloudAsrProvider>(DEFAULT_PROVIDER);
  let baseUrl = $state('');
  let model = $state('');
  let apiKey = $state('');
  // The real, currently-persisted key. Never bound to an input — only resent on a
  // save that doesn't touch the key, so masking never writes a blank over a real
  // key (mirrors the #194 Cloud-key-wipe regression fix; see CloudTtsForm).
  let savedApiKey = $state('');
  let hasSavedKey = $state(false);
  let editingKey = $state(false);
  let error = $state<string | null>(null);

  let hydrated = false;

  onMount(() => {
    void ensureLoaded();
  });

  $effect(() => {
    const asr = appConfigStore.asr;
    if (hydrated || !asr) return;
    hydrated = true;
    provider = asr.cloud_provider ?? DEFAULT_PROVIDER;
    baseUrl = asr.cloud_base_url.trim() || CLOUD_ASR_PRESETS[provider].base_url;
    model = asr.cloud_model.trim() || CLOUD_ASR_PRESETS[provider].model;
    hasSavedKey = asr.cloud_api_key.trim() !== '';
    savedApiKey = asr.cloud_api_key;
  });

  /** The API key is bearer-transmitted to this URL (`openai_compat.rs:59`), so plain
   *  http: is only safe for a loopback host — anything else must be https:. */
  function isLoopbackHost(hostname: string): boolean {
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]';
  }

  function isValidBaseUrl(raw: string): boolean {
    let parsed: URL;
    try {
      parsed = new URL(raw);
    } catch {
      return false;
    }
    if (parsed.protocol === 'https:') return true;
    return parsed.protocol === 'http:' && isLoopbackHost(parsed.hostname);
  }

  /** Reactive Cloud persist; no Save button. Only flips `backend` to `"cloud"` when the
   *  config is usable; a blank required field demotes an active `"cloud"` backend so the
   *  engine's blank-filling default can never keep a cleared endpoint live. */
  async function persistCloud(): Promise<void> {
    error = null;
    const trimmedBaseUrl = baseUrl.trim();
    if (trimmedBaseUrl !== '' && !isValidBaseUrl(trimmedBaseUrl)) {
      error =
        'Enter an https:// URL. Plain http:// is only accepted for localhost, 127.0.0.1, or [::1].';
      return;
    }
    const trimmedModel = model.trim();
    const keyToSave = editingKey && apiKey.trim() ? apiKey : hasSavedKey ? savedApiKey : apiKey;
    try {
      await persist((cfg) => {
        const usable =
          trimmedBaseUrl !== '' &&
          trimmedModel !== '' &&
          keyToSave.trim() !== '' &&
          cfg.audio_cloud_consent;
        let backend = cfg.asr.backend;
        if (usable) backend = 'cloud';
        else if (backend === 'cloud') backend = '';
        return {
          ...cfg,
          asr: {
            ...cfg.asr,
            cloud_provider: provider,
            cloud_base_url: trimmedBaseUrl,
            cloud_model: trimmedModel,
            cloud_api_key: keyToSave,
            backend
          }
        };
      });
      savedApiKey = keyToSave;
      hasSavedKey = keyToSave.trim() !== '';
      editingKey = false;
      apiKey = '';
      // Verbatim from the store, no preset fallback (unlike the mount-time hydration
      // above) — the engine may have rewritten what was just sent.
      if (appConfigStore.asr) {
        baseUrl = appConfigStore.asr.cloud_base_url;
        model = appConfigStore.asr.cloud_model;
      }
    } catch (err) {
      error = toLensError(err).message;
    }
  }

  /** A provider switch must never carry the previous provider's key to the new vendor —
   *  clear all key state so the new provider starts unconfigured (and thus inactive)
   *  until the user enters its own key. */
  function handleProviderChange(value: string): void {
    if (value === provider) return;
    provider = value as CloudAsrProvider;
    const preset = CLOUD_ASR_PRESETS[provider];
    baseUrl = preset.base_url;
    model = preset.model;
    apiKey = '';
    savedApiKey = '';
    hasSavedKey = false;
    editingKey = false;
    void persistCloud();
  }

  /** Mirrors CloudTtsForm's key commit: an emptied "replace" field re-masks instead of
   *  persisting, so blurring away from a cleared field never wipes the saved key. */
  function handleKeyCommit(): void {
    if (editingKey && !apiKey.trim()) {
      editingKey = false;
      return;
    }
    if (editingKey || (!hasSavedKey && apiKey.trim())) {
      void persistCloud();
    }
  }
</script>

<div role="group" aria-label="Cloud speech-to-text setup" class="flex flex-col gap-4">
  {#if !appConfigStore.audioCloudConsent}
    <p
      role="status"
      class="flex items-center gap-2 rounded-[10px] bg-destructive/10 px-3.5 py-3 text-[0.72rem] text-destructive ring-1 ring-destructive/30"
    >
      <CircleAlert class="size-3.5 shrink-0" aria-hidden="true" />
      Cloud transcription needs audio consent. Turn on "Allow cloud audio" in Privacy settings to enable
      this provider.
    </p>
  {/if}

  <div class="flex flex-col gap-1.5">
    <label for="asr-cloud-provider" class="text-[0.72rem] font-bold text-foreground">
      Provider
    </label>
    <Select
      type="single"
      value={provider}
      onValueChange={(v) => {
        if (v) handleProviderChange(v);
      }}
      items={PROVIDER_IDS.map((id) => ({ value: id, label: PROVIDER_LABELS[id] }))}
    >
      <SelectTrigger id="asr-cloud-provider" class="w-full">
        <SelectValue placeholder="Select a provider" />
      </SelectTrigger>
      <SelectContent
        class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
      >
        {#each PROVIDER_IDS as id (id)}
          <SelectItem value={id} label={PROVIDER_LABELS[id]}>{PROVIDER_LABELS[id]}</SelectItem>
        {/each}
      </SelectContent>
    </Select>
  </div>

  <div class="flex flex-col gap-1.5">
    <label for="asr-cloud-base-url" class="text-[0.72rem] font-bold text-foreground">
      Base URL
    </label>
    <Input
      id="asr-cloud-base-url"
      type="text"
      bind:value={baseUrl}
      placeholder={CLOUD_ASR_PRESETS[provider].base_url}
      autocomplete="off"
      onblur={() => void persistCloud()}
    />
  </div>

  <div class="flex flex-col gap-1.5">
    <label for="asr-cloud-model" class="text-[0.72rem] font-bold text-foreground">Model</label>
    <Input
      id="asr-cloud-model"
      type="text"
      bind:value={model}
      placeholder={CLOUD_ASR_PRESETS[provider].model}
      autocomplete="off"
      onblur={() => void persistCloud()}
    />
  </div>

  <ApiKeyField
    id="asr-cloud-key"
    bind:value={apiKey}
    bind:editing={editingKey}
    {hasSavedKey}
    oncommit={handleKeyCommit}
  />

  {#if error}
    <p class="text-[0.72rem] text-destructive" role="alert">{error}</p>
  {/if}
</div>
