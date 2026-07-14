<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Download from '@lucide/svelte/icons/download';
  import {
    downloadTtsModel,
    listTtsVoices,
    ttsModelDownloaded,
    saveTtsProvider,
    type TtsVoice
  } from '$lib/onboarding/system-check.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { SELECT_CLASS } from './styles.js';
  import { updateConfig } from '$lib/config.js';

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  // Orpheus is the current default local backend (#192); it needs both the
  // GGUF weights and the SNAC decoder. Full multi-engine selection is #194.
  // SYNC-CHECK: ids must match lens-core TTS_REGISTRY / TtsBackend::required_model_ids (orpheus, snac).
  const TTS_ENGINE = 'orpheus';
  const TTS_MODEL_IDS = ['orpheus', 'snac'] as const;

  type TtsTab = 'local' | 'cloud';
  let activeTab = $state<TtsTab>('local');

  let downloadProgress = $state<number | null>(null);
  let downloaded = $state(false);
  let downloadError = $state<string | null>(null);

  let voices = $state<TtsVoice[]>([]);
  let maleVoice = $state('');
  let femaleVoice = $state('');
  let savingVoices = $state(false);
  let saveError = $state<string | null>(null);
  // True once a download completed but listTtsVoices() returned nothing — we
  // surface an inline error and disable Save rather than persisting fake IDs.
  let voicesUnavailable = $state(false);

  const maleVoices = $derived(voices.filter((v) => v.gender === 'male'));
  const femaleVoices = $derived(voices.filter((v) => v.gender === 'female'));

  let cloudApiKey = $state('');
  let savingCloud = $state(false);
  let cloudError = $state<string | null>(null);

  // Saved ElevenLabs key is masked; Save re-enables only after the user enters a fresh key.
  let hasSavedKey = $state(false);
  let editingKey = $state(false);

  // Identical gate to LlmConfigPanel's cloud Save: disabled until a non-empty
  // key is typed; when a key is saved, masking re-requires fresh entry.
  const cloudSaveDisabled = $derived(
    savingCloud || (hasSavedKey ? !editingKey || !cloudApiKey.trim() : !cloudApiKey.trim())
  );

  /** True once every model artifact the local engine needs is on disk. */
  async function engineDownloaded(): Promise<boolean> {
    for (const model of TTS_MODEL_IDS) {
      if (!(await ttsModelDownloaded(TTS_ENGINE, model))) return false;
    }
    return true;
  }

  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      if (cfg.tts?.cloud && cfg.tts.cloud.api_key.trim() !== '') {
        hasSavedKey = true;
        cloudApiKey = '';
      }
      // If the local engine is already on disk, skip the download step and go
      // straight to voice selection — pre-filled from any previously saved
      // host/guest voices.
      if (await engineDownloaded()) {
        downloaded = true;
        voices = await listTtsVoices();
        voicesUnavailable = voices.length === 0;
        const host = cfg.voices?.host;
        const guest = cfg.voices?.guest;
        maleVoice = (typeof host === 'string' ? host : '') || maleVoices[0]?.id || '';
        femaleVoice = (typeof guest === 'string' ? guest : '') || femaleVoices[0]?.id || '';
      }
    } catch {
      // Non-fatal: fall back to the default empty Cloud form / download prompt.
    }
  });

  // Entering "editing" mode clears the masked field so the user types a fresh key.
  function startEditingKey(): void {
    if (hasSavedKey && !editingKey) {
      editingKey = true;
      cloudApiKey = '';
    }
  }

  async function handleDownload(): Promise<void> {
    downloadError = null;
    downloadProgress = 0;
    try {
      for (const [i, model] of TTS_MODEL_IDS.entries()) {
        await downloadTtsModel(TTS_ENGINE, model, (pct) => {
          downloadProgress = Math.round(((i + pct / 100) / TTS_MODEL_IDS.length) * 100);
        });
      }
      downloadProgress = 100;
      downloaded = true;
      // Load available voices from the engine. No stubs: if the catalog comes
      // back empty the engine isn't really available, so we flag it.
      voices = await listTtsVoices();
      voicesUnavailable = voices.length === 0;
      if (maleVoices.length > 0) maleVoice = maleVoices[0].id;
      if (femaleVoices.length > 0) femaleVoice = femaleVoices[0].id;
    } catch (err) {
      downloadError = err instanceof Error ? err.message : 'Download failed.';
      downloadProgress = null;
    }
  }

  async function handleSaveVoices(): Promise<void> {
    savingVoices = true;
    saveError = null;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        voices: { host: maleVoice, guest: femaleVoice }
      }));
      await oncheck();
      oncollapse();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save voice settings.';
    } finally {
      savingVoices = false;
    }
  }

  async function handleSaveCloud(): Promise<void> {
    savingCloud = true;
    cloudError = null;
    try {
      await saveTtsProvider({ provider: 'cloud', apiKey: cloudApiKey });
      await oncheck();
      oncollapse();
    } catch (err) {
      cloudError = err instanceof Error ? err.message : 'Could not save configuration.';
    } finally {
      savingCloud = false;
    }
  }
</script>

<div class="pt-3">
  <div
    class="bg-muted flex w-full items-center rounded-lg p-0.5"
    role="tablist"
    aria-label="Text-to-speech provider type"
  >
    <button
      role="tab"
      aria-selected={activeTab === 'local'}
      aria-controls="tts-panel-local"
      id="tts-tab-local"
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
      aria-controls="tts-panel-cloud"
      id="tts-tab-cloud"
      class={cn(
        'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        activeTab === 'cloud'
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
      onclick={() => (activeTab = 'cloud')}
    >
      Cloud
    </button>
  </div>

  <div
    id="tts-panel-local"
    role="tabpanel"
    aria-labelledby="tts-tab-local"
    tabindex={activeTab === 'local' ? 0 : -1}
    class={cn('mt-3 flex flex-col gap-3', activeTab !== 'local' && 'hidden')}
  >
    {#if !downloaded}
      <div class="flex flex-col gap-2">
        <p class="text-muted-foreground text-[0.78rem] leading-relaxed">
          This open-weight TTS engine runs entirely on-device. Download once — no internet required
          for synthesis.
        </p>
        <div class="flex items-center justify-between text-[0.75rem] text-muted-foreground">
          <span>Local voice engine</span>
          <span>On-device · Offline</span>
        </div>

        {#if downloadProgress !== null && downloadProgress < 100}
          <div class="w-full bg-muted rounded-full h-1.5 overflow-hidden">
            <div
              class="bg-primary h-full rounded-full transition-all duration-300"
              style:width="{downloadProgress}%"
            ></div>
          </div>
          <p class="text-[0.72rem] text-muted-foreground text-center">
            {downloadProgress}% downloaded
          </p>
        {/if}

        {#if downloadError}
          <p class="text-destructive text-[0.75rem]" role="alert">{downloadError}</p>
        {/if}

        <Button
          class="h-10 w-full"
          onclick={handleDownload}
          disabled={downloadProgress !== null && downloadProgress < 100}
        >
          {#if downloadProgress !== null && downloadProgress < 100}
            <LoaderCircle class="size-4 animate-spin" />
            Downloading…
          {:else}
            <Download class="size-4" />
            Download voice engine
          {/if}
        </Button>
      </div>
    {:else}
      <div class="flex items-center gap-2 text-[0.78rem] text-primary" role="status">
        <CircleCheck class="size-4" />
        Voice engine ready
      </div>

      {#if voicesUnavailable}
        <p class="text-destructive text-[0.75rem]" role="alert">
          Couldn't load voices — is the engine installed?
        </p>
      {:else}
        <div class="flex flex-col gap-1.5">
          <label
            for="tts-male-voice"
            class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
          >
            Host voice (male)
          </label>
          <select id="tts-male-voice" bind:value={maleVoice} class={SELECT_CLASS}>
            {#each maleVoices as voice (voice.id)}
              <option value={voice.id}>{voice.name}</option>
            {/each}
          </select>
        </div>

        <div class="flex flex-col gap-1.5">
          <label
            for="tts-female-voice"
            class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
          >
            Co-host voice (female)
          </label>
          <select id="tts-female-voice" bind:value={femaleVoice} class={SELECT_CLASS}>
            {#each femaleVoices as voice (voice.id)}
              <option value={voice.id}>{voice.name}</option>
            {/each}
          </select>
        </div>
      {/if}

      {#if saveError}
        <p class="text-destructive text-[0.75rem]" role="alert">{saveError}</p>
      {/if}

      <Button
        class="h-10 w-full"
        onclick={handleSaveVoices}
        disabled={savingVoices || voicesUnavailable}
      >
        {savingVoices ? 'Saving…' : 'Save voice settings'}
      </Button>
    {/if}
  </div>

  <div
    id="tts-panel-cloud"
    role="tabpanel"
    aria-labelledby="tts-tab-cloud"
    tabindex={activeTab === 'cloud' ? 0 : -1}
    class={cn('mt-3 flex flex-col gap-3', activeTab !== 'cloud' && 'hidden')}
  >
    <p class="text-muted-foreground text-[0.78rem] leading-relaxed">
      ElevenLabs synthesizes high-quality voices in the cloud. Paste your API key to enable it — no
      local download required.
    </p>

    <div class="flex flex-col gap-1.5">
      <label
        for="tts-cloud-key"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        API Key
      </label>
      <Input
        id="tts-cloud-key"
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

    {#if cloudError}
      <p class="text-destructive text-[0.75rem]" role="alert">{cloudError}</p>
    {/if}

    <Button class="h-10 w-full" onclick={handleSaveCloud} disabled={cloudSaveDisabled}>
      {savingCloud ? 'Saving…' : 'Save'}
    </Button>
  </div>
</div>
