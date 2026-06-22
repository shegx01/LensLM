<script lang="ts">
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Download from '@lucide/svelte/icons/download';
  import {
    downloadTtsEngine,
    listTtsVoices,
    saveTtsProvider,
    type TtsVoice
  } from '$lib/onboarding/system-check.js';
  import { SELECT_CLASS } from './styles.js';
  import { updateConfig } from '$lib/config.js';

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  // --- Segmented tab state (mirrors LlmConfigPanel's Local | Cloud control) ---
  type TtsTab = 'local' | 'cloud';
  let activeTab = $state<TtsTab>('local');

  // --- Local tab: Kokoro download + voice selection ---
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

  // --- Cloud tab: ElevenLabs API key ---
  let cloudApiKey = $state('');
  let savingCloud = $state(false);
  let cloudError = $state<string | null>(null);

  async function handleDownload(): Promise<void> {
    downloadError = null;
    downloadProgress = 0;
    try {
      await downloadTtsEngine((pct) => {
        downloadProgress = pct;
      });
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

  // Cloud save: persist the ElevenLabs provider config, re-run the system check,
  // then collapse — same shape as LlmConfigPanel's cloud Save.
  async function handleSaveCloud(): Promise<void> {
    savingCloud = true;
    cloudError = null;
    try {
      await saveTtsProvider({ provider: 'elevenlabs', apiKey: cloudApiKey });
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
  <!-- Segmented tabs: Local | Cloud (mirrors LlmConfigPanel) -->
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

  <!-- Local tab panel: Kokoro download + voice selectors -->
  <div
    id="tts-panel-local"
    role="tabpanel"
    aria-labelledby="tts-tab-local"
    tabindex={activeTab === 'local' ? 0 : -1}
    class={cn('mt-3 flex flex-col gap-3', activeTab !== 'local' && 'hidden')}
  >
    {#if !downloaded}
      <!-- Pre-download: show download button -->
      <div class="flex flex-col gap-2">
        <p class="text-muted-foreground text-[0.78rem] leading-relaxed">
          Kokoro is an open-weight TTS engine that runs entirely on-device. Download once — no
          internet required for synthesis.
        </p>
        <div class="flex items-center justify-between text-[0.75rem] text-muted-foreground">
          <span>Kokoro-82M</span>
          <span>~86 MB · CPU · Fast</span>
        </div>

        {#if downloadProgress !== null && downloadProgress < 100}
          <!-- Progress bar -->
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
            Download Kokoro
          {/if}
        </Button>
      </div>
    {:else}
      <!-- Post-download: voice selectors -->
      <div class="flex items-center gap-2 text-[0.78rem] text-primary">
        <CircleCheck class="size-4" />
        Kokoro engine ready
      </div>

      {#if voicesUnavailable}
        <p class="text-destructive text-[0.75rem]" role="alert">
          Couldn't load voices — is the engine installed?
        </p>
      {:else}
        <!-- Male voice selector -->
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

        <!-- Female voice selector -->
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

  <!-- Cloud tab panel: ElevenLabs API key -->
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

    <!-- API KEY -->
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
        placeholder="Paste API key…"
        autocomplete="new-password"
      />
    </div>

    {#if cloudError}
      <p class="text-destructive text-[0.75rem]" role="alert">{cloudError}</p>
    {/if}

    <Button
      class="h-10 w-full"
      onclick={handleSaveCloud}
      disabled={!cloudApiKey.trim() || savingCloud}
    >
      {savingCloud ? 'Saving…' : 'Save'}
    </Button>
  </div>
</div>
