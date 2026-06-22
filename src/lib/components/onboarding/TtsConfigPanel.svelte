<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Download from '@lucide/svelte/icons/download';
  import { downloadTtsEngine, listTtsVoices, type TtsVoice } from '$lib/onboarding/system-check.js';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import type { AppConfig } from '$lib/theme/types.js';

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  // Download state
  let downloadProgress = $state<number | null>(null);
  let downloaded = $state(false);
  let downloadError = $state<string | null>(null);

  // Voice selection (post-download)
  let voices = $state<TtsVoice[]>([]);
  let maleVoice = $state('');
  let femaleVoice = $state('');
  let savingVoices = $state(false);
  let saveError = $state<string | null>(null);

  const maleVoices = $derived(voices.filter((v) => v.gender === 'male'));
  const femaleVoices = $derived(voices.filter((v) => v.gender === 'female'));

  async function handleDownload(): Promise<void> {
    downloadError = null;
    downloadProgress = 0;
    try {
      await downloadTtsEngine((pct) => {
        downloadProgress = pct;
      });
      downloadProgress = 100;
      downloaded = true;
      // Load available voices
      voices = await listTtsVoices();
      if (voices.length === 0) {
        // Stub voices for non-Tauri dev
        voices = [
          { id: 'kokoro_male_1', name: 'Oliver', gender: 'male' },
          { id: 'kokoro_male_2', name: 'Ethan', gender: 'male' },
          { id: 'kokoro_female_1', name: 'Emma', gender: 'female' },
          { id: 'kokoro_female_2', name: 'Aria', gender: 'female' }
        ];
      }
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
      if (isTauri()) {
        const cfg = await invoke<AppConfig>('get_config');
        await invoke<void>('set_config', {
          config: { ...cfg, voices: { host: maleVoice, guest: femaleVoice } }
        });
      }
      await oncheck();
      oncollapse();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save voice settings.';
    } finally {
      savingVoices = false;
    }
  }
</script>

<div class="border-border mt-3 border-t pt-3 flex flex-col gap-3">
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

    <!-- Male voice selector -->
    <div class="flex flex-col gap-1.5">
      <label
        for="tts-male-voice"
        class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        Host voice (male)
      </label>
      <select
        id="tts-male-voice"
        bind:value={maleVoice}
        class="border-input bg-transparent dark:bg-input/30 focus-visible:border-ring focus-visible:ring-ring/50 h-8 w-full min-w-0 rounded-lg border px-2.5 py-1 text-sm outline-none transition-colors focus-visible:ring-3 text-foreground"
      >
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
      <select
        id="tts-female-voice"
        bind:value={femaleVoice}
        class="border-input bg-transparent dark:bg-input/30 focus-visible:border-ring focus-visible:ring-ring/50 h-8 w-full min-w-0 rounded-lg border px-2.5 py-1 text-sm outline-none transition-colors focus-visible:ring-3 text-foreground"
      >
        {#each femaleVoices as voice (voice.id)}
          <option value={voice.id}>{voice.name}</option>
        {/each}
      </select>
    </div>

    {#if saveError}
      <p class="text-destructive text-[0.75rem]" role="alert">{saveError}</p>
    {/if}

    <Button class="h-10 w-full" onclick={handleSaveVoices} disabled={savingVoices}>
      {savingVoices ? 'Saving…' : 'Save voice settings'}
    </Button>
  {/if}
</div>
