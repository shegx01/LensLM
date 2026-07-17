<script lang="ts">
  import { onMount } from 'svelte';
  import { fade } from 'svelte/transition';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Download from '@lucide/svelte/icons/download';
  import {
    downloadTtsModel,
    prepareQwenModel,
    ttsModelDownloaded,
    saveTtsProvider,
    nextTtsConfig,
    ttsEngineCatalog,
    type TtsVoice,
    type TtsEngineCatalogEntry,
    type TtsEngineId,
    type TtsProvider
  } from '$lib/onboarding/system-check.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';
  import { updateConfig } from '$lib/config.js';

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  // The static capability catalog (#194) is the single source of truth for the
  // selector's engines/gating/size/language label — never `list_tts_voices`.
  let catalog = $state<TtsEngineCatalogEntry[]>([]);
  let selectedEngine = $state<TtsEngineId>('orpheus');

  const localEngines = $derived(catalog.filter((e) => e.id !== 'cloud'));
  const selectedEntry = $derived(catalog.find((e) => e.id === selectedEngine) ?? null);

  function engineLabel(id: TtsEngineId): string {
    if (id === 'orpheus') return 'Orpheus';
    if (id === 'qwen3_local') return 'Qwen3-TTS';
    return 'Cloud';
  }

  function engineToProvider(id: TtsEngineId): TtsProvider {
    if (id === 'qwen3_local') return 'qwen3';
    if (id === 'cloud') return 'cloud';
    return 'orpheus';
  }

  /** Human-readable on-disk size for the always-visible label, e.g. "~2.3 GB". */
  function formatSize(bytes: number | null): string | null {
    if (bytes === null || bytes <= 0) return null;
    return `~${(bytes / 1_000_000_000).toFixed(1)} GB`;
  }

  // Registry ids the selected engine needs on disk, from the catalog DTO (authority:
  // TtsBackend::required_model_ids). Empty for engines that fetch weights lazily (Qwen3Local).
  const modelIds = $derived<readonly string[]>(selectedEntry?.required_model_ids ?? []);

  type TtsTab = 'local' | 'cloud';
  let activeTab = $state<TtsTab>('local');

  let downloadProgress = $state<number | null>(null);
  let downloaded = $state(false);
  let downloadError = $state<string | null>(null);
  // True when a finished download failed its post-download presence re-check
  // (a truncated/partial artifact). Drives the "Model incomplete — re-download"
  // affordance so an incomplete local model always has a recovery path.
  let incomplete = $state(false);

  let voices = $state<TtsVoice[]>([]);
  let maleVoice = $state('');
  let femaleVoice = $state('');
  let saveError = $state<string | null>(null);
  // True only if the catalog carries no preset voices for the engine — surface
  // an inline error and disable Save rather than persisting fake IDs.
  let voicesUnavailable = $state(false);

  const maleVoices = $derived(voices.filter((v) => v.gender === 'male'));
  const femaleVoices = $derived(voices.filter((v) => v.gender === 'female'));

  const CUSTOM_VOICE = '__custom__';

  let cloudApiKey = $state('');
  // The real, currently-persisted key. Never bound to an input — only used to
  // resend the key on saves that touch other fields (base URL/voices) without
  // the user re-typing it, so masking never risks writing a blank key over a
  // real one (see the #194 Cloud-key-wipe regression this mirrors the fix for).
  let savedCloudApiKey = $state('');
  let cloudError = $state<string | null>(null);

  // Saved key is masked; Save re-enables only after the user enters a fresh key.
  let hasSavedKey = $state(false);
  let editingKey = $state(false);

  let cloudBaseUrl = $state('https://api.openai.com');
  let cloudHostPreset = $state('');
  let cloudGuestPreset = $state('');
  let cloudHostCustom = $state('');
  let cloudGuestCustom = $state('');

  const cloudEntry = $derived(catalog.find((e) => e.id === 'cloud') ?? null);
  const cloudVoices = $derived(cloudEntry?.preset_voices ?? []);
  const cloudMaleVoices = $derived(cloudVoices.filter((v) => v.gender === 'male'));
  const cloudFemaleVoices = $derived(cloudVoices.filter((v) => v.gender === 'female'));

  // The resolved host/guest voice id to persist: the curated pick, unless the
  // user chose the free-text escape hatch (or no curated voices exist yet).
  const cloudHostVoice = $derived(
    cloudMaleVoices.length > 0 && cloudHostPreset !== CUSTOM_VOICE
      ? cloudHostPreset
      : cloudHostCustom.trim()
  );
  const cloudGuestVoice = $derived(
    cloudFemaleVoices.length > 0 && cloudGuestPreset !== CUSTOM_VOICE
      ? cloudGuestPreset
      : cloudGuestCustom.trim()
  );

  /** Splits a saved voice id into {preset, custom}: a known curated id selects
   *  itself; anything else (or no curated list yet) falls back to free-text. */
  function classifyCloudVoice(
    saved: string,
    presets: TtsVoice[]
  ): { preset: string; custom: string } {
    // Nothing saved yet: leave `preset` empty so the caller's `|| presets[0]?.id`
    // fallback picks the default — CUSTOM_VOICE here would short-circuit that `||`.
    if (!saved) return { preset: '', custom: '' };
    if (presets.some((v) => v.id === saved)) return { preset: saved, custom: '' };
    return { preset: presets.length > 0 ? CUSTOM_VOICE : '', custom: saved };
  }

  function prefersReducedMotion(): boolean {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    try {
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch {
      return false;
    }
  }

  /** Crossfade duration, collapsed to 0 under reduced-motion. */
  function motionMs(ms: number): number {
    return prefersReducedMotion() ? 0 : ms;
  }

  /** True once the selected local engine's weights are on disk. Qwen has no
   *  registry-tracked artifacts (`required_model_ids` is empty) — presence is
   *  instead an HF-snapshot check via `tts_model_downloaded`. */
  async function engineDownloaded(): Promise<boolean> {
    if (selectedEngine === 'qwen3_local') {
      return ttsModelDownloaded('qwen3_local', '');
    }
    for (const model of modelIds) {
      if (!(await ttsModelDownloaded(selectedEngine, model))) return false;
    }
    return true;
  }

  onMount(async () => {
    if (!isTauri()) return;
    try {
      catalog = (await ttsEngineCatalog()) ?? [];
    } catch {
      catalog = [];
    }
    try {
      const cfg = await invoke<AppConfig>('get_config');
      if (cfg.tts?.cloud && cfg.tts.cloud.api_key.trim() !== '') {
        hasSavedKey = true;
        savedCloudApiKey = cfg.tts.cloud.api_key;
        cloudApiKey = '';
      }
      cloudBaseUrl = cfg.tts?.cloud?.base_url?.trim() || 'https://api.openai.com';
      // Voices are shared across engines; only trust them as Cloud picks when
      // Cloud is the currently-active backend (otherwise they belong to whatever
      // local engine is active and would be nonsense cloud voice ids).
      const backendIsCloud =
        typeof cfg.tts?.backend === 'object' &&
        cfg.tts.backend !== null &&
        'cloud' in cfg.tts.backend;
      const savedHost =
        backendIsCloud && typeof cfg.voices?.host === 'string' ? cfg.voices.host : '';
      const savedGuest =
        backendIsCloud && typeof cfg.voices?.guest === 'string' ? cfg.voices.guest : '';
      const hostClass = classifyCloudVoice(savedHost, cloudMaleVoices);
      cloudHostPreset = hostClass.preset || cloudMaleVoices[0]?.id || '';
      cloudHostCustom = hostClass.custom;
      const guestClass = classifyCloudVoice(savedGuest, cloudFemaleVoices);
      cloudGuestPreset = guestClass.preset || cloudFemaleVoices[0]?.id || '';
      cloudGuestCustom = guestClass.custom;
      selectedEngine = cfg.tts?.backend === 'qwen3_local' ? 'qwen3_local' : 'orpheus';
      // If the local engine is already on disk, skip the download step and go
      // straight to voice selection — pre-filled from any previously saved
      // host/guest voices.
      if (await engineDownloaded()) {
        downloaded = true;
        voices = selectedEntry?.preset_voices ?? [];
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

  /** Switch the Local-tab engine. Selection persists reactively via persistLocalTts
   *  (see nextTtsConfig in system-check.ts for the Cloud-key-preserving rule). */
  async function pickEngine(id: TtsEngineId): Promise<void> {
    if (id === 'cloud' || id === selectedEngine) return;
    const entry = catalog.find((e) => e.id === id);
    if (entry && !entry.available) return;

    selectedEngine = id;
    downloaded = false;
    incomplete = false;
    downloadProgress = null;
    downloadError = null;
    voices = [];
    voicesUnavailable = false;
    maleVoice = '';
    femaleVoice = '';

    if (await engineDownloaded()) {
      downloaded = true;
      voices = selectedEntry?.preset_voices ?? [];
      voicesUnavailable = voices.length === 0;
      if (maleVoices.length > 0) maleVoice = maleVoices[0].id;
      if (femaleVoices.length > 0) femaleVoice = femaleVoices[0].id;
      // Don't persist fake/empty voice IDs when the catalog has none for this engine.
      if (!voicesUnavailable) void persistLocalTts();
    }
  }

  // Entering "editing" mode clears the masked field so the user types a fresh key.
  function startEditingKey(): void {
    if (hasSavedKey && !editingKey) {
      editingKey = true;
      cloudApiKey = '';
    }
  }

  async function handleDownload(): Promise<void> {
    downloadError = null;
    incomplete = false;
    downloadProgress = 0;
    try {
      if (selectedEngine === 'qwen3_local') {
        await prepareQwenModel((pct) => {
          downloadProgress = pct;
        });
      } else {
        for (const [i, model] of modelIds.entries()) {
          await downloadTtsModel(selectedEngine, model, (pct) => {
            downloadProgress = Math.round(((i + pct / 100) / modelIds.length) * 100);
          });
        }
      }
      downloadProgress = 100;
      // Re-run the on-disk presence check before flipping to "ready": a download
      // that reported done can still be truncated/partial. If it isn't actually
      // complete, surface the re-download affordance instead of a false-ready.
      if (!(await engineDownloaded())) {
        incomplete = true;
        downloadProgress = null;
        return;
      }
      downloaded = true;
      // list_tts_voices is reserved for runtime synth — the sidecar may not be running during setup.
      voices = selectedEntry?.preset_voices ?? [];
      voicesUnavailable = voices.length === 0;
      if (maleVoices.length > 0) maleVoice = maleVoices[0].id;
      if (femaleVoices.length > 0) femaleVoice = femaleVoices[0].id;
      if (!voicesUnavailable) void persistLocalTts();
    } catch (err) {
      downloadError = err instanceof Error ? err.message : 'Download failed.';
      downloadProgress = null;
    }
  }

  /** Persist the current host/guest voice + backend selection via the shared
   *  cloud-preserving helper (see nextTtsConfig in system-check.ts). */
  async function persistLocalTts(): Promise<void> {
    saveError = null;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        voices: { host: maleVoice, guest: femaleVoice },
        tts: nextTtsConfig(cfg.tts, { provider: engineToProvider(selectedEngine), apiKey: '' })
      }));
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save voice settings.';
    }
  }

  /** Re-fetch the catalog so Cloud's backend-derived `available` reflects the
   *  just-saved key immediately (Critic #3) — without this the user saves a
   *  valid key but Cloud stays reported unselectable until app restart. */
  async function refreshCatalog(): Promise<void> {
    try {
      catalog = (await ttsEngineCatalog()) ?? [];
    } catch {
      // Keep the previous catalog on a transient re-fetch failure.
    }
  }

  /** Reactive Cloud persist — mirrors `persistLocalTts`: every field writes
   *  through immediately (Select) or on blur (text inputs), no Save button.
   *  Deliberately does NOT call `oncheck()`/`oncollapse()` — those were
   *  Save-button-only side effects; a reactive edit must not collapse the panel
   *  or re-run the system check on every keystroke-adjacent write. */
  async function persistCloud(): Promise<void> {
    cloudError = null;
    try {
      // Resend the already-saved key when the user isn't actively replacing it
      // (base URL/voice-only edits), but still send a freshly-typed key on the
      // very first save (before any key has ever been persisted).
      const apiKey = editingKey || !hasSavedKey ? cloudApiKey : savedCloudApiKey;
      await saveTtsProvider({
        provider: 'cloud',
        apiKey,
        baseUrl: cloudBaseUrl,
        hostVoice: cloudHostVoice,
        guestVoice: cloudGuestVoice
      });
      savedCloudApiKey = apiKey;
      hasSavedKey = apiKey.trim() !== '';
      editingKey = false;
      await refreshCatalog();
    } catch (err) {
      cloudError = err instanceof Error ? err.message : 'Could not save configuration.';
    }
  }

  /** API-key field's blur handler: persists a freshly-typed key (first-time
   *  entry or an explicit replace), but blurring an emptied "replace" field
   *  re-masks instead of persisting — never wipes a real saved key with blank. */
  function handleKeyBlur(): void {
    if (editingKey && !cloudApiKey.trim()) {
      editingKey = false;
      return;
    }
    if (editingKey || (!hasSavedKey && cloudApiKey.trim())) {
      void persistCloud();
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
    {#if localEngines.length > 0}
      <div class="flex flex-col gap-1.5" role="radiogroup" aria-label="Local voice engine">
        {#each localEngines as entry (entry.id)}
          {@const isSel = selectedEngine === entry.id}
          {@const isAvailable = entry.available}
          <button
            type="button"
            role="radio"
            aria-checked={isSel}
            aria-disabled={!isAvailable}
            disabled={!isAvailable}
            onclick={() => pickEngine(entry.id)}
            class={cn(
              'flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left transition-colors',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              isSel
                ? 'border-primary bg-primary/10 ring-1 ring-primary'
                : 'border-border bg-card hover:text-foreground',
              !isAvailable && 'cursor-not-allowed opacity-60'
            )}
          >
            <span class="flex flex-col">
              <span class="text-[0.78rem] font-bold text-foreground">{engineLabel(entry.id)}</span>
              {#if !isAvailable && entry.unavailable_reason}
                <span class="text-[0.68rem] text-destructive">{entry.unavailable_reason}</span>
              {/if}
            </span>
            <span class="text-[0.68rem] text-muted-foreground">
              {entry.language_capability_label}
            </span>
          </button>
        {/each}
      </div>
    {/if}

    {#if !downloaded}
      <div class="flex flex-col gap-2">
        <p class="text-muted-foreground text-[0.78rem] leading-relaxed">
          This open-weight TTS engine runs entirely on-device. Download once — no internet required
          for synthesis.
        </p>
        <div class="flex items-center justify-between text-[0.75rem] text-muted-foreground">
          <span>{selectedEntry?.language_capability_label ?? 'Local voice engine'}</span>
          <span class="tabular-nums"
            >{formatSize(selectedEntry?.model_size_bytes ?? null) ?? 'On-device · Offline'}</span
          >
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

        {#if incomplete && !(downloadProgress !== null && downloadProgress < 100)}
          <p class="text-destructive text-[0.75rem]" role="alert">
            The download didn't complete. Re-download to finish setting up this voice engine.
          </p>
        {/if}

        <Button
          class="h-10 w-full"
          onclick={handleDownload}
          disabled={downloadProgress !== null && downloadProgress < 100}
        >
          {#if downloadProgress !== null && downloadProgress < 100}
            <LoaderCircle class="size-4 animate-spin" />
            Downloading…
          {:else if incomplete}
            <Download class="size-4" />
            Model incomplete — re-download
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
          <Select
            type="single"
            value={maleVoice}
            onValueChange={(v) => {
              if (v) {
                maleVoice = v;
                void persistLocalTts();
              }
            }}
            items={maleVoices.map((voice) => ({ value: voice.id, label: voice.name }))}
          >
            <SelectTrigger id="tts-male-voice" class="w-full">
              <SelectValue placeholder="Select a voice" />
            </SelectTrigger>
            <SelectContent>
              {#each maleVoices as voice (voice.id)}
                <SelectItem value={voice.id} label={voice.name}>{voice.name}</SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>

        <div class="flex flex-col gap-1.5">
          <label
            for="tts-female-voice"
            class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
          >
            Co-host voice (female)
          </label>
          <Select
            type="single"
            value={femaleVoice}
            onValueChange={(v) => {
              if (v) {
                femaleVoice = v;
                void persistLocalTts();
              }
            }}
            items={femaleVoices.map((voice) => ({ value: voice.id, label: voice.name }))}
          >
            <SelectTrigger id="tts-female-voice" class="w-full">
              <SelectValue placeholder="Select a voice" />
            </SelectTrigger>
            <SelectContent>
              {#each femaleVoices as voice (voice.id)}
                <SelectItem value={voice.id} label={voice.name}>{voice.name}</SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>
      {/if}

      {#if saveError}
        <p class="text-destructive text-[0.75rem]" role="alert">{saveError}</p>
      {/if}
    {/if}
  </div>

  <div
    id="tts-panel-cloud"
    role="tabpanel"
    aria-labelledby="tts-tab-cloud"
    tabindex={activeTab === 'cloud' ? 0 : -1}
    class={cn('mt-3 flex flex-col gap-4', activeTab !== 'cloud' && 'hidden')}
  >
    <p class="text-muted-foreground text-pretty text-[0.78rem] leading-relaxed">
      Connect any endpoint that implements OpenAI's speech API (POST /v1/audio/speech) — OpenAI
      itself, hosted providers like Groq or DeepInfra, or a self-hosted server such as LocalAI.
      Enter the API root; no local model download required.
    </p>

    {#if cloudEntry}
      {#if !cloudEntry.available}
        <p
          transition:fade={{ duration: motionMs(160) }}
          class="rounded-lg px-3 py-2 text-[0.75rem] text-destructive ring-1 ring-destructive/30 bg-destructive/10"
          role="status"
        >
          {cloudEntry.unavailable_reason ?? 'Cloud is unavailable.'} Add an API key below to enable it.
        </p>
      {:else}
        <p
          transition:fade={{ duration: motionMs(160) }}
          class="flex items-center gap-2 rounded-lg px-3 py-2 text-[0.75rem] text-primary ring-1 ring-primary/30 bg-primary/10"
          role="status"
        >
          <CircleCheck class="size-3.5" />
          Cloud is available
        </p>
      {/if}
    {/if}

    <section class="flex flex-col gap-3 rounded-xl p-3 shadow-xs ring-1 ring-foreground/10">
      <h3
        class="text-muted-foreground text-balance text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        Connection
      </h3>

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
          onblur={handleKeyBlur}
        />
        {#if hasSavedKey && !editingKey}
          <p
            transition:fade={{ duration: motionMs(160) }}
            class="text-muted-foreground text-pretty text-[0.72rem] leading-relaxed"
          >
            A key is already saved. Click the field to replace it.
          </p>
        {/if}
      </div>

      <div class="flex flex-col gap-1.5">
        <label
          for="tts-cloud-base-url"
          class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
        >
          Base URL
        </label>
        <Input
          id="tts-cloud-base-url"
          type="text"
          bind:value={cloudBaseUrl}
          placeholder="https://api.openai.com"
          autocomplete="off"
          onblur={() => void persistCloud()}
        />
        <p class="text-muted-foreground text-pretty text-[0.72rem] leading-relaxed">
          API root only — no trailing <code>/v1</code>; it's appended automatically.
        </p>
      </div>
    </section>

    <section class="flex flex-col gap-3 rounded-xl p-3 shadow-xs ring-1 ring-foreground/10">
      <h3
        class="text-muted-foreground text-balance text-[0.68rem] font-semibold tracking-widest uppercase"
      >
        OpenAI voices
      </h3>

      <div class="flex flex-col gap-1.5">
        <label
          for="tts-cloud-host-voice"
          class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
        >
          Host speaker
        </label>
        {#if cloudMaleVoices.length > 0}
          <Select
            type="single"
            value={cloudHostPreset}
            onValueChange={(v) => {
              if (v) {
                cloudHostPreset = v;
                void persistCloud();
              }
            }}
            items={[
              ...cloudMaleVoices.map((voice) => ({ value: voice.id, label: voice.name })),
              { value: CUSTOM_VOICE, label: 'Custom voice ID…' }
            ]}
          >
            <SelectTrigger id="tts-cloud-host-voice" class="w-full">
              <SelectValue placeholder="Select a voice" />
            </SelectTrigger>
            <SelectContent
              class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
            >
              {#each cloudMaleVoices as voice (voice.id)}
                <SelectItem value={voice.id} label={voice.name}>{voice.name}</SelectItem>
              {/each}
              <SelectItem value={CUSTOM_VOICE} label="Custom voice ID…">Custom voice ID…</SelectItem
              >
            </SelectContent>
          </Select>
          {#if cloudHostPreset === CUSTOM_VOICE}
            <Input
              id="tts-cloud-host-voice-custom"
              type="text"
              bind:value={cloudHostCustom}
              placeholder="e.g. alloy"
              autocomplete="off"
              onblur={() => void persistCloud()}
            />
          {/if}
        {:else}
          <Input
            id="tts-cloud-host-voice"
            type="text"
            bind:value={cloudHostCustom}
            placeholder="Voice ID (e.g. alloy)"
            autocomplete="off"
            onblur={() => void persistCloud()}
          />
        {/if}
      </div>

      <div class="flex flex-col gap-1.5">
        <label
          for="tts-cloud-guest-voice"
          class="text-muted-foreground text-[0.68rem] font-semibold tracking-widest uppercase"
        >
          Guest speaker
        </label>
        {#if cloudFemaleVoices.length > 0}
          <Select
            type="single"
            value={cloudGuestPreset}
            onValueChange={(v) => {
              if (v) {
                cloudGuestPreset = v;
                void persistCloud();
              }
            }}
            items={[
              ...cloudFemaleVoices.map((voice) => ({ value: voice.id, label: voice.name })),
              { value: CUSTOM_VOICE, label: 'Custom voice ID…' }
            ]}
          >
            <SelectTrigger id="tts-cloud-guest-voice" class="w-full">
              <SelectValue placeholder="Select a voice" />
            </SelectTrigger>
            <SelectContent
              class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
            >
              {#each cloudFemaleVoices as voice (voice.id)}
                <SelectItem value={voice.id} label={voice.name}>{voice.name}</SelectItem>
              {/each}
              <SelectItem value={CUSTOM_VOICE} label="Custom voice ID…">Custom voice ID…</SelectItem
              >
            </SelectContent>
          </Select>
          {#if cloudGuestPreset === CUSTOM_VOICE}
            <Input
              id="tts-cloud-guest-voice-custom"
              type="text"
              bind:value={cloudGuestCustom}
              placeholder="e.g. onyx"
              autocomplete="off"
              onblur={() => void persistCloud()}
            />
          {/if}
        {:else}
          <Input
            id="tts-cloud-guest-voice"
            type="text"
            bind:value={cloudGuestCustom}
            placeholder="Voice ID (e.g. onyx)"
            autocomplete="off"
            onblur={() => void persistCloud()}
          />
        {/if}
      </div>

      <p class="text-muted-foreground text-pretty text-[0.72rem] leading-relaxed">
        Using another provider? Enter its own voice IDs.
      </p>
    </section>

    {#if cloudError}
      <p class="text-destructive text-[0.75rem]" role="alert">{cloudError}</p>
    {/if}
  </div>
</div>
