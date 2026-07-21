<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Button } from '$lib/components/ui/button/index.js';
  import ProgressBar from '$lib/components/ui/ProgressBar.svelte';
  import { cn } from '$lib/utils.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Download from '@lucide/svelte/icons/download';
  import {
    downloadTtsModel,
    prepareQwenModel,
    cancelPrepare,
    ttsModelStatus,
    nextTtsConfig,
    ttsEngineCatalog,
    type TtsVoice,
    type TtsEngineCatalogEntry,
    type TtsEngineId,
    type TtsProvider,
    type TtsModelStatus
  } from '$lib/onboarding/system-check.js';
  import { toLensError } from '../../sources/lens-error.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';
  import { updateConfig } from '$lib/config.js';

  // Bound so this form can self-fetch `catalog` on mount (writing back through the
  // bind), keeping its catalog-then-config init self-contained regardless of
  // parent/child mount order. See TtsConfigPanel for the ownership rationale.
  let {
    catalog = $bindable(),
    active
  }: {
    catalog: TtsEngineCatalogEntry[];
    active: boolean;
  } = $props();

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

  let downloadProgress = $state<number | null>(null);
  let downloadIndeterminate = $state(false);
  const isDownloading = $derived(downloadProgress !== null && downloadProgress < 100);
  // Single tri-state per engine; `downloaded`/`incomplete` derive from it — the
  // invalid "both true" state the two-boolean model allowed is unrepresentable.
  let status = $state<TtsModelStatus>('absent');
  const downloaded = $derived(status === 'complete');
  const incomplete = $derived(status === 'partial');
  let downloadError = $state<string | null>(null);

  let voices = $state<TtsVoice[]>([]);
  let maleVoice = $state('');
  let femaleVoice = $state('');
  let saveError = $state<string | null>(null);
  // True only if the catalog carries no preset voices for the engine — surface
  // an inline error and disable Save rather than persisting fake IDs.
  let voicesUnavailable = $state(false);

  const maleVoices = $derived(voices.filter((v) => v.gender === 'male'));
  const femaleVoices = $derived(voices.filter((v) => v.gender === 'female'));

  /** Aggregate the engine's per-model status into one tri-state, probing each
   *  model once: Complete iff all Complete; Partial iff any Partial; else Absent
   *  (so Orpheus complete + SNAC absent is Absent, not Partial). */
  async function engineStatus(): Promise<TtsModelStatus> {
    if (selectedEngine === 'qwen3_local') {
      return ttsModelStatus('qwen3_local', '');
    }
    let allComplete = true;
    let anyPartial = false;
    for (const model of modelIds) {
      const s = await ttsModelStatus(selectedEngine, model);
      if (s !== 'complete') allComplete = false;
      if (s === 'partial') anyPartial = true;
    }
    if (allComplete) return 'complete';
    return anyPartial ? 'partial' : 'absent';
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
      selectedEngine = cfg.tts?.backend === 'qwen3_local' ? 'qwen3_local' : 'orpheus';
      // If the local engine is already on disk, skip the download step and go
      // straight to voice selection — pre-filled from any previously saved
      // host/guest voices.
      status = await engineStatus();
      if (downloaded) {
        // Read the just-fetched catalog directly (not the `selectedEntry`/`maleVoices`
        // deriveds, which can be stale when read in this async continuation).
        const preset = (catalog.find((e) => e.id === selectedEngine)?.preset_voices ?? []).slice();
        voices = preset;
        voicesUnavailable = preset.length === 0;
        const host = cfg.voices?.host;
        const guest = cfg.voices?.guest;
        const male = preset.filter((v) => v.gender === 'male');
        const female = preset.filter((v) => v.gender === 'female');
        maleVoice = (typeof host === 'string' ? host : '') || male[0]?.id || '';
        femaleVoice = (typeof guest === 'string' ? guest : '') || female[0]?.id || '';
      }
    } catch {
      // Non-fatal: fall back to the default download prompt.
    }
  });

  /** Switch the Local-tab engine. Selection persists reactively via persistLocalTts
   *  (see nextTtsConfig in system-check.ts for the Cloud-key-preserving rule). */
  async function pickEngine(id: TtsEngineId): Promise<void> {
    if (id === 'cloud' || id === selectedEngine) return;
    const entry = catalog.find((e) => e.id === id);
    if (entry && !entry.available) return;

    selectedEngine = id;
    status = 'absent';
    downloadProgress = null;
    downloadIndeterminate = false;
    downloadError = null;
    voices = [];
    voicesUnavailable = false;
    maleVoice = '';
    femaleVoice = '';

    status = await engineStatus();
    if (downloaded) {
      voices = selectedEntry?.preset_voices ?? [];
      voicesUnavailable = voices.length === 0;
      if (maleVoices.length > 0) maleVoice = maleVoices[0].id;
      if (femaleVoices.length > 0) femaleVoice = femaleVoices[0].id;
      // Don't persist fake/empty voice IDs when the catalog has none for this engine.
      if (!voicesUnavailable) void persistLocalTts();
    }
  }

  /** Apply one progress callback tick: `null` (unknown total) flips the
   *  indeterminate flag without touching `downloadProgress`, so `isDownloading`
   *  stays true; a known percentage clears the flag and updates the value. */
  function applyProgress(pct: number | null, computeDeterminate: (p: number) => number): void {
    if (pct === null) {
      downloadIndeterminate = true;
      return;
    }
    downloadIndeterminate = false;
    downloadProgress = computeDeterminate(pct);
  }

  async function handleDownload(): Promise<void> {
    downloadError = null;
    status = 'absent';
    downloadProgress = 0;
    downloadIndeterminate = false;
    try {
      if (selectedEngine === 'qwen3_local') {
        await prepareQwenModel((pct) => applyProgress(pct, (p) => p));
      } else {
        for (const [i, model] of modelIds.entries()) {
          await downloadTtsModel(selectedEngine, model, (pct) =>
            applyProgress(pct, (p) => Math.round(((i + p / 100) / modelIds.length) * 100))
          );
        }
      }
      downloadIndeterminate = false;
      downloadProgress = 100;
      // Re-run the on-disk presence check before flipping to "ready": a download
      // that reported done can still be truncated/partial. If it isn't actually
      // complete, surface the re-download affordance instead of a false-ready.
      if ((await engineStatus()) !== 'complete') {
        status = 'partial';
        downloadProgress = null;
        return;
      }
      status = 'complete';
      // list_tts_voices is reserved for runtime synth — the sidecar may not be running during setup.
      voices = selectedEntry?.preset_voices ?? [];
      voicesUnavailable = voices.length === 0;
      if (maleVoices.length > 0) maleVoice = maleVoices[0].id;
      if (femaleVoices.length > 0) femaleVoice = femaleVoices[0].id;
      if (!voicesUnavailable) void persistLocalTts();
    } catch (err) {
      downloadIndeterminate = false;
      downloadProgress = null;
      // A deliberate cancel (unmount during a Qwen download) isn't a failure —
      // don't surface "Download failed" for it.
      if (toLensError(err).kind === 'Cancelled') return;
      downloadError = err instanceof Error ? err.message : 'Download failed.';
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

  // TtsConfigPanel is single-@render-mounted by SettingsShell, so unmount here means
  // Settings nav-away/close — the intended cancel trigger. Engine-guarded because
  // `cancel_prepare` is macOS-aarch64-only (cancelPrepare() no-ops off it anyway).
  onDestroy(() => {
    if (isDownloading && selectedEngine === 'qwen3_local') {
      void cancelPrepare();
    }
  });
</script>

<div
  id="tts-panel-local"
  role="tabpanel"
  aria-labelledby="tts-tab-local"
  tabindex={active ? 0 : -1}
  class={cn('mt-3 flex flex-col gap-3', !active && 'hidden')}
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

      {#if isDownloading}
        <ProgressBar value={downloadIndeterminate ? null : downloadProgress} />
        {#if !downloadIndeterminate}
          <p class="text-[0.72rem] text-muted-foreground text-center">
            {downloadProgress}% downloaded
          </p>
        {/if}
      {/if}

      {#if downloadError}
        <p class="text-destructive text-[0.75rem]" role="alert">{downloadError}</p>
      {/if}

      {#if incomplete && !isDownloading}
        <p class="text-destructive text-[0.75rem]" role="alert">
          The download didn't complete. Re-download to finish setting up this voice engine.
        </p>
      {/if}

      <Button class="h-10 w-full" onclick={handleDownload} disabled={isDownloading}>
        {#if isDownloading}
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
