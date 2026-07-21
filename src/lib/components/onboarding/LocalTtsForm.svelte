<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Button } from '$lib/components/ui/button/index.js';
  import ProgressBar from '$lib/components/ui/ProgressBar.svelte';
  import { cn } from '$lib/utils.js';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
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

  // `engine` is owned by the parent's engine list. `catalog` is $bindable so the form
  // can self-fetch when mounted standalone (unit tests); the panel passes it populated.
  // See TtsConfigPanel for the parent/child ownership rationale.
  let {
    catalog = $bindable(),
    engine,
    active,
    onactivated
  }: {
    catalog: TtsEngineCatalogEntry[];
    engine: TtsEngineId;
    active: boolean;
    onactivated?: (id: TtsEngineId) => void;
  } = $props();

  const selectedEntry = $derived(catalog.find((e) => e.id === engine) ?? null);

  function engineToProvider(id: TtsEngineId): TtsProvider {
    if (id === 'qwen3_local') return 'qwen3';
    if (id === 'cloud') return 'cloud';
    return 'orpheus';
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

  /** Human-readable on-disk size for the always-visible label, e.g. "~2.3 GB". */
  function formatSize(bytes: number | null): string | null {
    if (bytes === null || bytes <= 0) return null;
    return `~${(bytes / 1_000_000_000).toFixed(1)} GB`;
  }

  /** Aggregate the engine's per-model tri-states (Complete iff all; Partial iff any;
   *  else Absent). Takes an explicit `id`, not the reactive `engine`, so a stale
   *  in-flight probe stays self-consistent. */
  async function engineStatus(id: TtsEngineId): Promise<TtsModelStatus> {
    if (id === 'qwen3_local') {
      return ttsModelStatus('qwen3_local', '');
    }
    const ids = catalog.find((e) => e.id === id)?.required_model_ids ?? [];
    let allComplete = ids.length > 0;
    let anyPartial = false;
    for (const model of ids) {
      const s = await ttsModelStatus(id, model);
      if (s !== 'complete') allComplete = false;
      if (s === 'partial') anyPartial = true;
    }
    if (allComplete) return 'complete';
    return anyPartial ? 'partial' : 'absent';
  }

  // Read preset voices straight from the freshly-fetched catalog (not a derived,
  // which can be stale in an async continuation).
  function presetVoicesFor(id: TtsEngineId): TtsVoice[] {
    return (catalog.find((e) => e.id === id)?.preset_voices ?? []).slice();
  }

  /** Populate the pickers from `preset`, computing male/female locally (never the
   *  possibly-stale deriveds); prefer a saved id, else the first of each gender. */
  function seedVoicePickers(preset: TtsVoice[], savedHost = '', savedGuest = ''): void {
    voices = preset;
    voicesUnavailable = preset.length === 0;
    const male = preset.filter((v) => v.gender === 'male');
    const female = preset.filter((v) => v.gender === 'female');
    maleVoice = savedHost || male[0]?.id || '';
    femaleVoice = savedGuest || female[0]?.id || '';
  }

  // Set by activate() when the parent re-picks this engine before its load finished;
  // the in-flight load persists on completion (see loadEngine). Reset per fresh load.
  let pendingActivate = false;

  /** Probe `id`'s on-disk status and, when complete, populate the voice pickers.
   *  `prefillFromConfig` (initial engine only) seeds host/guest from saved config;
   *  a switch uses catalog defaults. `persist` pins an already-installed engine. */
  async function loadEngine(
    id: TtsEngineId,
    opts: { persist: boolean; prefillFromConfig: boolean }
  ): Promise<void> {
    status = 'absent';
    downloadProgress = null;
    downloadIndeterminate = false;
    downloadError = null;
    voices = [];
    voicesUnavailable = false;
    maleVoice = '';
    femaleVoice = '';
    pendingActivate = false;

    if (!isTauri()) return;

    let probed: TtsModelStatus;
    try {
      probed = await engineStatus(id);
    } catch {
      // A transient probe failure is non-fatal: leave the download prompt (status
      // was reset to 'absent' above) rather than throwing out of the effect.
      return;
    }
    // A newer selection superseded this load while probing — don't clobber its state.
    if (engine !== id) return;
    status = probed;
    if (probed !== 'complete') return;

    let savedHost = '';
    let savedGuest = '';
    if (opts.prefillFromConfig) {
      try {
        const cfg = await invoke<AppConfig>('get_config');
        savedHost = typeof cfg.voices?.host === 'string' ? cfg.voices.host : '';
        savedGuest = typeof cfg.voices?.guest === 'string' ? cfg.voices.guest : '';
      } catch {
        // Non-fatal: fall back to catalog defaults below.
      }
    }
    if (engine !== id) return;
    seedVoicePickers(presetVoicesFor(id), savedHost, savedGuest);

    if ((opts.persist || pendingActivate) && !voicesUnavailable) {
      pendingActivate = false;
      void persistLocalTts();
    }
  }

  // Only `engine` is tracked (untrack guards the rest) so a catalog refresh never
  // re-triggers a load; `lastLoaded` dedupes and the `engine !== id` checks in
  // loadEngine act as its cancellation token against superseded loads.
  let lastLoaded: TtsEngineId | null = null;
  $effect(() => {
    const id = engine;
    untrack(() => {
      if (id === lastLoaded) return;
      void handleEngine(id);
    });
  });

  /** Pin this engine when the parent re-picks it without changing the `engine` prop
   *  (the load effect can't observe that). Persists now if ready, else defers to the
   *  in-flight load — never activates an uninstalled engine. */
  export function activate(): void {
    if (status === 'complete' && !voicesUnavailable) void persistLocalTts();
    else pendingActivate = true;
  }

  /** Fetch the catalog first when standalone, THEN load — `engineStatus` reads
   *  `modelIds` (catalog-derived), which would wrongly aggregate to `complete` over
   *  an empty list if probed before the catalog resolved. */
  async function handleEngine(id: TtsEngineId): Promise<void> {
    const isInitial = lastLoaded === null;
    lastLoaded = id;
    if (catalog.length === 0 && isTauri()) {
      try {
        catalog = (await ttsEngineCatalog()) ?? [];
      } catch {
        catalog = [];
      }
    }
    await loadEngine(id, { persist: !isInitial, prefillFromConfig: isInitial });
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
    // Pin the target so a mid-download engine switch can't make the completion
    // path probe/persist/reveal-voices for the newly-selected engine instead.
    const dlId = engine;
    downloadError = null;
    status = 'absent';
    downloadProgress = 0;
    downloadIndeterminate = false;
    // Pin the model list at the start: `modelIds` is a derived that recomputes to
    // the newly-selected engine on a mid-download switch (0-length for Qwen →
    // division by zero), and the `engine === dlId` guard drops stale progress ticks.
    const ids = [...modelIds];
    try {
      if (dlId === 'qwen3_local') {
        await prepareQwenModel((pct) => {
          if (engine === dlId) applyProgress(pct, (p) => p);
        });
      } else {
        for (const [i, model] of ids.entries()) {
          await downloadTtsModel(dlId, model, (pct) => {
            if (engine === dlId)
              applyProgress(pct, (p) => Math.round(((i + p / 100) / ids.length) * 100));
          });
        }
      }
      if (engine !== dlId) return;
      downloadIndeterminate = false;
      downloadProgress = 100;
      // Re-run the on-disk presence check before flipping to "ready": a download
      // that reported done can still be truncated/partial. If it isn't actually
      // complete, surface the re-download affordance instead of a false-ready.
      const rechecked = await engineStatus(dlId);
      if (engine !== dlId) return;
      if (rechecked !== 'complete') {
        status = 'partial';
        downloadProgress = null;
        return;
      }
      status = 'complete';
      // list_tts_voices is reserved for runtime synth — the sidecar may not be running during setup.
      seedVoicePickers(presetVoicesFor(dlId));
      if (!voicesUnavailable) void persistLocalTts();
    } catch (err) {
      if (engine !== dlId) return;
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
        tts: nextTtsConfig(cfg.tts, { provider: engineToProvider(engine), apiKey: '' })
      }));
      onactivated?.(engine);
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save voice settings.';
    }
  }

  // Unmount here means Settings nav-away/close — the intended cancel trigger. Engine-guarded
  // because `cancel_prepare` is macOS-aarch64-only (cancelPrepare() no-ops off it anyway).
  onDestroy(() => {
    if (isDownloading && engine === 'qwen3_local') {
      void cancelPrepare();
    }
  });
</script>

{#snippet voiceSelect(opts: {
  id: string;
  label: string;
  value: string;
  voices: TtsVoice[];
  onpick: (v: string) => void;
})}
  <div class="mt-3 flex flex-col gap-1.5">
    <label for={opts.id} class="text-[0.72rem] font-bold text-foreground">{opts.label}</label>
    <Select
      type="single"
      value={opts.value}
      onValueChange={(v) => {
        if (v) opts.onpick(v);
      }}
      items={opts.voices.map((voice) => ({ value: voice.id, label: voice.name }))}
    >
      <SelectTrigger id={opts.id} class="w-full">
        <SelectValue placeholder="Select a voice" />
      </SelectTrigger>
      <SelectContent
        class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
      >
        {#each opts.voices as voice (voice.id)}
          <SelectItem value={voice.id} label={voice.name}>{voice.name}</SelectItem>
        {/each}
      </SelectContent>
    </Select>
  </div>
{/snippet}

<div
  role="group"
  aria-label="Local voice engine setup"
  class={cn('flex flex-col gap-4', !active && 'hidden')}
>
  {#if !downloaded}
    <div>
      <p class="text-pretty text-[0.72rem] leading-relaxed text-muted-foreground">
        This open-weight engine runs entirely on-device. Download once — no internet required for
        synthesis.
      </p>
      <div class="mt-3 flex items-center justify-between text-[0.72rem] text-muted-foreground">
        <span>{selectedEntry?.language_capability_label ?? 'Local voice engine'}</span>
        <span class="tabular-nums">
          {formatSize(selectedEntry?.model_size_bytes ?? null) ?? 'On-device · Offline'}
        </span>
      </div>

      {#if isDownloading}
        <div class="mt-3">
          <ProgressBar value={downloadIndeterminate ? null : downloadProgress} />
          {#if !downloadIndeterminate}
            <p class="mt-1 text-center text-[0.7rem] tabular-nums text-muted-foreground">
              {downloadProgress}% downloaded
            </p>
          {/if}
        </div>
      {/if}

      {#if downloadError}
        <p class="mt-3 text-[0.72rem] text-destructive" role="alert">{downloadError}</p>
      {/if}

      {#if incomplete && !isDownloading}
        <p class="mt-3 text-[0.72rem] text-destructive" role="alert">
          The download didn't complete. Re-download to finish setting up this voice engine.
        </p>
      {/if}

      <Button class="mt-4 h-10 w-full" onclick={handleDownload} disabled={isDownloading}>
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
  {:else if voicesUnavailable}
    <p class="text-[0.72rem] text-destructive" role="alert">
      Couldn't load voices — is the engine installed?
    </p>
  {:else}
    <div>
      <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
        Voices
      </p>
      {@render voiceSelect({
        id: 'tts-male-voice',
        label: 'Host voice (male)',
        value: maleVoice,
        voices: maleVoices,
        onpick: (v) => {
          maleVoice = v;
          void persistLocalTts();
        }
      })}
      {@render voiceSelect({
        id: 'tts-female-voice',
        label: 'Co-host voice (female)',
        value: femaleVoice,
        voices: femaleVoices,
        onpick: (v) => {
          femaleVoice = v;
          void persistLocalTts();
        }
      })}
    </div>
  {/if}

  {#if saveError}
    <p class="text-[0.72rem] text-destructive" role="alert">{saveError}</p>
  {/if}
</div>
