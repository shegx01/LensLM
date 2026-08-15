<!--
  TranscriptionWhisperPane — Local Whisper model picker + downloader for the
  Transcription settings pane. Fail-closed: presence is reported upward via
  onPresenceChange so the shell can withhold "Active" until a model is on disk.
-->
<script module lang="ts">
  // Module-scope (not per-instance): a Settings-tab remount recreates this
  // component, but a download already streaming to disk must stay visible and
  // un-restartable, or a second invoke would race the first onto the same
  // `.part` path (see download.rs `File::create`).
  type DownloadState = { progress: number | null; indeterminate: boolean };
  let activeDownloads = $state<Record<string, DownloadState>>({});

  /** Test hook: clears in-flight download state between tests (mirrors
   *  `resetConfig` in app-config.svelte.ts). */
  export function resetActiveDownloads(): void {
    activeDownloads = {};
  }
</script>

<script lang="ts">
  import { onMount } from 'svelte';
  import Download from '@lucide/svelte/icons/download';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import { Button } from '$lib/components/ui/button/index.js';
  import ProgressBar from '$lib/components/ui/ProgressBar.svelte';
  import { cn } from '$lib/utils.js';
  import {
    listWhisperModels,
    whisperModelDownloaded,
    downloadWhisperModel,
    type WhisperModelInfo
  } from '$lib/asr/ipc.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import { toLensError } from '$lib/sources/lens-error.js';

  let { onPresenceChange }: { onPresenceChange: (modelId: string, downloaded: boolean) => void } =
    $props();

  let models = $state<WhisperModelInfo[]>([]);
  let selected = $state('');
  let downloadedMap = $state<Record<string, boolean>>({});
  let loadError = $state<string | null>(null);
  let downloadError = $state<string | null>(null);

  onMount(async () => {
    await ensureLoaded();
    try {
      models = await listWhisperModels();
    } catch (err) {
      loadError = toLensError(err).message;
      return;
    }
    selected =
      appConfigStore.asr?.whisper_model || models.find((m) => m.is_default)?.id || models[0]?.id;

    const probes = await Promise.all(
      models.map(async (m) => [m.id, await whisperModelDownloaded(m.id)] as const)
    );
    downloadedMap = Object.fromEntries(probes);

    if (selected) onPresenceChange(selected, downloadedMap[selected] ?? false);
  });

  async function selectModel(id: string): Promise<void> {
    if (id === selected) return;
    selected = id;
    // Fail-closed: only persist a model that's actually on disk, or browsing an
    // undownloaded row would silently swap a working config for a broken one —
    // (LocalWhisper, undownloaded) has no degradation arm in the engine.
    if (downloadedMap[id]) {
      await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, whisper_model: id } }));
    }
    onPresenceChange(id, downloadedMap[id] ?? false);
  }

  async function handleDownload(id: string): Promise<void> {
    if (id in activeDownloads) return;
    activeDownloads = { ...activeDownloads, [id]: { progress: 0, indeterminate: false } };
    downloadError = null;
    try {
      await downloadWhisperModel(id, (pct) => {
        activeDownloads = {
          ...activeDownloads,
          [id]:
            pct === null
              ? { progress: activeDownloads[id]?.progress ?? null, indeterminate: true }
              : { progress: pct, indeterminate: false }
        };
      });
      // download.rs early-returns `done` for an already-complete file (its own
      // sha256+rename already ran before this event) — re-probe disk rather
      // than trust the event so presence never drifts from what's actually there.
      const downloaded = await whisperModelDownloaded(id);
      downloadedMap = { ...downloadedMap, [id]: downloaded };
      if (downloaded && id === selected) {
        await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, whisper_model: id } }));
      }
      onPresenceChange(id, downloaded);
    } catch (err) {
      downloadError = toLensError(err).message;
    } finally {
      const { [id]: _removed, ...rest } = activeDownloads;
      activeDownloads = rest;
    }
  }
</script>

<div class="flex flex-col gap-3">
  <p class="text-pretty text-[0.72rem] leading-relaxed text-muted-foreground">
    Runs entirely on-device via whisper.cpp. A model must be downloaded before Local Whisper can
    transcribe.
  </p>

  {#if loadError}
    <p class="text-[0.72rem] text-destructive" role="alert">{loadError}</p>
  {:else}
    <div class="flex flex-col gap-2" role="radiogroup" aria-label="Local Whisper model">
      {#each models as m (m.id)}
        {@const isSelected = selected === m.id}
        {@const isDownloaded = downloadedMap[m.id] ?? false}
        {@const isDownloadingThis = m.id in activeDownloads}
        <div
          class={cn(
            'flex items-center gap-3 rounded-[10px] border px-4 py-3.5 transition-colors',
            isSelected ? 'border-primary/45 bg-primary/5' : 'border-border bg-card'
          )}
        >
          <button
            type="button"
            role="radio"
            aria-checked={isSelected}
            aria-label={m.id}
            onclick={() => selectModel(m.id)}
            class="flex min-w-0 flex-1 items-center gap-3 text-left"
          >
            <span
              class={cn(
                'size-3.5 shrink-0 rounded-full',
                isSelected ? 'bg-primary' : 'ring-[1.5px] ring-inset ring-muted-foreground'
              )}
              aria-hidden="true"
            ></span>

            <div class="min-w-0 flex-1">
              <div class="flex items-center gap-2">
                <span class="font-mono text-[0.8rem] font-bold text-foreground">{m.id}</span>
                {#if m.is_default}
                  <span
                    class="rounded-full bg-primary/15 px-2 py-0.5 text-[0.6rem] font-bold uppercase tracking-[0.05em] text-primary"
                  >
                    Recommended
                  </span>
                {/if}
              </div>
              <p class="mt-0.5 text-[0.68rem] text-muted-foreground">{m.approx_mb} MB</p>

              {#if isDownloadingThis}
                {@const dl = activeDownloads[m.id] ?? { progress: null, indeterminate: false }}
                <div class="mt-2">
                  <ProgressBar value={dl.indeterminate ? null : dl.progress} />
                </div>
              {/if}
            </div>
          </button>

          <div class="shrink-0">
            {#if isDownloaded}
              <span class="flex items-center gap-1 text-[0.72rem] font-bold text-primary">
                <CircleCheck class="size-3.5" />
                Downloaded
              </span>
            {:else}
              <Button
                type="button"
                size="sm"
                aria-label={`Download ${m.id}`}
                disabled={isDownloadingThis}
                onclick={() => handleDownload(m.id)}
              >
                {#if isDownloadingThis}
                  <LoaderCircle class="size-3.5 animate-spin" />
                  Downloading…
                {:else}
                  <Download class="size-3.5" />
                  Download
                {/if}
              </Button>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}

  {#if downloadError}
    <p class="text-[0.72rem] text-destructive" role="alert">{downloadError}</p>
  {/if}
</div>
