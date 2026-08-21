<!--
  TranscriptionWhisperPane — Local Whisper model picker + downloader for the
  Transcription settings pane. Fail-closed: presence is reported upward via
  onPresenceChange so the shell can withhold "Active" until a model is on disk.
-->
<script module lang="ts">
  // Module-scope (not per-instance): survives a Settings-tab remount, so a download
  // already streaming to disk stays visible. Not the race guard — download.rs owns a
  // per-`.part` write guard; a second invoke queues, then skips the finished file.
  type DownloadState = {
    progress: number | null;
    indeterminate: boolean;
    cancelling: boolean;
    token: symbol;
  };
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
  import X from '@lucide/svelte/icons/x';
  import { Button } from '$lib/components/ui/button/index.js';
  import ProgressBar from '$lib/components/ui/ProgressBar.svelte';
  import { cn } from '$lib/utils.js';
  import {
    listWhisperModels,
    whisperModelDownloaded,
    downloadWhisperModel,
    cancelDownload,
    type WhisperModelInfo
  } from '$lib/asr/ipc.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import { toLensError } from '$lib/sources/lens-error.js';

  let { onPresenceChange }: { onPresenceChange: (modelId: string, downloaded: boolean) => void } =
    $props();

  // Sized against download.rs's 30 s idle-read timeout, and re-armed by its
  // attempt-start progress tick, so a retry's ≈96 s worst case never trips it. Fires
  // only when the bridge stops delivering events altogether.
  const WEDGE_TIMEOUT_MS = 45_000;
  const WEDGE_MESSAGE = 'Download stalled — no response from the server. Try again.';
  const STORAGE_POINTER = 'Free up space in Settings → Storage.';

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
      try {
        await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, whisper_model: id } }));
      } catch (err) {
        downloadError = toLensError(err).message;
      }
    }
    onPresenceChange(id, downloadedMap[id] ?? false);
  }

  function clearWedge(id: string, token: symbol): void {
    if (activeDownloads[id]?.token !== token) return;
    const { [id]: _removed, ...rest } = activeDownloads;
    activeDownloads = rest;
    downloadError = WEDGE_MESSAGE;
  }

  function armWatchdog(id: string, token: symbol): ReturnType<typeof setTimeout> {
    return setTimeout(() => clearWedge(id, token), WEDGE_TIMEOUT_MS);
  }

  async function handleDownload(id: string): Promise<void> {
    // Keyed per model id, not globally: downloading tiny and small at once is
    // intentional — each streams to its own `.part` file, so nothing is shared.
    if (id in activeDownloads) return;
    const token = Symbol(id);
    activeDownloads = {
      ...activeDownloads,
      [id]: { progress: 0, indeterminate: false, cancelling: false, token }
    };
    downloadError = null;
    let watchdog = armWatchdog(id, token);
    try {
      await downloadWhisperModel(id, (pct) => {
        if (activeDownloads[id]?.token !== token) return;
        clearTimeout(watchdog);
        watchdog = armWatchdog(id, token);
        const entry = activeDownloads[id];
        if (pct === null) entry.indeterminate = true;
        else {
          entry.progress = pct;
          entry.indeterminate = false;
        }
      });
      if (activeDownloads[id]?.token !== token) return;
      // Re-probe disk rather than trust the event: download.rs early-returns `done`
      // for a file that was already complete, and that skip compares length only, so
      // the event alone never proves presence.
      const downloaded = await whisperModelDownloaded(id);
      downloadedMap = { ...downloadedMap, [id]: downloaded };
      if (downloaded && id === selected) {
        await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, whisper_model: id } }));
      }
      onPresenceChange(id, downloaded);
    } catch (err) {
      const lensErr = toLensError(err);
      if (lensErr.kind === 'Cancelled') return;
      if (activeDownloads[id]?.token !== token) return;
      downloadError =
        lensErr.kind === 'InsufficientSpace'
          ? `${lensErr.message} ${STORAGE_POINTER}`
          : lensErr.message;
    } finally {
      clearTimeout(watchdog);
      if (activeDownloads[id]?.token === token) {
        const { [id]: _removed, ...rest } = activeDownloads;
        activeDownloads = rest;
      }
    }
  }

  // Deliberately leaves `activeDownloads` alone: the entry is owned by handleDownload's
  // `finally`, so Download stays disabled until the invoke itself settles.
  async function handleCancel(id: string): Promise<void> {
    const entry = activeDownloads[id];
    if (!entry || entry.cancelling) return;
    const { token } = entry;
    entry.cancelling = true;
    try {
      await cancelDownload({ kind: 'whisper', id });
    } catch (err) {
      if (activeDownloads[id]?.token !== token) return;
      entry.cancelling = false;
      downloadError = toLensError(err).message;
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
        {@const isActive = isDownloaded && m.id === (appConfigStore.asr?.whisper_model ?? '')}
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
                {#if isActive}
                  <span
                    class="rounded-full bg-primary px-2 py-0.5 text-[0.6rem] font-bold uppercase tracking-[0.05em] text-primary-foreground"
                  >
                    Active
                  </span>
                {/if}
              </div>
              <p class="mt-0.5 text-[0.68rem] text-muted-foreground">{m.approx_mb} MB</p>

              {#if isDownloadingThis}
                {@const dl = activeDownloads[m.id]}
                <div class="mt-2">
                  <ProgressBar value={dl?.indeterminate ? null : (dl?.progress ?? null)} />
                </div>
              {/if}
            </div>
          </button>

          <div class="flex shrink-0 items-center gap-1.5">
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
              {#if isDownloadingThis}
                {@const dl = activeDownloads[m.id]}
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  aria-label={`Cancel ${m.id} download`}
                  disabled={dl?.cancelling ?? false}
                  onclick={() => handleCancel(m.id)}
                >
                  <X class="size-3.5" />
                  {dl?.cancelling ? 'Cancelling…' : 'Cancel'}
                </Button>
              {/if}
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
