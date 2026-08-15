<!--
  TranscriptionWhisperPane — Local Whisper model picker + downloader for the
  Transcription settings pane. Fail-closed: presence is reported upward via
  onPresenceChange so the shell can withhold "Active" until a model is on disk.
-->
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

  let { onPresenceChange }: { onPresenceChange: (modelId: string, downloaded: boolean) => void } =
    $props();

  let models = $state<WhisperModelInfo[]>([]);
  let selected = $state('');
  let downloadedMap = $state<Record<string, boolean>>({});
  let loadError = $state<string | null>(null);

  let downloadingId = $state<string | null>(null);
  let downloadProgress = $state<number | null>(null);
  let downloadIndeterminate = $state(false);
  let downloadError = $state<string | null>(null);

  // Monotonic generation token: guards a download's progress ticks and terminal
  // handling against a second download starting before the first settles.
  let gen = 0;

  onMount(async () => {
    await ensureLoaded();
    try {
      models = await listWhisperModels();
    } catch (err) {
      loadError = err instanceof Error ? err.message : 'Could not load Whisper models.';
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
    await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, whisper_model: id } }));
    onPresenceChange(id, downloadedMap[id] ?? false);
  }

  async function handleDownload(id: string): Promise<void> {
    const my = ++gen;
    downloadingId = id;
    downloadProgress = 0;
    downloadIndeterminate = false;
    downloadError = null;
    try {
      await downloadWhisperModel(id, (pct) => {
        if (gen !== my) return;
        if (pct === null) {
          downloadIndeterminate = true;
          return;
        }
        downloadIndeterminate = false;
        downloadProgress = pct;
      });
      if (gen !== my) return;
      // download.rs early-returns `done` for an already-complete file (its own
      // sha256+rename already ran before this event) — re-probe disk rather
      // than trust the event so presence never drifts from what's actually there.
      const downloaded = await whisperModelDownloaded(id);
      if (gen !== my) return;
      downloadedMap = { ...downloadedMap, [id]: downloaded };
      onPresenceChange(id, downloaded);
    } catch (err) {
      if (gen !== my) return;
      downloadError = err instanceof Error ? err.message : 'Download failed.';
    } finally {
      if (gen === my) {
        downloadingId = null;
        downloadProgress = null;
        downloadIndeterminate = false;
      }
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
        {@const isDownloadingThis = downloadingId === m.id}
        <div
          role="radio"
          tabindex="0"
          aria-checked={isSelected}
          onclick={() => selectModel(m.id)}
          onkeydown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              void selectModel(m.id);
            }
          }}
          class={cn(
            'flex cursor-pointer items-center gap-3 rounded-[10px] border px-4 py-3.5 transition-colors',
            isSelected ? 'border-primary/45 bg-primary/5' : 'border-border bg-card'
          )}
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
              <div class="mt-2">
                <ProgressBar value={downloadIndeterminate ? null : downloadProgress} />
              </div>
            {/if}
          </div>

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
                disabled={downloadingId !== null && !isDownloadingThis}
                onclick={(e) => {
                  e.stopPropagation();
                  void handleDownload(m.id);
                }}
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
