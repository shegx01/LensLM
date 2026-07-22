<!--
  StorageSection — the "Storage" panel inside the global Preferences view.
  Data location + relocation (`relocate_data_dir` / `restart_app`), the per-bucket
  usage breakdown, model-cache offload (`offload_cache` / `reset_cache_location`),
  a soft cache quota, and reclaiming the cache via `clear_model_cache` (#238).
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { open } from '@tauri-apps/plugin-dialog';
  import { revealItemInDir } from '@tauri-apps/plugin-opener';
  import { writeText } from '@tauri-apps/plugin-clipboard-manager';
  import { audioOverviewStore } from '$lib/sources/audio-state.svelte.js';
  import { formatBytes } from '$lib/format/bytes.js';
  import { updateConfig } from '$lib/config.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter
  } from '$lib/components/ui/dialog/index.js';
  import FolderOpen from '@lucide/svelte/icons/folder-open';
  import FolderInput from '@lucide/svelte/icons/folder-input';
  import HardDriveDownload from '@lucide/svelte/icons/hard-drive-download';
  import Copy from '@lucide/svelte/icons/copy';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import type { AppConfig, StorageStats } from '$lib/theme/types.js';

  const GB = 1_000_000_000;

  let dataDir = $state('');
  let stats = $state<StorageStats | null>(null);
  let loadError = $state<string | null>(null);
  let revealError = $state<string | null>(null);
  let copied = $state(false);
  let clearing = $state(false);
  let clearError = $state<string | null>(null);
  let confirmOpen = $state(false);

  let relocating = $state(false);
  let relocateError = $state<string | null>(null);
  let moveConfirmOpen = $state(false);
  let restartDialogOpen = $state(false);
  let pendingNewDataPath = $state('');

  let cacheDir = $state<string | null>(null);
  let offloading = $state(false);
  let resetting = $state(false);
  let offloadError = $state<string | null>(null);
  let offloadedBytes = $state<number | null>(null);
  let cacheRestartOpen = $state(false);

  // A native <input type="number"> binds `value` as number|null (Svelte coerces
  // an empty field to null), not a string — mirror that here.
  let quotaGbInput = $state<number | null>(null);
  let quotaBytes = $state<number | null>(null);
  let quotaError = $state<string | null>(null);

  let copiedTimer: ReturnType<typeof setTimeout> | null = null;

  const corpusBreakdown = $derived(
    stats
      ? [
          { label: 'Database', value: stats.db_bytes },
          { label: 'Vectors', value: stats.vectors_bytes },
          { label: 'Sources', value: stats.sources_bytes },
          { label: 'Audio', value: stats.audio_bytes }
        ]
      : []
  );

  // Advisory only — nothing auto-deletes when the cache grows past this.
  const overQuota = $derived(
    quotaBytes != null && quotaBytes > 0 && (stats?.reclaimable_cache_bytes ?? 0) > quotaBytes
  );

  const generating = $derived(audioOverviewStore.overviewStatus === 'generating');

  function bucketPct(value: number): number {
    const total = stats?.corpus_bytes ?? 0;
    return total > 0 ? Math.min(100, (value / total) * 100) : 0;
  }

  async function loadStats(): Promise<void> {
    stats = await invoke<StorageStats>('get_storage_stats');
  }

  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      dataDir = cfg.paths.data_dir;
      cacheDir = cfg.paths.cache_dir ?? null;
      quotaBytes = cfg.storage?.cache_quota_bytes ?? null;
      quotaGbInput = quotaBytes != null ? quotaBytes / GB : null;
      await loadStats();
    } catch (err) {
      loadError = err instanceof Error ? err.message : 'Could not load storage information.';
    }
  });

  async function handleReveal(): Promise<void> {
    revealError = null;
    try {
      await revealItemInDir(dataDir);
    } catch {
      // Reveal may require a fresh permission grant — surfaces as a graceful
      // inline hint rather than an unhandled rejection.
      revealError = 'Could not open Finder. This may need an app restart.';
    }
  }

  async function handleCopy(): Promise<void> {
    try {
      await writeText(dataDir);
      copied = true;
      if (copiedTimer) clearTimeout(copiedTimer);
      copiedTimer = setTimeout(() => {
        copied = false;
      }, 2000);
    } catch {
      // Non-fatal: leave the button in its normal state.
    }
  }

  async function handleClearConfirmed(): Promise<void> {
    confirmOpen = false;
    clearing = true;
    clearError = null;
    try {
      await invoke<number>('clear_model_cache');
    } catch (err) {
      clearError = err instanceof Error ? err.message : 'Could not clear the model cache.';
      return;
    } finally {
      clearing = false;
    }
    // A failed refetch after a successful clear must not report the clear as failed.
    try {
      await loadStats();
    } catch (err) {
      console.error('StorageSection: stats refetch after clear failed', err);
    }
  }

  async function pickFolder(title: string): Promise<string | null> {
    try {
      return await open({ directory: true, multiple: false, title });
    } catch {
      return null;
    }
  }

  async function handleMoveDataClick(): Promise<void> {
    relocateError = null;
    const dir = await pickFolder('Choose a new data location');
    if (!dir) return;
    pendingNewDataPath = dir;
    moveConfirmOpen = true;
  }

  async function handleMoveConfirmed(): Promise<void> {
    relocating = true;
    relocateError = null;
    try {
      await invoke<void>('relocate_data_dir', { new_path: pendingNewDataPath });
      moveConfirmOpen = false;
      restartDialogOpen = true;
    } catch (err) {
      relocateError = err instanceof Error ? err.message : 'Could not move the data folder.';
    } finally {
      relocating = false;
    }
  }

  async function handleRestartNow(): Promise<void> {
    try {
      // AppHandle::restart diverges (the process relaunches) — a resolved promise
      // here only ever means the restart itself failed to kick off.
      await invoke<void>('restart_app');
    } catch (err) {
      relocateError = err instanceof Error ? err.message : 'Could not restart the app.';
    }
  }

  async function handleMoveCacheClick(): Promise<void> {
    offloadError = null;
    const dir = await pickFolder('Choose a new model cache location');
    if (!dir) return;
    offloading = true;
    offloadedBytes = null;
    try {
      const moved = await invoke<number>('offload_cache', { new_path: dir });
      cacheDir = dir;
      offloadedBytes = moved;
      cacheRestartOpen = true;
    } catch (err) {
      offloadError = err instanceof Error ? err.message : 'Could not move the model cache.';
      return;
    } finally {
      offloading = false;
    }
    // A failed refetch after a successful move must not report the move as failed.
    try {
      await loadStats();
    } catch (err) {
      console.error('StorageSection: stats refetch after cache move failed', err);
    }
  }

  async function handleResetCacheLocation(): Promise<void> {
    offloadError = null;
    resetting = true;
    try {
      const moved = await invoke<number>('reset_cache_location');
      cacheDir = null;
      offloadedBytes = moved;
      cacheRestartOpen = true;
    } catch (err) {
      offloadError = err instanceof Error ? err.message : 'Could not reset the cache location.';
      return;
    } finally {
      resetting = false;
    }
    // A failed refetch after a successful reset must not report the reset as failed.
    try {
      await loadStats();
    } catch (err) {
      console.error('StorageSection: stats refetch after cache reset failed', err);
    }
  }

  function quotaGbToBytes(gb: number | null): number | null {
    return gb != null && Number.isFinite(gb) && gb > 0 ? Math.round(gb * GB) : null;
  }

  async function handleQuotaBlur(): Promise<void> {
    const next = quotaGbToBytes(quotaGbInput);
    if (next === quotaBytes) return;
    quotaError = null;
    const previous = quotaBytes;
    quotaBytes = next;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        storage: { ...cfg.storage, cache_quota_bytes: next }
      }));
    } catch (err) {
      quotaError = err instanceof Error ? err.message : 'Could not save the cache limit.';
      quotaBytes = previous;
      quotaGbInput = previous != null ? previous / GB : null;
    }
  }

  onDestroy(() => {
    if (copiedTimer) clearTimeout(copiedTimer);
  });
</script>

<section class="flex flex-col" aria-label="Storage settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Storage</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Everything lives on-device. Manage where your data is kept and reclaim downloaded models.
  </p>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Data location
    </p>

    <div
      class="mt-3 flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">Notebook data folder</span>
        <span class="mt-0.5 block truncate text-[0.68rem] text-muted-foreground" title={dataDir}>
          {dataDir || '—'}
        </span>
      </span>
      <div class="flex shrink-0 items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onclick={handleReveal}
          disabled={!dataDir}
          aria-label="Reveal data folder in Finder"
        >
          <FolderOpen class="size-3.5" />
          Reveal in Finder
        </Button>
        <Button
          variant="outline"
          size="sm"
          onclick={handleCopy}
          disabled={!dataDir}
          aria-label="Copy data folder path"
        >
          <Copy class="size-3.5" />
          {copied ? 'Copied' : 'Copy path'}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onclick={handleMoveDataClick}
          disabled={!dataDir || relocating || generating}
          aria-label="Move data folder"
        >
          {#if relocating}
            <LoaderCircle class="size-3.5 animate-spin" />
            Moving…
          {:else}
            <FolderInput class="size-3.5" />
            Move…
          {/if}
        </Button>
      </div>
    </div>
    {#if revealError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{revealError}</p>
    {/if}
    {#if generating}
      <p class="mt-2 text-[0.72rem] text-muted-foreground">
        Unavailable while an audio overview is generating.
      </p>
    {/if}
    {#if relocateError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{relocateError}</p>
    {/if}
  </div>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Usage
    </p>

    <div class="mt-3 flex flex-col gap-2">
      <div class="rounded-[10px] border border-border bg-card">
        <div class="flex items-center justify-between gap-4 px-4 py-3.5">
          <span class="min-w-0 flex-1">
            <span class="block text-[0.78rem] font-bold text-foreground">Notebook corpus</span>
            <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
              Your notebooks, sources, and generated audio.
            </span>
          </span>
          <span class="shrink-0 text-[0.9rem] font-bold tabular-nums text-foreground">
            {formatBytes(stats?.corpus_bytes ?? 0)}
          </span>
        </div>
        {#if stats}
          <div class="flex flex-col gap-1.5 border-t border-border/60 px-4 py-3">
            {#each corpusBreakdown as bucket (bucket.label)}
              <div class="flex items-center gap-3">
                <span class="w-16 shrink-0 text-[0.68rem] text-muted-foreground">
                  {bucket.label}
                </span>
                <span class="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                  <span
                    class="block h-full rounded-full bg-primary/50"
                    style="width: {bucketPct(bucket.value)}%"
                  ></span>
                </span>
                <span
                  class="w-16 shrink-0 text-right text-[0.72rem] font-semibold tabular-nums text-foreground/80"
                >
                  {formatBytes(bucket.value)}
                </span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <div
        class="flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
      >
        <span class="min-w-0 flex-1">
          <span class="block text-[0.78rem] font-bold text-foreground">Reclaimable model cache</span
          >
          <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
            Downloaded voice, ASR, and inactive embedding models. Safe to clear — they re-download
            on next use.
          </span>
        </span>
        <span class="shrink-0 text-[0.9rem] font-bold tabular-nums text-foreground">
          {formatBytes(stats?.reclaimable_cache_bytes ?? 0)}
        </span>
      </div>

      {#if stats}
        <div
          class="flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
        >
          <span class="min-w-0 flex-1">
            <span class="block text-[0.78rem] font-bold text-muted-foreground">Required (kept)</span
            >
            <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
              Your active embedding model and the model catalog — never cleared.
            </span>
          </span>
          <span class="shrink-0 text-[0.85rem] font-semibold tabular-nums text-muted-foreground">
            {formatBytes(stats.retained_bytes)}
          </span>
        </div>
      {/if}
    </div>

    {#if loadError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{loadError}</p>
    {/if}

    <Button
      variant="destructive"
      class="mt-4 h-10 w-full"
      onclick={() => (confirmOpen = true)}
      disabled={clearing || !stats || generating}
    >
      {clearing ? 'Clearing…' : 'Clear reclaimable model cache'}
    </Button>

    {#if clearError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{clearError}</p>
    {/if}
  </div>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Model cache location
    </p>

    <div
      class="mt-3 flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">Downloaded model cache</span>
        <span
          class="mt-0.5 block truncate text-[0.68rem] text-muted-foreground"
          title={cacheDir ?? undefined}
        >
          {cacheDir || 'Default (with your data)'}
        </span>
      </span>
      <div class="flex shrink-0 items-center gap-2">
        {#if cacheDir}
          <Button
            variant="outline"
            size="sm"
            onclick={handleResetCacheLocation}
            disabled={offloading || resetting || generating}
          >
            {resetting ? 'Resetting…' : 'Reset to default'}
          </Button>
        {/if}
        <Button
          variant="outline"
          size="sm"
          onclick={handleMoveCacheClick}
          disabled={offloading || resetting || generating}
        >
          {#if offloading}
            <LoaderCircle class="size-3.5 animate-spin" />
            Moving…
          {:else}
            <HardDriveDownload class="size-3.5" />
            Move cache…
          {/if}
        </Button>
      </div>
    </div>
    <p class="mt-2 text-[0.68rem] text-muted-foreground">
      Text-to-speech voices use the new location after the next restart.
    </p>
    {#if offloadedBytes != null}
      <p class="mt-2 text-[0.72rem] text-foreground" role="status">
        Moved {formatBytes(offloadedBytes)}.
      </p>
    {/if}
    {#if generating}
      <p class="mt-2 text-[0.72rem] text-muted-foreground">
        Unavailable while an audio overview is generating.
      </p>
    {/if}
    {#if offloadError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{offloadError}</p>
    {/if}
  </div>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Cache limit
    </p>

    <div
      class="mt-3 flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">
          Soft limit for the model cache
        </span>
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          Warns you when downloaded models grow past this size. Nothing is deleted automatically.
          Leave empty for no limit.
        </span>
      </span>
      <div class="flex shrink-0 items-center gap-1.5">
        <Input
          type="number"
          step="0.5"
          min="0"
          inputmode="decimal"
          class="h-8 w-20 text-right tabular-nums"
          placeholder="No limit"
          bind:value={quotaGbInput}
          onblur={() => void handleQuotaBlur()}
          aria-label="Model cache limit in gigabytes"
        />
        <span class="text-[0.72rem] text-muted-foreground">GB</span>
      </div>
    </div>

    {#if quotaError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{quotaError}</p>
    {/if}

    {#if overQuota}
      <div
        class="mt-3 flex items-start gap-2.5 rounded-lg border border-amber-500/30 bg-amber-500/15 px-3 py-2.5"
      >
        <TriangleAlert class="mt-0.5 size-4 shrink-0 text-amber-500" />
        <span class="min-w-0 flex-1 text-[0.78rem] leading-relaxed text-amber-500">
          Model cache exceeds your limit ({formatBytes(stats?.reclaimable_cache_bytes ?? 0)} of {formatBytes(
            quotaBytes ?? 0
          )}).
        </span>
        <Button
          variant="outline"
          size="sm"
          class="shrink-0 border-amber-500/40 text-amber-600 hover:bg-amber-500/10 dark:text-amber-400"
          onclick={() => (confirmOpen = true)}
          disabled={clearing || !stats || generating}
        >
          Reclaim now
        </Button>
      </div>
    {/if}
  </div>
</section>

<Dialog bind:open={confirmOpen}>
  <DialogContent class="max-w-md">
    <DialogHeader>
      <DialogTitle class="flex items-center gap-2">
        <TriangleAlert class="size-5 text-amber-500" />
        Clear the model cache?
      </DialogTitle>
      <DialogDescription class="leading-relaxed">
        This removes downloaded voice, ASR, and inactive embedding models. They'll re-download
        automatically the next time you use them. Your active embedding model, notebooks, and
        sources are never touched.
      </DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button variant="outline" onclick={() => (confirmOpen = false)}>Cancel</Button>
      <Button
        variant="destructive"
        onclick={handleClearConfirmed}
        aria-label="Confirm clear model cache"
      >
        Clear cache
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>

<Dialog bind:open={moveConfirmOpen}>
  <DialogContent class="max-w-md">
    <DialogHeader>
      <DialogTitle class="flex items-center gap-2">
        <TriangleAlert class="size-5 text-amber-500" />
        Move your data folder?
      </DialogTitle>
      <DialogDescription class="leading-relaxed">
        Lens will copy all notebooks, sources, and settings to {pendingNewDataPath} and verify the copy.
        The app must restart to switch over — the old copy is cleaned up automatically after a successful
        restart.
      </DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button variant="outline" onclick={() => (moveConfirmOpen = false)} disabled={relocating}>
        Cancel
      </Button>
      <Button
        onclick={handleMoveConfirmed}
        disabled={relocating}
        aria-label="Confirm move data folder"
      >
        {relocating ? 'Moving…' : 'Move data'}
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>

<!--
  Mandatory restart: the engine keeps writing to the OLD dir until relaunch, and the
  next boot deletes it — so there is no safe "Later". Gate `open` (ignore any close
  request) and drop the close button / Escape / outside-click affordances.
-->
<Dialog open={restartDialogOpen} onOpenChange={(v) => v && (restartDialogOpen = v)}>
  <DialogContent
    class="max-w-md"
    showCloseButton={false}
    escapeKeydownBehavior="ignore"
    interactOutsideBehavior="ignore"
  >
    <DialogHeader>
      <DialogTitle>Restart to finish the move</DialogTitle>
      <DialogDescription class="leading-relaxed">
        Your data was copied to the new location. Lens must restart now to switch over and clean up
        the old copy — until it does, changes still write to the previous folder.
      </DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button onclick={handleRestartNow} aria-label="Restart now">Restart now</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>

<Dialog bind:open={cacheRestartOpen}>
  <DialogContent class="max-w-md">
    <DialogHeader>
      <DialogTitle>Cache moved</DialogTitle>
      <DialogDescription class="leading-relaxed">
        Embedding and speech-recognition models use the new location right away. Restart to finish
        moving your text-to-speech voices.
      </DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button variant="outline" onclick={() => (cacheRestartOpen = false)}>Later</Button>
      <Button onclick={handleRestartNow} aria-label="Restart now">Restart now</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
