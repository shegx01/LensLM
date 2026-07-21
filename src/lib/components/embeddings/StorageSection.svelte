<!--
  StorageSection — the "Storage" panel inside the global Preferences view.
  Read-only data-dir + usage figures via `get_storage_stats`, plus `clear_model_cache`.
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { revealItemInDir } from '@tauri-apps/plugin-opener';
  import { writeText } from '@tauri-apps/plugin-clipboard-manager';
  import { audioOverviewStore } from '$lib/sources/audio-state.svelte.js';
  import { formatBytes } from '$lib/format/bytes.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter
  } from '$lib/components/ui/dialog/index.js';
  import FolderOpen from '@lucide/svelte/icons/folder-open';
  import Copy from '@lucide/svelte/icons/copy';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import type { AppConfig, StorageStats } from '$lib/theme/types.js';

  let dataDir = $state('');
  let stats = $state<StorageStats | null>(null);
  let loadError = $state<string | null>(null);
  let revealError = $state<string | null>(null);
  let copied = $state(false);
  let clearing = $state(false);
  let clearError = $state<string | null>(null);
  let confirmOpen = $state(false);

  let copiedTimer: ReturnType<typeof setTimeout> | null = null;

  async function loadStats(): Promise<void> {
    stats = await invoke<StorageStats>('get_storage_stats');
  }

  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      dataDir = cfg.paths.data_dir;
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
      </div>
    </div>
    {#if revealError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{revealError}</p>
    {/if}
  </div>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Usage
    </p>

    <div class="mt-3 flex flex-col gap-2">
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

      <div
        class="flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
      >
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
      disabled={clearing || !stats || audioOverviewStore.overviewStatus === 'generating'}
    >
      {clearing ? 'Clearing…' : 'Clear reclaimable model cache'}
    </Button>

    {#if audioOverviewStore.overviewStatus === 'generating'}
      <p class="mt-2 text-[0.72rem] text-muted-foreground">
        Unavailable while an audio overview is generating.
      </p>
    {/if}

    {#if clearError}
      <p class="mt-2 text-[0.72rem] text-destructive" role="alert">{clearError}</p>
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
