<!-- SourcesRail fills the 320px right <aside> in AppShell.
     Displays all sources for the active notebook with per-source status pills,
     an "Add source" control (file picker + paste-text dialog), and a counter badge.
     All colours are CSS-variable tokens — no hardcoded hex. -->
<script lang="ts">
  import { open as openDialog } from '@tauri-apps/plugin-dialog';
  import { isTauri } from '@tauri-apps/api/core';
  import Plus from '@lucide/svelte/icons/plus';
  import File from '@lucide/svelte/icons/file';
  import Check from '@lucide/svelte/icons/check';
  import X from '@lucide/svelte/icons/x';
  import { cn } from '$lib/utils.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    sourcesStore,
    loadSources,
    ingest,
    toggleSelected
  } from '$lib/sources/sources-state.svelte.js';
  import { addFileSource, addTextSource } from '$lib/sources/ipc.js';
  import { notebookStore } from '$lib/notebooks/notebooks-state.svelte.js';
  import type { SourceStatus } from '$lib/sources/types.js';

  // ---------------------------------------------------------------------------
  // Local state
  // ---------------------------------------------------------------------------

  /** Controls the "Add source" dropdown menu visibility */
  let menuOpen = $state(false);

  /** Controls the paste-text dialog visibility */
  let pasteDialogOpen = $state(false);

  /** Paste-text dialog fields */
  let pasteTitle = $state('');
  let pasteText = $state('');
  let pasteError = $state<string | null>(null);
  let pasteSubmitting = $state(false);

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------

  const activeNotebookId = $derived(notebookStore.activeNotebookId);
  const sources = $derived(sourcesStore.sources);
  const sourceCount = $derived(sources.length);

  // ---------------------------------------------------------------------------
  // Status pill helpers — token-only colours
  // ---------------------------------------------------------------------------

  function statusPillClass(status: SourceStatus): string {
    switch (status) {
      case 'indexed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300';
      case 'parsing':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300';
      case 'embedding':
        return 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300';
      case 'queued':
      case 'pending':
        return 'bg-muted text-muted-foreground';
      case 'error':
        return 'bg-destructive/10 text-destructive dark:bg-destructive/20';
      default:
        return 'bg-muted text-muted-foreground';
    }
  }

  function statusLabel(status: SourceStatus): string {
    switch (status) {
      case 'indexed':
        return 'Indexed';
      case 'parsing':
        return 'Parsing…';
      case 'embedding':
        return 'Embedding…';
      case 'queued':
        return 'Queued';
      case 'pending':
        return 'Pending';
      case 'error':
        return 'Error';
      default:
        return status;
    }
  }

  // ---------------------------------------------------------------------------
  // File picker
  // ---------------------------------------------------------------------------

  async function browseFile(): Promise<void> {
    menuOpen = false;
    if (!isTauri() || !activeNotebookId) return;
    try {
      const selected = await openDialog({
        multiple: false,
        filters: [{ name: 'Documents', extensions: ['md', 'txt'] }]
      });
      if (!selected) return;
      const path = Array.isArray(selected) ? selected[0] : selected;
      const name = path.split('/').pop() ?? path;
      const source = await addFileSource(activeNotebookId, name, path);
      await loadSources(activeNotebookId);
      void ingest(source.id);
    } catch (err) {
      console.error('SourcesRail: browseFile failed', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Paste-text dialog
  // ---------------------------------------------------------------------------

  function openPasteDialog(): void {
    menuOpen = false;
    pasteTitle = '';
    pasteText = '';
    pasteError = null;
    pasteSubmitting = false;
    pasteDialogOpen = true;
  }

  function closePasteDialog(): void {
    pasteDialogOpen = false;
  }

  async function submitPaste(): Promise<void> {
    if (!activeNotebookId) return;
    if (!pasteTitle.trim()) {
      pasteError = 'Please enter a title.';
      return;
    }
    if (!pasteText.trim()) {
      pasteError = 'Please paste some text.';
      return;
    }
    pasteError = null;
    pasteSubmitting = true;
    try {
      const source = await addTextSource(
        activeNotebookId,
        pasteTitle.trim(),
        pasteText.trim(),
        'text'
      );
      pasteDialogOpen = false;
      await loadSources(activeNotebookId);
      void ingest(source.id);
    } catch (err) {
      pasteError = 'Could not add source. Please try again.';
      console.error('SourcesRail: submitPaste failed', err);
    } finally {
      pasteSubmitting = false;
    }
  }
</script>

<!-- Right aside header: drag region with "Sources" title + counter badge + Add button -->
<div data-tauri-drag-region class="flex h-[var(--titlebar-h)] shrink-0 items-center gap-2 px-4">
  <span class="flex-1 text-xs font-semibold tracking-wide text-foreground">Sources</span>
  {#if sourceCount > 0}
    <span
      class="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-muted px-1.5 text-[10px] font-semibold text-muted-foreground"
    >
      {sourceCount}
    </span>
  {/if}

  <!-- Add source button + dropdown -->
  <div class="relative">
    <Button
      variant="ghost"
      class="h-[26px] w-[26px] rounded-md p-0 text-muted-foreground hover:bg-muted hover:text-foreground"
      onclick={() => (menuOpen = !menuOpen)}
      aria-label="Add source"
      aria-expanded={menuOpen}
    >
      <Plus class="size-[14px]" strokeWidth={2} />
    </Button>

    {#if menuOpen}
      <!-- Click-away backdrop -->
      <button
        class="fixed inset-0 z-10 cursor-default"
        aria-hidden="true"
        onclick={() => (menuOpen = false)}
        tabindex="-1"
        type="button"
      ></button>
      <!-- Dropdown menu -->
      <div
        class="absolute right-0 top-full z-20 mt-1 w-44 overflow-hidden rounded-lg border border-border bg-popover py-1 shadow-lg"
        role="menu"
      >
        <button
          class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-[12px] font-medium text-popover-foreground hover:bg-muted"
          onclick={browseFile}
          type="button"
          role="menuitem"
        >
          <File class="size-[13px] shrink-0 text-muted-foreground" strokeWidth={1.75} />
          Browse file…
        </button>
        <button
          class="flex w-full items-center gap-2.5 px-3 py-2 text-left text-[12px] font-medium text-popover-foreground hover:bg-muted"
          onclick={openPasteDialog}
          type="button"
          role="menuitem"
        >
          <Plus class="size-[13px] shrink-0 text-muted-foreground" strokeWidth={1.75} />
          Paste text…
        </button>
      </div>
    {/if}
  </div>
</div>

<!-- Divider -->
<div class="shrink-0 border-t border-border"></div>

<!-- Scrollable source list -->
<div class="flex flex-1 flex-col overflow-y-auto">
  {#if sources.length === 0}
    <!-- Empty state -->
    <div class="flex flex-1 flex-col items-center justify-center gap-2 px-4 py-10">
      <File class="size-8 text-muted-foreground/30" strokeWidth={1.25} />
      <p class="text-center text-[12px] font-medium text-muted-foreground">No sources yet</p>
      <p class="text-center text-[11px] text-muted-foreground/60">
        Add a file or paste text to ground this notebook.
      </p>
    </div>
  {:else}
    <ul class="flex flex-col gap-px p-2">
      {#each sources as source (source.id)}
        <li
          class="group flex items-center gap-2.5 rounded-lg px-2.5 py-2 hover:bg-muted/50 transition-colors duration-100"
        >
          <!-- File icon -->
          <div class="flex size-7 shrink-0 items-center justify-center rounded-[6px] bg-muted">
            <File class="size-3 text-muted-foreground" strokeWidth={1.75} />
          </div>

          <!-- Title + status -->
          <div class="min-w-0 flex-1">
            <div class="truncate text-[12px] font-medium text-foreground">
              {source.title}
            </div>
            <div class="mt-[2px]">
              <span
                class={cn(
                  'inline-flex items-center rounded-[4px] px-[6px] py-[1px] text-[10px] font-semibold',
                  statusPillClass(source.status as SourceStatus)
                )}
              >
                {statusLabel(source.status as SourceStatus)}
              </span>
            </div>
          </div>

          <!-- Select / deselect toggle -->
          <button
            class={cn(
              'flex size-[18px] shrink-0 cursor-pointer items-center justify-center rounded-[5px] transition-all duration-[130ms] border',
              source.selected === 1
                ? 'border-primary bg-primary'
                : 'border-border bg-transparent hover:border-muted-foreground'
            )}
            onclick={() => void toggleSelected(source.id)}
            type="button"
            aria-label={source.selected === 1
              ? `Deselect source ${source.title}`
              : `Select source ${source.title}`}
            aria-pressed={source.selected === 1}
          >
            {#if source.selected === 1}
              <Check class="size-[11px] text-primary-foreground" strokeWidth={3} />
            {/if}
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<!-- Paste-text dialog -->
{#if pasteDialogOpen}
  <!-- Modal backdrop -->
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
    role="dialog"
    aria-modal="true"
    aria-label="Add text source"
  >
    <div
      class="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-2xl"
      role="document"
    >
      <!-- Dialog header -->
      <div class="mb-4 flex items-center justify-between">
        <h2 class="text-[15px] font-semibold text-foreground">Paste text</h2>
        <button
          class="flex size-7 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          onclick={closePasteDialog}
          type="button"
          aria-label="Close"
        >
          <X class="size-[14px]" strokeWidth={2} />
        </button>
      </div>

      <!-- Title input -->
      <div class="mb-3">
        <label
          class="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground"
          for="paste-title"
        >
          Title
        </label>
        <input
          id="paste-title"
          class="w-full rounded-md border border-border bg-background px-3 py-2 text-[13px] text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-ring"
          type="text"
          placeholder="My notes…"
          bind:value={pasteTitle}
          disabled={pasteSubmitting}
        />
      </div>

      <!-- Text area -->
      <div class="mb-4">
        <label
          class="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground"
          for="paste-text"
        >
          Content
        </label>
        <textarea
          id="paste-text"
          class="h-[160px] w-full resize-none rounded-md border border-border bg-background px-3 py-2 text-[13px] text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-ring"
          placeholder="Paste or type your text here…"
          bind:value={pasteText}
          disabled={pasteSubmitting}
        ></textarea>
      </div>

      <!-- Error -->
      {#if pasteError}
        <p class="mb-3 text-[12px] text-destructive" role="alert">{pasteError}</p>
      {/if}

      <!-- Actions -->
      <div class="flex justify-end gap-2">
        <Button
          variant="outline"
          class="h-[36px] text-[13px]"
          onclick={closePasteDialog}
          disabled={pasteSubmitting}
        >
          Cancel
        </Button>
        <Button
          class="h-[36px] text-[13px] font-semibold"
          onclick={submitPaste}
          disabled={pasteSubmitting || !pasteTitle.trim() || !pasteText.trim()}
        >
          {pasteSubmitting ? 'Adding…' : 'Add source'}
        </Button>
      </div>
    </div>
  </div>
{/if}
