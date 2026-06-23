<script lang="ts">
  import { onMount } from 'svelte';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import BookOpen from '@lucide/svelte/icons/book-open';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import { cn } from '$lib/utils.js';
  import {
    notebookStore,
    loadTrashed,
    restoreNotebookAction,
    purgeNotebookAction,
    notebookAccentClass,
    formatRelativeTime
  } from '$lib/notebooks/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
    DialogDescription
  } from '$lib/components/ui/dialog/index.js';

  // ---------------------------------------------------------------------------
  // Confirm-dialog state — one pending purge at a time
  // ---------------------------------------------------------------------------

  /** The id of the notebook awaiting a purge confirmation, or null if none. */
  let pendingPurgeId = $state<string | null>(null);

  /** The title of the notebook awaiting purge (used in the dialog copy). */
  const pendingPurgeTitle = $derived(
    pendingPurgeId
      ? (notebookStore.trashedNotebooks.find((n) => n.id === pendingPurgeId)?.title ?? '')
      : ''
  );

  /** Whether the confirm dialog is open. Synced with pendingPurgeId. */
  const confirmOpen = $derived(pendingPurgeId !== null);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  function openConfirm(id: string): void {
    pendingPurgeId = id;
  }

  function closeConfirm(): void {
    pendingPurgeId = null;
  }

  async function handlePurge(): Promise<void> {
    if (!pendingPurgeId) return;
    const id = pendingPurgeId;
    pendingPurgeId = null;
    await purgeNotebookAction(id);
  }

  function goBack(): void {
    notebookStore.viewMode = 'notebook';
  }

  // ---------------------------------------------------------------------------
  // Load trashed list on mount
  // ---------------------------------------------------------------------------

  onMount(() => {
    void loadTrashed();
  });

  // ---------------------------------------------------------------------------
  // Derived display helpers
  // ---------------------------------------------------------------------------

  function sourcesLabel(count: number): string {
    return count === 1 ? '1 source' : `${count} sources`;
  }
</script>

<!--
  TrashView — center-pane content for viewMode === 'trash'.
  Lists trashed notebooks with Restore and Delete-forever actions.
  Delete-forever requires explicit confirmation via a Dialog (no AlertDialog
  primitive exists; we compose Dialog + destructive Button directly, per plan §Risk).
-->
<div class="flex h-full flex-col">
  <!-- ── Header ────────────────────────────────────────────────────────────── -->
  <div class="flex shrink-0 items-center gap-3 border-b border-border px-6 py-4" data-trash-header>
    <Button
      variant="ghost"
      size="icon-sm"
      onclick={goBack}
      aria-label="Back to notebooks"
      data-back-btn
    >
      <ArrowLeft class="size-4" />
    </Button>

    <div class="min-w-0 flex-1">
      <h1 class="text-base font-semibold text-foreground">Trash</h1>
      <p class="text-xs text-muted-foreground">Restore notebooks or delete them permanently</p>
    </div>
  </div>

  <!-- ── Body ─────────────────────────────────────────────────────────────── -->
  <ScrollArea class="min-h-0 flex-1">
    <div class="px-4 py-3" data-trash-list>
      {#if notebookStore.trashedNotebooks.length === 0}
        <!-- Empty state -->
        <div
          class="flex flex-col items-center justify-center gap-3 py-16 text-center"
          data-empty-state
        >
          <div
            class="flex size-12 items-center justify-center rounded-xl bg-muted/60 text-muted-foreground"
          >
            <Trash2 class="size-6" />
          </div>
          <p class="text-sm font-medium text-muted-foreground">Trash is empty</p>
        </div>
      {:else}
        <ul role="list" class="flex flex-col gap-1">
          {#each notebookStore.trashedNotebooks as notebook (notebook.id)}
            {@const accentClass = notebookAccentClass(notebook.id)}
            {@const relTime = notebook.trashed_at ? formatRelativeTime(notebook.trashed_at) : ''}

            <li
              class={cn(
                'flex items-center gap-3 rounded-lg px-3 py-3',
                'hover:bg-muted/40 transition-colors'
              )}
              data-trash-row
            >
              <!-- Color icon -->
              <div
                class={cn(
                  'flex size-9 shrink-0 items-center justify-center rounded-lg',
                  accentClass
                )}
                aria-hidden="true"
              >
                <BookOpen class="size-4" />
              </div>

              <!-- Title + subtitle -->
              <div class="min-w-0 flex-1">
                <p
                  class="truncate text-sm font-medium leading-tight text-foreground"
                  title={notebook.title}
                >
                  {notebook.title}
                </p>
                <p class="truncate text-xs leading-tight text-muted-foreground mt-0.5">
                  {sourcesLabel(notebook.source_count)} · trashed {relTime}
                </p>
              </div>

              <!-- Actions -->
              <div class="flex shrink-0 items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  onclick={() => void restoreNotebookAction(notebook.id)}
                  aria-label={`Restore ${notebook.title}`}
                  data-restore-btn
                >
                  <RotateCcw class="size-3.5" />
                  Restore
                </Button>

                <Button
                  variant="destructive"
                  size="sm"
                  onclick={() => openConfirm(notebook.id)}
                  aria-label={`Delete ${notebook.title} forever`}
                  data-delete-forever-btn
                >
                  <Trash2 class="size-3.5" />
                  Delete forever
                </Button>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </ScrollArea>
</div>

<!-- ── Confirm purge Dialog ──────────────────────────────────────────────────
  There is no AlertDialog primitive — built from Dialog + destructive Button.
  The `open` binding drives visibility via `pendingPurgeId`.
  `showCloseButton=false` keeps the header clean; Cancel/Delete-forever buttons
  in the footer are the only affordances.
-->
<Dialog
  open={confirmOpen}
  onOpenChange={(v) => {
    if (!v) closeConfirm();
  }}
>
  <DialogContent showCloseButton={false} data-confirm-dialog>
    <DialogHeader>
      <DialogTitle>Delete forever?</DialogTitle>
      <DialogDescription>
        Delete "<span class="font-medium text-foreground">{pendingPurgeTitle}</span>" permanently?
        This removes the notebook and its sources. This cannot be undone.
      </DialogDescription>
    </DialogHeader>

    <DialogFooter class="gap-2 sm:gap-2">
      <Button variant="outline" size="sm" onclick={closeConfirm} data-cancel-btn>Cancel</Button>
      <Button
        variant="destructive"
        size="sm"
        onclick={() => void handlePurge()}
        data-confirm-purge-btn
      >
        <Trash2 class="size-3.5" />
        Delete forever
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
