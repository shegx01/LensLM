<script lang="ts">
  import BookOpen from '@lucide/svelte/icons/book-open';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import { cn } from '$lib/utils.js';
  import {
    notebookStore,
    restoreNotebookAction,
    purgeNotebookAction,
    notebookAccentClass,
    formatRelativeTime,
    formatSourceCount
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
  // Modal open state — driven by the shared store flag (`trashOpen`).
  // ---------------------------------------------------------------------------

  const trashOpen = $derived(notebookStore.trashOpen);

  function closeTrash(): void {
    notebookStore.trashOpen = false;
  }

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
</script>

<!--
  TrashView — centered modal dialog (per design source "Trash" modal).
  Opens when `notebookStore.trashOpen` is true; loaded via `openTrash()`.
  Header: small trash icon + "Trash" title + subtitle, with the shadcn close
  (×) affordance top-right. Body lists trashed notebooks with Restore +
  Delete-forever (confirm). Empty state = centered trash icon + copy.
-->
<Dialog
  open={trashOpen}
  onOpenChange={(v) => {
    if (!v) closeTrash();
  }}
>
  <DialogContent
    class="flex max-h-[640px] flex-col gap-0 overflow-hidden p-0 sm:max-w-[520px]"
    data-trash-modal
  >
    <!-- ── Header ──────────────────────────────────────────────────────────── -->
    <DialogHeader class="shrink-0 space-y-0 px-6 pt-6 pb-4 text-left" data-trash-header>
      <div class="flex items-center gap-2.5">
        <div
          class="flex size-[30px] shrink-0 items-center justify-center rounded-[9px] bg-muted text-muted-foreground"
          aria-hidden="true"
        >
          <Trash2 class="size-3.5" />
        </div>
        <div class="min-w-0">
          <DialogTitle class="text-base font-bold tracking-[-0.3px] text-foreground"
            >Trash</DialogTitle
          >
          <DialogDescription class="mt-px text-[11px] text-muted-foreground">
            Deleted notebooks, sources and notes
          </DialogDescription>
        </div>
      </div>
    </DialogHeader>

    <!-- ── Body ────────────────────────────────────────────────────────────── -->
    <ScrollArea class="min-h-0 flex-1">
      <div class="px-6 pb-4" data-trash-list>
        {#if notebookStore.trashedNotebooks.length === 0}
          <!-- Empty state -->
          <div class="flex flex-col items-center gap-0 py-12 text-center" data-empty-state>
            <div
              class="mb-3.5 flex size-11 items-center justify-center rounded-xl bg-muted text-muted-foreground"
              aria-hidden="true"
            >
              <Trash2 class="size-5" />
            </div>
            <p class="mb-1 text-sm font-semibold text-muted-foreground">Trash is empty</p>
            <p class="text-xs text-muted-foreground">Deleted items will appear here</p>
          </div>
        {:else}
          <ul role="list" class="flex flex-col gap-1">
            {#each notebookStore.trashedNotebooks as notebook (notebook.id)}
              {@const accentClass = notebookAccentClass(notebook.id)}
              {@const relTime = notebook.trashed_at ? formatRelativeTime(notebook.trashed_at) : ''}

              <li
                class={cn('flex items-center gap-2.5 rounded-[10px] bg-muted/60 px-3 py-2.5')}
                data-trash-row
              >
                <!-- Color icon -->
                <div
                  class={cn(
                    'flex size-8 shrink-0 items-center justify-center rounded-lg opacity-70',
                    accentClass
                  )}
                  aria-hidden="true"
                >
                  <BookOpen class="size-3.5" />
                </div>

                <!-- Title + subtitle -->
                <div class="min-w-0 flex-1">
                  <p
                    class="truncate text-[13px] font-semibold leading-tight text-foreground"
                    title={notebook.title}
                  >
                    {notebook.title}
                  </p>
                  <p class="mt-0.5 truncate text-[10px] leading-tight text-muted-foreground">
                    {formatSourceCount(notebook.source_count)} · trashed {relTime}
                  </p>
                </div>

                <!-- Actions -->
                <div class="flex shrink-0 items-center gap-1">
                  <Button
                    variant="default"
                    size="sm"
                    class="h-7 rounded-lg px-2.5 text-[11px] font-semibold"
                    onclick={() => void restoreNotebookAction(notebook.id)}
                    aria-label={`Restore ${notebook.title}`}
                    data-restore-btn
                  >
                    <RotateCcw class="size-3" />
                    Restore
                  </Button>

                  <Button
                    variant="destructive"
                    size="sm"
                    class="h-7 rounded-lg px-2.5 text-[11px] font-semibold"
                    onclick={() => openConfirm(notebook.id)}
                    aria-label={`Delete ${notebook.title} forever`}
                    data-delete-forever-btn
                  >
                    <Trash2 class="size-3" />
                    Delete
                  </Button>
                </div>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    </ScrollArea>
  </DialogContent>
</Dialog>

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

    <!-- No divider: neutralise DialogFooter's baked-in border-t + muted bg. -->
    <DialogFooter class="mx-0 mb-0 gap-2 border-t-0 bg-transparent p-0 pt-2 sm:gap-2">
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
