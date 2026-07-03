<!--
  EmbeddingsInspector — dev/QA-only, read-only split-view overlay for the M4
  extract→chunk→enrich→embed→LanceDB pipeline.

  Mounted at shell root (DEV-gated dynamic import in AppShell), toggled by
  `notebookStore.inspectorOpen`. Built as a shadcn <Dialog> so Escape /
  focus-trap / portal / aria-modal come free — same pattern as TrashView.

  LEFT pane: the active notebook's sources (SourceRow). RIGHT pane: the selected
  source's chunks (ChunkCard) + per-model embedding stats. This component is the
  layout shell + fetch orchestration; row/card rendering lives in the children.

  Error affordances (#73):
  - SourceRow gets an `onretry` prop so the row can trigger per-source retry.
  - Right pane renders a destructive alert block for an errored selected source.
  - "Retry all failed" button in the right-pane header retries every error source.
-->
<script lang="ts">
  import Loader from '@lucide/svelte/icons/loader-circle';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import { notebookStore } from '$lib/notebooks/index.js';
  import { sourcesStore, retrySource, retryAllFailed } from '$lib/sources/sources-state.svelte.js';
  import { listSourceChunks } from '$lib/inspector/ipc.js';
  import type { InspectorChunk, EmbeddingStats } from '$lib/inspector/types.js';
  import { Dialog, DialogContent } from '$lib/components/ui/dialog/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import SourceRow from './SourceRow.svelte';
  import ChunkCard from './ChunkCard.svelte';

  // Component-local state (NOT in any store — per consensus fix #6).
  const open = $derived(notebookStore.inspectorOpen);

  /** The id of the source whose chunks are shown in the right pane. */
  let selectedSourceId = $state<string | null>(null);
  let chunks = $state<InspectorChunk[]>([]);
  let stats = $state<EmbeddingStats[]>([]);
  let loading = $state(false);
  let errorMsg = $state<string | null>(null);

  const sources = $derived(sourcesStore.sources);
  const selectedSource = $derived(sources.find((s) => s.id === selectedSourceId) ?? null);
  const hasFailedSources = $derived(sources.some((s) => s.status === 'error' && !s.trashed_at));

  /**
   * Load a source's chunks + stats into the right pane.
   *
   * Guards against a stale-response race: if the user clicks another source
   * while this request is in flight, `selectedSourceId` has moved on by the time
   * we resolve, so we drop the late result instead of overwriting the newer one.
   */
  async function selectSource(sourceId: string): Promise<void> {
    selectedSourceId = sourceId;
    loading = true;
    errorMsg = null;
    try {
      const res = await listSourceChunks(sourceId, notebookStore.activeNotebookId ?? '');
      if (selectedSourceId !== sourceId) return; // a newer selection won — drop this one
      chunks = res.chunks;
      stats = res.stats;
    } catch (err) {
      if (selectedSourceId !== sourceId) return;
      console.error('EmbeddingsInspector: listSourceChunks failed', err);
      errorMsg = String(err);
      chunks = [];
      stats = [];
    } finally {
      if (selectedSourceId === sourceId) loading = false;
    }
  }

  async function handleRetrySource(sourceId: string): Promise<void> {
    await retrySource(sourceId);
    // Reload chunks for the retried source if it is still selected.
    if (selectedSourceId === sourceId) {
      await selectSource(sourceId);
    }
  }

  async function handleRetryAllFailed(): Promise<void> {
    const notebookId = notebookStore.activeNotebookId;
    if (!notebookId) return;
    await retryAllFailed(notebookId);
  }
</script>

<Dialog
  {open}
  onOpenChange={(v) => {
    if (!v) notebookStore.inspectorOpen = false;
  }}
>
  <DialogContent
    class="flex h-[calc(100vh-4rem)] w-[calc(100vw-4rem)] max-w-none flex-row gap-0 overflow-hidden p-0 sm:max-w-none"
    aria-label="Embeddings Inspector"
    data-embeddings-inspector
  >
    <!-- ── LEFT pane: sources ──────────────────────────────────────────────── -->
    <div class="flex w-[280px] shrink-0 flex-col border-r border-border bg-card">
      <div class="shrink-0 px-4 pt-5 pb-3">
        <p class="text-sm font-bold tracking-[-0.3px] text-foreground">Embeddings Inspector</p>
        <p class="mt-0.5 text-[11px] text-muted-foreground">Dev/QA — read-only pipeline view</p>
      </div>
      <div class="shrink-0 border-t border-border"></div>
      <ScrollArea class="min-h-0 flex-1">
        <div class="p-2">
          {#if sources.length === 0}
            <p class="px-2 py-6 text-center text-xs text-muted-foreground">No sources</p>
          {:else}
            <ul role="list" aria-label="Sources" class="flex flex-col gap-px">
              {#each sources as source (source.id)}
                <SourceRow
                  {source}
                  selected={selectedSourceId === source.id}
                  onselect={() => void selectSource(source.id)}
                  onretry={source.status === 'error'
                    ? () => void handleRetrySource(source.id)
                    : undefined}
                />
              {/each}
            </ul>
          {/if}
        </div>
      </ScrollArea>
    </div>

    <!-- ── RIGHT pane: chunks + stats ──────────────────────────────────────── -->
    <div class="flex min-w-0 flex-1 flex-col bg-background">
      <!-- Header -->
      <div class="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-5 py-4">
        {#if selectedSource}
          <span class="text-sm font-semibold text-foreground">{selectedSource.title}</span>
          <span class="text-xs text-muted-foreground tabular-nums">
            {chunks.length} chunk{chunks.length === 1 ? '' : 's'}
          </span>
          <div class="flex-1"></div>
          {#if stats.length === 0}
            <span class="text-xs text-muted-foreground italic">Not yet embedded</span>
          {:else}
            {#each stats as stat (stat.model + '-' + stat.dim)}
              <Badge variant="secondary" class="text-[0.6875rem]">
                {stat.model} · {stat.dim}d
              </Badge>
            {/each}
          {/if}
        {:else}
          <span class="text-sm text-muted-foreground">Select a source to inspect its chunks</span>
          <div class="flex-1"></div>
        {/if}

        <!-- "Retry all failed" — visible whenever any source in the notebook is in error. -->
        {#if hasFailedSources}
          <button
            type="button"
            aria-label="Retry all failed sources"
            data-retry-all-failed-btn
            onclick={() => void handleRetryAllFailed()}
            class="flex items-center gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-2.5 py-1 text-[0.6875rem] font-medium text-destructive transition-colors hover:bg-destructive/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <RotateCcw class="size-3 shrink-0" strokeWidth={2} />
            Retry all failed
          </button>
        {/if}
      </div>

      <!-- Body -->
      <ScrollArea class="min-h-0 flex-1">
        <div class="p-5">
          {#if loading}
            <div
              role="status"
              aria-label="Loading chunks"
              class="flex flex-col items-center justify-center gap-2 py-16 text-muted-foreground"
            >
              <Loader class="size-5 animate-spin" />
              <span class="text-xs">Loading chunks…</span>
            </div>
          {:else if errorMsg}
            <div
              role="alert"
              class="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
            >
              {errorMsg}
            </div>
          {:else if selectedSource && selectedSource.status === 'error'}
            <!-- Ingest error alert block — mirrors the destructive alert pattern
                 at EmbeddingsInspector.svelte:139-145 (errorMsg branch above). -->
            <div
              role="alert"
              data-source-error-alert
              class="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3"
            >
              <p class="text-sm font-medium text-destructive">
                {selectedSource.error_meta
                  ? selectedSource.error_meta.message
                  : 'Ingest failed (no details captured)'}
              </p>
              {#if selectedSource.error_meta}
                <p class="mt-1 text-xs text-destructive/70">
                  Error kind: {selectedSource.error_meta.kind}
                  {#if selectedSource.error_meta.attempt_count > 1}
                    · {selectedSource.error_meta.attempt_count} attempts
                  {/if}
                  {#if selectedSource.error_meta.timestamp}
                    · {new Date(selectedSource.error_meta.timestamp).toLocaleString()}
                  {/if}
                </p>
              {/if}
              <button
                type="button"
                aria-label="Retry ingesting {selectedSource.title}"
                data-retry-source-btn
                onclick={() => void handleRetrySource(selectedSource.id)}
                class="mt-3 flex items-center gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-1.5 text-xs font-medium text-destructive transition-colors hover:bg-destructive/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <RotateCcw class="size-3 shrink-0" strokeWidth={2} />
                Retry
              </button>
            </div>
          {:else if selectedSourceId === null}
            <p class="py-16 text-center text-sm text-muted-foreground">
              Select a source from the left to view its chunks.
            </p>
          {:else if chunks.length === 0}
            <p class="py-16 text-center text-sm text-muted-foreground">No chunks found</p>
          {:else}
            <ul role="list" aria-label="Chunks" class="flex flex-col gap-3">
              {#each chunks as chunk (chunk.id)}
                <ChunkCard {chunk} />
              {/each}
            </ul>
          {/if}
        </div>
      </ScrollArea>
    </div>
  </DialogContent>
</Dialog>
