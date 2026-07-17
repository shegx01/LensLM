<!-- SourcesRail — the right rail. Header is a drag region; all interactive children
     carry -webkit-app-region: no-drag so clicks don't conflict with window drag.
     Motion is gated through --rail-motion / --ease-out (shared with the left rail)
     so it stays consistent and honors the Animations preference.
     All colours are CSS-variable tokens — no hardcoded hex. -->
<script lang="ts" module>
  /** How long (ms) the reveal highlight stays on a focused source row. */
  export const PULSE_MS = 1000;
</script>

<script lang="ts">
  import FileText from '@lucide/svelte/icons/file-text';
  import Check from '@lucide/svelte/icons/check';
  import Minus from '@lucide/svelte/icons/minus';
  import Plus from '@lucide/svelte/icons/plus';
  import Trash from '@lucide/svelte/icons/trash';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import PanelRight from '@lucide/svelte/icons/panel-right';
  import PanelRightClose from '@lucide/svelte/icons/panel-right-close';
  import Headphones from '@lucide/svelte/icons/headphones';
  import { cn } from '$lib/utils.js';
  import { onDestroy, tick, untrack } from 'svelte';
  import { fadeRise, popIn } from '$lib/motion/index.js';
  import {
    sourcesStore,
    toggleSelected,
    toggleAllSelected,
    removeSource,
    undoRemove,
    disposeTrashTimers,
    retrySource
  } from '$lib/sources/sources-state.svelte.js';
  import { notebookStore } from '$lib/notebooks/index.js';
  import type { SourceStatus } from '$lib/sources/types.js';
  import { statusDotClass } from '$lib/sources/status.js';
  import AddSourcesModal from './AddSourcesModal.svelte';
  import StudioPanel from './StudioPanel.svelte';

  let modalOpen = $state(false);

  const collapsed = $derived(notebookStore.rightRailCollapsed);

  // Clear all pending trash-undo timers when the rail unmounts (e.g. notebook
  // switch). Without this, orphan timers fire after unmount and mutate trashQueue
  // state belonging to a different notebook session.
  onDestroy(disposeTrashTimers);

  // Reveal-in-rail (#23b): a citation chip sets sourcesStore.focusedSourceId;
  // scroll that row into view and highlight it briefly. A single stored timer is
  // cleared on re-fire and on destroy so it can't leak across notebook switches.
  let pulsingId = $state<string | null>(null);
  let pulseTimer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    if (sourcesStore.focusedSourceId === null) return;
    // Reading focusNonce keeps it a dependency so re-clicking the same chip re-fires.
    void sourcesStore.focusNonce;
    const targetId = sourcesStore.focusedSourceId;

    // untrack the collapsed read: this effect writes it below, so tracking it would
    // re-run the effect on expand and scroll twice on the collapsed path.
    if (untrack(() => notebookStore.rightRailCollapsed)) {
      notebookStore.rightRailCollapsed = false;
    }

    void tick().then(() => {
      const el = document.querySelector(
        `[data-sources-scroll] [data-source-id="${CSS.escape(targetId)}"]`
      );
      // Cancel any prior pulse before the null-guard so a re-fire onto a missing
      // element still clears the previous timer rather than leaking it.
      clearTimeout(pulseTimer);
      if (!el) return;
      el.scrollIntoView({ block: 'nearest' });
      pulsingId = targetId;
      pulseTimer = setTimeout(() => {
        pulsingId = null;
      }, PULSE_MS);
    });
  });

  onDestroy(() => clearTimeout(pulseTimer));

  function toggleCollapse(): void {
    notebookStore.rightRailCollapsed = !notebookStore.rightRailCollapsed;
  }

  const sources = $derived(sourcesStore.sources);
  const totalCount = $derived(sources.length);
  const selectedCount = $derived(sources.filter((s) => s.selected === 1).length);
  const allSelected = $derived(totalCount > 0 && selectedCount === totalCount);
  const someSelected = $derived(selectedCount > 0 && selectedCount < totalCount);

  /** Short display badge derived from the source's kind + locator/title. */
  function typeBadge(kind: string, locator: string, title: string): string {
    if (kind === 'url') return 'URL';

    const path = locator || title || '';
    const ext = path.split('.').pop()?.toLowerCase() ?? '';

    switch (ext) {
      case 'pdf':
        return 'PDF';
      case 'docx':
      case 'doc':
        return 'DOCX';
      case 'rtf':
        return 'RTF';
      case 'odt':
        return 'ODT';
      case 'epub':
        return 'EPUB';
      case 'md':
      case 'markdown':
        return 'MD';
      case 'txt':
        return 'TXT';
      case 'xlsx':
        return 'XLSX';
      case 'xls':
        return 'XLS';
      case 'csv':
        return 'CSV';
      case 'json':
        return 'JSON';
      case 'jsonl':
        return 'JSONL';
      case 'yaml':
      case 'yml':
        return 'YAML';
      case 'xml':
        return 'XML';
      case 'pptx':
      case 'ppt':
        return 'PPTX';
      case 'mp3':
      case 'wav':
      case 'm4a':
      case 'flac':
      case 'ogg':
      case 'aac':
      case 'opus':
        return 'AUDIO';
      case 'mp4':
      case 'mov':
      case 'webm':
        return 'VIDEO';
      default:
        if (kind === 'text') return 'TXT';
        return 'FILE';
    }
  }

  /** Human-readable metadata line (approximate word count from token_count). */
  // TODO(M6): extract typeBadge + metaLine to src/lib/sources/format.ts when Studio reuses badges.
  function metaLine(tokenCount: number | null): string {
    if (tokenCount !== null && tokenCount > 0) {
      const approxWords = Math.round(tokenCount * 0.75);
      if (approxWords >= 1000) {
        return `~${(approxWords / 1000).toFixed(1)}k words`;
      }
      return `~${approxWords} words`;
    }
    return '';
  }

  // statusDotClass is shared with EmbeddingsInspector — see $lib/sources/status.ts.
  function statusDotLabel(status: SourceStatus): string {
    switch (status) {
      case 'indexed':
        return 'Indexed';
      case 'error':
        return 'Error';
      case 'parsing':
        return 'Parsing';
      case 'embedding':
        return 'Embedding';
      case 'queued':
        return 'Queued';
      case 'pending':
        return 'Pending';
      case 'needs_ocr':
        return 'Needs OCR';
      case 'needs_js':
        return 'Needs JS';
      case 'render_failed':
        return 'Render failed';
      default:
        return status;
    }
  }
</script>

{#if collapsed}
  <div data-tauri-drag-region class="flex h-14 shrink-0 items-center justify-center"></div>

  <div class="flex flex-1 flex-col items-center gap-1.5 px-1.5 pt-1.5">
    <button
      type="button"
      data-right-rail-collapse-btn
      aria-label="Expand sources"
      title="Expand sources"
      onclick={toggleCollapse}
      class="rail-icon-btn text-muted-foreground hover:bg-muted hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <PanelRight class="size-4" strokeWidth={2} />
    </button>

    <div class="my-1 h-px w-6 bg-border"></div>

    <button
      type="button"
      aria-label="Sources ({totalCount})"
      title="Sources"
      onclick={toggleCollapse}
      class="rail-icon-btn relative bg-primary/10 text-primary hover:bg-primary/15"
      style="-webkit-app-region: no-drag;"
    >
      <FileText class="size-4" strokeWidth={2} />
      {#if totalCount > 0}
        <span
          class="absolute -top-0.5 -right-0.5 flex size-3.5 items-center justify-center rounded-full bg-primary text-[0.5rem] font-bold tabular-nums text-primary-foreground"
          aria-hidden="true"
        >
          {totalCount > 9 ? '9+' : totalCount}
        </span>
      {/if}
    </button>

    <button
      type="button"
      aria-label="Add source"
      title="Add source"
      onclick={() => (modalOpen = true)}
      class="rail-icon-btn text-muted-foreground hover:bg-muted hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <Plus class="size-4" strokeWidth={2.5} />
    </button>

    <div class="flex-1"></div>

    <button
      type="button"
      aria-label="Studio"
      title="Studio"
      onclick={toggleCollapse}
      class="rail-icon-btn mb-2 text-muted-foreground hover:bg-muted hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <Headphones class="size-4" strokeWidth={2} />
    </button>
  </div>
{:else}
  <div data-tauri-drag-region class="flex h-14 shrink-0 items-center gap-2.5 px-3">
    {#if totalCount > 0}
      <button
        type="button"
        role="checkbox"
        aria-checked={allSelected ? 'true' : someSelected ? 'mixed' : 'false'}
        aria-label={allSelected ? 'Deselect all sources' : 'Select all sources'}
        title={allSelected ? 'Deselect all' : 'Select all'}
        onclick={() => void toggleAllSelected()}
        class={cn(
          'checkbox-box size-[18px] transition-transform active:scale-90',
          allSelected || someSelected
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border hover:border-primary/60'
        )}
        style="-webkit-app-region: no-drag;"
      >
        {#if allSelected}
          <span class="flex" in:popIn={{ duration: 200 }}>
            <Check class="size-[11px]" strokeWidth={3} />
          </span>
        {:else if someSelected}
          <span class="flex" in:popIn={{ duration: 200 }}>
            <Minus class="size-[11px]" strokeWidth={3} />
          </span>
        {/if}
      </button>
    {/if}

    <!-- Panel header — deliberately one notch below the app brand ("Lens",
         text-base) and the centered notebook title, so this reads as a panel
         label, not a competing app/page title. -->
    <span class="flex-1 text-sm font-semibold text-foreground">Sources</span>

    {#if totalCount > 0}
      <span
        class="inline-flex h-[18px] min-w-[34px] items-center justify-center rounded-full bg-muted px-1.5 text-xs font-semibold tabular-nums text-muted-foreground"
        aria-label="{selectedCount} of {totalCount} sources selected"
        style="-webkit-app-region: no-drag;"
      >
        {selectedCount}/{totalCount}
      </span>
    {/if}

    <button
      class="rail-pill-btn text-foreground"
      type="button"
      aria-label="Add source"
      title="Add source"
      onclick={() => (modalOpen = true)}
      style="-webkit-app-region: no-drag;"
    >
      <Plus class="size-3.5" strokeWidth={2.5} />
    </button>

    <button
      type="button"
      data-right-rail-collapse-btn
      aria-label="Collapse sources"
      title="Collapse sources"
      onclick={toggleCollapse}
      class="rail-pill-btn text-muted-foreground/70 hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <PanelRightClose class="size-3.5" strokeWidth={2} />
    </button>
  </div>

  <div class="shrink-0 border-t border-border"></div>

  <!-- min-h floor keeps a few rows visible+scrollable even when Studio is tall. -->
  <div data-sources-scroll class="no-scrollbar flex min-h-[128px] flex-1 flex-col overflow-y-auto">
    {#if sources.length === 0}
      <div class="flex flex-1 flex-col items-center justify-center gap-2 px-4 py-12">
        <div class="empty-tile flex size-11 items-center justify-center" aria-hidden="true">
          <FileText class="size-[18px] text-muted-foreground/40" strokeWidth={1.5} />
        </div>
        <p class="mt-1 text-center text-[12px] font-semibold text-foreground">No sources yet</p>
        <p class="max-w-[180px] text-center text-[11px] leading-relaxed text-muted-foreground/60">
          Add a file or paste text to ground this notebook.
        </p>
        <button
          class="mt-2 flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-[12px] font-semibold text-primary-foreground transition-[opacity,transform] hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:scale-[0.97]"
          type="button"
          aria-label="Add first source"
          onclick={() => (modalOpen = true)}
        >
          <Plus class="size-[11px]" strokeWidth={2.5} />
          Add source
        </button>
      </div>
    {:else}
      <ul class="flex flex-col gap-px p-2" role="list" aria-label="Sources">
        {#each sources as source, i (source.id)}
          {@const badge = typeBadge(source.kind, source.locator, source.title)}
          {@const meta = metaLine(source.token_count)}
          {@const status = source.status as SourceStatus}
          <li
            data-source-id={source.id}
            data-pulsing={pulsingId === source.id}
            class="src-row group flex items-center gap-2.5 rounded-[10px] px-2.5 py-2.5"
            use:fadeRise={{ y: 6, duration: 0.34, delay: Math.min(i, 8) * 0.035 }}
          >
            <button
              class={cn(
                'checkbox-box size-[16px] transition-[background-color,border-color,transform] duration-150 active:scale-90',
                source.selected === 1
                  ? 'border-primary bg-primary'
                  : 'border-border bg-transparent hover:border-primary/60'
              )}
              onclick={() => void toggleSelected(source.id)}
              type="button"
              aria-label={source.selected === 1
                ? `Deselect source ${source.title}`
                : `Select source ${source.title}`}
              aria-pressed={source.selected === 1}
            >
              {#if source.selected === 1}
                <span class="flex" in:popIn={{ duration: 200 }}>
                  <Check class="size-[9px] text-primary-foreground" strokeWidth={3} />
                </span>
              {/if}
            </button>

            <div class="min-w-0 flex-1">
              <div class="truncate text-sm font-medium leading-tight text-foreground">
                {source.title}
              </div>
              <div class="mt-0.5 flex flex-wrap items-center gap-1">
                <span
                  class="text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  {badge}
                </span>
                {#if meta}
                  <span class="text-[0.6875rem] text-muted-foreground/40" aria-hidden="true">·</span
                  >
                  <span class="text-xs text-muted-foreground/50">{meta}</span>
                {/if}
              </div>
            </div>

            <!-- Fixed-width slot: status dot fades out on hover, action buttons fade in.
                 Error sources get a wider slot with a tooltip + retry button.
                 All buttons are no-drag so the titlebar region doesn't swallow clicks. -->
            <div
              class={cn(
                'relative flex shrink-0 items-center justify-end gap-0.5',
                status === 'error' ? 'w-auto' : 'size-5'
              )}
              aria-label="Status: {statusDotLabel(status)}"
            >
              <!-- group-hover:animate-none prevents animate-pulse from bleeding through on hover. -->
              {#if status === 'error'}
                <div class="relative flex items-center">
                  <span
                    class={cn(
                      'pointer-events-none block size-[7px] rounded-full transition-opacity duration-150 group-hover:animate-none group-hover:opacity-0',
                      statusDotClass(status)
                    )}
                    aria-hidden="true"
                  ></span>
                  <div
                    class={cn(
                      'pointer-events-none absolute right-0 bottom-full z-10 mb-1.5',
                      'w-max max-w-[200px] rounded-md border border-destructive/30 bg-popover px-2.5 py-1.5 shadow-md',
                      'opacity-0 transition-opacity duration-150 group-hover:opacity-100'
                    )}
                    role="tooltip"
                    data-error-tooltip
                  >
                    <p class="text-[0.6875rem] font-medium leading-snug text-destructive">
                      {source.error_meta
                        ? source.error_meta.message
                        : 'Ingest failed (no details captured)'}
                    </p>
                    {#if source.error_meta?.kind}
                      <p class="mt-0.5 text-[0.625rem] text-muted-foreground">
                        {source.error_meta.kind}
                        {#if source.error_meta.attempt_count > 1}
                          · attempt {source.error_meta.attempt_count}
                        {/if}
                      </p>
                    {/if}
                  </div>
                </div>

                <button
                  type="button"
                  aria-label="Retry ingesting {source.title}"
                  data-retry-source-btn
                  onclick={(e) => {
                    e.stopPropagation();
                    void retrySource(source.id);
                  }}
                  class={cn(
                    'row-action flex size-5 items-center justify-center rounded-[6px]',
                    'text-muted-foreground/40 opacity-0 transition-[opacity,background-color,transform] duration-150',
                    'hover:bg-destructive/15 hover:text-destructive active:scale-90',
                    'focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                    'group-hover:opacity-100'
                  )}
                  style="-webkit-app-region: no-drag;"
                >
                  <RotateCcw class="size-3" strokeWidth={2} />
                </button>
              {:else}
                <span
                  class={cn(
                    'pointer-events-none absolute block size-[7px] rounded-full transition-opacity duration-150 group-hover:animate-none group-hover:opacity-0',
                    statusDotClass(status)
                  )}
                  aria-hidden="true"
                ></span>
              {/if}

              <button
                type="button"
                aria-label="Delete source"
                data-delete-source-btn
                onclick={(e) => {
                  e.stopPropagation();
                  void removeSource(source.id);
                }}
                class={cn(
                  'row-action flex size-5 items-center justify-center rounded-[6px]',
                  status !== 'error' && 'absolute',
                  'text-muted-foreground/40 opacity-0 transition-[opacity,background-color,transform] duration-150',
                  'hover:bg-destructive/15 hover:text-destructive active:scale-90',
                  'focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  'group-hover:opacity-100'
                )}
                style="-webkit-app-region: no-drag;"
              >
                <Trash class="size-3" strokeWidth={2} />
              </button>
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  </div>

  <!-- Undo bar — shown transiently after a soft-delete. Pinned above Studio.
       Auto-dismisses via the store's 6 s timeout; "Undo" button calls undoRemove().
       Built inline (no toast primitive exists) using tokens only. no-drag. -->
  {#if sourcesStore.recentlyTrashed}
    <div
      class="undo-bar mx-2 mb-1.5 flex shrink-0 items-center justify-between gap-2 rounded-lg border border-border bg-muted/60 px-3 py-2 text-xs shadow-sm"
      role="status"
      aria-live="polite"
      aria-label="Source moved to trash"
      style="-webkit-app-region: no-drag;"
    >
      <span class="truncate text-muted-foreground">Source moved to trash</span>
      <button
        type="button"
        onclick={() => void undoRemove(notebookStore.activeNotebookId ?? undefined)}
        class="shrink-0 rounded-[6px] px-2 py-0.5 text-xs font-semibold text-foreground transition-[background-color,transform] hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:scale-95"
        style="-webkit-app-region: no-drag;"
      >
        Undo
      </button>
    </div>
  {/if}

  <StudioPanel {selectedCount} {totalCount} />
{/if}

<AddSourcesModal open={modalOpen} onclose={() => (modalOpen = false)} />

<style>
  /* Collapsed-strip icon buttons — uniform hit target + spring press, matching the
     left rail's collapsed action feel. */
  .rail-icon-btn {
    display: flex;
    width: 36px;
    height: 36px;
    align-items: center;
    justify-content: center;
    border: 0;
    border-radius: 10px;
    background: transparent;
    cursor: pointer;
    transition:
      background-color 0.16s var(--ease-out, ease),
      color 0.16s var(--ease-out, ease),
      transform 0.16s var(--ease-out, ease);
  }
  .rail-icon-btn:active {
    transform: scale(calc(1 - 0.06 * var(--rail-motion, 1)));
  }
  .rail-icon-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }

  /* Expanded-header round action pills (add / collapse). */
  .rail-pill-btn {
    display: flex;
    width: 26px;
    height: 26px;
    flex: none;
    align-items: center;
    justify-content: center;
    border: 0;
    border-radius: 999px;
    background: var(--muted);
    cursor: pointer;
    transition:
      background-color 0.16s var(--ease-out, ease),
      color 0.16s var(--ease-out, ease),
      transform 0.16s var(--ease-out, ease);
  }
  .rail-pill-btn:hover {
    background: color-mix(in oklch, var(--muted) 70%, transparent);
  }
  .rail-pill-btn:active {
    transform: scale(calc(1 - 0.06 * var(--rail-motion, 1)));
  }
  .rail-pill-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }

  /* Shared checkbox chrome (header select-all + per-row). */
  .checkbox-box {
    display: flex;
    flex: none;
    align-items: center;
    justify-content: center;
    border-width: 1px;
    border-style: solid;
    border-radius: 5px;
    cursor: pointer;
    outline: none;
  }
  .checkbox-box:focus-visible {
    box-shadow: 0 0 0 2px var(--ring);
  }

  .empty-tile {
    border-radius: 14px;
    background: var(--muted);
  }

  /* Source rows: fast hover, plus a reveal highlight (#23b) driven by data-pulsing.
     The ring fades IN fast (0.18s, to catch the eye) and OUT slow (0.5s, gentle). */
  .src-row {
    position: relative;
    transition:
      background-color 0.16s var(--ease-out, ease),
      box-shadow 0.5s var(--ease-out, ease);
  }
  .src-row:hover {
    background-color: color-mix(in oklch, var(--muted) 50%, transparent);
  }
  .src-row[data-pulsing='true'] {
    background-color: color-mix(in oklch, var(--primary) 9%, transparent);
    box-shadow: inset 0 0 0 1.5px color-mix(in oklch, var(--primary) 55%, transparent);
    transition:
      background-color 0.18s var(--ease-out, ease),
      box-shadow 0.18s var(--ease-out, ease);
  }

  /* row-action buttons already fade via Tailwind opacity; the press scale is gated
     so calm mode drops the movement but keeps the hover affordance. */
  .row-action:active {
    transform: scale(calc(1 - 0.1 * var(--rail-motion, 1)));
  }

  @media (prefers-reduced-motion: reduce) {
    .src-row,
    .src-row[data-pulsing='true'] {
      transition:
        background-color 0.16s ease,
        box-shadow 0.16s ease;
    }
  }
</style>
