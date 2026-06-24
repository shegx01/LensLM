<!-- SourcesRail fills the 320px right <aside> in AppShell.
     Header: accent rounded-square icon · "Sources" · selected/total counter · secondary icon · "+" add.
     Source rows: accent checkbox · doc icon · truncated title · type badge · metadata · status dot.
     "Add sources" opens the tabbed AddSourcesModal.
     macOS drag region: header row = data-tauri-drag-region; every interactive child has
     style="-webkit-app-region: no-drag;" so window-drag and button-click don't conflict.
     All colours are CSS-variable tokens — no hardcoded hex. -->
<script lang="ts">
  import File from '@lucide/svelte/icons/file';
  import FileText from '@lucide/svelte/icons/file-text';
  import Check from '@lucide/svelte/icons/check';
  import Plus from '@lucide/svelte/icons/plus';
  import Trash from '@lucide/svelte/icons/trash';
  import PanelRight from '@lucide/svelte/icons/panel-right';
  import PanelRightClose from '@lucide/svelte/icons/panel-right-close';
  import Headphones from '@lucide/svelte/icons/headphones';
  import { cn } from '$lib/utils.js';
  import {
    sourcesStore,
    toggleSelected,
    removeSource,
    undoRemove
  } from '$lib/sources/sources-state.svelte.js';
  import { notebookStore } from '$lib/notebooks/index.js';
  import type { SourceStatus } from '$lib/sources/types.js';
  import AddSourcesModal from './AddSourcesModal.svelte';
  import StudioPanel from './StudioPanel.svelte';

  // ---------------------------------------------------------------------------
  // Local state
  // ---------------------------------------------------------------------------

  /** Controls the "Add sources" modal */
  let modalOpen = $state(false);

  // ---------------------------------------------------------------------------
  // Collapse — mirrors the left rail's sidebarCollapsed (store field
  // rightRailCollapsed). The AppShell grid's THIRD column width follows this
  // value (320px expanded / 104px collapsed icon strip, matching the left rail) and animates.
  // ---------------------------------------------------------------------------

  const collapsed = $derived(notebookStore.rightRailCollapsed);

  function toggleCollapse(): void {
    notebookStore.rightRailCollapsed = !notebookStore.rightRailCollapsed;
  }

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------

  const sources = $derived(sourcesStore.sources);
  const totalCount = $derived(sources.length);
  const selectedCount = $derived(sources.filter((s) => s.selected === 1).length);

  // ---------------------------------------------------------------------------
  // Type badge helpers
  // ---------------------------------------------------------------------------

  /**
   * Derive a short type badge from the source's kind + locator/title.
   * This is display-only — purely informational.
   */
  function typeBadge(kind: string, locator: string, title: string): string {
    // kind 'url' → 'URL'
    if (kind === 'url') return 'URL';

    // Derive from locator extension first, then fall back to title
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
        // For text/paste sources with no extension
        if (kind === 'text') return 'TXT';
        return 'FILE';
    }
  }

  /**
   * Derive a human-readable metadata line.
   * Phase 1: for text/md sources we typically only have token_count.
   * Show token count if available; otherwise gracefully omit.
   */
  // TODO(M6): extract typeBadge + metaLine to src/lib/sources/format.ts when Studio reuses badges.
  function metaLine(tokenCount: number | null): string {
    if (tokenCount !== null && tokenCount > 0) {
      // Approximate word count from tokens (~0.75 words/token)
      const approxWords = Math.round(tokenCount * 0.75);
      if (approxWords >= 1000) {
        return `~${(approxWords / 1000).toFixed(1)}k words`;
      }
      return `~${approxWords} words`;
    }
    return '';
  }

  // ---------------------------------------------------------------------------
  // Status dot helpers
  // ---------------------------------------------------------------------------

  /**
   * Map SourceStatus to a dot color class.
   * indexed → green, error → destructive/red, queued/pending/parsing/embedding → amber (pulsing)
   */
  function statusDotClass(status: SourceStatus): string {
    switch (status) {
      case 'indexed':
        return 'bg-green-primary';
      case 'error':
        return 'bg-destructive';
      case 'parsing':
      case 'embedding':
      case 'queued':
      case 'pending':
        return 'bg-amber-500 animate-pulse';
      default:
        return 'bg-muted-foreground/40';
    }
  }

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
      default:
        return status;
    }
  }
</script>

{#if collapsed}
  <!-- ──────────────────────────────────────────────────────────────────────
       COLLAPSED ICON STRIP — mirrors the left rail's minimized vibe (shot5).
       Sources icon + count badge near the top; Studio/headphones icon near the
       bottom. The top drag bar stays a drag region; every button is no-drag.
  ────────────────────────────────────────────────────────────────────────── -->
  <!-- Top drag bar — h-14 matches the left rail's traffic-lights spacer -->
  <div data-tauri-drag-region class="flex h-14 shrink-0 items-center justify-center"></div>

  <div class="flex flex-1 flex-col items-center gap-1.5 px-1.5 pt-1.5">
    <!-- Expand button — no-drag -->
    <button
      type="button"
      data-right-rail-collapse-btn
      aria-label="Expand sources"
      title="Expand sources"
      onclick={toggleCollapse}
      class="flex size-8 items-center justify-center rounded-lg border-0 bg-transparent text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <PanelRight class="size-4" strokeWidth={2} />
    </button>

    <div class="my-1 h-px w-6 bg-border"></div>

    <!-- Sources icon with count badge — no-drag -->
    <button
      type="button"
      aria-label="Sources ({totalCount})"
      title="Sources"
      onclick={toggleCollapse}
      class="relative flex size-8 items-center justify-center rounded-lg border-0 bg-primary/10 text-primary transition-colors hover:bg-primary/15"
      style="-webkit-app-region: no-drag;"
    >
      <FileText class="size-4" strokeWidth={2} />
      {#if totalCount > 0}
        <span
          class="absolute -top-0.5 -right-0.5 flex size-3.5 items-center justify-center rounded-full bg-primary text-[0.5rem] font-bold text-primary-foreground"
          aria-hidden="true"
        >
          {totalCount > 9 ? '9+' : totalCount}
        </span>
      {/if}
    </button>

    <!-- Add source — no-drag -->
    <button
      type="button"
      aria-label="Add source"
      title="Add source"
      onclick={() => (modalOpen = true)}
      class="flex size-8 items-center justify-center rounded-lg border-0 bg-transparent text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <Plus class="size-4" strokeWidth={2.5} />
    </button>

    <div class="flex-1"></div>

    <!-- Studio / headphones icon near the bottom — no-drag -->
    <button
      type="button"
      aria-label="Studio"
      title="Studio"
      onclick={toggleCollapse}
      class="mb-2 flex size-8 items-center justify-center rounded-lg border-0 bg-transparent text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <Headphones class="size-4" strokeWidth={2} />
    </button>
  </div>
{:else}
  <!-- ──────────────────────────────────────────────────────────────────────
       EXPANDED LAYOUT — Sources (flex-1) on top, Studio (capped) on the bottom.
  ────────────────────────────────────────────────────────────────────────── -->

  <!-- Rail header — h-14 matches the left rail's traffic-lights spacer height (56px),
       giving equal vertical breathing room top and bottom. data-tauri-drag-region on the
       outer wrapper; all interactive children carry -webkit-app-region: no-drag. -->
  <div data-tauri-drag-region class="flex h-14 shrink-0 items-center gap-2 px-3">
    <!-- Collapse toggle (mirrors the left rail) — no-drag -->
    <button
      type="button"
      data-right-rail-collapse-btn
      aria-label="Collapse sources"
      title="Collapse sources"
      onclick={toggleCollapse}
      class="flex size-[26px] shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground/70 transition-colors hover:opacity-60 hover:text-foreground"
      style="-webkit-app-region: no-drag;"
    >
      <PanelRightClose class="size-3.5" strokeWidth={2} />
    </button>

    <!-- Panel header — deliberately one notch below the app brand ("Lens",
         text-base) and the centered notebook title, so this reads as a panel
         label, not a competing app/page title. -->
    <span class="flex-1 text-sm font-semibold text-foreground">Sources</span>

    <!-- selected/total counter -->
    {#if totalCount > 0}
      <span
        class="inline-flex h-[18px] min-w-[30px] items-center justify-center rounded-full bg-muted px-1.5 text-xs font-semibold tabular-nums text-muted-foreground"
        aria-label="{selectedCount} of {totalCount} sources selected"
        style="-webkit-app-region: no-drag;"
      >
        {selectedCount}/{totalCount}
      </span>
    {/if}

    <!-- Add source button — no-drag -->
    <button
      class="flex size-[26px] shrink-0 items-center justify-center rounded-full bg-muted text-foreground transition-colors hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      type="button"
      aria-label="Add source"
      onclick={() => (modalOpen = true)}
      style="-webkit-app-region: no-drag;"
    >
      <Plus class="size-3.5" strokeWidth={2.5} />
    </button>
  </div>

  <!-- Hairline divider -->
  <div class="shrink-0 border-t border-border"></div>

  <!-- Scrollable source list — flex-1, hidden scrollbar (no-scrollbar utility). -->
  <div data-sources-scroll class="no-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto">
    {#if sources.length === 0}
      <!-- Empty state -->
      <div class="flex flex-1 flex-col items-center justify-center gap-2 px-4 py-12">
        <div
          class="flex size-10 items-center justify-center rounded-xl bg-muted"
          aria-hidden="true"
        >
          <FileText class="size-4 text-muted-foreground/40" strokeWidth={1.5} />
        </div>
        <p class="mt-1 text-center text-[12px] font-semibold text-foreground">No sources yet</p>
        <p class="text-center text-[11px] text-muted-foreground/60 leading-relaxed max-w-[180px]">
          Add a file or paste text to ground this notebook.
        </p>
        <button
          class="mt-2 flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-[12px] font-semibold text-primary-foreground transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
        {#each sources as source (source.id)}
          {@const badge = typeBadge(source.kind, source.locator, source.title)}
          {@const meta = metaLine(source.token_count)}
          {@const status = source.status as SourceStatus}
          <li
            class="group flex items-start gap-2.5 rounded-lg px-2.5 py-2.5 transition-colors duration-100 hover:bg-muted/50"
          >
            <!-- Accent checkbox — no-drag -->
            <button
              class={cn(
                'mt-0.5 flex size-[16px] shrink-0 cursor-pointer items-center justify-center rounded-[4px] transition-all duration-[130ms] border',
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
                <Check class="size-[9px] text-primary-foreground" strokeWidth={3} />
              {/if}
            </button>

            <!-- Document icon tile -->
            <div
              class="flex size-[28px] shrink-0 items-center justify-center rounded-[6px] bg-muted"
              aria-hidden="true"
            >
              <File class="size-[13px] text-muted-foreground" strokeWidth={1.75} />
            </div>

            <!-- Content: title + badge + meta — type scale matches left rail notebook rows -->
            <div class="min-w-0 flex-1">
              <div class="truncate text-sm font-medium leading-tight text-foreground">
                {source.title}
              </div>
              <div class="mt-0.5 flex items-center gap-1.5 flex-wrap">
                <!-- Type badge -->
                <span
                  class="inline-flex items-center rounded-[4px] bg-muted px-[5px] py-px text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  {badge}
                </span>
                <!-- Metadata line (word/page count) -->
                {#if meta}
                  <span class="text-xs text-muted-foreground/50">{meta}</span>
                {/if}
              </div>
            </div>

            <!-- Right-side affordance: fixed-width 20px reserved slot.
                 Status dot is visible by default; on row hover/focus the dot
                 fades out and the trash button fades in — occupying the same
                 slot so they never overlap. The trash button is sized to 20px
                 (matching the reserved slot) so there is no layout shift.
                 -webkit-app-region: no-drag on the button prevents the titlebar
                 drag region from swallowing the click. -->
            <div
              class="relative mt-1 flex size-5 shrink-0 items-center justify-center"
              aria-label="Status: {statusDotLabel(status)}"
            >
              <!-- Status dot — fades out on group-hover, invisible when trash is shown.
                   group-hover:animate-none stops the in-progress `animate-pulse`
                   keyframes from re-driving opacity (which would otherwise bleed the
                   pulsing dot through under the trash icon). -->
              <span
                class={cn(
                  'pointer-events-none absolute block size-[7px] rounded-full transition-opacity duration-150 group-hover:animate-none group-hover:opacity-0',
                  statusDotClass(status)
                )}
                aria-hidden="true"
              ></span>

              <!-- Delete button — invisible by default, fades in on hover/focus.
                   Sized to fill the same 20px reserved slot as the dot wrapper
                   so there is zero layout collision. -->
              <button
                type="button"
                aria-label="Delete source"
                data-delete-source-btn
                onclick={(e) => {
                  e.stopPropagation();
                  void removeSource(source.id);
                }}
                class={cn(
                  'absolute flex size-5 items-center justify-center rounded-[5px]',
                  'opacity-0 transition-opacity duration-150',
                  'bg-transparent text-muted-foreground/40',
                  'hover:bg-destructive/15 hover:text-destructive',
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
      class="mx-2 mb-1.5 flex shrink-0 items-center justify-between gap-2 rounded-lg border border-border bg-muted/60 px-3 py-2 text-xs shadow-sm"
      role="status"
      aria-live="polite"
      aria-label="Source moved to trash"
      style="-webkit-app-region: no-drag;"
    >
      <span class="truncate text-muted-foreground">Source moved to trash</span>
      <button
        type="button"
        onclick={() => void undoRemove()}
        class="shrink-0 rounded-[5px] px-2 py-0.5 text-xs font-semibold text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        style="-webkit-app-region: no-drag;"
      >
        Undo
      </button>
    </div>
  {/if}

  <!-- Studio (bottom) — visual shell, own capped scroll. -->
  <StudioPanel {selectedCount} {totalCount} />
{/if}

<!-- Add sources modal (tabbed) -->
<AddSourcesModal open={modalOpen} onclose={() => (modalOpen = false)} />
