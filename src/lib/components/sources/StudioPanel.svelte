<!-- StudioPanel — the bottom region of the right rail (M4 visual shell).
     TODO(M6/M7): this is a non-functional visual shell. The "Generate Audio
     Overview" action lands with the Audio-Overview milestone (M7) and the study
     tools (Study Guide, Briefing Doc, …) land with the Notes milestone (M6).
     Every action here is DISABLED with a "coming soon" affordance.

     Layout: a "Studio" header with a "RESEARCH" tag, an accent Audio Overview
     card, then a 2-column grid of study-tool buttons. The panel owns its own
     vertical scroll (max-height + no-scrollbar) so it never crowds out the
     Sources list above it.

     Drag region: this panel sits BELOW the rail's drag bar, so it is not a drag
     region. Every interactive control still carries -webkit-app-region: no-drag
     (belt + suspenders) so a future drag bar can't swallow clicks.

     Theming: tokens only — the accent drives the Generate button (bg-primary).
     No hardcoded hex. -->
<script lang="ts">
  import Headphones from '@lucide/svelte/icons/headphones';
  import BookOpen from '@lucide/svelte/icons/book-open';
  import FileText from '@lucide/svelte/icons/file-text';
  import FileChartColumn from '@lucide/svelte/icons/file-chart-column';
  import Presentation from '@lucide/svelte/icons/presentation';
  import Layers from '@lucide/svelte/icons/layers';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import Clock from '@lucide/svelte/icons/clock';
  import ChartBar from '@lucide/svelte/icons/chart-bar';
  import Brain from '@lucide/svelte/icons/brain';
  import Image from '@lucide/svelte/icons/image';
  import Table2 from '@lucide/svelte/icons/table-2';
  import type { Component } from 'svelte';

  // ---------------------------------------------------------------------------
  // Props — selection counts so the Audio Overview card can read "{n} of {m}".
  // ---------------------------------------------------------------------------

  let {
    selectedCount = 0,
    totalCount = 0
  }: {
    /** Number of currently-selected sources. */
    selectedCount?: number;
    /** Total number of sources in the notebook. */
    totalCount?: number;
  } = $props();

  // ---------------------------------------------------------------------------
  // Study tools — visual shell only (all disabled). TODO(M6) wire to Notes.
  //
  // Layout: Study Guide is the "hero" tool and spans a full row on its own
  // (matching the shot1 reference). The remaining tools fill a 2-col grid below.
  // ---------------------------------------------------------------------------

  /** The hero tool — rendered full-width above the grid. */
  const heroTool: { label: string; sub: string; icon: Component } = {
    label: 'Study Guide',
    sub: 'Key terms & review Qs',
    icon: BookOpen
  };

  /** The remaining tools — rendered in a 2-column grid. */
  const gridTools: Array<{ label: string; sub: string; icon: Component }> = [
    { label: 'Briefing Doc', sub: 'One-page summary', icon: FileText },
    { label: 'Report', sub: 'Structured write-up', icon: FileChartColumn },
    { label: 'Slide Deck', sub: 'Presentation outline', icon: Presentation },
    { label: 'Flashcards', sub: 'Spaced recall', icon: Layers },
    { label: 'Quiz', sub: 'Self-test questions', icon: Sparkles },
    { label: 'Timeline', sub: 'Chronological view', icon: Clock },
    { label: 'FAQ', sub: 'Anticipated questions', icon: ChartBar },
    { label: 'Mind Map', sub: 'Concept graph', icon: Brain },
    { label: 'Infographic', sub: 'Visual summary', icon: Image },
    { label: 'Data Table', sub: 'Extracted facts', icon: Table2 }
  ];
</script>

<!-- Studio region — own scroll, hidden scrollbar, capped height. -->
<section
  class="no-scrollbar flex max-h-[55%] shrink-0 flex-col gap-3 overflow-y-auto border-t border-border px-3 py-3"
  aria-label="Studio"
>
  <!-- Studio header with a RESEARCH tag — text-sm matches left rail section labels -->
  <div class="flex items-center gap-2">
    <span class="text-sm font-semibold text-foreground">Studio</span>
    <span
      class="inline-flex items-center rounded-[4px] bg-muted px-[5px] py-px text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground"
    >
      Research
    </span>
  </div>

  <!-- Audio Overview card -->
  <div class="rounded-xl border border-border bg-muted/30 p-3">
    <div class="flex items-center gap-2">
      <div
        class="flex size-[26px] shrink-0 items-center justify-center rounded-lg bg-primary/10"
        aria-hidden="true"
      >
        <Headphones class="size-[14px] text-primary" strokeWidth={2} />
      </div>
      <div class="min-w-0">
        <p class="text-sm font-semibold leading-tight text-foreground">Audio Overview</p>
        <p class="text-xs leading-tight text-muted-foreground">
          {selectedCount} of {totalCount} sources selected
        </p>
      </div>
    </div>

    <!-- Generate button — accent, disabled (M7). no-drag. -->
    <button
      type="button"
      class="mt-3 flex w-full items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground opacity-60 transition-opacity disabled:cursor-not-allowed"
      disabled
      aria-label="Generate Audio Overview (coming soon)"
      title="Coming soon"
      style="-webkit-app-region: no-drag;"
    >
      <Sparkles class="size-[12px]" strokeWidth={2} />
      Generate Audio Overview
    </button>
    <p class="mt-2 text-center text-xs leading-relaxed text-muted-foreground/70">
      Two AI hosts discuss your selected sources in a natural conversation.
    </p>
  </div>

  <!-- Study tools — Study Guide hero (full row) + 2-col grid for the rest.
       Per shot1: Study Guide spans the full width as the first item. -->
  <div aria-label="Study tools">
    <!-- Hero tool: Study Guide — full-width button -->
    <button
      type="button"
      class="mb-1.5 flex w-full items-center gap-2 rounded-lg border border-border bg-card px-2.5 py-2 text-left opacity-60 transition-opacity disabled:cursor-not-allowed"
      disabled
      aria-label="{heroTool.label} (coming soon)"
      title="Coming soon"
      style="-webkit-app-region: no-drag;"
    >
      <heroTool.icon class="size-[14px] shrink-0 text-muted-foreground" strokeWidth={1.75} />
      <span class="min-w-0">
        <span class="block truncate text-xs font-semibold leading-tight text-foreground"
          >{heroTool.label}</span
        >
        <span class="block truncate text-[0.6875rem] leading-tight text-muted-foreground/70"
          >{heroTool.sub}</span
        >
      </span>
    </button>

    <!-- Remaining tools — 2-col grid -->
    <div class="grid grid-cols-2 gap-1.5">
      {#each gridTools as tool (tool.label)}
        <button
          type="button"
          class="flex items-center gap-2 rounded-lg border border-border bg-card px-2.5 py-2 text-left opacity-60 transition-opacity disabled:cursor-not-allowed"
          disabled
          aria-label="{tool.label} (coming soon)"
          title="Coming soon"
          style="-webkit-app-region: no-drag;"
        >
          <tool.icon class="size-[14px] shrink-0 text-muted-foreground" strokeWidth={1.75} />
          <span class="min-w-0">
            <span class="block truncate text-xs font-semibold leading-tight text-foreground"
              >{tool.label}</span
            >
            <span class="block truncate text-[0.6875rem] leading-tight text-muted-foreground/70"
              >{tool.sub}</span
            >
          </span>
        </button>
      {/each}
    </div>
  </div>
</section>
