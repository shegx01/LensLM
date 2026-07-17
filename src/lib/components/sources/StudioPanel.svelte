<!-- StudioPanel — non-functional visual shell (M4). All actions are disabled and
     labelled "coming soon"; the section reads as an intentional preview, not a
     broken control. TODO(M6/M7): study tools land with M6 (Notes), Audio Overview
     lands with M7. Tokens only — no hardcoded hex. -->
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
  import { fadeRise } from '$lib/motion/index.js';

  let {
    selectedCount = 0,
    totalCount = 0
  }: {
    /** Number of currently-selected sources. */
    selectedCount?: number;
    /** Total number of sources in the notebook. */
    totalCount?: number;
  } = $props();

  // The study/learning trio renders as full-width rows for prominence; the
  // remaining document/visual tools fill a 2-column grid below.
  const heroTools: Array<{ label: string; sub: string; icon: Component }> = [
    { label: 'Study Guide', sub: 'Key terms & review Qs', icon: BookOpen },
    { label: 'Flashcards', sub: 'Spaced recall', icon: Layers },
    { label: 'Quiz', sub: 'Self-test questions', icon: Sparkles }
  ];

  const gridTools: Array<{ label: string; sub: string; icon: Component }> = [
    { label: 'Briefing Doc', sub: 'One-page summary', icon: FileText },
    { label: 'Report', sub: 'Structured write-up', icon: FileChartColumn },
    { label: 'Slide Deck', sub: 'Presentation outline', icon: Presentation },
    { label: 'Timeline', sub: 'Chronological view', icon: Clock },
    { label: 'FAQ', sub: 'Anticipated questions', icon: ChartBar },
    { label: 'Mind Map', sub: 'Concept graph', icon: Brain },
    { label: 'Infographic', sub: 'Visual summary', icon: Image },
    { label: 'Data Table', sub: 'Extracted facts', icon: Table2 }
  ];
</script>

<section
  class="no-scrollbar flex min-h-0 flex-[0_1_auto] flex-col gap-3 overflow-y-auto border-t border-border px-3 py-3"
  aria-label="Studio"
>
  <div class="flex items-center gap-2">
    <span class="text-sm font-semibold text-foreground">Studio</span>
    <span
      class="inline-flex items-center rounded-[4px] bg-muted px-[5px] py-px text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground"
    >
      Research
    </span>
    <span
      class="ml-auto inline-flex items-center gap-1 text-[0.625rem] font-medium text-muted-foreground/60"
    >
      <span class="soon-dot" aria-hidden="true"></span>
      Coming soon
    </span>
  </div>

  <div class="audio-card p-3" use:fadeRise={{ y: 8, duration: 0.36 }}>
    <div class="flex items-center gap-2.5">
      <div class="audio-icon" aria-hidden="true">
        <Headphones class="size-[15px] text-primary" strokeWidth={2} />
      </div>
      <div class="min-w-0">
        <p class="text-sm font-semibold leading-tight text-foreground">Audio Overview</p>
        <p class="text-xs leading-tight text-muted-foreground tabular-nums">
          {selectedCount} of {totalCount} sources selected
        </p>
      </div>
    </div>

    <button
      type="button"
      class="mt-3 flex w-full items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground opacity-70 transition-opacity disabled:cursor-not-allowed"
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

  <div aria-label="Study tools" use:fadeRise={{ y: 8, duration: 0.36, delay: 0.05 }}>
    <div class="mb-1.5 flex flex-col gap-1.5">
      {#each heroTools as tool (tool.label)}
        <button
          type="button"
          class="tool-tile hero-tile flex w-full items-center gap-2.5 px-2.5 py-2 text-left disabled:cursor-not-allowed"
          disabled
          aria-label="{tool.label} (coming soon)"
          title="Coming soon"
          style="-webkit-app-region: no-drag;"
        >
          <span class="tool-icon" aria-hidden="true">
            <tool.icon class="size-[14px] text-primary" strokeWidth={1.75} />
          </span>
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

    <div class="grid grid-cols-2 gap-1.5">
      {#each gridTools as tool (tool.label)}
        <button
          type="button"
          class="tool-tile flex items-center gap-2 px-2.5 py-2 text-left disabled:cursor-not-allowed"
          disabled
          aria-label="{tool.label} (coming soon)"
          title="Coming soon"
          style="-webkit-app-region: no-drag;"
        >
          <span class="tool-icon-sm" aria-hidden="true">
            <tool.icon class="size-[13px] text-muted-foreground" strokeWidth={1.75} />
          </span>
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

<style>
  .soon-dot {
    width: 5px;
    height: 5px;
    border-radius: 999px;
    background: color-mix(in oklch, var(--primary) 70%, transparent);
  }

  /* Audio Overview: the one hero surface in Studio — tinted card with a soft
     primary wash so it reads as the section's headline action. */
  .audio-card {
    border-radius: 14px;
    border: 1px solid color-mix(in oklch, var(--primary) 16%, var(--border));
    background: color-mix(in oklch, var(--primary) 5%, var(--card));
  }
  .audio-icon {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    flex: none;
    border-radius: 9px;
    background: color-mix(in oklch, var(--primary) 12%, transparent);
  }

  /* Study-tool tiles — a locked preview: clean surface (not a 60% grey wash),
     but non-interactive. */
  .tool-tile {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--card);
    opacity: 0.85;
  }
  .hero-tile {
    background: color-mix(in oklch, var(--muted) 40%, var(--card));
  }
  .tool-icon {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    flex: none;
    border-radius: 7px;
    background: color-mix(in oklch, var(--primary) 10%, transparent);
  }
  .tool-icon-sm {
    display: grid;
    place-items: center;
    flex: none;
  }
</style>
