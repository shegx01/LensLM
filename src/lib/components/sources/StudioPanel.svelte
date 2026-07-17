<!-- StudioPanel — a "coming soon" preview (M4). The tools aren't wired yet, so every
     action is aria-disabled and tagged "Coming soon"; the hover/press states are the
     designed preview feel, not a working control. TODO(M6/M7): study tools land with
     M6 (Notes), Audio Overview lands with M7. Tokens only — no hardcoded hex. -->
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
  class="studio-tray no-scrollbar flex min-h-0 flex-[0_1_auto] flex-col gap-3 overflow-y-auto px-3 py-3"
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
      class="press mt-3 flex w-full items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground transition-[opacity,transform] hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      aria-disabled="true"
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
          class="tool-tile hero-tile flex w-full items-center gap-2.5 px-2.5 py-2.5 text-left"
          aria-disabled="true"
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
          class="tool-tile flex items-center gap-2 px-2.5 py-2.5 text-left"
          aria-disabled="true"
          aria-label="{tool.label} (coming soon)"
          title="Coming soon"
          style="-webkit-app-region: no-drag;"
        >
          <span class="tool-icon-sm" aria-hidden="true">
            <tool.icon class="size-[13px]" strokeWidth={1.75} />
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

  /* Gated press-scale (calm mode / reduced-motion drops it); the button's
     transition-[opacity,transform] utility animates it. */
  .press:active {
    transform: scale(calc(1 - 0.02 * var(--rail-motion, 1)));
  }

  /* Audio Overview: the section's headline surface — an elevated card (stronger
     shadow than the tool tiles) with the accent carried by its icon + CTA. */
  .audio-card {
    border-radius: 14px;
    background: var(--card);
    box-shadow: var(--shadow-bar);
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

  /* Container and elements all share the card surface. The top edge combines a hairline
     with an upward scroll-edge shadow, so the sources list reads as scrolling UNDER the
     Studio panel rather than bleeding into it. Elements separate by elevation alone. */
  .studio-tray {
    background: var(--card);
    border-top: 1px solid var(--border);
    box-shadow: var(--shadow-scroll-edge);
  }

  /* Study-tool tiles — borderless cards that lift off the tray on a soft shadow;
     hover deepens the shadow and raises them a hair. The action is still "coming
     soon" (aria-disabled on the button); this is the designed preview state. */
  .tool-tile {
    border-radius: 10px;
    background: var(--card);
    box-shadow: var(--shadow-tile);
    cursor: pointer;
    transition:
      box-shadow 0.18s var(--ease-out, ease),
      transform 0.18s var(--ease-out, ease);
  }
  .tool-tile:hover {
    box-shadow: var(--shadow-bar);
    transform: translateY(calc(-1px * var(--rail-motion, 1)));
  }
  .tool-tile:active {
    transform: scale(calc(1 - 0.02 * var(--rail-motion, 1)));
  }
  .tool-tile:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }
  /* The study trio rests a touch higher than the grid via a stronger shadow. */
  .hero-tile {
    box-shadow: var(--shadow-bar);
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
    color: var(--muted-foreground);
    transition: color 0.16s var(--ease-out, ease);
  }
  .tool-tile:hover .tool-icon-sm {
    color: var(--primary);
  }
</style>
