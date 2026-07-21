<!-- StudioPanel — Studio surface for the right rail (M4). The Audio Overview card
     (#29) is a live per-notebook lifecycle: idle (length + Generate) → generating
     (phase + progress + Cancel) → ready/stale (player + regenerate) → failed/missing
     (functional error + retry). The study/report tools below remain a "coming soon"
     preview (M6/M7 land those separately) — every action stays aria-disabled there. -->
<script lang="ts">
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
  import Headphones from '@lucide/svelte/icons/headphones';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import Square from '@lucide/svelte/icons/square';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import type { Component } from 'svelte';
  import { fadeRise } from '$lib/motion/index.js';
  import { notebookStore } from '$lib/notebooks/index.js';
  import { formatRelativeTime } from '$lib/notebooks/format-time.js';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';
  import AudioPlayer from '$lib/components/audio/AudioPlayer.svelte';
  import {
    audioOverviewStore,
    generateOverview,
    cancelOverview
  } from '$lib/sources/audio-state.svelte.js';
  import { sourcesStore } from '$lib/sources/sources-state.svelte.js';
  import type { Length } from '$lib/sources/audio-ipc.js';

  let {
    selectedCount = 0,
    totalCount = 0
  }: {
    selectedCount?: number;
    totalCount?: number;
  } = $props();

  const notebookId = $derived(notebookStore.activeNotebookId);
  const status = $derived(audioOverviewStore.overviewStatus);
  const hasPlayableAudio = $derived(status === 'ready' || status === 'stale');

  let length = $state<Length>('medium');
  const LENGTH_OPTIONS: { value: Length; label: string }[] = [
    { value: 'short', label: 'Short' },
    { value: 'medium', label: 'Med' },
    { value: 'long', label: 'Long' }
  ];

  function phaseLabel(): string {
    const { phase, turn, total } = audioOverviewStore;
    if (phase === 'synthesizing') {
      return total ? `Synthesizing turn ${turn}/${total}` : 'Synthesizing…';
    }
    if (phase === 'stitching') return 'Stitching audio…';
    if (phase === 'encoding') return 'Encoding…';
    return 'Preparing script…';
  }

  const progressPct = $derived(
    audioOverviewStore.phase === 'synthesizing' && audioOverviewStore.total
      ? Math.round(((audioOverviewStore.turn ?? 0) / audioOverviewStore.total) * 100)
      : null
  );

  /**
   * `null` when Generate is allowed; otherwise the reason to surface. Reads selection
   * from the store (same source as `canGenerate`) so the gate and its hint never disagree.
   */
  const blockedReason = $derived<'no-sources' | 'model-not-ready' | null>(
    sourcesStore.selectedCount === 0
      ? 'no-sources'
      : !audioOverviewStore.modelReady
        ? 'model-not-ready'
        : null
  );

  const regenerateTooltip = $derived(
    blockedReason === 'no-sources'
      ? 'Select at least one source to regenerate'
      : blockedReason === 'model-not-ready'
        ? 'Download a voice engine in Settings to regenerate'
        : 'Regenerate overview'
  );

  async function handleGenerate(): Promise<void> {
    if (!notebookId || !audioOverviewStore.canGenerate) return;
    await generateOverview(notebookId, length);
  }

  async function handleCancel(): Promise<void> {
    if (!notebookId) return;
    await cancelOverview(notebookId);
  }

  function openTtsSettings(): void {
    notebookStore.openSettings('text_to_speech');
  }

  // Full-width rows give the study trio prominence over the grid below.
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
  </div>

  <TooltipProvider>
    <div class="audio-card p-3" use:fadeRise={{ y: 8, duration: 0.36 }}>
      <div class="flex items-center gap-2.5">
        <div class="audio-icon" aria-hidden="true">
          <Headphones class="size-[15px] text-primary" strokeWidth={2} />
        </div>
        <div class="min-w-0 flex-1">
          <p class="text-sm font-semibold leading-tight text-foreground">Audio Overview</p>
          {#if status === 'ready' || status === 'stale'}
            <p class="text-xs leading-tight text-muted-foreground tabular-nums">
              {#if audioOverviewStore.generatedAt}
                Generated {formatRelativeTime(audioOverviewStore.generatedAt)}
              {:else}
                Ready
              {/if}
            </p>
          {:else}
            <p class="text-xs leading-tight text-muted-foreground tabular-nums">
              {selectedCount} of {totalCount} sources selected
            </p>
          {/if}
        </div>

        {#if hasPlayableAudio}
          <Tooltip>
            <TooltipTrigger>
              <button
                type="button"
                class="press grid size-7 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground transition-transform hover:bg-muted/70 hover:text-foreground active:scale-[0.94] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-muted disabled:hover:text-muted-foreground"
                aria-label="Regenerate overview"
                title={regenerateTooltip}
                disabled={!audioOverviewStore.canGenerate}
                style="-webkit-app-region: no-drag;"
                onclick={handleGenerate}
              >
                <RotateCw class="size-[13px]" strokeWidth={2} />
              </button>
            </TooltipTrigger>
            <TooltipContent side="top">{regenerateTooltip}</TooltipContent>
          </Tooltip>
        {/if}
      </div>

      {#if status === 'stale'}
        <div class="stale-hint mt-2 flex items-center gap-1.5 text-[0.7rem] text-amber-500">
          <TriangleAlert class="size-3 shrink-0" strokeWidth={2} />
          Sources changed — regenerate for an up-to-date overview.
        </div>
      {/if}

      {#if status === 'generating'}
        <div class="card-body mt-3 flex flex-col justify-center gap-3">
          <div class="flex items-center justify-between gap-2">
            <p class="text-xs font-medium text-foreground" role="status">{phaseLabel()}</p>
            <button
              type="button"
              class="press flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground hover:bg-muted/70 hover:text-foreground"
              aria-label="Cancel generation"
              onclick={handleCancel}
            >
              <Square class="size-2.5" fill="currentColor" strokeWidth={0} />
            </button>
          </div>
          <div class="progress-track">
            {#if progressPct !== null}
              <div class="progress-fill" style:width="{progressPct}%"></div>
            {:else}
              <div class="progress-fill indeterminate"></div>
            {/if}
          </div>
        </div>
      {:else if hasPlayableAudio && notebookId && audioOverviewStore.overviewPath}
        <div class="card-body mt-3 flex flex-col justify-center">
          <AudioPlayer path={audioOverviewStore.overviewPath} />
        </div>
      {:else}
        <div class="card-body mt-3 flex flex-col justify-center gap-2">
          <div class="seg3" role="radiogroup" aria-label="Overview length" data-len={length}>
            <span class="seg-ind" aria-hidden="true"></span>
            {#each LENGTH_OPTIONS as opt (opt.value)}
              <button
                type="button"
                role="radio"
                aria-checked={length === opt.value}
                class="seg-btn"
                data-active={length === opt.value}
                onclick={() => (length = opt.value)}
              >
                {opt.label}
              </button>
            {/each}
          </div>

          {#if status === 'failed' && audioOverviewStore.errorMessage}
            <p class="text-[0.7rem] text-destructive" role="alert">
              {audioOverviewStore.errorMessage}
            </p>
          {:else if status === 'missing'}
            <p class="text-[0.7rem] text-muted-foreground">
              The overview file is missing. Regenerate to create a new one.
            </p>
          {/if}

          <button
            type="button"
            class="press flex w-full items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground transition-[opacity,transform] hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!audioOverviewStore.canGenerate}
            style="-webkit-app-region: no-drag;"
            onclick={handleGenerate}
          >
            <Sparkles class="size-[12px]" strokeWidth={2} />
            {status === 'failed' ? 'Retry' : 'Generate Audio Overview'}
          </button>

          {#if blockedReason === 'no-sources'}
            <p class="text-center text-xs leading-relaxed text-muted-foreground/70">
              Select at least one source to generate an overview.
            </p>
          {:else if blockedReason === 'model-not-ready'}
            <button
              type="button"
              class="flex items-center justify-center gap-1 text-center text-xs font-medium text-primary hover:underline"
              style="-webkit-app-region: no-drag;"
              onclick={openTtsSettings}
            >
              <Settings2 class="size-3" strokeWidth={2} />
              Download a voice engine in Settings
            </button>
          {:else}
            <p class="text-center text-xs leading-relaxed text-muted-foreground/70">
              Two AI hosts discuss your selected sources in a natural conversation.
            </p>
          {/if}
        </div>
      {/if}
    </div>
  </TooltipProvider>

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
  /* Gated press-scale (calm mode / reduced-motion drops it); the button's
     transition-[opacity,transform] utility animates it. */
  .press:active {
    transform: scale(calc(1 - 0.02 * var(--rail-motion, 1)));
  }

  /* Audio Overview headline surface: the accent border is softened (not solid --primary)
     so it reads as an accent frame, not a selected state. */
  .audio-card {
    border-radius: 14px;
    background: var(--card);
    border: 1px solid color-mix(in oklch, var(--primary) 45%, transparent);
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

  /* Shared floor across idle/generating/player content so switching lifecycle
     state can't shrink the card (generating renders far less markup than idle). */
  .card-body {
    min-height: 9.5rem;
  }

  .progress-track {
    position: relative;
    height: 5px;
    border-radius: 999px;
    background: var(--muted);
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    border-radius: 999px;
    background: var(--primary);
    transition: width 0.3s var(--ease-out, ease);
  }
  .progress-fill.indeterminate {
    position: absolute;
    top: 0;
    width: 40%;
    left: -40%;
    animation: indeterminate-slide calc(1.1s / max(var(--rail-motion, 1), 0.0001)) ease-in-out
      infinite;
  }
  @keyframes indeterminate-slide {
    0% {
      left: -40%;
    }
    100% {
      left: 100%;
    }
  }

  .seg3 {
    position: relative;
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: 1fr;
    padding: 2px;
    border-radius: 10px;
    background: var(--muted);
  }
  .seg-ind {
    position: absolute;
    top: 2px;
    left: 2px;
    height: calc(100% - 4px);
    width: calc(33.333% - 1.33px);
    border-radius: 8px;
    background: var(--card);
    box-shadow: var(--shadow-tile);
    transform: translateX(0);
    transition: transform calc(0.32s * var(--rail-motion, 1)) var(--ease-spring, ease);
  }
  .seg3[data-len='medium'] .seg-ind {
    transform: translateX(100%);
  }
  .seg3[data-len='long'] .seg-ind {
    transform: translateX(200%);
  }
  .seg-btn {
    position: relative;
    z-index: 1;
    height: 26px;
    border: 0;
    background: transparent;
    border-radius: 8px;
    font-size: 0.7rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--muted-foreground);
    transition: color 0.18s var(--ease-out, ease);
    outline: none;
  }
  .seg-btn[data-active='true'] {
    color: var(--card-foreground);
  }
  .seg-btn:not([data-active='true']):hover {
    color: var(--foreground);
  }
  .seg-btn:focus-visible {
    box-shadow: 0 0 0 2px var(--ring);
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
