<script lang="ts">
  // NotebookTopBar — floating pill header for the center pane (Apple Music style).
  //
  // ALWAYS rendered (the header stays visible even with no notebook selected, so
  // share + settings remain reachable). The title + Chat/Notes segmented toggle
  // are notebook-contextual and appear only when `activeNotebook` is non-null.
  //
  // Layout (per design source "Floating pill header — top right"):
  //   Outer row: right-aligned (justify-end), padding 10px 12px 2px — sits on the
  //   window drag region so the window stays draggable from the empty top area.
  //   The pill itself is an inline-flex rounded-full elevated surface (bg-popover)
  //   with a soft drop shadow, reading as a floating control over the canvas.
  //
  //   Inside the pill, in order:
  //     [title span] [Chat|Notes segmented] [Share circle] [Settings circle]
  //
  // Chat|Notes is real state via `notebookStore.activeTab` ('chat'|'notes').
  // Share + Settings are honestly disabled with "Available soon" tooltips (M8+).
  //
  // The outer row carries `data-tauri-drag-region`; the pill is interactive and is
  // NOT a drag region (native Tauri: interactive children receive pointer events).

  import Share2 from '@lucide/svelte/icons/share-2';
  import Settings from '@lucide/svelte/icons/settings';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';
  import { notebookStore } from '$lib/notebooks/index.js';

  // Reactive reads from the shared store
  const activeNotebook = $derived(notebookStore.activeNotebook);
  const activeTab = $derived(notebookStore.activeTab);

  function setTab(tab: 'chat' | 'notes'): void {
    notebookStore.activeTab = tab;
  }
</script>

<!--
  Outer drag row — right-aligned. `data-tauri-drag-region` lets the window be
  dragged by the empty space to the left of the pill. The pill is interactive
  and intentionally NOT a drag region.
-->
<!--
  The pill is ALWAYS rendered (the header must be visible without a notebook
  selected, so share/settings stay reachable). The title + Chat/Notes segmented
  toggle are notebook-contextual and only appear when a notebook is active.
-->
<div data-tauri-drag-region class="flex shrink-0 justify-end px-3 pt-2.5 pb-0.5">
  <!-- Floating pill — elevated surface (bg-popover) with a soft drop shadow -->
  <div
    role="toolbar"
    aria-label="Notebook toolbar"
    class="inline-flex items-center gap-[3px] rounded-full bg-popover py-1 pr-1 shadow-[0_4px_16px_rgba(0,0,0,0.12)] {activeNotebook
      ? 'pl-3.5'
      : 'pl-1'}"
  >
    {#if activeNotebook}
      <!-- Notebook title — lives inside the pill (not large/bold) -->
      <span
        class="max-w-[180px] truncate pr-1.5 text-xs font-semibold tracking-[-0.1px] text-popover-foreground"
        title={activeNotebook.title}
      >
        {activeNotebook.title}
      </span>
    {/if}

    <TooltipProvider>
      {#if activeNotebook}
        <!-- Chat | Notes segmented toggle -->
        <div
          role="group"
          aria-label="View toggle"
          class="flex items-center gap-px rounded-full bg-muted p-0.5"
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'chat'}
            aria-controls="notebook-tab-panel"
            class="h-[26px] rounded-full px-[13px] text-[11px] font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
              {activeTab === 'chat'
              ? 'bg-popover text-popover-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'}"
            onclick={() => setTab('chat')}
          >
            Chat
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'notes'}
            aria-controls="notebook-tab-panel"
            class="h-[26px] rounded-full px-[13px] text-[11px] font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
              {activeTab === 'notes'
              ? 'bg-popover text-popover-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'}"
            onclick={() => setTab('notes')}
          >
            Notes
          </button>
        </div>
      {/if}

      <!-- Share — circular icon button, disabled, "Available soon" -->
      <Tooltip>
        <TooltipTrigger>
          <Button
            variant="ghost"
            size="icon"
            disabled
            class="size-[30px] rounded-full"
            aria-label="Share notebook (available soon)"
          >
            <Share2 class="size-[13px]" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Available soon</TooltipContent>
      </Tooltip>

      <!-- Settings gear — circular icon button, disabled, "Available soon" -->
      <Tooltip>
        <TooltipTrigger>
          <Button
            variant="ghost"
            size="icon"
            disabled
            class="size-[30px] rounded-full"
            aria-label="Notebook settings (available soon)"
          >
            <Settings class="size-[13px]" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Available soon</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  </div>
</div>
