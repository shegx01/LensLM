<script lang="ts">
  // Floating pill header. Always rendered; only the title is conditional on
  // `activeNotebook`. The outer row is the drag region; the pill is interactive
  // and is NOT a drag region (Tauri: interactive children receive pointer events).

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
      <span
        class="max-w-[180px] truncate pr-1.5 text-xs font-semibold tracking-[-0.1px] text-popover-foreground"
        title={activeNotebook.title}
      >
        {activeNotebook.title}
      </span>
    {/if}

    <TooltipProvider>
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

      <!-- Share — circular icon button, disabled, "Available soon" -->
      <Tooltip>
        <TooltipTrigger>
          <Button
            variant="ghost"
            size="icon"
            disabled
            class="size-[30px] rounded-full bg-muted text-muted-foreground hover:bg-muted/70"
            aria-label="Share notebook (available soon)"
          >
            <Share2 class="size-[13px]" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Available soon</TooltipContent>
      </Tooltip>

      <!-- Settings gear — opens the per-notebook "{notebook} settings" sheet.
           Only interactive when a notebook is active (it edits THAT notebook's
           embedding coordinate); honestly disabled otherwise. -->
      {#if activeNotebook}
        <Tooltip>
          <TooltipTrigger>
            <Button
              variant="ghost"
              size="icon"
              class="size-[30px] rounded-full bg-muted text-muted-foreground hover:bg-muted/70 hover:text-foreground"
              aria-label="Notebook settings"
              onclick={() => (notebookStore.notebookSettingsOpen = true)}
            >
              <Settings class="size-[13px]" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Notebook settings</TooltipContent>
        </Tooltip>
      {:else}
        <Tooltip>
          <TooltipTrigger>
            <Button
              variant="ghost"
              size="icon"
              disabled
              class="size-[30px] rounded-full bg-muted text-muted-foreground hover:bg-muted/70"
              aria-label="Notebook settings (no active notebook)"
            >
              <Settings class="size-[13px]" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Select a notebook</TooltipContent>
        </Tooltip>
      {/if}
    </TooltipProvider>
  </div>
</div>
