<script lang="ts">
  // Compact floating pill header — right-aligned, hugging its contents. Same
  // family as the left rail: bg-card surface + the layered --shadow-bar lift, and
  // a Chat/Notes segmented control whose active pill SLIDES on the rail's spring
  // (--ease-spring, gated by --rail-motion). The outer row + pill are Tauri drag
  // regions; interactive children (toggle, buttons) stay real controls.

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

<!-- pr-6 lands the pill's right edge on the same 24px line as the composer + message content. -->
<div data-tauri-drag-region class="flex shrink-0 justify-end pt-2.5 pr-6 pb-1 pl-3">
  <div
    data-tauri-drag-region
    role="toolbar"
    aria-label="Notebook toolbar"
    class="inline-flex items-center gap-1 rounded-full bg-card py-1 pr-1 shadow-[var(--shadow-bar)] {activeNotebook
      ? 'pl-3.5'
      : 'pl-1'}"
  >
    {#if activeNotebook}
      <span
        data-tauri-drag-region
        class="max-w-[180px] truncate pr-1 text-xs font-semibold tracking-[-0.1px] text-card-foreground"
        title={activeNotebook.title}
      >
        {activeNotebook.title}
      </span>
    {/if}

    <TooltipProvider>
      <!-- Segmented toggle: absolutely-positioned pill slides between the two
           tabs on the rail spring. Equal-width tabs → translateX(100%) lands the
           pill exactly on the second tab (its own width == a tab's width). -->
      <div class="seg" role="group" aria-label="View toggle" data-tab={activeTab}>
        <span class="seg-ind" aria-hidden="true"></span>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'chat'}
          aria-controls="notebook-tab-panel"
          class="seg-btn"
          data-active={activeTab === 'chat'}
          onclick={() => setTab('chat')}
        >
          Chat
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'notes'}
          aria-controls="notebook-tab-panel"
          class="seg-btn"
          data-active={activeTab === 'notes'}
          onclick={() => setTab('notes')}
        >
          Notes
        </button>
      </div>

      <Tooltip>
        <TooltipTrigger>
          <Button
            variant="ghost"
            size="icon"
            disabled
            class="size-[30px] rounded-full bg-muted text-muted-foreground transition-transform hover:bg-muted/70 active:scale-[0.96]"
            aria-label="Share notebook (available soon)"
          >
            <Share2 class="size-[13px]" strokeWidth={2} />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Available soon</TooltipContent>
      </Tooltip>

      {#if activeNotebook}
        <Tooltip>
          <TooltipTrigger>
            <Button
              variant="ghost"
              size="icon"
              class="size-[30px] rounded-full bg-muted text-muted-foreground transition-transform hover:bg-muted/70 hover:text-foreground active:scale-[0.96]"
              aria-label="Notebook settings"
              onclick={() => (notebookStore.notebookSettingsOpen = true)}
            >
              <Settings class="size-[13px]" strokeWidth={2} />
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
              <Settings class="size-[13px]" strokeWidth={2} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Select a notebook</TooltipContent>
        </Tooltip>
      {/if}
    </TooltipProvider>
  </div>
</div>

<style>
  /* ---- segmented toggle ---- */
  .seg {
    position: relative;
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: 1fr;
    padding: 2px;
    border-radius: 999px;
    background: var(--muted);
  }
  .seg-ind {
    position: absolute;
    top: 2px;
    left: 2px;
    height: calc(100% - 4px);
    width: calc(50% - 2px);
    border-radius: 999px;
    background: var(--card);
    box-shadow:
      0 1px 2px oklch(0.2 0.02 293 / 0.12),
      0 1px 1px oklch(0.2 0.02 293 / 0.06);
    transform: translateX(0);
    transition: transform calc(0.42s * var(--rail-motion, 1)) var(--ease-spring, ease);
  }
  .seg[data-tab='notes'] .seg-ind {
    transform: translateX(100%);
  }
  .seg-btn {
    position: relative;
    z-index: 1;
    min-width: 52px;
    height: 26px;
    padding: 0 13px;
    border: 0;
    background: transparent;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: -0.01em;
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
</style>
