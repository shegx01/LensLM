<script lang="ts">
  // Floating card top bar — same family as the left rail: bg-card surface, the
  // layered --shadow-bar lift, and a Chat/Notes segmented control whose active
  // pill SLIDES on the rail's spring (--ease-spring, gated by --rail-motion). The
  // whole bar is a Tauri drag region; interactive children (toggle, buttons) opt
  // out by being real controls, which still receive pointer events.

  import BookOpen from '@lucide/svelte/icons/book-open';
  import Share2 from '@lucide/svelte/icons/share-2';
  import Settings from '@lucide/svelte/icons/settings';
  import { cn } from '$lib/utils.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';
  import { notebookStore, notebookColorClass } from '$lib/notebooks/index.js';

  const activeNotebook = $derived(notebookStore.activeNotebook);
  const activeTab = $derived(notebookStore.activeTab);
  const accentClass = $derived(activeNotebook ? notebookColorClass(activeNotebook.id) : '');

  function setTab(tab: 'chat' | 'notes'): void {
    notebookStore.activeTab = tab;
  }
</script>

<div data-tauri-drag-region class="shrink-0 px-3 pt-2.5 pb-1">
  <div
    data-tauri-drag-region
    role="toolbar"
    aria-label="Notebook toolbar"
    class="flex h-12 items-center gap-2 rounded-2xl bg-card pr-1.5 pl-2 shadow-[var(--shadow-bar)]"
  >
    <!-- Identity — colored icon tile + title, echoing the rail's notebook rows. -->
    {#if activeNotebook}
      <div class={cn('nb-mark', accentClass)} aria-hidden="true">
        <BookOpen class="size-4" />
      </div>
      <span
        data-tauri-drag-region
        class="min-w-0 flex-1 truncate text-[13px] font-semibold tracking-[-0.01em] text-card-foreground"
        title={activeNotebook.title}
      >
        {activeNotebook.title}
      </span>
    {:else}
      <div class="nb-mark nb-mark--muted" aria-hidden="true">
        <BookOpen class="size-4" />
      </div>
      <span
        data-tauri-drag-region
        class="min-w-0 flex-1 truncate text-[13px] text-muted-foreground"
      >
        No notebook selected
      </span>
    {/if}

    <TooltipProvider>
      <!-- Segmented toggle: absolutely-positioned pill slides between the two
           tabs on the rail spring. Equal-width tabs → translateX(100%) lands
           the pill exactly on the second tab (its own width == a tab's width). -->
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
            class="size-8 rounded-full text-muted-foreground transition-transform hover:bg-muted active:scale-[0.96]"
            aria-label="Share notebook (available soon)"
          >
            <Share2 class="size-[15px]" strokeWidth={2} />
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
              class="size-8 rounded-full text-muted-foreground transition-transform hover:bg-muted hover:text-foreground active:scale-[0.96]"
              aria-label="Notebook settings"
              onclick={() => (notebookStore.notebookSettingsOpen = true)}
            >
              <Settings class="size-[15px]" strokeWidth={2} />
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
              class="size-8 rounded-full text-muted-foreground hover:bg-muted"
              aria-label="Notebook settings (no active notebook)"
            >
              <Settings class="size-[15px]" strokeWidth={2} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Select a notebook</TooltipContent>
        </Tooltip>
      {/if}
    </TooltipProvider>
  </div>
</div>

<style>
  /* Icon tile — matches the rail's 30px accented notebook tile (concentric 9px). */
  .nb-mark {
    width: 30px;
    height: 30px;
    flex: none;
    display: grid;
    place-items: center;
    border-radius: 9px;
    transition: transform calc(0.5s * var(--rail-motion, 1)) var(--ease-spring, ease);
  }
  .nb-mark--muted {
    background: var(--muted);
    color: var(--muted-foreground);
  }
  /* Playful nod to the rail's logo-rotate on hover. */
  [role='toolbar']:hover .nb-mark {
    transform: rotate(calc(-4deg * var(--rail-motion, 1)));
  }

  /* ---- segmented toggle ---- */
  .seg {
    position: relative;
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: 1fr;
    padding: 3px;
    border-radius: 13px;
    background: var(--muted);
  }
  .seg-ind {
    position: absolute;
    top: 3px;
    left: 3px;
    height: calc(100% - 6px);
    width: calc(50% - 3px);
    border-radius: 10px;
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
    min-width: 58px;
    height: 28px;
    padding: 0 12px;
    border: 0;
    background: transparent;
    border-radius: 10px;
    font-size: 11.5px;
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
