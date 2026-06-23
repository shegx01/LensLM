<script lang="ts">
  // NotebookTopBar — center pane top chrome.
  //
  // Renders only when `notebookStore.activeNotebook` is non-null; AppShell
  // decides when to mount this component, but we guard internally too.
  //
  // Layout (per middle_content header.png):
  //   [drag region — fills remaining left space] [title] [Chat|Notes pill] [Share] [Settings]
  //
  // The entire bar background carries `data-tauri-drag-region` so the window
  // is still draggable by the transparent left portion. The right-aligned
  // interactive cluster is NOT a drag target (its pointer events are normal).
  //
  // Chat|Notes is real state via `notebookStore.activeTab` ('chat'|'notes').
  // Share + Settings are honestly disabled with "Available soon" tooltips (M8+).

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
  Full-width bar, height matches `--titlebar-h` (28px) so it aligns with the
  sidebar and right-rail drag bars.

  `data-tauri-drag-region` is on the container: the left transparent portion
  is the drag zone; the right cluster sits inside a flex child that does NOT
  propagate the drag event (native Tauri behaviour — interactive elements
  within a drag-region container naturally receive pointer events first).
-->
{#if activeNotebook}
  <div
    role="toolbar"
    data-tauri-drag-region
    class="flex h-[var(--titlebar-h)] w-full shrink-0 items-center"
    aria-label="Notebook toolbar"
  >
    <!-- Left: empty drag region — fills all remaining space -->
    <div data-tauri-drag-region class="flex-1"></div>

    <!-- Right: interactive cluster — NOT a drag region -->
    <div class="flex shrink-0 items-center gap-2 pr-3">
      <!-- Notebook title -->
      <span
        class="max-w-[200px] truncate text-sm font-bold text-foreground"
        title={activeNotebook.title}
      >
        {activeNotebook.title}
      </span>

      <!-- Chat | Notes segmented toggle (rounded-full pill) -->
      <TooltipProvider>
        <div
          role="group"
          aria-label="View toggle"
          class="flex items-center rounded-full border border-border bg-muted p-0.5"
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'chat'}
            aria-controls="notebook-tab-panel"
            class="rounded-full px-3 py-0.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
              {activeTab === 'chat'
              ? 'bg-primary text-primary-foreground shadow-sm'
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
            class="rounded-full px-3 py-0.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
              {activeTab === 'notes'
              ? 'bg-primary text-primary-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'}"
            onclick={() => setTab('notes')}
          >
            Notes
          </button>
        </div>

        <!-- Share button — disabled, "Available soon" -->
        <Tooltip>
          <TooltipTrigger>
            <Button
              variant="ghost"
              size="icon-sm"
              disabled
              aria-label="Share notebook (available soon)"
            >
              <Share2 class="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Available soon</TooltipContent>
        </Tooltip>

        <!-- Settings gear — disabled, "Available soon" -->
        <Tooltip>
          <TooltipTrigger>
            <Button
              variant="ghost"
              size="icon-sm"
              disabled
              aria-label="Notebook settings (available soon)"
            >
              <Settings class="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Available soon</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  </div>
{/if}
