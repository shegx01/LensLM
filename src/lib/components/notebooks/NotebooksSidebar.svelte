<script lang="ts">
  import Aperture from '@lucide/svelte/icons/aperture';
  import PanelLeft from '@lucide/svelte/icons/panel-left';
  import PanelLeftClose from '@lucide/svelte/icons/panel-left-close';
  import Search from '@lucide/svelte/icons/search';
  import Plus from '@lucide/svelte/icons/plus';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import { cn } from '$lib/utils.js';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import { Separator } from '$lib/components/ui/separator/index.js';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';
  import ThemeSwitcher from '$lib/components/ThemeSwitcher.svelte';
  import NotebookRow from '$lib/components/notebooks/NotebookRow.svelte';
  import AccountFooter from '$lib/components/notebooks/AccountFooter.svelte';
  import { notebookStore, openTrash, getInitials } from '$lib/notebooks/index.js';

  /**
   * Callback fired when the user clicks "New notebook".
   *
   * Integration note: AppShell (Wave 3) must wire this to open NotebookCreateDialog,
   * e.g.:  <NotebooksSidebar onnewnotebook={() => (createDialogOpen = true)} />
   */
  let {
    onnewnotebook,
    userName = ''
  }: {
    onnewnotebook?: () => void;
    /**
     * Display name from AppConfig `user_name`. Passed down to AccountFooter.
     * AppShell wires this from `config.user_name`.
     */
    userName?: string;
  } = $props();

  // ---------------------------------------------------------------------------
  // Derived state — reads from the reactive store
  // ---------------------------------------------------------------------------

  const notebooks = $derived(notebookStore.notebooks);
  const activeId = $derived(notebookStore.activeNotebookId);
  const collapsed = $derived(notebookStore.sidebarCollapsed);
  const trashCount = $derived(notebookStore.trashCount);

  function toggleCollapse(): void {
    notebookStore.sidebarCollapsed = !notebookStore.sidebarCollapsed;
  }

  function openPalette(): void {
    notebookStore.paletteOpen = true;
  }

  function handleNewNotebook(): void {
    onnewnotebook?.();
  }
</script>

<!--
  NotebooksSidebar: the full-height left rail. In EXPANDED mode it renders the
  brand row, search trigger, notebooks list, new/trash footer, and account row.
  In COLLAPSED mode it renders an icon-only rail with tooltips and a + button.

  Width transition is handled by the parent AppShell's aside element via the
  `sidebarCollapsed` store value; this component fills that element entirely.
-->
<div data-sidebar class={cn('flex h-full flex-col', 'text-sidebar-foreground')}>
  {#if !collapsed}
    <!-- ──────────────────────────────────────────────────────────────────────
         EXPANDED LAYOUT
    ────────────────────────────────────────────────────────────────────────── -->

    <!-- Traffic lights spacer (macOS native titlebar overlay) -->
    <div data-tauri-drag-region class="h-14 shrink-0"></div>

    <!-- Brand row -->
    <div class="flex items-center gap-2 px-3 pb-2">
      <!-- Logo -->
      <div
        class="flex size-7 shrink-0 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground"
        aria-hidden="true"
      >
        <Aperture class="size-4" />
      </div>
      <span class="flex-1 text-base font-semibold text-sidebar-foreground">Lens</span>

      <!-- Theme switcher -->
      <ThemeSwitcher
        class="size-7 border-0 bg-transparent shadow-none hover:bg-sidebar-accent/60 rounded-md"
      />

      <!-- Collapse button -->
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            aria-label="Collapse sidebar"
            data-collapse-btn
            onclick={toggleCollapse}
            class={cn(
              'flex size-7 items-center justify-center rounded-md',
              'text-sidebar-foreground/60 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
              'transition-colors cursor-pointer border-0 bg-transparent'
            )}
          >
            <PanelLeftClose class="size-4" />
          </TooltipTrigger>
          <TooltipContent side="right">Collapse sidebar</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>

    <!-- Search trigger (button that looks like an input) -->
    <div class="px-3 pb-3">
      <button
        type="button"
        aria-label="Search notebooks (⌘K)"
        data-search-trigger
        onclick={openPalette}
        class={cn(
          'flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5',
          'border border-sidebar-border bg-sidebar-accent/30',
          'text-sidebar-foreground/50 text-sm',
          'hover:bg-sidebar-accent/60 hover:text-sidebar-foreground/80',
          'transition-colors cursor-pointer outline-none',
          'focus-visible:ring-2 focus-visible:ring-sidebar-ring'
        )}
      >
        <Search class="size-3.5 shrink-0" />
        <span class="flex-1 text-left text-[0.8125rem]">Search notebooks</span>
        <kbd
          class={cn(
            'inline-flex items-center gap-0.5 rounded px-1.5 py-0.5',
            'border border-sidebar-border bg-sidebar text-[0.65rem]',
            'font-medium text-sidebar-foreground/40'
          )}
          aria-hidden="true"
        >
          ⌘K
        </kbd>
      </button>
    </div>

    <!-- "NOTEBOOKS" section label -->
    <p
      class="px-3 pb-1.5 text-[0.6875rem] font-medium tracking-widest text-sidebar-foreground/40 uppercase"
    >
      Notebooks
    </p>

    <!-- Notebook list -->
    <ScrollArea class="flex-1 min-h-0 px-1.5">
      <div class="flex flex-col gap-0.5 py-0.5">
        {#if notebooks.length === 0}
          <p class="px-2 py-4 text-center text-xs text-sidebar-foreground/40">No notebooks yet</p>
        {:else}
          {#each notebooks as nb (nb.id)}
            <NotebookRow notebook={nb} active={nb.id === activeId} />
          {/each}
        {/if}
      </div>
    </ScrollArea>

    <!-- Bottom actions: New notebook + Trash -->
    <div class="px-1.5 py-1.5 flex flex-col gap-0.5">
      <!-- New notebook button -->
      <button
        type="button"
        aria-label="New notebook"
        data-new-notebook-btn
        onclick={handleNewNotebook}
        class={cn(
          'flex w-full items-center gap-2 rounded-lg px-2 py-2',
          'text-sidebar-foreground/70 text-sm',
          'hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
          'transition-colors cursor-pointer outline-none',
          'focus-visible:ring-2 focus-visible:ring-sidebar-ring'
        )}
      >
        <div
          class="flex size-6 shrink-0 items-center justify-center rounded-md bg-sidebar-accent/60"
          aria-hidden="true"
        >
          <Plus class="size-3.5" />
        </div>
        <span>New notebook</span>
      </button>

      <!-- Trash entry -->
      <button
        type="button"
        aria-label={`Trash${trashCount > 0 ? ` (${trashCount} items)` : ''}`}
        data-trash-entry
        onclick={() => void openTrash()}
        class={cn(
          'flex w-full items-center gap-2 rounded-lg px-2 py-2',
          'text-sidebar-foreground/70 text-sm',
          'hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
          'transition-colors cursor-pointer outline-none',
          'focus-visible:ring-2 focus-visible:ring-sidebar-ring'
        )}
      >
        <div
          class="flex size-6 shrink-0 items-center justify-center rounded-md bg-sidebar-accent/60"
          aria-hidden="true"
        >
          <Trash2 class="size-3.5" />
        </div>
        <span class="flex-1">Trash</span>
        {#if trashCount > 0}
          <span
            class="inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-sidebar-foreground/15 px-1 text-[0.625rem] font-medium text-sidebar-foreground/60"
            aria-hidden="true"
          >
            {trashCount}
          </span>
        {/if}
      </button>
    </div>

    <Separator class="bg-sidebar-border/60" />

    <!-- Account footer -->
    <div class="px-1.5 pb-2 pt-1">
      <AccountFooter {userName} />
    </div>
  {:else}
    <!-- ──────────────────────────────────────────────────────────────────────
         COLLAPSED ICON RAIL
    ────────────────────────────────────────────────────────────────────────── -->

    <!-- Traffic lights spacer -->
    <div data-tauri-drag-region class="h-14 shrink-0"></div>

    <div class="flex flex-col items-center gap-1.5 px-1.5">
      <!-- Logo -->
      <div
        class="flex size-8 shrink-0 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground"
        aria-hidden="true"
      >
        <Aperture class="size-4" />
      </div>

      <!-- Expand button -->
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            aria-label="Expand sidebar"
            data-collapse-btn
            onclick={toggleCollapse}
            class={cn(
              'flex size-8 items-center justify-center rounded-lg',
              'text-sidebar-foreground/60 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
              'transition-colors cursor-pointer border-0 bg-transparent'
            )}
          >
            <PanelLeft class="size-4" />
          </TooltipTrigger>
          <TooltipContent side="right">Expand sidebar</TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <!-- Search icon -->
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            aria-label="Search notebooks (⌘K)"
            data-search-trigger
            onclick={openPalette}
            class={cn(
              'flex size-8 items-center justify-center rounded-lg',
              'text-sidebar-foreground/60 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
              'transition-colors cursor-pointer border-0 bg-transparent'
            )}
          >
            <Search class="size-4" />
          </TooltipTrigger>
          <TooltipContent side="right">Search notebooks (⌘K)</TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <Separator class="w-6 bg-sidebar-border/60 my-1" />

      <!-- Notebook icon list -->
      <ScrollArea class="flex-1 min-h-0 w-full">
        <div class="flex flex-col items-center gap-1.5 py-0.5">
          {#each notebooks as nb (nb.id)}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger class="cursor-pointer border-0 bg-transparent p-0">
                  <NotebookRow notebook={nb} active={nb.id === activeId} collapsed />
                </TooltipTrigger>
                <TooltipContent side="right">{nb.title}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          {/each}
        </div>
      </ScrollArea>
    </div>

    <!-- Spacer -->
    <div class="flex-1"></div>

    <!-- Bottom: + Trash Avatar -->
    <div class="flex flex-col items-center gap-1.5 px-1.5 pb-2">
      <!-- New notebook -->
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            aria-label="New notebook"
            data-new-notebook-btn
            onclick={handleNewNotebook}
            class={cn(
              'flex size-8 items-center justify-center rounded-lg',
              'bg-sidebar-accent/60 text-sidebar-foreground',
              'hover:bg-sidebar-accent hover:text-sidebar-foreground',
              'transition-colors cursor-pointer border-0'
            )}
          >
            <Plus class="size-4" />
          </TooltipTrigger>
          <TooltipContent side="right">New notebook</TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <!-- Trash (with count dot) -->
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            aria-label={`Trash${trashCount > 0 ? ` (${trashCount})` : ''}`}
            data-trash-entry
            onclick={() => void openTrash()}
            class={cn(
              'relative flex size-8 items-center justify-center rounded-lg',
              'text-sidebar-foreground/60 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
              'transition-colors cursor-pointer border-0 bg-transparent'
            )}
          >
            <Trash2 class="size-4" />
            {#if trashCount > 0}
              <span
                class="absolute -top-0.5 -right-0.5 flex size-3.5 items-center justify-center rounded-full bg-sidebar-primary text-sidebar-primary-foreground text-[0.5rem] font-bold"
                aria-hidden="true"
              >
                {trashCount > 9 ? '9+' : trashCount}
              </span>
            {/if}
          </TooltipTrigger>
          <TooltipContent side="right"
            >Trash{trashCount > 0 ? ` (${trashCount})` : ''}</TooltipContent
          >
        </Tooltip>
      </TooltipProvider>

      <!-- Initials avatar — collapsed version (no popover, just label) -->
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            aria-label={`Account: ${userName || 'user'}`}
            class={cn(
              'flex size-8 items-center justify-center rounded-full',
              'bg-sidebar-primary text-sidebar-primary-foreground text-xs font-semibold',
              'cursor-pointer border-0'
            )}
          >
            {getInitials(userName)}
          </TooltipTrigger>
          <TooltipContent side="right">{userName || 'Account'}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  {/if}
</div>
