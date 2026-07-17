<script lang="ts">
  import Aperture from '@lucide/svelte/icons/aperture';
  import PanelLeft from '@lucide/svelte/icons/panel-left';
  import PanelLeftClose from '@lucide/svelte/icons/panel-left-close';
  import Search from '@lucide/svelte/icons/search';
  import Plus from '@lucide/svelte/icons/plus';
  import Trash from '@lucide/svelte/icons/trash';
  import Settings from '@lucide/svelte/icons/settings';
  import Microscope from '@lucide/svelte/icons/microscope';
  import { cn } from '$lib/utils.js';
  import ThemeCycleButton from '$lib/components/ThemeCycleButton.svelte';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';
  import NotebookRow from '$lib/components/notebooks/NotebookRow.svelte';
  import AccountFooter from '$lib/components/notebooks/AccountFooter.svelte';
  import { notebookStore, openTrash, getInitials } from '$lib/notebooks/index.js';

  // `collapsed` is the effective layout state from AppShell; falls back to the
  // store's `sidebarCollapsed` when omitted so existing direct usage/tests work.
  let {
    onnewnotebook,
    userName = '',
    collapsed: collapsedProp
  }: {
    onnewnotebook?: () => void;
    userName?: string;
    collapsed?: boolean;
  } = $props();

  const notebooks = $derived(notebookStore.notebooks);
  const activeId = $derived(notebookStore.activeNotebookId);
  // Use the prop when provided (AppShell's effective collapsed state),
  // otherwise fall back to the persisted store value.
  const collapsed = $derived(collapsedProp ?? notebookStore.sidebarCollapsed);
  const trashCount = $derived(notebookStore.trashCount);
  const noActiveNotebook = $derived(activeId === null);

  let listEl = $state<HTMLElement | null>(null);
  let indicatorEl = $state<HTMLElement | null>(null);
  // First placement snaps (no glide from y=0); every later one glides.
  let firstPosition = true;

  // Slide the active indicator to the active row using a rect delta against the
  // list container (wrapper-agnostic). Row height is not transitioned, so the rect
  // is settled when we measure. happy-dom returns 0 here; position is verified in-app.
  // Returns true only when a row was actually positioned.
  function positionIndicator(animate = true): boolean {
    const el = indicatorEl;
    const list = listEl;
    if (!el || !list) return false;
    const idx = notebooks.findIndex((n) => n.id === activeId);
    if (idx < 0) {
      el.style.opacity = '0';
      return false;
    }
    const rows = list.querySelectorAll<HTMLElement>('[data-notebook-row]');
    const row = rows[idx];
    if (!row) {
      el.style.opacity = '0';
      return false;
    }
    const y = row.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop;
    if (!animate) el.style.transition = 'none';
    el.style.setProperty('--ind-y', `${y}px`);
    el.style.opacity = '1';
    if (!animate) {
      void el.offsetHeight; // flush the no-transition write before restoring
      el.style.transition = '';
    }
    return true;
  }

  // Reposition on active-notebook change, collapse toggle, and list changes. The
  // first SUCCESSFUL placement snaps (no glide from y=0); later ones glide. On cold
  // start `notebooks` is [] so the early return keeps `firstPosition` true until a
  // row actually places.
  $effect(() => {
    void activeId;
    void collapsed;
    void notebooks.length;
    const animate = !firstPosition;
    const raf = requestAnimationFrame(() => {
      if (positionIndicator(animate)) firstPosition = false;
    });
    return () => cancelAnimationFrame(raf);
  });

  $effect(() => {
    if (typeof window === 'undefined') return;
    const onResize = (): void => void positionIndicator(false);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  });

  function toggleCollapse(): void {
    notebookStore.sidebarCollapsed = !notebookStore.sidebarCollapsed;
  }

  function openPalette(): void {
    notebookStore.paletteOpen = true;
  }

  function handleNewNotebook(): void {
    onnewnotebook?.();
  }

  function openSettings(): void {
    notebookStore.settingsOpen = true;
  }

  function openInspector(): void {
    if (noActiveNotebook) return;
    notebookStore.inspectorOpen = true;
  }
</script>

<!--
  Single unified rail skeleton: the SAME DOM serves expanded and collapsed.
  `data-collapsed` on the root drives the label crossfade, the uniform 44×44
  collapsed boxes, and the sliding active indicator via scoped CSS.
-->
<TooltipProvider>
  <div data-sidebar data-collapsed={collapsed} class={cn('rail-root', 'text-sidebar-foreground')}>
    <!-- Traffic lights spacer (macOS native titlebar overlay) -->
    <div data-tauri-drag-region class="h-14 shrink-0"></div>

    <!-- brand -->
    <div class="brand">
      <span class="logo" aria-hidden="true">
        <Aperture class="size-4" />
      </span>
      <span class="wordmark lbl">Lens</span>
      <span class="brand-btns">
        <span class="brand-theme">
          <ThemeCycleButton variant="bare" />
        </span>
        <Tooltip disabled={!collapsed}>
          <TooltipTrigger
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            data-collapse-btn
            onclick={toggleCollapse}
            class={cn(
              'flex size-[26px] shrink-0 items-center justify-center rounded-full',
              'bg-muted text-sidebar-foreground/70 hover:text-sidebar-foreground hover:opacity-60',
              'cursor-pointer border-0 transition-opacity',
              'outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring'
            )}
          >
            {#if collapsed}
              <PanelLeft class="size-3.5" />
            {:else}
              <PanelLeftClose class="size-3.5" />
            {/if}
          </TooltipTrigger>
          <TooltipContent side="right">
            {collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          </TooltipContent>
        </Tooltip>
      </span>
    </div>

    <!-- search -->
    <Tooltip disabled={!collapsed}>
      <TooltipTrigger
        aria-label="Search notebooks (⌘K)"
        data-search-trigger
        onclick={openPalette}
        class={cn(
          'flex items-center rounded-[11px] border border-sidebar-border bg-surface-raised text-sm outline-none',
          'text-sidebar-foreground/55 cursor-text transition-[background,box-shadow,border-color]',
          'hover:bg-sidebar-accent/50 hover:text-sidebar-foreground/80',
          'focus-visible:border-sidebar-ring focus-visible:ring-[3px] focus-visible:ring-sidebar-ring/30',
          collapsed
            ? 'size-11 justify-center gap-0 self-center'
            : 'mb-3.5 h-[38px] w-full gap-[9px] px-2.5'
        )}
      >
        <Search class="size-4 shrink-0" aria-hidden="true" />
        <span class="lbl flex-1 text-left">Search notebooks</span>
        <kbd
          class={cn(
            'kbd inline-flex items-center gap-0.5 rounded px-1.5 py-0.5',
            'border border-sidebar-border bg-sidebar text-[0.65rem]',
            'font-medium text-sidebar-foreground/40'
          )}
          aria-hidden="true"
        >
          ⌘K
        </kbd>
      </TooltipTrigger>
      <TooltipContent side="right">Search notebooks (⌘K)</TooltipContent>
    </Tooltip>

    {#if !collapsed}
      <p class="sect-label">Notebooks</p>
    {/if}

    <!-- list + sliding active indicator -->
    <div class="list" bind:this={listEl}>
      {#if notebooks.length > 0}
        <div class="indicator" bind:this={indicatorEl} aria-hidden="true"></div>
      {/if}
      {#if notebooks.length === 0}
        <p class="px-2 py-4 text-center text-xs text-sidebar-foreground/40">No notebooks yet</p>
      {:else}
        {#each notebooks as nb (nb.id)}
          {#if collapsed}
            <Tooltip>
              <TooltipTrigger>
                {#snippet child({ props })}
                  <NotebookRow
                    notebook={nb}
                    active={nb.id === activeId}
                    {collapsed}
                    triggerProps={props}
                  />
                {/snippet}
              </TooltipTrigger>
              <TooltipContent side="right">{nb.title}</TooltipContent>
            </Tooltip>
          {:else}
            <NotebookRow notebook={nb} active={nb.id === activeId} {collapsed} />
          {/if}
        {/each}
      {/if}
    </div>

    <!-- actions -->
    <div class="actions">
      <Tooltip disabled={!collapsed}>
        <TooltipTrigger
          aria-label="New notebook"
          data-new-notebook-btn
          onclick={handleNewNotebook}
          class={cn(
            'flex items-center rounded-[11px] text-[13px] font-semibold outline-none cursor-pointer',
            'bg-primary text-primary-foreground shadow-sm transition-transform active:scale-[0.97]',
            'focus-visible:ring-2 focus-visible:ring-sidebar-ring',
            collapsed ? 'size-11 justify-center gap-0 self-center' : 'h-9 w-full gap-2.5 px-[11px]'
          )}
        >
          <Plus class="size-4 shrink-0" aria-hidden="true" />
          <span class="lbl">New notebook</span>
        </TooltipTrigger>
        <TooltipContent side="right">New notebook</TooltipContent>
      </Tooltip>

      <Tooltip disabled={!collapsed}>
        <TooltipTrigger
          aria-label={`Trash${trashCount > 0 ? ` (${trashCount} items)` : ''}`}
          data-trash-entry
          onclick={() => void openTrash()}
          class={cn(
            'flex items-center rounded-[11px] text-[13px] font-semibold outline-none cursor-pointer',
            'border border-sidebar-border bg-surface-raised text-sidebar-foreground',
            'transition-[transform,background] hover:bg-sidebar-accent/50 active:scale-[0.97]',
            'focus-visible:ring-2 focus-visible:ring-sidebar-ring',
            collapsed ? 'size-11 justify-center gap-0 self-center' : 'h-9 w-full gap-2.5 px-[11px]'
          )}
        >
          <Trash class="size-4 shrink-0" aria-hidden="true" />
          <span class="lbl">Trash</span>
          {#if trashCount > 0}
            <span class="count lbl" aria-hidden="true">{trashCount}</span>
          {/if}
        </TooltipTrigger>
        <TooltipContent side="right">
          Trash{trashCount > 0 ? ` (${trashCount})` : ''}
        </TooltipContent>
      </Tooltip>
    </div>

    <!-- footer: expanded = AccountFooter popup; collapsed = icon stack -->
    {#if !collapsed}
      <div class="foot px-1.5 pb-2 pt-1">
        <AccountFooter {userName} />
      </div>
    {:else}
      <div class="foot foot-icons" data-collapsed-footer>
        <Tooltip>
          <TooltipTrigger
            aria-label="Settings"
            data-settings-icon
            onclick={openSettings}
            class={cn(
              'flex size-[30px] items-center justify-center rounded-lg',
              'text-sidebar-foreground/60 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground',
              'transition-colors cursor-pointer border-0 bg-transparent'
            )}
          >
            <Settings class="size-4" />
          </TooltipTrigger>
          <TooltipContent side="right">Settings</TooltipContent>
        </Tooltip>

        <ThemeCycleButton variant="bare" />

        {#if import.meta.env.DEV}
          <Tooltip>
            <TooltipTrigger
              aria-label="Embeddings Inspector"
              data-embeddings-inspector-icon
              disabled={noActiveNotebook}
              aria-disabled={noActiveNotebook}
              onclick={openInspector}
              class={cn(
                'flex size-[30px] items-center justify-center rounded-lg transition-colors border-0 bg-transparent',
                noActiveNotebook
                  ? 'cursor-not-allowed text-sidebar-foreground/30'
                  : 'cursor-pointer text-sidebar-foreground/60 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground'
              )}
            >
              <Microscope class="size-4" />
            </TooltipTrigger>
            <TooltipContent side="right">
              {noActiveNotebook ? 'No active notebook' : 'Embeddings Inspector'}
            </TooltipContent>
          </Tooltip>
        {/if}

        <Tooltip>
          <TooltipTrigger
            aria-label={`Account: ${userName || 'user'} — expand sidebar`}
            data-account-avatar
            onclick={toggleCollapse}
            class={cn(
              'mt-1 flex size-7 items-center justify-center rounded-full',
              'bg-sidebar-primary text-sidebar-primary-foreground text-xs font-semibold',
              'cursor-pointer border-0 outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring',
              'transition-transform active:scale-95'
            )}
          >
            {getInitials(userName)}
          </TooltipTrigger>
          <TooltipContent side="right">{userName || 'Account'} — expand sidebar</TooltipContent>
        </Tooltip>
      </div>
    {/if}
  </div>
</TooltipProvider>

<style>
  .rail-root {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 0 12px 0;
  }

  /* Label crossfade. */
  .lbl {
    overflow: hidden;
    white-space: nowrap;
    max-width: 180px;
    opacity: 1;
    transition:
      opacity 0.28s var(--ease-out, ease),
      max-width calc(0.44s * var(--rail-motion, 1)) var(--ease-spring, ease),
      transform calc(0.34s * var(--rail-motion, 1)) var(--ease-out, ease),
      margin calc(0.44s * var(--rail-motion, 1)) var(--ease-spring, ease);
  }
  .rail-root[data-collapsed='true'] .lbl {
    opacity: 0;
    max-width: 0;
    transform: translateX(calc(-6px * var(--rail-motion, 1)));
    margin: 0 !important;
  }
  .rail-root[data-collapsed='true'] .kbd,
  .rail-root[data-collapsed='true'] .count,
  .rail-root[data-collapsed='true'] .wordmark,
  .rail-root[data-collapsed='true'] .brand-theme {
    display: none;
  }

  /* ---- brand ---- */
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 2px 4px 12px;
    min-height: 40px;
  }
  .rail-root[data-collapsed='true'] .brand {
    flex-direction: column;
    justify-content: center;
    gap: 8px;
    padding: 2px 0 14px;
    min-height: 0;
  }
  .brand-btns {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .rail-root[data-collapsed='true'] .brand-btns {
    gap: 0;
  }
  .logo {
    width: 30px;
    height: 30px;
    flex: none;
    display: grid;
    place-items: center;
    border-radius: 9px;
    color: var(--sidebar-primary-foreground);
    background: var(--sidebar-primary);
    box-shadow: inset 0 1px 0 rgb(255 255 255 / 0.18);
    transition: transform calc(0.5s * var(--rail-motion, 1)) var(--ease-spring, ease);
  }
  .rail-root:hover .logo {
    transform: rotate(calc(24deg * var(--rail-motion, 1)));
  }
  .wordmark {
    font-size: 16px;
    font-weight: 650;
    letter-spacing: -0.02em;
    white-space: nowrap;
    margin-right: auto;
  }

  /* ---- section label ---- */
  .sect-label {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: color-mix(in oklch, var(--sidebar-foreground) 40%, transparent);
    padding: 0 6px 8px;
  }

  /* ---- notebook list + sliding indicator ---- */
  /* Native overflow (not the app ScrollArea): the indicator measures against this
     position:relative container. */
  .list {
    position: relative;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 2px;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
  }
  .rail-root[data-collapsed='true'] .list {
    align-items: center;
  }
  /* The moving accent surface sits behind the rows via z-order. */
  .indicator {
    position: absolute;
    left: 2px;
    right: 2px;
    top: 0;
    height: 46px;
    border-radius: 12px;
    background: color-mix(in oklch, var(--primary) 12%, transparent);
    box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--primary) 14%, transparent);
    transform: translateY(var(--ind-y, 0px));
    opacity: 0;
    z-index: 0;
    pointer-events: none;
    transition:
      transform calc(0.5s * var(--rail-motion, 1)) var(--ease-spring, ease),
      opacity 0.3s var(--ease-out, ease);
  }
  .indicator::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    width: 3px;
    height: 20px;
    border-radius: 3px;
    background: var(--primary);
    transform: translateY(-50%);
    transition: opacity 0.2s var(--ease-out, ease);
  }
  .rail-root[data-collapsed='true'] .indicator {
    left: 50%;
    right: auto;
    margin-left: -22px;
    width: 44px;
    height: 44px;
  }
  .rail-root[data-collapsed='true'] .indicator::before {
    opacity: 0;
  }

  /* ---- actions ---- */
  .actions {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 2px 4px;
  }
  .rail-root[data-collapsed='true'] .actions {
    align-items: center;
  }
  .count {
    margin-left: auto;
    font-size: 11px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    min-width: 20px;
    height: 20px;
    padding: 0 6px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    background: color-mix(in oklch, var(--sidebar-accent) 60%, transparent);
    color: color-mix(in oklch, var(--sidebar-foreground) 70%, transparent);
  }

  /* ---- footer ---- */
  .foot {
    position: relative;
  }
  .foot-icons {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    padding: 9px 0 8px;
    border-top: 1px solid var(--sidebar-border);
  }
</style>
