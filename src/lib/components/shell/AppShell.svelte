<script lang="ts">
  import Aperture from '@lucide/svelte/icons/aperture';
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import type { AppConfig } from '$lib/theme/types.js';
  import SidebarRail from '$lib/components/shell/SidebarRail.svelte';
  import NotebookTopBar from '$lib/components/notebooks/NotebookTopBar.svelte';
  import TrashView from '$lib/components/notebooks/TrashView.svelte';
  import CommandPalette from '$lib/components/notebooks/CommandPalette.svelte';
  import NotebookCreateDialog from '$lib/components/notebooks/NotebookCreateDialog.svelte';
  import { notebookStore, loadNotebooks } from '$lib/notebooks/index.js';

  // ---------------------------------------------------------------------------
  // Local state
  // ---------------------------------------------------------------------------

  /** Controls NotebookCreateDialog visibility; opened by the sidebar's New button. */
  let createOpen = $state(false);

  /** User display name (AppConfig.user_name) for the sidebar account footer. */
  let userName = $state('');

  // ---------------------------------------------------------------------------
  // Reactive reads from the shared store
  // ---------------------------------------------------------------------------

  const activeNotebook = $derived(notebookStore.activeNotebook);

  // Left grid column width is driven ONLY by the persisted collapse state:
  // expanded = 256px, collapsed icon rail = 88px. Hover NEVER changes this — the
  // collapsed flyout (SidebarRail) expands as an absolute overlay over the centre
  // content, so the centre/right regions never reflow on hover. 88px is wide
  // enough that the floating panel (m-2/8px left gutter → inner edge ≈ x8) clears
  // the native traffic lights.
  const gridCols = $derived(
    notebookStore.sidebarCollapsed ? 'grid-cols-[88px_1fr_320px]' : 'grid-cols-[256px_1fr_320px]'
  );

  // ---------------------------------------------------------------------------
  // Global ⌘K handler (Step 4.10) — macOS-first / metaKey only for M3.
  // ---------------------------------------------------------------------------

  function isTypingTarget(el: Element | null): boolean {
    if (!el) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || (el as HTMLElement).isContentEditable === true;
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (!(e.metaKey && e.key === 'k')) return;
    e.preventDefault();
    if (notebookStore.paletteOpen) {
      // Unconditional close — no active-element guard so the palette's own
      // search input can't block its own close.
      notebookStore.paletteOpen = false;
      return;
    }
    // Guard on open only: don't steal focus while typing in an input/textarea/
    // contenteditable (e.g. the create-dialog name field or inline rename).
    if (isTypingTarget(document.activeElement)) return;
    notebookStore.paletteOpen = true;
  }

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);

    // Load notebooks + user name on mount (guarded for non-Tauri test/SSR env).
    void loadNotebooks();
    if (isTauri()) {
      invoke<AppConfig>('get_config')
        .then((cfg) => {
          userName = cfg.user_name ?? '';
        })
        .catch((err) => {
          console.error('AppShell: get_config failed', err);
        });
    }

    return () => {
      window.removeEventListener('keydown', handleKeydown);
    };
  });
</script>

<!-- Full-viewport app shell on a canvas ("container wall", bg-background). The
     LEFT rail is a floating panel inset from the window edges (subtle border +
     tiny shadow for a crisp elevation); the macOS native traffic lights
     (titleBarStyle "Overlay") sit on its top row. Each region has a top drag bar
     (data-tauri-drag-region) so the window can be moved by its top edge.
     The left column WIDTH is reactive to sidebarCollapsed and animates. -->
<div
  class={[
    'grid h-svh w-full bg-background transition-[grid-template-columns] duration-200 ease-out',
    gridCols
  ].join(' ')}
>
  <!-- LEFT: floating sidebar rail. SidebarRail owns the collapse / hover-flyout
       logic: the grid cell width above is fixed by sidebarCollapsed (88 / 256px),
       and when collapsed, hovering floats an expanded panel OVER the centre as an
       absolute overlay — the centre/right content never moves. Native traffic
       lights overlay the panel's top drag row. -->
  <SidebarRail onnewnotebook={() => (createOpen = true)} {userName} />

  <!-- CENTER: workspace on the canvas — top drag bar, then state-driven content.
       The floating pill header (NotebookTopBar) sits within the top area; Trash
       is a centered modal (mounted at shell root), not a center-pane view. -->
  <main class="flex flex-col overflow-hidden">
    <!-- Floating pill header — always present (its full-width outer row is the
         window drag region); the pill shows the title + Chat/Notes only when a
         notebook is active, and always exposes share + settings. -->
    <NotebookTopBar />
    {#if activeNotebook}
      <!-- Empty content region — chat/notes fill this in M5/M6. -->
      <div class="flex flex-1 flex-col overflow-hidden"></div>
    {:else}
      <div class="flex flex-1 flex-col items-center justify-center gap-2">
        <Aperture class="size-8 text-muted-foreground/40" />
        <p class="text-sm text-muted-foreground">Your workspace</p>
        <p class="text-xs text-muted-foreground/60">Select or create a notebook to begin</p>
      </div>
    {/if}
  </main>

  <!-- RIGHT: sources & studio rail — flush panel with a hairline divider; M4 fills -->
  <aside class="flex flex-col overflow-hidden border-l border-border bg-card text-card-foreground">
    <div data-tauri-drag-region class="flex h-[var(--titlebar-h)] items-center px-4">
      <span class="text-xs font-medium tracking-wide text-muted-foreground uppercase"
        >Sources &amp; Studio</span
      >
    </div>
  </aside>
</div>

<!-- Overlays mounted at shell root -->
<CommandPalette />
<TrashView />
<NotebookCreateDialog open={createOpen} onOpenChange={(v) => (createOpen = v)} />
