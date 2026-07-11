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
  import {
    notebookStore,
    loadNotebooks,
    refreshTrashed,
    refreshTrashedSources,
    selectNotebook
  } from '$lib/notebooks/index.js';
  import SourcesRail from '$lib/components/sources/SourcesRail.svelte';
  import PreferencesShell from '$lib/components/embeddings/PreferencesShell.svelte';
  import NotebookSettingsSheet from '$lib/components/embeddings/NotebookSettingsSheet.svelte';
  import ChatPane from '$lib/components/chat/ChatPane.svelte';

  let createOpen = $state(false);
  let userName = $state('');

  const activeNotebook = $derived(notebookStore.activeNotebook);

  // Preferences renders IN-PLACE (col-span-2), not as a floating overlay.
  const settingsOpen = $derived(notebookStore.settingsOpen);

  // Four combinations spelled out as STATIC class literals (not interpolated) so
  // Tailwind v4's static extractor emits each grid-cols rule. Collapsed = 104px
  // (fits the macOS traffic-light cluster with margin); expanded = 256px / 320px.
  const gridCols = $derived.by(() => {
    const left = notebookStore.sidebarCollapsed;
    const right = notebookStore.rightRailCollapsed;
    if (left && right) return 'grid-cols-[104px_1fr_104px]';
    if (left && !right) return 'grid-cols-[104px_1fr_320px]';
    if (!left && right) return 'grid-cols-[256px_1fr_104px]';
    return 'grid-cols-[256px_1fr_320px]';
  });

  // ⌘K handler — macOS-first / metaKey only for M3.

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

    // Load notebooks + config in parallel. Auto-select the most-recently-active
    // notebook only after BOTH have resolved, so there is no race between the
    // notebook list and the reopen_last_notebook flag.
    void (async () => {
      // Fetch notebooks and config in parallel; isolate get_config so a failure
      // does not abort auto-select — loadNotebooks() never rejects (it catches
      // internally), so notebooks are always populated regardless.
      const cfgPromise = isTauri()
        ? invoke<AppConfig>('get_config').catch((err) => {
            console.error('AppShell: get_config failed, defaulting reopen to true', err);
            return null;
          })
        : Promise.resolve(null);
      const [, cfg] = await Promise.all([loadNotebooks(), cfgPromise]);
      if (cfg) userName = (cfg as AppConfig).user_name ?? '';
      // default-on: null/undefined ⇒ open; only explicit `false` suppresses it.
      if (
        (cfg as AppConfig | null)?.reopen_last_notebook !== false &&
        !notebookStore.activeNotebookId &&
        notebookStore.notebooks.length > 0
      ) {
        selectNotebook(notebookStore.notebooks[0].id);
      }
    })();

    // Pre-load trash counts so the badge is correct from startup without a
    // loading flash. Uses raw refresh helpers (not loadTrashed/loadTrashedSources)
    // to avoid toggling the shared `loading` flag and flashing the UI.
    void refreshTrashed().catch(() => {});
    void refreshTrashedSources().catch(() => {});

    return () => {
      window.removeEventListener('keydown', handleKeydown);
    };
  });
</script>

<div
  class={[
    'grid h-svh w-full bg-background transition-[grid-template-columns] duration-200 ease-out',
    gridCols
  ].join(' ')}
>
  <SidebarRail onnewnotebook={() => (createOpen = true)} {userName} />

  {#if settingsOpen}
    <div class="col-span-2 flex min-w-0 overflow-hidden">
      <PreferencesShell />
    </div>
  {:else}
    <main class="flex flex-col overflow-hidden">
      <NotebookTopBar />
      {#if activeNotebook}
        <div class="flex flex-1 flex-col overflow-hidden">
          {#if notebookStore.activeTab === 'chat'}
            {#key activeNotebook.id}
              <ChatPane notebookId={activeNotebook.id} />
            {/key}
          {/if}
        </div>
      {:else if !notebookStore.loading}
        <!-- Gate on !loading to prevent an empty-state flash before auto-select fires. -->
        <div class="flex flex-1 flex-col items-center justify-center gap-2">
          <Aperture class="size-8 text-muted-foreground/40" />
          <p class="text-sm text-muted-foreground">Your workspace</p>
          <p class="text-xs text-muted-foreground/60">Select or create a notebook to begin</p>
        </div>
      {/if}
    </main>

    <aside
      class="flex flex-col overflow-hidden border-l border-border bg-card text-card-foreground"
    >
      <SourcesRail />
    </aside>
  {/if}
</div>

<CommandPalette />
<TrashView />
<NotebookCreateDialog open={createOpen} onOpenChange={(v) => (createOpen = v)} />

<NotebookSettingsSheet />

<!-- DEV-gated dynamic import: tree-shaken out of release bundles. -->
{#if import.meta.env.DEV}
  {#await import('$lib/components/inspector/EmbeddingsInspector.svelte') then { default: Inspector }}
    <Inspector />
  {/await}
{/if}
