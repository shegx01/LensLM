<!-- AddSourcesModal — tabbed "Add sources" modal (M4).
     Three tabs: Upload | URL | Paste text.
     Upload: functional (md/txt via @tauri-apps/plugin-dialog + existing addFileSource/ingest).
     URL:    deferred (Phase 2) — input rendered, action disabled with inline hint.
     Paste:  functional (addTextSource + ingest).
     Drag region: modal and ALL its controls are data-tauri-drag-region=none (no-drag).
     Tokens only — no hardcoded hex. -->
<script lang="ts">
  import { open as openFilePicker } from '@tauri-apps/plugin-dialog';
  import { isTauri } from '@tauri-apps/api/core';
  import X from '@lucide/svelte/icons/x';
  import Upload from '@lucide/svelte/icons/upload';
  import Link from '@lucide/svelte/icons/link';
  import { cn } from '$lib/utils.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { addFileSource, addTextSource } from '$lib/sources/ipc.js';
  import {
    addSourceLocal,
    loadSources,
    ingest,
    sourcesStore
  } from '$lib/sources/sources-state.svelte.js';
  import { notebookStore } from '$lib/notebooks/index.js';
  import { registerDropTarget, PICKER_FILTERS } from '$lib/sources/dragDrop.js';

  // ---------------------------------------------------------------------------
  // Props
  // ---------------------------------------------------------------------------

  let {
    open = false,
    onclose
  }: {
    /** Whether the modal is visible. */
    open?: boolean;
    /** Called when the modal should close. */
    onclose?: () => void;
  } = $props();

  // ---------------------------------------------------------------------------
  // Local state
  // ---------------------------------------------------------------------------

  type Tab = 'upload' | 'url' | 'paste';

  let activeTab = $state<Tab>('upload');

  /** Upload tab — drag-over highlight */
  let dragOver = $state(false);

  /** URL tab */
  let urlValue = $state('');

  /** Paste tab */
  let pasteTitle = $state('');
  let pasteContent = $state('');
  let pasteError = $state<string | null>(null);
  let pasteSubmitting = $state(false);

  /** Upload submitting + error */
  let uploadSubmitting = $state(false);
  let uploadError = $state<string | null>(null);

  /** Drop zone element ref — used by the native drag-drop manager. */
  let dropZoneEl = $state<HTMLDivElement | undefined>(undefined);

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------

  const activeNotebookId = $derived(notebookStore.activeNotebookId);
  const activeNotebook = $derived(notebookStore.activeNotebook);

  const pasteCanSubmit = $derived(pasteContent.trim().length > 0 && !pasteSubmitting);

  // ---------------------------------------------------------------------------
  // Reset on open
  // ---------------------------------------------------------------------------

  $effect(() => {
    if (open) {
      activeTab = 'upload';
      urlValue = '';
      pasteTitle = '';
      pasteContent = '';
      pasteError = null;
      pasteSubmitting = false;
      uploadSubmitting = false;
      uploadError = null;
      dragOver = false;
    }
  });

  // ---------------------------------------------------------------------------
  // Keyboard close (Escape)
  // ---------------------------------------------------------------------------

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      onclose?.();
    }
  }

  // ---------------------------------------------------------------------------
  // Upload tab handlers
  // ---------------------------------------------------------------------------

  async function handleBrowse(): Promise<void> {
    if (!isTauri() || !activeNotebookId || uploadSubmitting) return;
    uploadError = null;
    uploadSubmitting = true;
    try {
      const selected = await openFilePicker({
        multiple: false,
        filters: PICKER_FILTERS
      });
      if (!selected) return;
      const path = Array.isArray(selected) ? selected[0] : selected;
      // Use both / and \ to extract the filename (Tauri returns OS-native paths).
      const name = path.split(/[\\/]/).pop() ?? path;
      const source = await addFileSource(activeNotebookId, name, path);
      // Optimistically insert the row BEFORE starting ingest so progress events
      // find the entry in the store immediately (avoids silent drops).
      addSourceLocal(source);
      void ingest(source.id);
      onclose?.();
      // Reconcile with backend ordering after the modal closes.
      void loadSources(activeNotebookId);
    } catch (err) {
      uploadError = 'Could not add file. Please try again.';
      console.error('AddSourcesModal: handleBrowse failed', err);
    } finally {
      uploadSubmitting = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Native drag-drop (Tauri v2) — registered via the app-level drop manager.
  // The drop zone only exists while the modal is open AND the Upload tab is
  // active (it lives behind `{#if open}` + `{#if activeTab === 'upload'}`), so a
  // one-shot onMount registration would miss it. Instead, (re)register whenever
  // the bound element mounts and clean up when it unmounts — an $effect keyed on
  // `dropZoneEl` handles open/close and tab switches in both directions.
  // ---------------------------------------------------------------------------

  $effect(() => {
    // Registering only while the drop-zone element is mounted (modal open +
    // Upload tab active) scopes the drop to this surface — the manager routes
    // drops to the active (last-registered) target. No coordinate hit-test.
    if (!dropZoneEl) return;
    return registerDropTarget({
      setHover: (h) => {
        dragOver = h;
      },
      onDrop: async (paths) => {
        if (!activeNotebookId || uploadSubmitting) return;
        uploadError = null;
        uploadSubmitting = true;
        let addedAny = false;
        try {
          for (const path of paths) {
            // Skip files already added to this notebook (dedup by locator).
            if (sourcesStore.sources.some((s) => s.locator === path)) continue;
            const name = path.split(/[\\/]/).pop() ?? path;
            try {
              const source = await addFileSource(activeNotebookId, name, path);
              addSourceLocal(source);
              void ingest(source.id);
              addedAny = true;
            } catch (err) {
              uploadError = 'Could not add file. Please try again.';
              console.error('AddSourcesModal: drop ingest failed for', path, err);
            }
          }
        } finally {
          uploadSubmitting = false;
          void loadSources(activeNotebookId);
        }
        // Dismiss the modal after a successful drop so the new sources are
        // visible in the rail (mirrors the browse flow). Stay open when nothing
        // was added (all duplicates / all failed) so the error remains visible.
        if (addedAny) onclose?.();
      }
    });
  });

  // ---------------------------------------------------------------------------
  // Paste tab handler
  // ---------------------------------------------------------------------------

  async function handlePasteSubmit(): Promise<void> {
    if (!pasteCanSubmit || !activeNotebookId) return;
    pasteError = null;
    pasteSubmitting = true;
    try {
      const title = pasteTitle.trim() || 'Untitled text';
      const source = await addTextSource(activeNotebookId, title, pasteContent.trim(), 'text');
      // Optimistically insert the row BEFORE starting ingest so progress events
      // find the entry in the store immediately (avoids silent drops).
      addSourceLocal(source);
      void ingest(source.id);
      onclose?.();
      // Reconcile with backend ordering after the modal closes.
      void loadSources(activeNotebookId);
    } catch (err) {
      pasteError = 'Could not add source. Please try again.';
      console.error('AddSourcesModal: handlePasteSubmit failed', err);
    } finally {
      pasteSubmitting = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Footer action dispatcher
  // ---------------------------------------------------------------------------

  async function handlePrimaryAction(): Promise<void> {
    if (activeTab === 'upload') await handleBrowse();
    else if (activeTab === 'paste') await handlePasteSubmit();
    // 'url' tab: action is disabled — no-op
  }

  // Note: pasteCanSubmit already incorporates !pasteSubmitting, so we don't need a
  // standalone pasteSubmitting clause here (which would wrongly disable the upload tab
  // while a paste submission is in flight on another tab).
  const primaryDisabled = $derived(
    activeTab === 'url' ||
      (activeTab === 'paste' && !pasteCanSubmit) ||
      (activeTab === 'upload' && uploadSubmitting)
  );
</script>

{#if open}
  <!-- Backdrop — full-screen, no-drag -->
  <!-- svelte-ignore a11y_interactive_supports_focus -->
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
    role="dialog"
    aria-modal="true"
    aria-label="Add sources"
    onkeydown={handleKeydown}
    style="-webkit-app-region: no-drag;"
  >
    <!-- Modal card — no-drag -->
    <div
      class="relative w-full max-w-[480px] rounded-2xl border border-border bg-card text-card-foreground shadow-2xl"
      style="-webkit-app-region: no-drag;"
      role="document"
    >
      <!-- ── Header ── -->
      <div class="flex items-start justify-between px-5 pt-5 pb-0">
        <div class="flex items-center gap-2.5">
          <!-- Upload icon pill -->
          <div
            class="flex size-7 shrink-0 items-center justify-center rounded-lg bg-primary/10"
            aria-hidden="true"
          >
            <Upload class="size-3.5 text-primary" strokeWidth={2} />
          </div>
          <div>
            <h2 class="text-[14px] font-semibold leading-tight text-card-foreground">
              Add sources
            </h2>
            {#if activeNotebook}
              <p class="text-[11px] text-muted-foreground leading-tight mt-0.5">
                {activeNotebook.title}
              </p>
            {/if}
          </div>
        </div>
        <!-- Close X — no-drag -->
        <button
          class="flex size-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          type="button"
          aria-label="Close"
          onclick={onclose}
          style="-webkit-app-region: no-drag;"
        >
          <X class="size-[13px]" strokeWidth={2} />
        </button>
      </div>

      <!-- ── Tabs ── -->
      <div class="flex items-center gap-1.5 px-5 pt-4 pb-0" role="tablist" aria-label="Source type">
        {#each ['upload', 'url', 'paste'] as Tab[] as tab (tab)}
          {@const label = tab === 'upload' ? 'Upload' : tab === 'url' ? 'URL' : 'Paste text'}
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            aria-controls={`sources-tab-panel-${tab}`}
            class={cn(
              'rounded-lg border px-3 py-1.5 text-[12px] font-semibold transition-all duration-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              activeTab === tab
                ? 'border-primary bg-primary/5 text-primary'
                : 'border-transparent bg-transparent text-muted-foreground hover:bg-muted hover:text-foreground'
            )}
            onclick={() => (activeTab = tab)}
            style="-webkit-app-region: no-drag;"
          >
            {label}
          </button>
        {/each}
      </div>

      <!-- ── Tab content ── -->
      <div class="px-5 py-4">
        <!-- UPLOAD TAB -->
        {#if activeTab === 'upload'}
          <div
            id="sources-tab-panel-upload"
            role="tabpanel"
            aria-label="Upload files"
            style="-webkit-app-region: no-drag;"
          >
            <!-- Drop zone -->
            <div
              bind:this={dropZoneEl}
              class={cn(
                'flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors duration-100',
                dragOver
                  ? 'border-primary bg-primary/5'
                  : 'border-border bg-muted/30 hover:border-primary/40 hover:bg-muted/50'
              )}
              role="region"
              aria-label="File drop zone"
            >
              <Upload class="size-6 text-muted-foreground/60" strokeWidth={1.5} />
              <div>
                <p class="text-[13px] font-medium text-foreground">Drop files here</p>
                <p class="mt-0.5 text-[12px] text-muted-foreground">
                  or
                  <button
                    type="button"
                    class="text-primary underline underline-offset-2 transition-opacity hover:opacity-70 focus-visible:outline-none"
                    onclick={handleBrowse}
                    disabled={uploadSubmitting}
                    style="-webkit-app-region: no-drag;"
                  >
                    browse your computer
                  </button>
                </p>
              </div>
            </div>

            <!-- Supported format list -->
            <div class="mt-4 space-y-1.5 text-center text-[11px] text-muted-foreground/70">
              <p>
                <span class="font-semibold uppercase tracking-wide text-muted-foreground/50"
                  >DOCUMENTS</span
                >
                &nbsp;PDF · DOCX · RTF · ODT · TXT · MD · EPUB
              </p>
              <p>
                <span class="font-semibold uppercase tracking-wide text-muted-foreground/50"
                  >JSON</span
                >
                &nbsp;JSON · JSONL · NDJSON · YAML · YML · XML
              </p>
              <p class="mt-1 italic text-muted-foreground/40">
                Supported: .md · .markdown · .mdx · .txt · .pdf · .docx · .rtf · .odt · .epub ·
                .xlsx · .xls · .csv · .json · .jsonl · .ndjson · .yaml · .yml · .xml
              </p>
            </div>

            <!-- Upload error feedback -->
            {#if uploadError}
              <p class="mt-3 text-[12px] text-destructive" role="alert">{uploadError}</p>
            {/if}
          </div>

          <!-- URL TAB -->
        {:else if activeTab === 'url'}
          <div
            id="sources-tab-panel-url"
            role="tabpanel"
            aria-label="Add by URL"
            style="-webkit-app-region: no-drag;"
          >
            <div class="mb-3">
              <label
                class="mb-1.5 block text-[10px] font-semibold uppercase tracking-widest text-muted-foreground"
                for="add-sources-url"
              >
                Web page URL
              </label>
              <input
                id="add-sources-url"
                class="w-full rounded-lg border border-border bg-background px-3 py-2 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-ring"
                type="url"
                placeholder="https://example.com/article"
                bind:value={urlValue}
                autocomplete="off"
                style="-webkit-app-region: no-drag;"
              />
            </div>
            <p class="text-[12px] text-muted-foreground/70 leading-relaxed">
              Supports web pages, blog posts, documentation and GitHub repos. Content is fetched and
              indexed locally.
            </p>
            <!-- Phase 2 deferral notice -->
            <p
              class="mt-3 flex items-center gap-1.5 rounded-lg bg-muted/60 px-3 py-2 text-[11px] text-muted-foreground"
              role="note"
            >
              <Link class="size-3 shrink-0" strokeWidth={2} />
              URL ingestion is available in the next update.
            </p>
          </div>

          <!-- PASTE TEXT TAB -->
        {:else}
          <div
            id="sources-tab-panel-paste"
            role="tabpanel"
            aria-label="Paste text"
            style="-webkit-app-region: no-drag;"
          >
            <!-- Title input -->
            <div class="mb-3">
              <label
                class="mb-1.5 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground"
                for="paste-title"
              >
                Title
                <span class="font-normal normal-case tracking-normal text-muted-foreground/50"
                  >— optional</span
                >
              </label>
              <input
                id="paste-title"
                class="w-full rounded-lg border border-border bg-background px-3 py-2 text-[13px] text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:ring-2 focus:ring-ring"
                type="text"
                placeholder="e.g. Meeting notes — 12 Jan"
                bind:value={pasteTitle}
                disabled={pasteSubmitting}
                style="-webkit-app-region: no-drag;"
              />
            </div>

            <!-- Content textarea -->
            <div class="mb-1">
              <label
                class="mb-1.5 block text-[10px] font-semibold uppercase tracking-widest text-muted-foreground"
                for="paste-content"
              >
                Content
              </label>
              <textarea
                id="paste-content"
                class="h-[140px] w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-[13px] text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder="Paste any text — notes, transcripts, research…"
                maxlength={500000}
                bind:value={pasteContent}
                disabled={pasteSubmitting}
                style="-webkit-app-region: no-drag;"
              ></textarea>
            </div>

            {#if pasteError}
              <p class="mt-2 text-[12px] text-destructive" role="alert">{pasteError}</p>
            {/if}
          </div>
        {/if}
      </div>

      <!-- ── Footer ── -->
      <div
        class="flex items-center justify-between border-t border-border px-5 py-3"
        style="-webkit-app-region: no-drag;"
      >
        <Button
          variant="ghost"
          class="h-[34px] px-4 text-[13px] text-muted-foreground"
          onclick={onclose}
          style="-webkit-app-region: no-drag;"
        >
          Cancel
        </Button>
        <Button
          class="h-[34px] px-4 text-[13px] font-semibold"
          disabled={primaryDisabled}
          onclick={handlePrimaryAction}
          style="-webkit-app-region: no-drag;"
        >
          {#if activeTab === 'upload' && uploadSubmitting}
            Adding…
          {:else if activeTab === 'paste' && pasteSubmitting}
            Adding…
          {:else}
            Add to notebook →
          {/if}
        </Button>
      </div>
    </div>
  </div>
{/if}
