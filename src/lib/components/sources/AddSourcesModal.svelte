<!-- AddSourcesModal — tabbed "Add sources" modal (Upload | URL | Paste).
     All controls carry -webkit-app-region: no-drag. Tokens only — no hardcoded hex.
     URL tab: static pages use fast extract; JS/SPA pages fall through to offscreen webview (#78). -->
<script lang="ts">
  import { open as openFilePicker } from '@tauri-apps/plugin-dialog';
  import { isTauri } from '@tauri-apps/api/core';
  import X from '@lucide/svelte/icons/x';
  import Upload from '@lucide/svelte/icons/upload';
  import Check from '@lucide/svelte/icons/check';
  import { cn } from '$lib/utils.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { addFileSource, addTextSource, addUrlSource } from '$lib/sources/ipc.js';
  import { addSourceLocal, loadSources, ingest } from '$lib/sources/sources-state.svelte.js';
  import { notebookStore } from '$lib/notebooks/index.js';
  import { registerDropTarget, PICKER_FILTERS } from '$lib/sources/dragDrop.js';
  import { showToast } from '$lib/sources/toast.svelte.js';

  let {
    open = false,
    onclose
  }: {
    /** Whether the modal is visible. */
    open?: boolean;
    /** Called when the modal should close. */
    onclose?: () => void;
  } = $props();

  type Tab = 'upload' | 'url' | 'paste';

  let activeTab = $state<Tab>('upload');

  /** Upload tab — drag-over highlight */
  let dragOver = $state(false);

  /** URL tab */
  let urlValue = $state('');
  let urlError = $state<string | null>(null);
  let urlSubmitting = $state(false);
  /** #78: mark the URL as a JS app / SPA so ingest always JS-renders it. */
  let urlIsSpa = $state(false);

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

  const activeNotebookId = $derived(notebookStore.activeNotebookId);
  const activeNotebook = $derived(notebookStore.activeNotebook);

  const pasteCanSubmit = $derived(pasteContent.trim().length > 0 && !pasteSubmitting);

  /** A URL is submittable when it parses as an absolute http(s) URL. */
  function isValidHttpUrl(value: string): boolean {
    let parsed: URL;
    try {
      parsed = new URL(value.trim());
    } catch {
      return false;
    }
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  }

  const urlCanSubmit = $derived(isValidHttpUrl(urlValue) && !urlSubmitting);

  $effect(() => {
    if (open) {
      activeTab = 'upload';
      urlValue = '';
      urlError = null;
      urlSubmitting = false;
      urlIsSpa = false;
      pasteTitle = '';
      pasteContent = '';
      pasteError = null;
      pasteSubmitting = false;
      uploadSubmitting = false;
      uploadError = null;
      dragOver = false;
    }
  });

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      onclose?.();
    }
  }

  /**
   * Shared ingest dispatch for browse picker and native drop (DRY).
   * `wasExisting` (backend content-hash dedup, #96) determines skip vs. add;
   * per-file failures are caught and counted without aborting the batch.
   */
  async function ingestPaths(
    notebookId: string,
    paths: string[]
  ): Promise<{ added: number; failed: number; skipped: number }> {
    let added = 0;
    let failed = 0;
    let skipped = 0;
    for (const path of paths) {
      // Use both / and \ to extract the filename (Tauri returns OS-native paths).
      const name = path.split(/[\\/]/).pop() ?? path;
      try {
        const { source, wasExisting } = await addFileSource(notebookId, name, path);
        if (wasExisting) {
          // Backend detected a content-dedup hit — do NOT addSourceLocal/ingest.
          skipped++;
        } else {
          // Optimistically insert the row BEFORE starting ingest so progress
          // events find the entry in the store immediately (avoids silent drops).
          addSourceLocal(source);
          void ingest(source.id);
          added++;
        }
      } catch (err) {
        failed++;
        uploadError = 'Could not add file. Please try again.';
        console.error('ingestPaths: failed for', path, err);
      }
    }
    return { added, failed, skipped };
  }

  /** Shows a summary toast when anything was skipped or failed; clean batches close silently. */
  function showBatchSummary(result: { added: number; failed: number; skipped: number }): void {
    if (result.skipped === 0 && result.failed === 0) return;
    const parts: string[] = [];
    if (result.added > 0) parts.push(`${result.added} added`);
    if (result.skipped > 0) parts.push(`${result.skipped} already in notebook`);
    if (result.failed > 0) parts.push(`${result.failed} failed`);
    showToast(parts.join(', '));
  }

  async function handleBrowse(): Promise<void> {
    if (!isTauri() || !activeNotebookId || uploadSubmitting) return;
    uploadError = null;
    uploadSubmitting = true;
    try {
      const selected = await openFilePicker({
        multiple: true,
        filters: PICKER_FILTERS
      });
      if (!selected) return;
      const paths = Array.isArray(selected) ? selected : [selected];
      if (paths.length === 0) return;

      const result = await ingestPaths(activeNotebookId, paths);
      showBatchSummary(result);
      // Close only if at least one source was added (mirrors the drop flow).
      if (result.added > 0) onclose?.();
    } catch (err) {
      uploadError = 'Could not add file. Please try again.';
      console.error('AddSourcesModal: handleBrowse failed', err);
    } finally {
      uploadSubmitting = false;
      // Reconcile with backend ordering.
      void loadSources(activeNotebookId);
    }
  }

  // $effect keyed on dropZoneEl re-registers on mount and cleans up on unmount,
  // handling open/close and tab switches. No coordinate hit-test needed.
  $effect(() => {
    if (!dropZoneEl) return;
    return registerDropTarget({
      setHover: (h) => {
        dragOver = h;
      },
      onDrop: async (paths) => {
        if (!activeNotebookId || uploadSubmitting) return;
        uploadError = null;
        uploadSubmitting = true;
        try {
          const result = await ingestPaths(activeNotebookId, paths);
          showBatchSummary(result);
          // Stay open when nothing was added (all duplicates / all failed).
          if (result.added > 0) onclose?.();
        } finally {
          uploadSubmitting = false;
          void loadSources(activeNotebookId);
        }
      }
    });
  });

  async function handlePasteSubmit(): Promise<void> {
    if (!pasteCanSubmit || !activeNotebookId) return;
    pasteError = null;
    pasteSubmitting = true;
    try {
      const title = pasteTitle.trim() || 'Untitled text';
      const { source, wasExisting } = await addTextSource(
        activeNotebookId,
        title,
        pasteContent.trim(),
        'text'
      );
      if (wasExisting) {
        // Backend detected a content-dedup hit (#100) — do NOT insert or ingest.
        showToast('Already in notebook');
      } else {
        // Optimistically insert the row BEFORE starting ingest so progress events
        // find the entry in the store immediately (avoids silent drops).
        addSourceLocal(source);
        void ingest(source.id);
      }
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

  /** Derive a human-readable title from a URL (hostname, else the raw URL). */
  function titleFromUrl(value: string): string {
    try {
      return new URL(value.trim()).hostname || value.trim();
    } catch {
      return value.trim();
    }
  }

  async function handleUrlSubmit(): Promise<void> {
    if (!urlCanSubmit || !activeNotebookId) return;
    urlError = null;
    urlSubmitting = true;
    try {
      const url = urlValue.trim();
      const { source, wasExisting } = await addUrlSource(
        activeNotebookId,
        titleFromUrl(url),
        url,
        urlIsSpa
      );
      if (wasExisting) {
        // Backend content-dedup hit (#100) — do NOT insert or ingest.
        showToast('Already in notebook');
      } else {
        // Optimistically insert the row BEFORE ingest so progress events find the
        // entry in the store immediately (mirrors the paste flow).
        addSourceLocal(source);
        void ingest(source.id);
      }
      onclose?.();
      void loadSources(activeNotebookId);
    } catch (err) {
      urlError = 'Could not add URL. Please check the address and try again.';
      console.error('AddSourcesModal: handleUrlSubmit failed', err);
    } finally {
      urlSubmitting = false;
    }
  }

  async function handlePrimaryAction(): Promise<void> {
    if (activeTab === 'upload') await handleBrowse();
    else if (activeTab === 'url') await handleUrlSubmit();
    else if (activeTab === 'paste') await handlePasteSubmit();
  }

  // Note: paste/urlCanSubmit already incorporate their own !*Submitting flag, so we
  // don't add a standalone submitting clause here (which would wrongly disable a
  // different tab while a submission is in flight).
  const primaryDisabled = $derived(
    (activeTab === 'url' && !urlCanSubmit) ||
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
      <div class="flex items-start justify-between px-5 pt-5 pb-0">
        <div class="flex items-center gap-2.5">
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

      <div class="px-5 py-4">
        {#if activeTab === 'upload'}
          <div
            id="sources-tab-panel-upload"
            role="tabpanel"
            aria-label="Upload files"
            style="-webkit-app-region: no-drag;"
          >
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

            {#if uploadError}
              <p class="mt-3 text-[12px] text-destructive" role="alert">{uploadError}</p>
            {/if}
          </div>
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
            <!-- #78: SPA / JS-render opt-in. Token-styled (appearance-none) so it
                 follows light/dark mode AND the selected accent — a native
                 checkbox ignores the app theme (no color-scheme is set). -->
            <label
              class="mb-3 flex items-start gap-2 text-[12px] text-foreground"
              for="add-sources-url-spa"
              style="-webkit-app-region: no-drag;"
            >
              <span class="relative mt-0.5 inline-flex size-4 shrink-0 items-center justify-center">
                <input
                  id="add-sources-url-spa"
                  type="checkbox"
                  class="peer size-4 shrink-0 cursor-pointer appearance-none rounded border border-input bg-background outline-none checked:border-primary checked:bg-primary focus-visible:ring-2 focus-visible:ring-ring/50"
                  bind:checked={urlIsSpa}
                  style="-webkit-app-region: no-drag;"
                />
                <Check
                  class="pointer-events-none absolute size-3 text-primary-foreground opacity-0 peer-checked:opacity-100"
                  strokeWidth={3}
                />
              </span>
              <span class="leading-relaxed">
                This page needs JavaScript to load
                <span class="text-muted-foreground/70">(render it before extracting)</span>
              </span>
            </label>
            <p class="text-[12px] text-muted-foreground/70 leading-relaxed">
              Supports web pages, blog posts, documentation and GitHub repos. Content is fetched and
              indexed locally.
            </p>
            {#if urlError}
              <p class="mt-3 text-[12px] text-destructive" role="alert">{urlError}</p>
            {/if}
          </div>
        {:else}
          <div
            id="sources-tab-panel-paste"
            role="tabpanel"
            aria-label="Paste text"
            style="-webkit-app-region: no-drag;"
          >
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
          {:else if activeTab === 'url' && urlSubmitting}
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
