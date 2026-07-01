<!-- PROP CONTRACT (do not change without updating +layout.svelte):
     oncomplete: () => void  — finish onboarding (after persisting sources + onboarding_complete)
     onback:     () => void  — return to 'create-notebook'
     Reads/writes the shared draft via $lib/components/onboarding/onboarding-state.svelte.ts
     (draft.selectedSources, draft.notebookId). -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { open as openDialog } from '@tauri-apps/plugin-dialog';
  import Upload from '@lucide/svelte/icons/upload';
  import File from '@lucide/svelte/icons/file';
  import Check from '@lucide/svelte/icons/check';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import { Card } from '$lib/components/ui/card/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import {
    draft,
    type DraftSource,
    type RecentDocument
  } from '$lib/components/onboarding/onboarding-state.svelte.js';
  import ProgressDots from '$lib/components/onboarding/ProgressDots.svelte';
  import OnboardingBackButton from '$lib/components/onboarding/OnboardingBackButton.svelte';
  import { completeOnboarding } from '$lib/onboarding/completeOnboarding.js';
  import { registerDropTarget, PICKER_FILTERS } from '$lib/sources/dragDrop.js';
  import { showToast } from '$lib/sources/toast.svelte.js';
  import type { Source } from '$lib/sources/types.js';

  let { oncomplete, onback }: { oncomplete: () => void; onback: () => void } = $props();

  // ── State ─────────────────────────────────────────────────────────────────
  let suggestions = $state<RecentDocument[]>([]);
  let dropHover = $state(false);
  let finishing = $state(false);
  let completeError = $state<string | null>(null);

  let dropZoneEl: HTMLElement | undefined;
  let unregisterDrop: (() => void) | undefined;

  // Paths that `add_source` has already accepted. Persists across retries (it is
  // NOT reset on each click) so a failure partway through the loop can be retried
  // without re-inserting the sources that already landed (no duplicate rows).
  const addedPaths = new Set<string>();

  // ── Derived ───────────────────────────────────────────────────────────────
  const selectedPaths = $derived(new Set(draft.selectedSources.map((s) => s.path)));

  // Cap suggestions display at 4 rows.
  const visibleSuggestions = $derived(suggestions.slice(0, 4));

  const sourceCountLabel = $derived(
    draft.selectedSources.length === 0
      ? 'No sources selected'
      : draft.selectedSources.length === 1
        ? '1 source selected'
        : `${draft.selectedSources.length} sources selected`
  );

  // Added files: first 4 shown, remainder count for "+N more".
  const visibleAddedFiles = $derived(draft.selectedSources.slice(0, 4));
  const hiddenAddedCount = $derived(Math.max(0, draft.selectedSources.length - 4));

  // ── Mount: load recent documents + register native drop target ───────────
  onMount(() => {
    void loadSuggestions();

    if (!dropZoneEl) return;
    unregisterDrop = registerDropTarget({
      onDrop: (paths: string[]) => {
        for (const p of paths) {
          if (selectedPaths.has(p)) continue;
          const name = p.split(/[\\/]/).pop() ?? p;
          const extMatch = name.match(/\.([^.]+)$/);
          const ext = extMatch ? extMatch[1].toLowerCase() : '';
          const src: DraftSource = { path: p, name, ext, size: 0, mtime: 0 };
          draft.selectedSources = [...draft.selectedSources, src];
        }
      },
      setHover: (h: boolean) => {
        dropHover = h;
      }
    });
  });

  onDestroy(() => {
    unregisterDrop?.();
  });

  async function loadSuggestions(): Promise<void> {
    if (!isTauri()) return;
    try {
      const docs = await invoke<RecentDocument[]>('list_recent_documents');
      suggestions = docs ?? [];
    } catch {
      // Best-effort: if the call fails, hide suggestions section silently.
      suggestions = [];
    }
  }

  // ── Toggle a suggested document in/out of selectedSources ─────────────────
  function toggleSuggestion(doc: RecentDocument): void {
    if (selectedPaths.has(doc.path)) {
      draft.selectedSources = draft.selectedSources.filter((s) => s.path !== doc.path);
    } else {
      draft.selectedSources = [
        ...draft.selectedSources,
        { path: doc.path, name: doc.name, ext: doc.ext, size: doc.size, mtime: doc.mtime }
      ];
    }
  }

  // ── Remove a file from selectedSources (ADDED FILES delete action) ─────────
  function removeSource(path: string): void {
    draft.selectedSources = draft.selectedSources.filter((s) => s.path !== path);
  }

  // ── Open native file picker ───────────────────────────────────────────────
  async function browse(): Promise<void> {
    if (!isTauri()) return;
    try {
      const selected = await openDialog({
        multiple: true,
        filters: PICKER_FILTERS
      });
      if (!selected) return;
      const paths = Array.isArray(selected) ? selected : [selected];
      for (const p of paths) {
        const name = p.split(/[\\/]/).pop() ?? p;
        const extMatch = name.match(/\.([^.]+)$/);
        const ext = extMatch ? extMatch[1].toLowerCase() : '';
        const src: DraftSource = { path: p, name, ext, size: 0, mtime: 0 };
        if (!selectedPaths.has(p)) {
          draft.selectedSources = [...draft.selectedSources, src];
        }
      }
    } catch {
      // Picker cancelled or failed — ignore silently.
    }
  }

  // ── Finish onboarding ──────────────────────────────────────────────────────
  // Shared by both footer actions. `persistSources` is true for "Launch Lens"
  // (insert the selected sources first) and false for "Skip for now". The
  // source-insert loop skips paths already accepted on a prior attempt (see
  // `addedPaths`) so a retry after a partial failure never duplicates rows.
  async function finish(persistSources: boolean): Promise<void> {
    finishing = true;
    completeError = null;
    try {
      let skipped = 0;
      if (persistSources && isTauri() && draft.notebookId) {
        for (const src of draft.selectedSources) {
          // Client-side belt-and-suspenders guard: skip paths already accepted on
          // a prior (partially-failed) attempt to avoid redundant IPC round-trips.
          // The backend `raw_content_hash` dedup (#100) is the authority.
          if (addedPaths.has(src.path)) continue;
          const { wasExisting } = await invoke<{ source: Source; wasExisting: boolean }>(
            'add_source',
            {
              notebookId: draft.notebookId,
              title: src.name,
              locator: src.path
            }
          );
          if (wasExisting) skipped++;
          addedPaths.add(src.path);
        }
      }
      if (skipped > 0) {
        showToast(skipped === 1 ? 'Already in notebook' : `${skipped} already in notebook`);
      }
      await completeOnboarding();
      oncomplete();
    } catch {
      completeError = 'Could not save your setup. Please try again.';
    } finally {
      finishing = false;
    }
  }

  // ── Ext badge colour (subtle muted palette) ───────────────────────────────
  function extBadgeClass(ext: string): string {
    switch (ext.toLowerCase()) {
      case 'pdf':
        return 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400';
      case 'docx':
      case 'doc':
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400';
      case 'md':
        return 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-400';
      default:
        return 'bg-muted text-muted-foreground';
    }
  }

  // ── Format bytes to human-readable size ───────────────────────────────────
  function formatSize(bytes: number): string {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
</script>

<!-- macOS drag region (titleBarStyle Overlay): the empty canvas drags the window;
     the Card carries -webkit-app-region: no-drag so every inner control (Back, the
     dropzone, suggestions, footer buttons) stays clickable (mirrors SourcesRail.svelte). -->
<main data-tauri-drag-region class="flex min-h-svh items-center justify-center p-6">
  <div class="w-full max-w-[520px]" style="-webkit-app-region: no-drag;">
    <Card class="w-full gap-0 rounded-[14px] px-10 pt-9 pb-8 shadow-2xl ring-0">
      <!-- Header: Back + 3-dot progress -->
      <div class="mb-7 flex items-center justify-between">
        <OnboardingBackButton {onback} />
        <ProgressDots current={3} total={3} />
      </div>

      <!-- Title + subtitle -->
      <h1 class="text-foreground mb-1.5 text-[20px] font-bold tracking-[-0.35px]">Add sources</h1>
      <p class="text-muted-foreground mb-5 text-[13px]">
        Attach documents, PDFs, or notes. You can also add more later.
      </p>

      <!-- Drop zone -->
      <button
        bind:this={dropZoneEl}
        class={cn(
          'mb-[18px] w-full cursor-pointer rounded-xl border-[1.5px] border-dashed px-6 py-6 text-center transition-colors duration-150',
          dropHover ? 'border-primary' : 'border-border'
        )}
        onclick={browse}
        aria-label="Drop files here or click to browse"
        type="button"
      >
        <div
          class="bg-muted mx-auto mb-2.5 flex size-10 items-center justify-center rounded-[10px]"
        >
          <Upload class="text-muted-foreground size-[18px]" strokeWidth={1.75} />
        </div>
        <div class="text-foreground mb-1 text-[13px] font-semibold">Drop files here</div>
        <div class="text-muted-foreground text-[11px]">
          PDF, DOCX, RTF, EPUB & more — or
          <!-- svelte-ignore a11y_interactive_supports_focus -->
          <span
            class="text-primary cursor-pointer underline"
            onclick={(e) => {
              e.stopPropagation();
              void browse();
            }}
            role="button"
            tabindex="0"
            onkeydown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') void browse();
            }}>browse</span
          >
        </div>
      </button>

      <!-- Suggested from your library (hidden when no suggestions; capped at 4 rows) -->
      {#if visibleSuggestions.length > 0}
        <div class="text-muted-foreground mb-1.5 text-[10px] font-bold uppercase tracking-[0.08em]">
          Suggested from your library
        </div>
        <div class="mb-[22px] flex flex-col gap-1">
          {#each visibleSuggestions as doc (doc.path)}
            {@const selected = selectedPaths.has(doc.path)}
            <button
              class={cn(
                'flex w-full cursor-pointer items-center gap-2.5 rounded-[9px] border px-3 py-[9px] text-left transition-all duration-[130ms]',
                selected
                  ? 'border-border bg-muted/40'
                  : 'border-transparent bg-transparent hover:bg-muted/20'
              )}
              onclick={() => toggleSuggestion(doc)}
              type="button"
              aria-pressed={selected}
              aria-label={`${selected ? 'Deselect' : 'Select'} ${doc.name}`}
            >
              <!-- File icon tile -->
              <div class="bg-muted flex size-7 shrink-0 items-center justify-center rounded-[6px]">
                <File class="text-muted-foreground size-3" strokeWidth={1.75} />
              </div>

              <!-- Name + meta -->
              <div class="min-w-0 flex-1">
                <div
                  class={cn(
                    'truncate text-[12px] font-semibold',
                    selected ? 'text-primary' : 'text-foreground'
                  )}
                >
                  {doc.name}
                </div>
                <div class="text-muted-foreground mt-[1px] flex items-center gap-1.5 text-[10px]">
                  {#if doc.ext}
                    <span
                      class={cn(
                        'rounded-[3px] px-[5px] py-[1px] text-[9px] font-bold uppercase',
                        extBadgeClass(doc.ext)
                      )}
                    >
                      {doc.ext.toUpperCase()}
                    </span>
                  {/if}
                  {#if doc.size}
                    <span>{formatSize(doc.size)}</span>
                  {/if}
                </div>
              </div>

              <!-- Checkbox -->
              <div
                class={cn(
                  'flex size-[18px] shrink-0 items-center justify-center rounded-[5px] transition-all duration-[130ms]',
                  selected ? 'bg-primary' : 'border border-border'
                )}
                aria-hidden="true"
              >
                {#if selected}
                  <Check class="size-[11px] text-white" strokeWidth={3} />
                {/if}
              </div>
            </button>
          {/each}
        </div>
      {/if}

      <!-- ADDED FILES (hidden when selectedSources is empty) -->
      {#if draft.selectedSources.length > 0}
        <div class="text-muted-foreground mb-1.5 text-[10px] font-bold uppercase tracking-[0.08em]">
          Added files
        </div>
        <div class="mb-[18px] flex flex-col gap-1">
          {#each visibleAddedFiles as src (src.path)}
            <div
              class="bg-muted/30 border-border flex w-full items-center gap-2.5 rounded-[9px] border px-3 py-[9px]"
            >
              <!-- File icon tile -->
              <div class="bg-muted flex size-7 shrink-0 items-center justify-center rounded-[6px]">
                <File class="text-muted-foreground size-3" strokeWidth={1.75} />
              </div>

              <!-- Name + meta -->
              <div class="min-w-0 flex-1">
                <div class="text-foreground truncate text-[12px] font-semibold">
                  {src.name}
                </div>
                <div class="text-muted-foreground mt-[1px] flex items-center gap-1.5 text-[10px]">
                  {#if src.ext}
                    <span
                      class={cn(
                        'rounded-[3px] px-[5px] py-[1px] text-[9px] font-bold uppercase',
                        extBadgeClass(src.ext)
                      )}
                    >
                      {src.ext.toUpperCase()}
                    </span>
                  {/if}
                  {#if src.size}
                    <span>{formatSize(src.size)}</span>
                  {/if}
                </div>
              </div>

              <!-- Remove button -->
              <button
                class="text-muted-foreground hover:text-destructive flex size-[26px] shrink-0 cursor-pointer items-center justify-center rounded-[5px] transition-colors duration-[130ms]"
                onclick={() => removeSource(src.path)}
                type="button"
                aria-label={`Remove ${src.name}`}
              >
                <Trash2 class="size-[13px]" strokeWidth={1.75} />
              </button>
            </div>
          {/each}

          <!-- "+N more" overflow indicator -->
          {#if hiddenAddedCount > 0}
            <div
              class="text-muted-foreground px-3 py-[7px] text-[11px]"
              aria-label={`${hiddenAddedCount} more file${hiddenAddedCount === 1 ? '' : 's'} added`}
            >
              +{hiddenAddedCount} more
            </div>
          {/if}
        </div>
      {/if}

      <!-- Selected count label (only when nothing in ADDED FILES, to avoid redundancy) -->
      {#if draft.selectedSources.length === 0}
        <div class="text-muted-foreground mb-3.5 text-center text-[12px]">
          {sourceCountLabel}
        </div>
      {:else}
        <div class="mb-3.5"></div>
      {/if}

      <!-- Error message -->
      {#if completeError}
        <p class="text-destructive mb-3 w-full text-center text-sm" role="alert">
          {completeError}
        </p>
      {/if}

      <!-- Footer buttons -->
      <div class="flex gap-2">
        <Button
          variant="outline"
          class="h-[42px] flex-1 text-[13px] font-semibold"
          onclick={() => finish(false)}
          disabled={finishing}
        >
          Skip for now
        </Button>
        <Button
          class="h-[42px] flex-[2] text-[13px] font-semibold"
          onclick={() => finish(true)}
          disabled={finishing}
        >
          Launch Lens
          <ArrowRight class="size-[13px]" strokeWidth={2.5} />
        </Button>
      </div>
    </Card>
  </div>
</main>
