<!--
  CommandPalette.svelte — ⌘K notebook search modal (M3).

  NOTEBOOKS-ONLY IN M3: This palette searches notebooks only. SOURCES and CHATS
  sections are deferred to M4/M5 respectively. The placeholder text intentionally
  reads "Search notebooks" (not "Search notebooks, sources, chats...") to avoid
  implying cross-type search that doesn't exist yet.

  GLOBAL ⌘K HANDLER: The keydown listener that sets `notebookStore.paletteOpen = true`
  is registered in AppShell.svelte (Step 4.10). This component only reacts to
  `notebookStore.paletteOpen` — it does NOT register its own ⌘K listener.

  OVERLAY CHOICE: Custom overlay (not shadcn Dialog) — the palette has a distinct
  two-region layout (search input + keyboard-hint footer) that doesn't fit the
  Dialog's grid-gap content model, and direct DOM access is needed for focus trap
  management. bits-ui Dialog wraps in a Portal which makes sentinel-based trapping
  harder to reason about. A custom overlay with manual focus trapping is simpler and
  more explicit here.

  FOCUS TRAP: Manual implementation. On open: cache `document.activeElement`, move
  focus to the search input. Tab/Shift+Tab cycles through focusable children of the
  palette panel. On close: restore focus to the cached element.

  Z-INDEX SCALE (defined here for cross-component consistency):
    Backdrop + panel: z-[60]  (above Dialog z-50, below Tooltip z-70)
-->
<script lang="ts">
  import SearchIcon from '@lucide/svelte/icons/search';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import BookOpen from '@lucide/svelte/icons/book-open';
  import {
    notebookStore,
    selectNotebook,
    notebookAccentClass,
    formatRelativeTime,
    formatSourceCount
  } from '$lib/notebooks/index.js';

  // ---------------------------------------------------------------------------
  // Local state
  // ---------------------------------------------------------------------------

  /** Index of the keyboard-highlighted result row. Resets to 0 when query changes. */
  let highlightedIndex = $state(0);

  /** The element that had focus when the palette opened — restored on close. */
  let previouslyFocusedEl: Element | null = null;

  /** Ref to the search input — used for autofocus on open. */
  let searchInput: HTMLInputElement | null = $state(null);

  /** Ref to the palette panel — used as the focus-trap boundary. */
  let panelEl: HTMLDivElement | null = $state(null);

  // ---------------------------------------------------------------------------
  // Derived helpers
  // ---------------------------------------------------------------------------

  const results = $derived(notebookStore.paletteResults);

  /** Reset highlight whenever results change (query changed). */
  $effect(() => {
    // Touch `results` to subscribe; reset highlight on any change.
    void results;
    highlightedIndex = 0;
  });

  // ---------------------------------------------------------------------------
  // Open/close side-effects
  // ---------------------------------------------------------------------------

  $effect(() => {
    if (notebookStore.paletteOpen) {
      previouslyFocusedEl = document.activeElement;
      // Defer focus until the DOM is painted.
      requestAnimationFrame(() => {
        searchInput?.focus();
      });
    } else {
      // Restore focus to the element that was focused before the palette opened.
      if (previouslyFocusedEl && 'focus' in previouslyFocusedEl) {
        (previouslyFocusedEl as HTMLElement).focus();
      }
      previouslyFocusedEl = null;
    }
  });

  // ---------------------------------------------------------------------------
  // Keyboard handlers
  // ---------------------------------------------------------------------------

  /** Close the palette (resets query via the store setter). */
  function close() {
    notebookStore.paletteOpen = false;
  }

  /** Select the notebook at `index` and close. */
  function selectAt(index: number) {
    const result = results[index];
    if (!result) return;
    selectNotebook(result.id);
    close();
  }

  /** Handle keydown on the palette panel for navigation. */
  function handlePanelKeydown(e: KeyboardEvent) {
    switch (e.key) {
      case 'Escape':
        e.preventDefault();
        e.stopPropagation();
        close();
        break;
      case 'ArrowDown':
        e.preventDefault();
        highlightedIndex = results.length > 0 ? (highlightedIndex + 1) % results.length : 0;
        scrollHighlightedIntoView();
        break;
      case 'ArrowUp':
        e.preventDefault();
        highlightedIndex =
          results.length > 0 ? (highlightedIndex - 1 + results.length) % results.length : 0;
        scrollHighlightedIntoView();
        break;
      case 'Enter':
        e.preventDefault();
        selectAt(highlightedIndex);
        break;
      case 'Tab':
        // Trap focus within the panel.
        trapTab(e);
        break;
    }
  }

  /** Scroll the highlighted result row into view inside the results container. */
  function scrollHighlightedIntoView() {
    requestAnimationFrame(() => {
      const row = panelEl?.querySelector<HTMLElement>(`[data-result-index="${highlightedIndex}"]`);
      row?.scrollIntoView({ block: 'nearest' });
    });
  }

  // ---------------------------------------------------------------------------
  // Focus trap
  // ---------------------------------------------------------------------------

  /** Returns all keyboard-focusable elements within the palette panel. */
  function getFocusable(): HTMLElement[] {
    if (!panelEl) return [];
    return Array.from(
      panelEl.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
    ).filter((el) => !el.hasAttribute('disabled') && el.tabIndex !== -1);
  }

  function trapTab(e: KeyboardEvent) {
    const focusable = getFocusable();
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement as HTMLElement;

    if (e.shiftKey) {
      if (active === first || !panelEl?.contains(active)) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (active === last || !panelEl?.contains(active)) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Backdrop click
  // ---------------------------------------------------------------------------

  function handleBackdropClick(e: MouseEvent) {
    // Only close if the click was on the backdrop itself, not the panel.
    if (e.target === e.currentTarget) {
      close();
    }
  }
</script>

{#if notebookStore.paletteOpen}
  <!--
    Backdrop — covers the full viewport, dims the UI, closes on click-outside.
    z-[60]: above shadcn Dialog (z-50), below Tooltip (z-70).
  -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="fixed inset-0 z-[60] flex items-start justify-center bg-black/40 pt-[15vh] backdrop-blur-sm"
    role="presentation"
    onclick={handleBackdropClick}
  >
    <!--
      Palette panel — centered modal.
      role="dialog" + aria-modal="true" declare it as a dialog to AT.
      aria-label provides an accessible name.
    -->
    <div
      bind:this={panelEl}
      role="dialog"
      aria-modal="true"
      aria-label="Search notebooks"
      tabindex="-1"
      class="bg-popover text-popover-foreground ring-foreground/10 w-full max-w-[480px] rounded-xl shadow-2xl ring-1 overflow-hidden flex flex-col"
      style="max-height: min(560px, calc(100vh - 30vh - 2rem));"
      onkeydown={handlePanelKeydown}
    >
      <!-- ------------------------------------------------------------------ -->
      <!-- Search header                                                        -->
      <!-- ------------------------------------------------------------------ -->
      <div class="flex items-center gap-3 px-4 py-3">
        <SearchIcon class="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />

        <input
          bind:this={searchInput}
          type="text"
          role="combobox"
          aria-autocomplete="list"
          aria-controls="palette-results"
          aria-expanded={results.length > 0}
          aria-activedescendant={results.length > 0
            ? `palette-result-${results[highlightedIndex]?.id}`
            : undefined}
          aria-label="Search notebooks"
          placeholder="Search notebooks"
          class="min-w-0 flex-1 bg-transparent text-[14px] text-foreground placeholder:text-muted-foreground outline-none"
          value={notebookStore.paletteQuery}
          oninput={(e) => {
            notebookStore.paletteQuery = (e.currentTarget as HTMLInputElement).value;
          }}
        />

        <!-- Esc affordance / close button -->
        <button
          type="button"
          aria-label="Close search (Escape)"
          class="shrink-0 rounded px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground border border-border hover:text-foreground hover:border-foreground/30 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onclick={close}
        >
          Esc
        </button>
      </div>

      <!-- ------------------------------------------------------------------ -->
      <!-- Results body                                                         -->
      <!-- ------------------------------------------------------------------ -->
      <div class="flex-1 overflow-y-auto min-h-0 border-t border-border">
        {#if results.length === 0}
          <!-- Empty state -->
          <div class="px-4 py-8 text-center text-[13px] text-muted-foreground">
            No notebooks found
          </div>
        {:else}
          <!-- NOTEBOOKS section -->
          <div class="px-4 pt-3 pb-1">
            <div
              class="text-[10px] font-bold tracking-[0.1em] uppercase text-muted-foreground mb-1"
            >
              Notebooks
            </div>
          </div>

          <!--
            Results list.
            role="listbox" + each row role="option" satisfies the combobox ARIA pattern.
            aria-selected marks the keyboard-highlighted item.
          -->
          <ul id="palette-results" role="listbox" aria-label="Notebook results" class="px-2 pb-2">
            {#each results as notebook, i (notebook.id)}
              {@const isHighlighted = i === highlightedIndex}
              {@const accentCls = notebookAccentClass(notebook.id)}
              <li
                id="palette-result-{notebook.id}"
                data-result-index={i}
                role="option"
                aria-selected={isHighlighted}
                class={[
                  'group flex items-center gap-3 rounded-lg px-3 py-2.5 cursor-pointer transition-colors outline-none',
                  isHighlighted ? 'bg-primary/10 text-foreground' : 'text-foreground hover:bg-muted'
                ].join(' ')}
                onclick={() => selectAt(i)}
                onmouseenter={() => {
                  highlightedIndex = i;
                }}
                tabindex="-1"
              >
                <!-- Deterministic color icon — self-applying `nb-{accent}` class
                     drives the tinted background (--nb-bg) and glyph color (--nb-fg),
                     matching NotebooksSidebar / NotebookRow / TrashView. -->
                <div
                  class={[
                    'size-8 shrink-0 rounded-[7px] flex items-center justify-center',
                    accentCls
                  ].join(' ')}
                  aria-hidden="true"
                >
                  <BookOpen class="size-4" />
                </div>

                <!-- Title + subtitle -->
                <div class="min-w-0 flex-1">
                  <div class="truncate text-[14px] font-bold leading-snug">
                    {notebook.title}
                  </div>
                  <div class="truncate text-[12px] text-muted-foreground leading-snug mt-0.5">
                    {formatSourceCount(notebook.source_count)} · {formatRelativeTime(
                      notebook.updated_at
                    )}
                  </div>
                </div>

                <!-- Chevron -->
                <ChevronRight
                  class={[
                    'size-4 shrink-0 transition-colors',
                    isHighlighted ? 'text-primary' : 'text-muted-foreground/50'
                  ].join(' ')}
                  aria-hidden="true"
                />
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <!-- ------------------------------------------------------------------ -->
      <!-- Footer hint bar                                                      -->
      <!-- ------------------------------------------------------------------ -->
      <div
        class="flex items-center gap-4 border-t border-border px-4 py-2 text-[11px] text-muted-foreground"
        aria-hidden="true"
      >
        <span>↑↓ navigate</span>
        <span>↵ open</span>
        <span>⌘ anywhere</span>
      </div>
    </div>
  </div>
{/if}
