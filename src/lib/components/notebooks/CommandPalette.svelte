<!--
  ⌘K palette — notebooks-only (M3). SOURCES/CHATS deferred to M4/M5.
  Custom overlay (not bits-ui Dialog) for manual focus-trap + two-region layout.
  z-[60]: above Dialog z-50, below Tooltip z-70. ⌘K listener lives in AppShell.
-->
<script lang="ts">
  import SearchIcon from '@lucide/svelte/icons/search';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import BookOpen from '@lucide/svelte/icons/book-open';
  import {
    notebookStore,
    selectNotebook,
    notebookColorClass,
    formatRelativeTime,
    formatSourceCount
  } from '$lib/notebooks/index.js';

  let highlightedIndex = $state(0);
  let previouslyFocusedEl: Element | null = null;
  let searchInput: HTMLInputElement | null = $state(null);
  let panelEl: HTMLDivElement | null = $state(null);

  const results = $derived(notebookStore.paletteResults);

  $effect(() => {
    // Touch `results` to subscribe; reset highlight on any change.
    void results;
    highlightedIndex = 0;
  });

  $effect(() => {
    // $effect is deferred (microtask); if it flushes after the component/jsdom is
    // torn down, the bare `document` global may be gone — guard it.
    if (typeof document === 'undefined') return;
    if (notebookStore.paletteOpen) {
      previouslyFocusedEl = document.activeElement;
      // Defer focus until the DOM is painted; cancel if we tear down first.
      const raf = requestAnimationFrame(() => {
        searchInput?.focus();
      });
      return () => cancelAnimationFrame(raf);
    } else {
      // Restore focus to the element that was focused before the palette opened.
      if (previouslyFocusedEl && 'focus' in previouslyFocusedEl) {
        (previouslyFocusedEl as HTMLElement).focus();
      }
      previouslyFocusedEl = null;
    }
  });

  function close() {
    notebookStore.paletteOpen = false;
  }

  function selectAt(index: number) {
    const result = results[index];
    if (!result) return;
    selectNotebook(result.id);
    close();
  }

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

  function scrollHighlightedIntoView() {
    requestAnimationFrame(() => {
      const row = panelEl?.querySelector<HTMLElement>(`[data-result-index="${highlightedIndex}"]`);
      row?.scrollIntoView({ block: 'nearest' });
    });
  }

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

  function handleBackdropClick(e: MouseEvent) {
    // Only close if the click was on the backdrop itself, not the panel.
    if (e.target === e.currentTarget) {
      close();
    }
  }
</script>

{#if notebookStore.paletteOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="fixed inset-0 z-[60] flex items-start justify-center bg-black/40 pt-[15vh] backdrop-blur-sm"
    role="presentation"
    onclick={handleBackdropClick}
  >
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

        <button
          type="button"
          aria-label="Close search (Escape)"
          class="shrink-0 rounded px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground border border-border hover:text-foreground hover:border-foreground/30 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onclick={close}
        >
          Esc
        </button>
      </div>

      <div class="flex-1 overflow-y-auto min-h-0 border-t border-border">
        {#if results.length === 0}
          <div class="px-4 py-8 text-center text-[13px] text-muted-foreground">
            No notebooks found
          </div>
        {:else}
          <div class="px-4 pt-3 pb-1">
            <div
              class="text-[10px] font-bold tracking-[0.1em] uppercase text-muted-foreground mb-1"
            >
              Notebooks
            </div>
          </div>

          <ul id="palette-results" role="listbox" aria-label="Notebook results" class="px-2 pb-2">
            {#each results as notebook, i (notebook.id)}
              {@const isHighlighted = i === highlightedIndex}
              {@const accentCls = notebookColorClass(notebook.id)}
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
                <div
                  class={[
                    'size-8 shrink-0 rounded-[7px] flex items-center justify-center',
                    accentCls
                  ].join(' ')}
                  aria-hidden="true"
                >
                  <BookOpen class="size-4" />
                </div>

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
