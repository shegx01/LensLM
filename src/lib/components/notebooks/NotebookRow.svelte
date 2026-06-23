<script lang="ts">
  import BookOpen from '@lucide/svelte/icons/book-open';
  import Trash from '@lucide/svelte/icons/trash';
  import { cn } from '$lib/utils.js';
  import type { NotebookSummary } from '$lib/notebooks/types.js';
  import {
    notebookAccentClass,
    formatRelativeTime,
    formatSourceCount
  } from '$lib/notebooks/index.js';
  import {
    renameNotebookAction,
    trashNotebookAction,
    selectNotebook
  } from '$lib/notebooks/index.js';

  /**
   * The notebook data to render.
   */
  let {
    notebook,
    active = false,
    collapsed = false
  }: {
    notebook: NotebookSummary;
    active?: boolean;
    /**
     * When true the row renders as a compact icon-only square (collapsed rail).
     * The parent supplies collapsed=true when sidebarCollapsed is active.
     */
    collapsed?: boolean;
  } = $props();

  // ---------------------------------------------------------------------------
  // Inline rename state
  // ---------------------------------------------------------------------------

  let renaming = $state(false);
  let draft = $state('');
  let inputEl = $state<HTMLInputElement | null>(null);
  // Re-entrancy guard: Enter sets `renaming=false`, which unmounts the input and
  // fires its `onblur` → a second `commitRename()`. This flag swallows that
  // second call so the rename IPC fires exactly once.
  let committing = false;

  function startRename(): void {
    draft = notebook.title;
    renaming = true;
    // focus after microtask so the input is in the DOM
    setTimeout(() => inputEl?.select(), 0);
  }

  function cancelRename(): void {
    renaming = false;
    draft = notebook.title;
  }

  async function commitRename(): Promise<void> {
    if (committing) return;
    committing = true;
    try {
      const trimmed = draft.trim();
      if (trimmed && trimmed !== notebook.title) {
        await renameNotebookAction(notebook.id, trimmed);
      }
      renaming = false;
    } finally {
      committing = false;
    }
  }

  function handleTitleDblClick(e: MouseEvent): void {
    e.stopPropagation(); // don't re-trigger row click
    startRename();
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Enter') {
      e.preventDefault();
      void commitRename();
    } else if (e.key === 'Escape') {
      cancelRename();
    }
  }

  // ---------------------------------------------------------------------------
  // Derived display values
  // ---------------------------------------------------------------------------

  const sourceLabel = $derived(formatSourceCount(notebook.source_count));
  const relTime = $derived(formatRelativeTime(notebook.updated_at));
  const accentClass = $derived(notebookAccentClass(notebook.id));
</script>

{#if collapsed}
  <!-- Collapsed: icon-only square with tooltip (provided by parent NotebooksSidebar) -->
  <button
    type="button"
    aria-label={notebook.title}
    aria-pressed={active}
    onclick={() => selectNotebook(notebook.id)}
    class={cn(
      'flex size-9 shrink-0 items-center justify-center rounded-lg transition-all',
      'hover:ring-2 hover:ring-sidebar-ring/40',
      accentClass,
      active && 'ring-2 ring-sidebar-ring'
    )}
  >
    <BookOpen class="size-4" />
  </button>
{:else}
  <!-- Expanded: full row -->
  <div
    role="button"
    tabindex="0"
    aria-label={notebook.title}
    aria-pressed={active}
    onclick={() => !renaming && selectNotebook(notebook.id)}
    onkeydown={(e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (!renaming) selectNotebook(notebook.id);
      }
    }}
    data-notebook-row
    class={cn(
      'group relative flex cursor-pointer items-center gap-2.5 rounded-lg px-2 py-2',
      'text-sidebar-foreground transition-colors outline-none',
      'hover:bg-sidebar-accent/60',
      'focus-visible:ring-2 focus-visible:ring-sidebar-ring',
      active && 'bg-sidebar-accent text-sidebar-accent-foreground'
    )}
  >
    <!-- Color icon square -->
    <div
      class={cn('flex size-8 shrink-0 items-center justify-center rounded-lg', accentClass)}
      aria-hidden="true"
    >
      <BookOpen class="size-4" />
    </div>

    <!-- Title + subtitle -->
    <div class="min-w-0 flex-1">
      {#if renaming}
        <input
          bind:this={inputEl}
          bind:value={draft}
          onkeydown={handleKeydown}
          onblur={() => void commitRename()}
          onclick={(e) => e.stopPropagation()}
          data-rename-input
          class={cn(
            'w-full rounded border border-sidebar-ring bg-sidebar px-1 py-0',
            'text-sm font-medium text-sidebar-foreground outline-none',
            'focus:ring-1 focus:ring-sidebar-ring'
          )}
          aria-label="Rename notebook"
        />
      {:else}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <p
          ondblclick={handleTitleDblClick}
          data-notebook-title
          class="truncate text-sm font-medium leading-tight"
          title={notebook.title}
        >
          {notebook.title}
        </p>
      {/if}
      <p class="truncate text-xs text-sidebar-foreground/50 leading-tight mt-0.5">
        {sourceLabel} · {relTime}
      </p>
    </div>

    <!-- Trash icon: visible on hover or when row is active -->
    <button
      type="button"
      aria-label={`Delete ${notebook.title}`}
      data-trash-btn
      onclick={(e) => {
        e.stopPropagation();
        void trashNotebookAction(notebook.id);
      }}
      class={cn(
        'ml-1.5 flex size-[22px] shrink-0 items-center justify-center rounded-[5px] opacity-0',
        'bg-transparent text-sidebar-foreground/40 transition-all',
        'hover:bg-destructive/15 hover:text-destructive',
        'group-hover:opacity-100',
        active && 'opacity-100'
      )}
    >
      <Trash class="size-3.5" />
    </button>
  </div>
{/if}
