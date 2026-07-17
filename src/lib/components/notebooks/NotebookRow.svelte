<script lang="ts">
  import BookOpen from '@lucide/svelte/icons/book-open';
  import Trash from '@lucide/svelte/icons/trash';
  import { cn } from '$lib/utils.js';
  import type { NotebookSummary } from '$lib/notebooks/types.js';
  import {
    notebookColorClass,
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
     * When true the row renders as a compact 44×44 icon square (collapsed rail).
     * The parent supplies collapsed=true when sidebarCollapsed is active. The DOM
     * is the SAME in both states — layout responds via `data-collapsed` + CSS.
     */
    collapsed?: boolean;
  } = $props();

  let renaming = $state(false);
  let draft = $state('');
  let inputEl = $state<HTMLInputElement | null>(null);
  let tileEl = $state<HTMLElement | null>(null);
  // Re-entrancy guard: Enter sets `renaming=false`, which unmounts the input and
  // fires its `onblur` → a second `commitRename()`. This flag swallows that
  // second call so the rename IPC fires exactly once.
  let committing = false;

  // Tile "pop" fires only on a genuine inactive→active transition (a fresh
  // selection). Re-selecting the already-active row must not re-pop. The first
  // effect run seeds `prevActive` so there is no pop on initial mount.
  let prevActive: boolean | undefined = undefined;
  $effect(() => {
    const now = active;
    if (prevActive === undefined) {
      prevActive = now;
      return;
    }
    if (now && !prevActive && tileEl) {
      tileEl.classList.remove('pop');
      void tileEl.offsetWidth; // reflow so the animation restarts
      tileEl.classList.add('pop');
    }
    prevActive = now;
  });

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

  const sourceLabel = $derived(formatSourceCount(notebook.source_count));
  const relTime = $derived(formatRelativeTime(notebook.updated_at));
  const accentClass = $derived(notebookColorClass(notebook.id));
</script>

<!--
  Single-root row: the same DOM serves expanded and collapsed. `data-collapsed`
  drives the label crossfade + 44×44 square via scoped CSS; all motion is gated
  by `--rail-motion` so calm mode keeps opacity fades but drops springs/translate.
-->
<div
  role="button"
  tabindex="0"
  aria-label={notebook.title}
  aria-pressed={active}
  title={collapsed ? notebook.title : undefined}
  data-notebook-row
  data-collapsed={collapsed}
  data-active={active}
  onclick={() => !renaming && selectNotebook(notebook.id)}
  onkeydown={(e) => {
    // While renaming, the inline input owns the keyboard — don't let the row's
    // Space/Enter "select" handler swallow the space (and other) keystrokes.
    if (renaming) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      selectNotebook(notebook.id);
    }
  }}
  class="nb-row group"
>
  <!-- Icon tile — active flips to the accent (primary) treatment; the tile pops
       on selection. -->
  <div
    bind:this={tileEl}
    class={cn('nb-tile', active ? 'bg-primary text-primary-foreground' : accentClass)}
    aria-hidden="true"
  >
    <BookOpen class="size-4" />
  </div>

  <div class="nb-rowtext">
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
        class="nb-title"
        title={notebook.title}
      >
        {notebook.title}
      </p>
    {/if}
    <p class="nb-rowsub">{sourceLabel} · {relTime}</p>
  </div>

  {#if !collapsed}
    <button
      type="button"
      aria-label={`Delete ${notebook.title}`}
      data-trash-btn
      onclick={(e) => {
        e.stopPropagation();
        void trashNotebookAction(notebook.id);
      }}
      class="nb-del"
    >
      <Trash class="size-3.5" />
    </button>
  {/if}
</div>

<style>
  .nb-row {
    position: relative;
    display: flex;
    align-items: center;
    gap: 11px;
    height: 46px;
    flex: none;
    width: 100%;
    padding: 0 10px;
    border-radius: 12px;
    border: 0;
    background: transparent;
    color: var(--sidebar-foreground);
    cursor: pointer;
    font: inherit;
    text-align: left;
    outline: none;
    /* Compose lean (hover) + press so they never overwrite each other; the
       translate vars scale by --rail-motion so calm mode zeroes the movement. */
    transform: translateX(calc(var(--row-x, 0px) * var(--rail-motion, 1)))
      translateY(calc(var(--row-y, 0px) * var(--rail-motion, 1))) scale(var(--row-s, 1));
    transition:
      background 0.18s var(--ease-out, ease),
      transform calc(0.24s * var(--rail-motion, 1)) var(--ease-out, ease),
      box-shadow 0.24s var(--ease-out, ease),
      gap 0.44s var(--ease-spring, ease);
  }
  .nb-row:hover:not([data-active='true']) {
    background: color-mix(in oklch, var(--sidebar-accent) 60%, transparent);
    --row-x: 3px;
    --row-y: -1px;
    box-shadow: 0 5px 16px rgb(0 0 0 / 0.07);
  }
  .nb-row:active {
    --row-s: 0.975;
  }
  .nb-row:focus-visible {
    box-shadow: 0 0 0 2px var(--sidebar-ring);
  }
  .nb-row[data-active='true'] .nb-title {
    color: var(--primary);
  }

  .nb-tile {
    width: 30px;
    height: 30px;
    flex: none;
    display: grid;
    place-items: center;
    border-radius: 9px;
    transition:
      transform calc(0.4s * var(--rail-motion, 1)) var(--ease-spring, ease),
      box-shadow 0.3s var(--ease-out, ease);
  }
  .nb-row:hover .nb-tile {
    transform: scale(calc(1 + 0.06 * var(--rail-motion, 1)))
      rotate(calc(-3deg * var(--rail-motion, 1)));
  }
  .nb-row[data-active='true'] .nb-tile {
    box-shadow:
      0 2px 8px color-mix(in oklch, var(--primary) 14%, transparent),
      inset 0 0 0 1px color-mix(in oklch, var(--primary) 10%, transparent);
  }
  /* Momentum pop when a notebook becomes active — a touch of overshoot is earned
     on a real selection. Duration scales by --rail-motion (0 → no pop). */
  @keyframes nbTilePop {
    0% {
      transform: scale(1);
    }
    38% {
      transform: scale(1.16) rotate(-5deg);
    }
    100% {
      transform: scale(1);
    }
  }
  .nb-tile.pop {
    animation: nbTilePop calc(0.5s * var(--rail-motion, 1)) var(--ease-spring, ease);
  }

  .nb-rowtext {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
    overflow: hidden;
    max-width: 180px;
    opacity: 1;
    /* Opacity fade stays in calm; the width collapse is spring-gated. */
    transition:
      opacity 0.28s var(--ease-out, ease),
      max-width calc(0.44s * var(--rail-motion, 1)) var(--ease-spring, ease);
  }
  .nb-title {
    font-size: 13.5px;
    font-weight: 550;
    line-height: 1.25;
    letter-spacing: -0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .nb-rowsub {
    margin-top: 1px;
    font-size: 11px;
    color: color-mix(in oklch, var(--sidebar-foreground) 50%, transparent);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-variant-numeric: tabular-nums;
  }

  .nb-del {
    margin-left: auto;
    flex: none;
    width: 26px;
    height: 26px;
    border-radius: 7px;
    border: 0;
    padding: 0;
    display: grid;
    place-items: center;
    cursor: pointer;
    background: transparent;
    color: color-mix(in oklch, var(--sidebar-foreground) 45%, transparent);
    opacity: 0;
    transform: translateX(4px);
    transition:
      opacity 0.18s var(--ease-out, ease),
      transform 0.18s var(--ease-out, ease),
      background 0.16s var(--ease-out, ease),
      color 0.16s var(--ease-out, ease);
  }
  .nb-row:hover .nb-del,
  .nb-row[data-active='true'] .nb-del {
    opacity: 1;
    transform: translateX(0);
  }
  .nb-del:hover {
    background: color-mix(in oklch, var(--destructive) 15%, transparent);
    color: var(--destructive);
  }

  /* ---- collapsed: uniform 44×44 square, label crossfaded out, icon anchored ---- */
  .nb-row[data-collapsed='true'] {
    width: 44px;
    height: 44px;
    align-self: center;
    justify-content: center;
    gap: 0;
    padding: 0;
    border-radius: 12px;
  }
  .nb-row[data-collapsed='true']:hover:not([data-active='true']) {
    --row-x: 0px;
    box-shadow: none;
  }
  .nb-row[data-collapsed='true'] .nb-rowtext {
    opacity: 0;
    max-width: 0;
  }
</style>
