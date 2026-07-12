<!-- Bubble-less assistant slot: ✦ avatar, sanitized markdown, version pager
     (only if >1 version, AC13), and the action row. Rendered per-turn by
     ChatTranscript; NOT rendered at all when a turn has zero versions (a
     reloaded cancelled/errored turn) — that case is handled by the caller. -->
<script lang="ts">
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import MessageActions from './MessageActions.svelte';
  import CitationChips from './CitationChips.svelte';
  import { renderMarkdown, stripCitationMarkers } from '$lib/chat/render-markdown.js';
  import { enhanceCodeBlocks } from '$lib/chat/code-copy.js';
  import { messageCitations } from '$lib/chat/chat-state.svelte.js';
  import { notesStore, toggleSave } from '$lib/notes/notes-state.svelte.js';
  import type { ChatMessage } from '$lib/chat/types.js';

  interface Props {
    notebookId: string;
    versions: ChatMessage[];
    oncopy: (content: string) => void;
    onregenerate: () => void;
    onfeedback: (messageId: string, next: 'up' | 'down') => void;
    regenerateDisabled?: boolean;
    highlightCode?: boolean;
    /** Whether this bubble is a finalized answer — the streaming bubble passes
     * `false` to hide Save (partial content has no stable message id to save). */
    finalized?: boolean;
  }

  let {
    notebookId,
    versions,
    oncopy,
    onregenerate,
    onfeedback,
    regenerateDisabled = false,
    highlightCode = true,
    finalized = true
  }: Props = $props();

  let selectedIndex = $state(0);
  let containerEl = $state<HTMLElement | null>(null);

  $effect(() => {
    // Follow newest version whenever a new one lands (regenerate appends).
    selectedIndex = versions.length - 1;
  });

  const current = $derived(versions[selectedIndex]);
  const citations = $derived(current ? messageCitations(current) : null);
  const ordinals = $derived(new Set((citations ?? []).map((c) => c.ordinal)));
  const html = $derived(
    current
      ? renderMarkdown(stripCitationMarkers(current.content, ordinals), {
          highlight: highlightCode
        })
      : ''
  );

  $effect(() => {
    // Read `html` so this re-runs after each {@html} render (post-DOM-update),
    // then enhance settled answers only — skip transient streaming bubbles.
    void html;
    if (!highlightCode || !containerEl) return;
    return enhanceCodeBlocks(containerEl);
  });

  function prevVersion(): void {
    selectedIndex = Math.max(0, selectedIndex - 1);
  }

  function nextVersion(): void {
    selectedIndex = Math.min(versions.length - 1, selectedIndex + 1);
  }
</script>

{#if current}
  <div class="px-4 pt-3">
    <div class="flex gap-2.5">
      <div
        class="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary"
        aria-hidden="true"
      >
        <Sparkles class="size-3.5" strokeWidth={1.75} />
      </div>

      <div class="min-w-0 flex-1">
        <!-- eslint-disable-next-line svelte/no-at-html-tags -->
        <div bind:this={containerEl} class="chat-markdown text-sm leading-relaxed text-foreground">
          {@html html}
        </div>

        {#if citations && citations.length > 0}
          <CitationChips {citations} />
        {/if}

        <div class="mt-1.5 flex items-center gap-2">
          <MessageActions
            feedback={current.feedback}
            saved={notesStore.savedMessageIds(notebookId).has(current.id)}
            disabled={regenerateDisabled}
            {finalized}
            oncopy={() => oncopy(current.content)}
            {onregenerate}
            onfeedback={(next) => onfeedback(current.id, next)}
            onsave={() => void toggleSave(notebookId, current)}
          />

          {#if versions.length > 1}
            <div
              class="flex items-center gap-0.5 text-xs text-muted-foreground/70"
              aria-label="Answer version {selectedIndex + 1} of {versions.length}"
            >
              <button
                type="button"
                aria-label="Previous version"
                disabled={selectedIndex === 0}
                onclick={prevVersion}
                class="flex size-5 items-center justify-center rounded disabled:opacity-30 hover:bg-muted"
              >
                <ChevronLeft class="size-3" strokeWidth={2.5} />
              </button>
              <span class="tabular-nums">{selectedIndex + 1}/{versions.length}</span>
              <button
                type="button"
                aria-label="Next version"
                disabled={selectedIndex === versions.length - 1}
                onclick={nextVersion}
                class="flex size-5 items-center justify-center rounded disabled:opacity-30 hover:bg-muted"
              >
                <ChevronRight class="size-3" strokeWidth={2.5} />
              </button>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  :global(.chat-markdown p) {
    margin: 0 0 0.5em;
  }
  :global(.chat-markdown p:last-child) {
    margin-bottom: 0;
  }
  :global(.chat-markdown ul),
  :global(.chat-markdown ol) {
    margin: 0.25em 0 0.5em;
    padding-left: 1.25em;
  }
  :global(.chat-markdown li) {
    margin: 0.15em 0;
  }
  :global(.chat-markdown li + li) {
    margin-top: 0.35em;
  }
  :global(.chat-markdown code) {
    background: var(--muted);
    border-radius: 0.25rem;
    padding: 0.1em 0.35em;
    font-size: 0.85em;
  }
  :global(.chat-markdown pre) {
    position: relative;
    background: var(--muted);
    border-radius: 0.5rem;
    padding: 0.6em 0.8em;
    overflow-x: auto;
    margin: 0.5em 0;
  }
  :global(.chat-markdown pre code) {
    background: none;
    padding: 0;
  }
  :global(.chat-markdown pre .code-copy-btn) {
    position: absolute;
    top: 0.4em;
    right: 0.4em;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.65rem;
    height: 1.65rem;
    padding: 0;
    border: 1px solid var(--border);
    border-radius: 0.375rem;
    background: var(--muted);
    color: var(--muted-foreground);
    cursor: pointer;
    opacity: 0;
    transition:
      opacity 0.12s ease,
      color 0.12s ease,
      background 0.12s ease;
  }
  :global(.chat-markdown pre:hover .code-copy-btn),
  :global(.chat-markdown pre .code-copy-btn:focus-visible) {
    opacity: 1;
  }
  :global(.chat-markdown pre .code-copy-btn:hover) {
    color: var(--foreground);
  }
  :global(.chat-markdown pre .code-copy-btn[data-copied='true']) {
    opacity: 1;
    color: var(--primary);
  }
  :global(.chat-markdown .code-copy-icon) {
    display: block;
  }
  :global(.chat-markdown .code-copy-icon--check) {
    display: none;
  }
  :global(.chat-markdown .code-copy-btn[data-copied='true'] .code-copy-icon--copy) {
    display: none;
  }
  :global(.chat-markdown .code-copy-btn[data-copied='true'] .code-copy-icon--check) {
    display: block;
  }
  :global(.chat-markdown a) {
    color: var(--primary);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  :global(.chat-markdown strong) {
    font-weight: 600;
  }
</style>
