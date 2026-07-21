<!-- Bubble-less assistant slot: ✦ avatar, sanitized markdown, version pager
     (only if >1 version, AC13), and the action row. Rendered per-turn by
     ChatTranscript; NOT rendered at all when a turn has zero versions (a
     reloaded cancelled/errored turn) — that case is handled by the caller. -->
<script lang="ts">
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import MessageActions from './MessageActions.svelte';
  import { renderMarkdown } from '$lib/chat/render-markdown.js';
  import { enhanceCodeBlocks } from '$lib/chat/code-copy.js';
  import { enhanceCitations, type CitationTarget } from '$lib/chat/citation-inline.js';
  import { hydrateMermaid } from '$lib/chat/mermaid.js';
  import { messageCitations } from '$lib/chat/chat-state.svelte.js';
  import { sourcesStore } from '$lib/sources/sources-state.svelte.js';
  import { notesStore, toggleSave } from '$lib/notes/notes-state.svelte.js';
  import { enterRise } from '$lib/motion/index.js';
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
  // Terminal-marker versions (Plan 2): a cancelled/errored turn shows a muted status
  // line and hides the action row — a partial/stopped answer is not one to rate/save.
  const markerLabel = $derived(
    current?.state === 'cancelled'
      ? 'Stopped'
      : current?.state === 'errored'
        ? "Couldn't complete"
        : null
  );
  const citations = $derived(current ? messageCitations(current) : null);
  const html = $derived(
    current ? renderMarkdown(current.content, { highlight: highlightCode }) : ''
  );

  // Ordinal → source, resolved against the live sources store so a removed source
  // flips its inline chip to the disabled state (AC5) without a full re-render.
  const citationTargets = $derived.by(() => {
    const map = new Map<number, CitationTarget>();
    for (const c of citations ?? []) {
      const src = sourcesStore.sources.find((s) => s.id === c.source_id);
      map.set(c.ordinal, {
        source_id: c.source_id,
        title: src?.title ?? 'Removed source',
        live: !!src,
        locators: c.locators
      });
    }
    return map;
  });

  $effect(() => {
    // Read `html` so this re-runs after each {@html} render (post-DOM-update),
    // then enhance settled answers only — skip transient streaming bubbles.
    void html;
    if (!highlightCode || !containerEl) return;
    return enhanceCodeBlocks(containerEl);
  });

  $effect(() => {
    // Re-runs on render (`html`) AND when the resolved targets change (source
    // added/removed), so inline chip liveness stays in sync. Streaming bubbles
    // carry no citations, so this is a no-op there.
    void html;
    const targets = citationTargets;
    if (!containerEl || targets.size === 0) return;
    return enhanceCitations(containerEl, (n) => targets.get(n) ?? null);
  });

  $effect(() => {
    // Final path only — streaming bubbles keep the raw fence so the growing
    // buffer never reaches the layout engine.
    void html;
    if (!highlightCode || !containerEl) return;
    void hydrateMermaid(containerEl);
  });

  function prevVersion(): void {
    selectedIndex = Math.max(0, selectedIndex - 1);
  }

  function nextVersion(): void {
    selectedIndex = Math.min(versions.length - 1, selectedIndex + 1);
  }
</script>

{#if current}
  <div class="px-4 pt-3" in:enterRise={{ y: finalized ? 6 : 0 }}>
    <div class="flex flex-col gap-2">
      <div
        class="ai-avatar inline-flex text-primary"
        data-streaming={!finalized}
        aria-hidden="true"
      >
        <Sparkles class="size-4" strokeWidth={2} />
      </div>

      <div class="min-w-0 flex-1">
        <!-- eslint-disable-next-line svelte/no-at-html-tags -->
        <div bind:this={containerEl} class="chat-markdown text-sm leading-relaxed text-foreground">
          {@html html}
        </div>

        {#if markerLabel}
          <p class="mt-1 text-xs font-medium text-muted-foreground/70">{markerLabel}</p>
        {:else}
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
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  /* While streaming, the bare ✦ glyph breathes a soft glow + pulse so the answer
     reads as actively arriving. Gated by --rail-motion (calm on reduce-motion). */
  .ai-avatar[data-streaming='true'] {
    animation: aiPulse calc(1.6s / max(var(--rail-motion, 1), 0.0001)) var(--ease-out, ease)
      infinite;
  }
  :global([data-motion='off']) .ai-avatar[data-streaming='true'],
  .ai-avatar[data-streaming='false'] {
    animation: none;
  }
  @keyframes aiPulse {
    0%,
    100% {
      opacity: 0.5;
      filter: drop-shadow(0 0 0 transparent);
    }
    50% {
      opacity: 1;
      filter: drop-shadow(0 0 5px color-mix(in oklch, var(--primary) 55%, transparent));
    }
  }

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
    font-size: 0.92em;
    letter-spacing: 0.015em;
    font-family: var(--font-mono);
    /* Programming ligatures (=>, !==, ->, etc.); calt drives JetBrains Mono's set. */
    font-feature-settings:
      'calt' 1,
      'liga' 1;
    font-variant-ligatures: contextual;
  }
  /* Collapsible code panel (built post-render by enhanceCodeBlocks). Bordered so
     it reads as more prominent than the borderless user-message bubble. */
  :global(.chat-markdown .code-block) {
    margin: 0.5em 0;
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    overflow: hidden;
    background: var(--muted);
  }
  :global(.chat-markdown .code-block__header) {
    display: flex;
    align-items: center;
    gap: 0.4em;
    width: 100%;
    padding: 0.45em 0.6em;
    border: 0;
    background: transparent;
    color: var(--muted-foreground);
    font-size: 0.8em;
    font-weight: 500;
    cursor: pointer;
    transition: color 0.12s ease;
  }
  :global(.chat-markdown .code-block__header:hover) {
    color: var(--foreground);
  }
  :global(.chat-markdown .code-block__header:focus-visible) {
    outline: none;
    box-shadow: inset 0 0 0 2px var(--ring);
  }
  :global(.chat-markdown .code-block__icon) {
    display: inline-flex;
    align-items: center;
  }
  :global(.chat-markdown .code-block__chevron) {
    display: inline-flex;
    align-items: center;
    margin-left: auto;
    transition: transform 0.15s ease;
  }
  :global(.chat-markdown .code-block[data-expanded='true'] .code-block__chevron) {
    transform: rotate(90deg);
  }
  :global(.chat-markdown .code-block[data-expanded='false'] .code-block__body) {
    display: none;
  }
  :global(.chat-markdown pre) {
    position: relative;
    background: var(--muted);
    border-radius: 0.5rem;
    padding: 0.6em 0.8em;
    overflow-x: auto;
    margin: 0.5em 0;
  }
  /* Inside the collapsible panel the container owns the radius/margin; the pre
     just fills the body and gets a divider under the header. */
  :global(.chat-markdown .code-block__body pre) {
    border-radius: 0;
    margin: 0;
    border-top: 1px solid var(--border);
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
  /* Inline citation chips are styled globally in app.css (.citation-chip) —
     shared with the notes preview. */
</style>
