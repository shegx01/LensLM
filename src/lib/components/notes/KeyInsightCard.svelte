<!-- A single saved chat note (#24): optional source-title pill, sanitized
     answer markdown, and a relative timestamp. Read-only — no edit/delete here
     (that's #25). -->
<script lang="ts">
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { renderMarkdown } from '$lib/chat/render-markdown.js';
  import { formatRelativeTime } from '$lib/notebooks/format-time.js';
  import type { Note } from '$lib/notes/types.js';

  interface Props {
    note: Note;
  }

  let { note }: Props = $props();

  const html = $derived(renderMarkdown(note.content));
</script>

<article class="rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm">
  {#if note.source_title}
    <Badge variant="secondary" class="mb-2 max-w-full truncate">{note.source_title}</Badge>
  {/if}

  <!-- eslint-disable-next-line svelte/no-at-html-tags -->
  <div class="note-markdown text-sm leading-relaxed text-foreground">
    {@html html}
  </div>

  <p class="mt-2 text-xs text-muted-foreground/70">{formatRelativeTime(note.created_at)}</p>
</article>

<style>
  :global(.note-markdown p) {
    margin: 0 0 0.5em;
  }
  :global(.note-markdown p:last-child) {
    margin-bottom: 0;
  }
  :global(.note-markdown ul),
  :global(.note-markdown ol) {
    margin: 0.25em 0 0.5em;
    padding-left: 1.25em;
  }
  :global(.note-markdown li) {
    margin: 0.15em 0;
  }
  :global(.note-markdown code) {
    background: var(--muted);
    border-radius: 0.25rem;
    padding: 0.1em 0.35em;
    font-size: 0.85em;
  }
  :global(.note-markdown pre) {
    background: var(--muted);
    border-radius: 0.5rem;
    padding: 0.6em 0.8em;
    overflow-x: auto;
    margin: 0.5em 0;
  }
  :global(.note-markdown pre code) {
    background: none;
    padding: 0;
  }
  :global(.note-markdown a) {
    color: var(--primary);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  :global(.note-markdown strong) {
    font-weight: 600;
  }
</style>
