<!-- A single saved chat note: optional source-title pill, sanitized answer
     markdown (collapsible past ~4 lines), a relative timestamp, and a delete
     action. Model content is sanitized (renderMarkdown/DOMPurify) — never run. -->
<script lang="ts">
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { renderMarkdown } from '$lib/chat/render-markdown.js';
  import CollapsibleText from './CollapsibleText.svelte';
  import NoteCard from './NoteCard.svelte';
  import type { Note } from '$lib/notes/types.js';

  interface Props {
    note: Note;
    notebookId: string;
  }

  let { note, notebookId }: Props = $props();

  const html = $derived(renderMarkdown(note.content));
</script>

<NoteCard {note} {notebookId}>
  {#snippet header()}
    {#if note.source_title}
      <Badge variant="secondary" class="mb-2 max-w-[calc(100%-2rem)] truncate">
        {note.source_title}
      </Badge>
    {/if}
  {/snippet}
  {#snippet body()}
    <CollapsibleText {html} />
  {/snippet}
</NoteCard>
