<!-- A single user-authored manual note: sanitized markdown body (collapsible
     past ~4 lines), a relative timestamp, and a delete action. No source pill —
     manual notes have no grounding. Content is sanitized, never executed. -->
<script lang="ts">
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
  {#snippet body()}
    <CollapsibleText {html} />
  {/snippet}
</NoteCard>
