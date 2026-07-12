<!-- A single user-authored manual note: sanitized markdown body (collapsible
     past ~4 lines), a relative timestamp, and a delete action. No source pill —
     manual notes have no grounding. Content is sanitized, never executed. -->
<script lang="ts">
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import { renderMarkdown } from '$lib/chat/render-markdown.js';
  import { formatRelativeTime } from '$lib/notebooks/format-time.js';
  import { remove } from '$lib/notes/notes-state.svelte.js';
  import CollapsibleText from './CollapsibleText.svelte';
  import type { Note } from '$lib/notes/types.js';

  interface Props {
    note: Note;
    notebookId: string;
  }

  let { note, notebookId }: Props = $props();

  const html = $derived(renderMarkdown(note.content));
</script>

<article
  data-note-id={note.id}
  class="group/note relative rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm"
>
  <button
    type="button"
    aria-label="Delete note"
    class="absolute top-2 right-2 flex size-7 items-center justify-center rounded-md text-muted-foreground/60 opacity-0 transition-all hover:bg-muted hover:text-destructive focus-visible:opacity-100 group-hover/note:opacity-100"
    onclick={() => void remove(notebookId, note.id)}
  >
    <Trash2 class="size-4" strokeWidth={1.75} />
  </button>

  <CollapsibleText {html} class="pr-6" />

  <p class="mt-2 text-xs text-muted-foreground/70">{formatRelativeTime(note.created_at)}</p>
</article>
