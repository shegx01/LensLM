<!-- Shared note-card shell: the article wrapper, hover-reveal delete action, an
     optional header (e.g. a source-title pill), the slotted body, and a relative
     timestamp. Callers supply the sanitized body; this owns the chrome. -->
<script lang="ts">
  import type { Snippet } from 'svelte';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import { formatRelativeTime } from '$lib/notebooks/format-time.js';
  import { remove } from '$lib/notes/notes-state.svelte.js';
  import type { Note } from '$lib/notes/types.js';

  interface Props {
    note: Note;
    notebookId: string;
    header?: Snippet;
    body: Snippet;
  }

  let { note, notebookId, header, body }: Props = $props();
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

  {@render header?.()}

  <div class="pr-6">
    {@render body()}
  </div>

  <p class="mt-2 text-xs text-muted-foreground/70">{formatRelativeTime(note.created_at)}</p>
</article>
