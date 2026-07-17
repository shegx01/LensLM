<!-- Shared note-card shell: the article wrapper, hover-reveal actions (pin, edit,
     delete), an optional header (e.g. a source-title pill), the slotted body or the
     in-place editor, an "edited" indicator, and a relative timestamp. Callers supply
     the sanitized body; this owns the chrome and the edit/pin affordances. -->
<script lang="ts">
  import type { Snippet } from 'svelte';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import Pencil from '@lucide/svelte/icons/pencil';
  import Pin from '@lucide/svelte/icons/pin';
  import PinOff from '@lucide/svelte/icons/pin-off';
  import { formatRelativeTime } from '$lib/notebooks/format-time.js';
  import { remove, setPinned } from '$lib/notes/notes-state.svelte.js';
  import NoteEditor from './NoteEditor.svelte';
  import type { Note } from '$lib/notes/types.js';

  interface Props {
    note: Note;
    notebookId: string;
    header?: Snippet;
    body: Snippet;
  }

  let { note, notebookId, header, body }: Props = $props();

  let editing = $state(false);
  const edited = $derived(note.updated_at !== note.created_at);
</script>

<article
  data-note-id={note.id}
  class="group/note relative rounded-lg border border-border bg-card p-4 text-card-foreground"
>
  {#if !editing}
    <div class="absolute top-2 right-2 flex items-center gap-0.5">
      <button
        type="button"
        aria-label={note.pinned ? 'Unpin note' : 'Pin note'}
        aria-pressed={note.pinned}
        class={[
          'flex size-7 items-center justify-center rounded-md transition-all hover:bg-muted focus-visible:opacity-100',
          note.pinned
            ? 'text-primary opacity-100'
            : 'text-muted-foreground/60 opacity-0 hover:text-foreground group-hover/note:opacity-100'
        ].join(' ')}
        onclick={() => void setPinned(notebookId, note.id, !note.pinned)}
      >
        {#if note.pinned}
          <PinOff class="size-4" strokeWidth={1.75} />
        {:else}
          <Pin class="size-4" strokeWidth={1.75} />
        {/if}
      </button>
      <button
        type="button"
        aria-label="Edit note"
        class="flex size-7 items-center justify-center rounded-md text-muted-foreground/60 opacity-0 transition-all hover:bg-muted hover:text-foreground focus-visible:opacity-100 group-hover/note:opacity-100"
        onclick={() => (editing = true)}
      >
        <Pencil class="size-4" strokeWidth={1.75} />
      </button>
      <button
        type="button"
        aria-label="Delete note"
        class="flex size-7 items-center justify-center rounded-md text-muted-foreground/60 opacity-0 transition-all hover:bg-muted hover:text-destructive focus-visible:opacity-100 group-hover/note:opacity-100"
        onclick={() => void remove(notebookId, note.id)}
      >
        <Trash2 class="size-4" strokeWidth={1.75} />
      </button>
    </div>
  {/if}

  {#if editing}
    <NoteEditor {note} {notebookId} onclose={() => (editing = false)} />
  {:else}
    {@render header?.()}

    <div class="pr-16">
      {@render body()}
    </div>

    <p class="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground/70">
      <span>{formatRelativeTime(note.created_at)}</span>
      {#if edited}
        <span aria-hidden="true">·</span>
        <span>edited</span>
      {/if}
    </p>
  {/if}
</article>
