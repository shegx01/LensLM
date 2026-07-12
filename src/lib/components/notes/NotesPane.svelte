<!-- Notes tab (#24): KEY INSIGHTS — saved chat answers grouped by source,
     newest-first. No PERSONAL NOTES section, composer, or per-card edit/delete
     here; that is #25's scope. -->
<script lang="ts">
  import BookmarkX from '@lucide/svelte/icons/bookmark-x';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import KeyInsightCard from './KeyInsightCard.svelte';
  import { notesStore, hydrate } from '$lib/notes/notes-state.svelte.js';

  interface Props {
    notebookId: string;
  }

  let { notebookId }: Props = $props();

  $effect(() => {
    void hydrate(notebookId);
  });

  const groups = $derived(notesStore.groupedBySource(notebookId));
  const isEmpty = $derived(groups.length === 0);
</script>

<div class="flex min-h-0 flex-1 flex-col overflow-hidden">
  {#if isEmpty}
    <div class="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center">
      <div
        class="flex size-11 items-center justify-center rounded-2xl bg-primary/10"
        aria-hidden="true"
      >
        <BookmarkX class="size-5 text-primary" strokeWidth={1.75} />
      </div>
      <p class="mt-1 text-sm font-semibold text-foreground">No saved notes yet</p>
      <p class="max-w-[280px] text-xs leading-relaxed text-muted-foreground/70">
        Save a grounded answer from the chat to see it here.
      </p>
    </div>
  {:else}
    <ScrollArea class="min-h-0 flex-1">
      <div class="flex flex-col gap-4 p-4">
        <h2 class="text-xs font-semibold tracking-wide text-muted-foreground/70">KEY INSIGHTS</h2>
        {#each groups as group (group.sourceId ?? 'untagged')}
          <div class="flex flex-col gap-2">
            {#each group.notes as note (note.id)}
              <KeyInsightCard {note} />
            {/each}
          </div>
        {/each}
      </div>
    </ScrollArea>
  {/if}
</div>
