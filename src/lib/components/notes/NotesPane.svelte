<!-- Notes tab: two labeled sections — KEY INSIGHTS (saved chat answers grouped by
     source) and PERSONAL NOTES (user-authored manual notes) — a bottom composer
     to add a manual note, and a right-edge timeline that jumps to any note. The
     empty state shows only when both sections are empty. -->
<script lang="ts">
  import { tick } from 'svelte';
  import BookmarkX from '@lucide/svelte/icons/bookmark-x';
  import Plus from '@lucide/svelte/icons/plus';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Scrubber, type ScrubberItem } from '$lib/components/ui/scrubber/index.js';
  import KeyInsightCard from './KeyInsightCard.svelte';
  import ManualNoteCard from './ManualNoteCard.svelte';
  import { notesStore, hydrate, addManualNote } from '$lib/notes/notes-state.svelte.js';
  import { truncateLabel } from '$lib/utils.js';
  import type { Note } from '$lib/notes/types.js';

  interface Props {
    notebookId: string;
  }

  let { notebookId }: Props = $props();

  $effect(() => {
    void hydrate(notebookId);
  });

  const groups = $derived(notesStore.groupedBySource(notebookId));
  const manual = $derived(notesStore.manualNotes(notebookId));
  const isEmpty = $derived(groups.length === 0 && manual.length === 0);

  const insightNotes = $derived(groups.flatMap((g) => g.notes));
  // Timeline order mirrors on-screen order: insights first, then personal notes.
  const timelineItems = $derived<ScrubberItem[]>(
    [...insightNotes, ...manual].map((note) => ({ id: note.id, label: noteLabel(note) }))
  );

  function noteLabel(note: Note): string {
    if (note.source_title) return note.source_title;
    return truncateLabel(note.content.trim().split('\n')[0] ?? '');
  }

  let viewportRef = $state<HTMLElement | null>(null);
  let activeNoteId = $state<string | null>(null);
  let draft = $state('');
  const canSave = $derived(draft.trim().length > 0);

  // A note counts as "in view" once its top has scrolled to/above this band.
  const ACTIVE_TOP_BAND_PX = 96;

  function updateActiveNote(): void {
    const vp = viewportRef;
    if (!vp) return;
    const vpTop = vp.getBoundingClientRect().top;
    let active: string | null = null;
    for (const el of vp.querySelectorAll<HTMLElement>('[data-note-id]')) {
      if (el.getBoundingClientRect().top - vpTop <= ACTIVE_TOP_BAND_PX) {
        active = el.dataset.noteId ?? active;
      } else {
        break;
      }
    }
    activeNoteId = active ?? timelineItems[0]?.id ?? null;
  }

  function scrollToNote(noteId: string): void {
    const el = viewportRef?.querySelector<HTMLElement>(`[data-note-id="${CSS.escape(noteId)}"]`);
    el?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  async function submit(): Promise<void> {
    if (!canSave) return;
    const content = draft;
    draft = '';
    try {
      await addManualNote(notebookId, content);
    } catch (err) {
      // Restore the composer so a failed save doesn't silently discard the text.
      draft = content;
      throw err;
    }
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void submit();
    }
  }

  $effect(() => {
    void timelineItems.length;
    tick().then(updateActiveNote);
  });

  // ScrollArea doesn't forward onscroll from its Viewport, so wire it imperatively.
  $effect(() => {
    const el = viewportRef;
    if (!el) return;
    el.addEventListener('scroll', updateActiveNote, { passive: true });
    return () => el.removeEventListener('scroll', updateActiveNote);
  });
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
      <p class="mt-1 text-sm font-semibold text-foreground">No notes yet</p>
      <p class="max-w-[280px] text-xs leading-relaxed text-muted-foreground/70">
        Save a grounded answer from the chat, or add a personal note below.
      </p>
    </div>
  {:else}
    <div class="relative min-h-0 flex-1">
      <ScrollArea bind:viewportRef scrollbarYClasses="hidden" class="min-h-0 flex-1">
        <div class="flex flex-col gap-6 p-4">
          {#if groups.length > 0}
            <section class="flex flex-col gap-3">
              <h2 class="text-xs font-semibold tracking-wide text-muted-foreground/70">
                KEY INSIGHTS
              </h2>
              {#each groups as group (group.sourceId ?? 'untagged')}
                <div class="flex flex-col gap-2">
                  {#each group.notes as note (note.id)}
                    <KeyInsightCard {note} {notebookId} />
                  {/each}
                </div>
              {/each}
            </section>
          {/if}

          {#if manual.length > 0}
            <section class="flex flex-col gap-3">
              <h2 class="text-xs font-semibold tracking-wide text-muted-foreground/70">
                PERSONAL NOTES
              </h2>
              <div class="flex flex-col gap-2">
                {#each manual as note (note.id)}
                  <ManualNoteCard {note} {notebookId} />
                {/each}
              </div>
            </section>
          {/if}
        </div>
      </ScrollArea>

      <Scrubber
        items={timelineItems}
        activeId={activeNoteId}
        onjump={scrollToNote}
        ariaLabel="Notes timeline"
      />
    </div>
  {/if}

  <div class="flex items-center gap-2 border-t border-border p-3">
    <Input
      bind:value={draft}
      placeholder="Add a note…"
      aria-label="Add a note"
      onkeydown={onKeydown}
    />
    <Button disabled={!canSave} onclick={submit} aria-label="Save note">
      <Plus class="size-4" strokeWidth={2} />
      Save
    </Button>
  </div>
</div>
