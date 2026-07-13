<!-- In-place note editor: a CodeMirror-6 markdown SOURCE buffer (left) and a live
     PREVIEW (right) that reuses the shared render seam + citation-chip overlay, so
     `[n]` markers are plain text in the buffer and chips in the preview — identical
     to chat, with no serializer. The buffer is inert markdown; the only sanitize is
     the existing render-time DOMPurify pass (content is display/copy-only, never run).
     Save blocks empty/whitespace; Cancel restores the rendered card. Editor chrome is
     themed with app tokens only (light/dark/accent), no hard-coded colors. -->
<script lang="ts">
  import { EditorView, keymap } from '@codemirror/view';
  import { EditorState } from '@codemirror/state';
  import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
  import { markdown } from '@codemirror/lang-markdown';
  import Check from '@lucide/svelte/icons/check';
  import X from '@lucide/svelte/icons/x';
  import { Button } from '$lib/components/ui/button/index.js';
  import { renderMarkdown } from '$lib/chat/render-markdown.js';
  import { enhanceCitations, type CitationTarget } from '$lib/chat/citation-inline.js';
  import { hydrateMermaid } from '$lib/chat/mermaid.js';
  import { parseCitations } from '$lib/chat/citations.js';
  import { sourcesStore } from '$lib/sources/sources-state.svelte.js';
  import { editNote } from '$lib/notes/notes-state.svelte.js';
  import type { Note } from '$lib/notes/types.js';

  interface Props {
    note: Note;
    notebookId: string;
    /** Called after a successful save or on cancel to exit edit mode. */
    onclose: () => void;
  }

  let { note, notebookId, onclose }: Props = $props();

  // The buffer is seeded once from the note's content when the editor mounts (read
  // inside the mount effect's closure); further edits live in `draft` (the
  // CodeMirror doc), never written back to the reactive `note` prop.
  let draft = $state('');
  let editorHostEl = $state<HTMLElement | null>(null);
  let previewEl = $state<HTMLElement | null>(null);
  let saving = $state(false);
  const canSave = $derived(draft.trim().length > 0);

  const previewHtml = $derived(renderMarkdown(draft));

  // Ordinal → source, resolved against the live sources store so preview chips
  // match chat exactly (a removed source flips its chip to the disabled state).
  const citationTargets = $derived.by(() => {
    const map = new Map<number, CitationTarget>();
    for (const c of parseCitations(note.citations) ?? []) {
      const src = sourcesStore.sources.find((s) => s.id === c.source_id);
      map.set(c.ordinal, {
        source_id: c.source_id,
        title: src?.title ?? 'Removed source',
        live: !!src
      });
    }
    return map;
  });

  // App-token-only theme: CodeMirror maps to the same CSS vars as the rest of the
  // app, so light/dark/accent all follow with no hard-coded colors.
  const themeExt = EditorView.theme({
    '&': {
      backgroundColor: 'var(--background)',
      color: 'var(--foreground)',
      fontSize: '0.8125rem',
      borderRadius: '0.5rem'
    },
    '.cm-content': {
      fontFamily: 'var(--font-mono)',
      padding: '0.5rem 0.65rem',
      caretColor: 'var(--foreground)'
    },
    '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--foreground)' },
    '&.cm-focused': { outline: 'none' },
    '.cm-scroller': { fontFamily: 'var(--font-mono)', lineHeight: '1.5' },
    '.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': {
      backgroundColor: 'color-mix(in oklab, var(--primary) 25%, transparent)'
    },
    '.cm-activeLine': { backgroundColor: 'color-mix(in oklab, var(--muted) 55%, transparent)' }
  });

  $effect(() => {
    const host = editorHostEl;
    if (!host) return;
    const initialContent = note.content;
    draft = initialContent;
    const view = new EditorView({
      parent: host,
      state: EditorState.create({
        doc: initialContent,
        extensions: [
          history(),
          keymap.of([...defaultKeymap, ...historyKeymap]),
          markdown(),
          themeExt,
          EditorView.lineWrapping,
          EditorView.updateListener.of((u) => {
            if (u.docChanged) draft = u.state.doc.toString();
          })
        ]
      })
    });
    return () => view.destroy();
  });

  // Re-run on render (`previewHtml`) and when resolved targets change so chip
  // liveness stays in sync — same discipline as AssistantMessage.
  $effect(() => {
    void previewHtml;
    const targets = citationTargets;
    if (!previewEl || targets.size === 0) return;
    return enhanceCitations(previewEl, (n) => targets.get(n) ?? null);
  });

  // Upgrade supported ```mermaid fences in the live preview (idempotent).
  $effect(() => {
    void previewHtml;
    if (previewEl) void hydrateMermaid(previewEl);
  });

  async function save(): Promise<void> {
    if (!canSave || saving) return;
    saving = true;
    try {
      await editNote(notebookId, note.id, draft);
      onclose();
    } finally {
      saving = false;
    }
  }
</script>

<div class="flex flex-col gap-2">
  <div class="grid gap-2 sm:grid-cols-2">
    <div
      bind:this={editorHostEl}
      class="min-h-[7rem] overflow-hidden rounded-lg border border-input bg-background focus-within:ring-2 focus-within:ring-ring"
      aria-label="Edit note (markdown source)"
    ></div>

    <div
      class="note-preview min-h-[7rem] overflow-auto rounded-lg border border-border bg-card p-3 text-sm leading-relaxed text-card-foreground"
    >
      <p class="mb-1.5 text-[10px] font-bold tracking-[0.1em] uppercase text-muted-foreground/70">
        Preview
      </p>
      <!-- eslint-disable-next-line svelte/no-at-html-tags -->
      <div bind:this={previewEl}>{@html previewHtml}</div>
    </div>
  </div>

  <div class="flex items-center justify-end gap-2">
    <Button variant="ghost" size="sm" onclick={onclose} aria-label="Cancel edit">
      <X class="size-4" strokeWidth={2} />
      Cancel
    </Button>
    <Button size="sm" disabled={!canSave || saving} onclick={save} aria-label="Save edit">
      <Check class="size-4" strokeWidth={2} />
      Save
    </Button>
  </div>
</div>
