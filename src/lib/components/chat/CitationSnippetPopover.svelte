<!-- Global inline-citation preview (issue #237): ONE instance mounted app-wide
     (AppShell), driven by the citation-preview store since chips are built OUTSIDE
     Svelte (citation-inline.ts). Anchors via bits-ui's `customAnchor` to the raw
     chip button. Lists one snippet per locator (a citation can carry several); a
     null-offset locator skips the snippet and only offers "view in source". Source
     text is rendered with plain interpolation only — never {@html}. -->
<script lang="ts">
  import ExternalLink from '@lucide/svelte/icons/external-link';
  import { Popover, PopoverContent } from '$lib/components/ui/popover/index.js';
  import {
    citationPreviewStore,
    cancelHideCitationPreview,
    scheduleHideCitationPreview,
    hideCitationPreviewNow
  } from '$lib/chat/citation-preview.svelte.js';
  import { resolveCitationSnippet } from '$lib/sources/source-text.js';
  import { openSourceViewer } from '$lib/sources/sources-state.svelte.js';
  import type { SnippetSegments } from '$lib/chat/types.js';

  type SnippetState =
    | { status: 'loading' }
    | { status: 'ready'; segments: SnippetSegments }
    | { status: 'error' };

  let snippets = $state<Map<number, SnippetState>>(new Map());

  const request = $derived(citationPreviewStore.request);
  const open = $derived(citationPreviewStore.open);

  $effect(() => {
    const req = request;
    if (!req) return;

    const next = new Map<number, SnippetState>();
    req.target.locators.forEach((loc, i) => {
      if (loc.char_start !== null && loc.char_end !== null) next.set(i, { status: 'loading' });
    });
    snippets = next;

    req.target.locators.forEach((loc, i) => {
      if (loc.char_start === null || loc.char_end === null) return;
      resolveCitationSnippet(req.target.source_id, loc.char_start, loc.char_end)
        .then((segments) => {
          if (citationPreviewStore.request !== req) return; // a newer chip took over
          snippets = new Map(snippets).set(i, { status: 'ready', segments });
        })
        .catch(() => {
          if (citationPreviewStore.request !== req) return;
          snippets = new Map(snippets).set(i, { status: 'error' });
        });
    });
  });

  function locatorLabel(loc: { page: number | null }, index: number, total: number): string {
    if (loc.page !== null) return `Page ${loc.page}`;
    return total > 1 ? `Excerpt ${index + 1}` : 'Excerpt';
  }

  // The engine's WINDOW snap (lens-core citation_source.rs) always returns a large
  // leading `before` (~240+ chars past the doc start), which used to push the
  // highlighted `marked` span below the fold under a whole-paragraph line-clamp.
  // Trim before/after for DISPLAY only, keeping the text nearest the mark so it is
  // always visible. Trim by code point (Array.from), not UTF-16 index, so an
  // astral emoji/surrogate pair at the cut boundary can't be split into U+FFFD.
  const BEFORE_DISPLAY_CHARS = 90;
  const AFTER_DISPLAY_CHARS = 130;

  function trimTail(text: string, maxChars: number): { text: string; trimmed: boolean } {
    const chars = Array.from(text);
    if (chars.length <= maxChars) return { text, trimmed: false };
    return { text: chars.slice(chars.length - maxChars).join(''), trimmed: true };
  }

  function trimHead(text: string, maxChars: number): { text: string; trimmed: boolean } {
    const chars = Array.from(text);
    if (chars.length <= maxChars) return { text, trimmed: false };
    return { text: chars.slice(0, maxChars).join(''), trimmed: true };
  }

  function trimForDisplay(segments: SnippetSegments): {
    before: string;
    after: string;
    showBeforeEllipsis: boolean;
    showAfterEllipsis: boolean;
  } {
    const before = trimTail(segments.before, BEFORE_DISPLAY_CHARS);
    const after = trimHead(segments.after, AFTER_DISPLAY_CHARS);
    return {
      before: before.text,
      after: after.text,
      showBeforeEllipsis: segments.truncated_before || before.trimmed,
      showAfterEllipsis: segments.truncated_after || after.trimmed
    };
  }

  function viewInSource(locatorIndex: number): void {
    if (!request) return;
    const loc = request.target.locators[locatorIndex];
    hideCitationPreviewNow();
    openSourceViewer(request.target.source_id, loc.char_start, loc.char_end);
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') hideCitationPreviewNow();
  }
</script>

{#if request}
  <Popover
    {open}
    onOpenChange={(v) => {
      if (!v) hideCitationPreviewNow();
    }}
  >
    <PopoverContent
      customAnchor={request.anchor}
      trapFocus={false}
      side="top"
      align="start"
      onpointerenter={cancelHideCitationPreview}
      onpointerleave={scheduleHideCitationPreview}
      onfocusin={cancelHideCitationPreview}
      onfocusout={scheduleHideCitationPreview}
      onkeydown={handleKeydown}
      class="w-80"
    >
      <p class="mb-2 truncate text-xs font-semibold text-foreground">{request.target.title}</p>
      <div class="flex flex-col gap-2.5">
        {#each request.target.locators as loc, i (i)}
          <div class="rounded-lg border border-border/60 bg-muted/40 p-2">
            <p class="mb-1 text-[10px] font-bold tracking-wide text-muted-foreground uppercase">
              {locatorLabel(loc, i, request.target.locators.length)}
            </p>
            {#if loc.char_start === null || loc.char_end === null}
              <p class="text-xs text-muted-foreground">No excerpt available for this reference.</p>
            {:else}
              {@const state = snippets.get(i)}
              {#if !state || state.status === 'loading'}
                <p class="text-xs text-muted-foreground">Loading excerpt…</p>
              {:else if state.status === 'error'}
                <p class="text-xs text-muted-foreground">Couldn't load this excerpt.</p>
              {:else}
                {@const display = trimForDisplay(state.segments)}
                <p class="text-xs leading-relaxed text-foreground">
                  {#if display.showBeforeEllipsis}<span aria-hidden="true">…</span
                    >{/if}{display.before}<mark
                    class="citation-highlight rounded-[3px] bg-primary/25 px-0.5 text-foreground"
                    >{state.segments.marked}</mark
                  >{display.after}{#if display.showAfterEllipsis}<span aria-hidden="true">…</span
                    >{/if}
                </p>
              {/if}
            {/if}
            <button
              type="button"
              onclick={() => viewInSource(i)}
              class="mt-1.5 inline-flex items-center gap-1 text-[11px] font-semibold text-primary hover:underline"
            >
              <ExternalLink class="size-3" strokeWidth={2.25} />
              View in source
            </button>
          </div>
        {/each}
      </div>
    </PopoverContent>
  </Popover>
{/if}
