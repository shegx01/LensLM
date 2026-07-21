<!-- Full-document "view in source" viewer (issue #237, AC2). Opened via
     sources-state's openSourceViewer/sourceViewerNonce bridge (mirrors
     focusSource/focusNonce — chips live outside Svelte). Lazily loads
     `load_source_view` and renders the retained ORIGINAL text as plain
     interpolation only (never {@html}) with the cited span highlighted and
     scrolled into view. A purged/removed source degrades to a friendly message,
     not a raw error. -->
<script lang="ts">
  import { tick } from 'svelte';
  import FileText from '@lucide/svelte/icons/file-text';
  import Loader2 from '@lucide/svelte/icons/loader-2';
  import AlertCircle from '@lucide/svelte/icons/alert-circle';
  import ExternalLink from '@lucide/svelte/icons/external-link';
  import X from '@lucide/svelte/icons/x';
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription
  } from '$lib/components/ui/dialog/index.js';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import { loadSourceView } from '$lib/sources/source-text.js';
  import { sourcesStore, focusSource } from '$lib/sources/sources-state.svelte.js';
  import { prefersReducedMotion } from '$lib/motion/index.js';
  import type { SourceView } from '$lib/chat/types.js';

  let open = $state(false);
  let loading = $state(false);
  let errored = $state(false);
  let view = $state<SourceView | null>(null);
  let markedEl = $state<HTMLElement | null>(null);
  let requestedSourceId = $state<string | null>(null);

  $effect(() => {
    const req = sourcesStore.sourceViewerRequest;
    // Read as a dependency (not the request's identity) so re-opening the SAME
    // source+span re-fires this effect, mirroring the focusSource/focusNonce bridge.
    void sourcesStore.sourceViewerNonce;
    if (!req) return;

    open = true;
    loading = true;
    errored = false;
    view = null;
    requestedSourceId = req.sourceId;

    // A fresh object every call (even for the same source+span, mirroring the
    // nonce bump), so reference equality alone detects a newer request taking over.
    const isStale = () => sourcesStore.sourceViewerRequest !== req;

    loadSourceView(req.sourceId, req.charStart, req.charEnd)
      .then((v) => {
        if (isStale()) return;
        view = v;
      })
      .catch((err) => {
        if (isStale()) return;
        console.error('SourceViewer: load_source_view failed', err);
        errored = true;
      })
      .finally(() => {
        if (isStale()) return;
        loading = false;
      });
  });

  $effect(() => {
    if (!view?.marked || !markedEl) return;
    const el = markedEl;
    void tick().then(() => {
      el.scrollIntoView({ block: 'center', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    });
  });

  function revealInRail(): void {
    if (!requestedSourceId) return;
    focusSource(requestedSourceId);
    open = false;
  }
</script>

<Dialog {open} onOpenChange={(v) => (open = v)}>
  <DialogContent
    showCloseButton={false}
    class="flex h-[80vh] w-full max-w-2xl flex-col gap-0 overflow-hidden rounded-xl border-border bg-card p-0"
  >
    <DialogHeader
      class="flex-row items-center justify-between gap-2 border-b border-border px-5 py-3.5 space-y-0"
    >
      <div class="flex min-w-0 items-center gap-2.5">
        <div
          class="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground"
          aria-hidden="true"
        >
          <FileText class="size-4" />
        </div>
        <div class="flex min-w-0 flex-col">
          <DialogTitle class="truncate text-sm font-bold text-foreground">
            {view?.title ?? 'Source'}
          </DialogTitle>
          {#if view}
            <DialogDescription class="text-[11px] text-muted-foreground">
              {view.kind}
            </DialogDescription>
          {/if}
        </div>
      </div>
      <div class="flex shrink-0 items-center gap-1">
        <button
          type="button"
          onclick={revealInRail}
          class="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold text-primary hover:bg-primary/10"
        >
          <ExternalLink class="size-3.5" strokeWidth={2.25} />
          Reveal in sources
        </button>
        <button
          type="button"
          aria-label="Close"
          onclick={() => (open = false)}
          class="flex size-7 items-center justify-center rounded-full bg-muted text-muted-foreground transition-opacity outline-none hover:opacity-65 focus-visible:ring-2 focus-visible:ring-ring/50"
        >
          <X class="size-3" strokeWidth={2.5} />
        </button>
      </div>
    </DialogHeader>

    <div class="min-h-0 flex-1">
      <ScrollArea class="h-full">
        <div class="px-5 py-4">
          {#if loading}
            <div class="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
              <Loader2 class="size-4 animate-spin" />
              Loading source…
            </div>
          {:else if errored}
            <div class="flex flex-col items-center gap-2 py-16 text-center">
              <AlertCircle class="size-6 text-muted-foreground/60" />
              <p class="text-sm font-medium text-foreground">This source is no longer available.</p>
              <p class="text-xs text-muted-foreground">
                It may have been removed or purged from the notebook.
              </p>
            </div>
          {:else if view}
            {#if view.truncated}
              <p class="mb-3 rounded-lg bg-muted/60 px-3 py-2 text-[11px] text-muted-foreground">
                This document is large — showing the area around the cited excerpt.
              </p>
            {/if}
            <p
              class="citation-source-text text-sm leading-relaxed whitespace-pre-wrap text-foreground"
            >
              {view.before}{#if view.marked}<mark
                  bind:this={markedEl}
                  class="citation-highlight scroll-mt-8 rounded-[3px] bg-primary/25 px-0.5 text-foreground"
                  >{view.marked}</mark
                >{/if}{view.after}
            </p>
          {/if}
        </div>
      </ScrollArea>
    </div>
  </DialogContent>
</Dialog>
