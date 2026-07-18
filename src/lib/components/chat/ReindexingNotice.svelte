<!-- Calm inline notice for the RT-1 reindexing gap: the notebook's embeddings are
     rebuilding, so retrieval had no active index. Distinct from ErrorCard (this is
     transient + retryable, not a failure) — muted styling, no destructive color. -->
<script lang="ts">
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';

  interface Props {
    onretry: () => void;
  }

  let { onretry }: Props = $props();
</script>

<div
  class="mx-4 mb-2 flex items-start justify-between gap-3 rounded-lg border border-border bg-muted/40 px-3 py-2.5"
  role="status"
>
  <div class="min-w-0">
    <p class="text-xs font-semibold text-foreground">Updating this notebook</p>
    <p class="mt-0.5 text-xs text-muted-foreground">
      The embeddings are still being built. Try again in a moment.
    </p>
  </div>
  <button
    type="button"
    onclick={onretry}
    class="flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs font-semibold text-foreground transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
  >
    <RefreshCw class="size-3" strokeWidth={2.5} />
    Retry
  </button>
</div>
