<!-- Inline muted error card for a non-cancel stream failure (AC7, AC21). Shows
     the sanitized {kind,message} LensError text only — no internal detail. -->
<script lang="ts">
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';

  interface Props {
    error: { kind: string; message: string };
    onretry: () => void;
  }

  let { error, onretry }: Props = $props();
</script>

<div
  class="mx-4 mb-2 flex items-start justify-between gap-3 rounded-lg border border-destructive/25 bg-destructive/5 px-3 py-2.5"
  role="alert"
>
  <div class="min-w-0">
    <p class="text-xs font-semibold text-destructive">{error.kind}</p>
    <p class="mt-0.5 text-xs text-muted-foreground">{error.message}</p>
  </div>
  <button
    type="button"
    onclick={onretry}
    class="flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs font-semibold text-destructive transition-colors hover:bg-destructive/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
  >
    <RefreshCw class="size-3" strokeWidth={2.5} />
    Retry
  </button>
</div>
