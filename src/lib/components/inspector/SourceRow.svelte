<!--
  SourceRow — one row in the EmbeddingsInspector left pane: doc icon, title,
  kind badge, and a status dot (shared statusDotClass, semantic tokens only).
  Presentational: all state lives in the parent.

  For error sources, the row renders a retry affordance (small icon button)
  visible on hover so the inspector can trigger a retry in place.
-->
<script lang="ts">
  import FileText from '@lucide/svelte/icons/file-text';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import { cn } from '$lib/utils.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { statusDotClass } from '$lib/sources/status.js';
  import type { Source, SourceStatus } from '$lib/sources/types.js';

  interface Props {
    source: Source;
    selected: boolean;
    onselect: () => void;
    onretry?: () => void;
  }

  const { source, selected, onselect, onretry }: Props = $props();
  const status = $derived(source.status as SourceStatus);
</script>

<li class="group">
  <button
    type="button"
    onclick={onselect}
    aria-pressed={selected}
    class={cn(
      'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors',
      'hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
      selected && 'bg-muted'
    )}
  >
    <div
      class="flex size-7 shrink-0 items-center justify-center rounded-[6px] bg-muted"
      aria-hidden="true"
    >
      <FileText class="size-3.5 text-muted-foreground" strokeWidth={1.75} />
    </div>
    <div class="min-w-0 flex-1">
      <div class="truncate text-sm font-medium leading-tight text-foreground">
        {source.title}
      </div>
      <div class="mt-0.5">
        <Badge variant="outline" class="px-1.5 py-0 text-[0.625rem] uppercase">
          {source.kind}
        </Badge>
      </div>
    </div>
    <span
      class={cn('block size-[7px] shrink-0 rounded-full', statusDotClass(status))}
      aria-label={`Status: ${source.status}`}
    ></span>
  </button>

  <!-- Retry button — only for error sources; appears on row hover. -->
  {#if status === 'error' && onretry}
    <div class="px-2.5 pb-1">
      <button
        type="button"
        aria-label="Retry ingesting {source.title}"
        data-retry-source-btn
        onclick={(e) => {
          e.stopPropagation();
          onretry();
        }}
        class={cn(
          'flex w-full items-center gap-1.5 rounded-[5px] px-2 py-1 text-[0.6875rem]',
          'opacity-0 transition-opacity duration-150 group-hover:opacity-100',
          'border border-destructive/30 bg-destructive/10 text-destructive',
          'hover:bg-destructive/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
        )}
      >
        <RotateCcw class="size-3 shrink-0" strokeWidth={2} />
        Retry
      </button>
    </div>
  {/if}
</li>
