<!--
  ChunkCard — one chunk in the EmbeddingsInspector right pane: level + block_type
  badges, section_path trail, char range, the canonical text (expand/collapse),
  the source_anchor JSON, and the enriched embedding_text (collapsible).

  Owns its OWN expand state locally ($state) — simpler than threading per-chunk
  Sets through the parent. All DB content renders via Svelte interpolation
  (auto-escaped); no {@html}.
-->
<script lang="ts">
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { cn } from '$lib/utils.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import type { InspectorChunk } from '$lib/inspector/types.js';

  interface Props {
    chunk: InspectorChunk;
  }

  const { chunk }: Props = $props();

  let textOpen = $state(false);
  let enrichedOpen = $state(false);
</script>

<li class="rounded-xl border border-border bg-card p-4 text-card-foreground">
  <!-- Meta row -->
  <div class="mb-2 flex flex-wrap items-center gap-2">
    <Badge variant="outline" class="text-[0.625rem] font-semibold uppercase">
      L{chunk.level}
    </Badge>
    {#if chunk.block_type}
      <Badge variant="secondary" class="text-[0.625rem]">{chunk.block_type}</Badge>
    {/if}
    <span class="truncate text-[11px] text-muted-foreground" title={chunk.section_path}>
      {chunk.section_path}
    </span>
    <div class="flex-1"></div>
    {#if chunk.char_start !== null && chunk.char_end !== null}
      <span class="shrink-0 text-[10px] tabular-nums text-muted-foreground/60">
        {chunk.char_start}..{chunk.char_end}
      </span>
    {/if}
  </div>

  <!-- Text -->
  <p
    class={cn(
      'whitespace-pre-wrap text-sm leading-relaxed text-foreground',
      !textOpen && 'line-clamp-3'
    )}
  >
    {chunk.text}
  </p>
  <button
    type="button"
    onclick={() => (textOpen = !textOpen)}
    class="mt-1 inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground"
  >
    {#if textOpen}
      <ChevronDown class="size-3" /> Collapse
    {:else}
      <ChevronRight class="size-3" /> Expand
    {/if}
  </button>

  <!-- source_anchor (JSON, if present) -->
  {#if chunk.source_anchor}
    <pre
      class="mt-2 overflow-x-auto rounded-md bg-muted px-2.5 py-1.5 text-[10px] leading-snug text-muted-foreground">{chunk.source_anchor}</pre>
  {/if}

  <!-- embedding_text (collapsible "Enriched text", if present) -->
  {#if chunk.embedding_text}
    <button
      type="button"
      onclick={() => (enrichedOpen = !enrichedOpen)}
      class="mt-2 inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground"
    >
      {#if enrichedOpen}
        <ChevronDown class="size-3" />
      {:else}
        <ChevronRight class="size-3" />
      {/if}
      Enriched text
    </button>
    {#if enrichedOpen}
      <p
        class="mt-1 whitespace-pre-wrap rounded-md border border-border bg-muted/40 px-2.5 py-2 text-[12px] leading-relaxed text-muted-foreground"
      >
        {chunk.embedding_text}
      </p>
    {/if}
  {/if}
</li>
