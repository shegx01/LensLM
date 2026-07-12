<!-- Footer row of per-source citation chips under a settled assistant answer.
     Resolves each Citation's title from the live sources store; a citation whose
     source is gone renders a disabled fallback chip (AC5). Clicking a live chip
     reveals its source in the SourcesRail (AC6) via the single focusSource seam. -->
<script lang="ts">
  import type { Citation } from '$lib/chat/types.js';
  import { sourcesStore, focusSource } from '$lib/sources/sources-state.svelte.js';
  import CitationChip from './CitationChip.svelte';

  interface Props {
    citations: Citation[];
  }

  let { citations }: Props = $props();

  const resolved = $derived(
    [...citations]
      .sort((a, b) => a.ordinal - b.ordinal)
      .map((c) => {
        const src = sourcesStore.sources.find((s) => s.id === c.source_id);
        return {
          source_id: c.source_id,
          ordinal: c.ordinal,
          live: !!src,
          label: src?.title ?? 'Removed source'
        };
      })
  );
</script>

{#if resolved.length > 0}
  <div class="mt-2 flex flex-wrap gap-1.5" aria-label="Sources cited in this answer">
    {#each resolved as c (c.source_id)}
      <CitationChip
        n={c.ordinal}
        label={c.label}
        live={c.live}
        onactivate={() => focusSource(c.source_id)}
      />
    {/each}
  </div>
{/if}
