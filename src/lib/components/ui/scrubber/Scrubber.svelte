<!-- Generic right-edge timeline scrubber. Collapsed: a column of ticks, one per
     item, the active one marked in accent. On hover the rail expands into a
     labeled panel (right-aligned previews + ticks); click a row to jump to it.
     Extracted from ChatScrubber so Notes can reuse it. -->
<script lang="ts">
  import { cn } from '$lib/utils.js';

  export interface ScrubberItem {
    id: string;
    label: string;
  }

  interface Props {
    items: ScrubberItem[];
    activeId: string | null;
    onjump: (id: string) => void;
    ariaLabel?: string;
  }

  let { items, activeId, onjump, ariaLabel = 'Timeline' }: Props = $props();
</script>

{#if items.length > 1}
  <div
    class="group/scrubber absolute inset-y-0 right-0 z-10 flex items-center"
    role="navigation"
    aria-label={ariaLabel}
  >
    <div
      class="no-scrollbar flex max-h-full flex-col items-stretch justify-center gap-0.5 overflow-y-auto rounded-l-2xl py-3 pr-1.5 pl-2 transition-all duration-150 group-hover/scrubber:bg-popover/95 group-hover/scrubber:pl-3 group-hover/scrubber:shadow-lg group-hover/scrubber:backdrop-blur-sm"
    >
      {#each items as item (item.id)}
        {@const active = item.id === activeId}
        <button
          type="button"
          class="flex items-center justify-end gap-2 rounded-md py-1 pr-0 pl-1.5 text-right transition-colors group-hover/scrubber:hover:bg-muted/70"
          aria-label={`Jump to: ${item.label}`}
          aria-current={active ? 'true' : undefined}
          onclick={() => onjump(item.id)}
        >
          <span
            class={cn(
              'max-w-0 truncate overflow-hidden text-[13px] leading-tight whitespace-nowrap opacity-0 transition-all duration-150 group-hover/scrubber:max-w-[240px] group-hover/scrubber:opacity-100',
              active ? 'font-semibold text-primary' : 'text-muted-foreground'
            )}
          >
            {item.label}
          </span>
          <span
            class={cn(
              'block h-[2px] shrink-0 rounded-full transition-all duration-150',
              active ? 'w-4 bg-primary' : 'w-2.5 bg-border'
            )}
          ></span>
        </button>
      {/each}
    </div>
  </div>
{/if}
