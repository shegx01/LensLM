<!-- Right-edge conversation scrubber. Collapsed: a column of ticks, one per turn,
     the in-view turn marked in accent. On hover the rail expands into a labeled
     panel (right-aligned question previews + ticks); click a row to jump to it.
     Replaces the default scrollbar (hidden on the transcript's ScrollArea), and
     lives in the transcript's right gutter so message padding stays symmetric. -->
<script lang="ts">
  import { cn } from '$lib/utils.js';
  import type { Turn } from '$lib/chat/types.js';

  interface Props {
    turns: Turn[];
    activeTurnId: string | null;
    onjump: (turnId: string) => void;
  }

  let { turns, activeTurnId, onjump }: Props = $props();

  /** Single-line question preview for the expanded label. */
  function snippet(turn: Turn): string {
    const s = turn.user.content.trim().replace(/\s+/g, ' ');
    return s.length > 60 ? `${s.slice(0, 60)}…` : s;
  }
</script>

{#if turns.length > 1}
  <div
    class="group/scrubber absolute inset-y-0 right-0 z-10 flex items-center"
    role="navigation"
    aria-label="Conversation timeline"
  >
    <div
      class="no-scrollbar flex max-h-full flex-col items-stretch justify-center gap-0.5 overflow-y-auto rounded-l-2xl py-3 pr-1.5 pl-2 transition-all duration-150 group-hover/scrubber:bg-popover/95 group-hover/scrubber:pl-3 group-hover/scrubber:shadow-lg group-hover/scrubber:backdrop-blur-sm"
    >
      {#each turns as turn (turn.turn_id)}
        {@const active = turn.turn_id === activeTurnId}
        <button
          type="button"
          class="flex items-center justify-end gap-2 rounded-md py-1 pr-0 pl-1.5 text-right transition-colors group-hover/scrubber:hover:bg-muted/70"
          aria-label={`Jump to: ${snippet(turn)}`}
          aria-current={active ? 'true' : undefined}
          onclick={() => onjump(turn.turn_id)}
        >
          <span
            class={cn(
              'max-w-0 truncate overflow-hidden text-[13px] leading-tight whitespace-nowrap opacity-0 transition-all duration-150 group-hover/scrubber:max-w-[240px] group-hover/scrubber:opacity-100',
              active ? 'font-semibold text-primary' : 'text-muted-foreground'
            )}
          >
            {snippet(turn)}
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
