<!-- Right-edge conversation scrubber: a thin wrapper mapping `turns` → the
     generic Scrubber's `items` (label = a single-line question preview). See
     ui/scrubber/Scrubber.svelte for the tick/hover-expand interaction. -->
<script lang="ts">
  import { Scrubber } from '$lib/components/ui/scrubber/index.js';
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

  const items = $derived(turns.map((turn) => ({ id: turn.turn_id, label: snippet(turn) })));
</script>

<Scrubber {items} activeId={activeTurnId} {onjump} ariaLabel="Conversation timeline" />
