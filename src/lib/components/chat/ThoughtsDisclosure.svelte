<!-- Collapsible "Thoughts" panel fed by the ephemeral thinkingBuffer (AC3).
     Collapsed by default; only rendered by the caller when thinking tokens exist. -->
<script lang="ts">
  import { Collapsible } from 'bits-ui';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';

  interface Props {
    thinking: string;
  }

  let { thinking }: Props = $props();
  let open = $state(false);
</script>

<Collapsible.Root bind:open class="mx-4 mb-1">
  <Collapsible.Trigger
    class="flex items-center gap-1 rounded-md px-1.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
  >
    <ChevronRight class="size-3 transition-transform {open ? 'rotate-90' : ''}" strokeWidth={2.5} />
    Thoughts
  </Collapsible.Trigger>
  <Collapsible.Content>
    <div
      class="mt-1 rounded-lg bg-muted/60 px-3 py-2 text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap"
    >
      {thinking}
    </div>
  </Collapsible.Content>
</Collapsible.Root>
