<!-- Shared onboarding "Back" affordance: muted ChevronLeft + label, fires `onback`. -->
<script lang="ts">
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';

  let { onback }: { onback: () => void } = $props();
</script>

<button
  type="button"
  onclick={onback}
  aria-label="Back"
  class="text-muted-foreground inline-flex cursor-pointer items-center gap-[5px] border-none bg-transparent p-0 text-xs font-normal transition-opacity hover:opacity-70"
>
  <ChevronLeft class="size-3.5" />
  Back
</button>
