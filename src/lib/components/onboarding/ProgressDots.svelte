<!-- Shared onboarding step indicator: `total` bars, the first `current` (1-based)
     filled with the accent, the rest the inactive `bg-border` token. One ARIA
     progressbar pattern across all onboarding screens. -->
<script lang="ts">
  let { current, total = 3 }: { current: number; total?: number } = $props();
</script>

<div
  class="flex items-center gap-[5px]"
  role="progressbar"
  aria-valuenow={current}
  aria-valuemin={1}
  aria-valuemax={total}
  aria-label={`Step ${current} of ${total}`}
>
  {#each { length: total } as _, i (i)}
    <div
      class={['h-[3px] w-[22px] rounded-full', i + 1 <= current ? 'bg-primary' : 'bg-border'].join(
        ' '
      )}
    ></div>
  {/each}
</div>
