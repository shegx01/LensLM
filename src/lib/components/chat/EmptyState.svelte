<!-- Centered empty state (AC20): no messages yet. -->
<script lang="ts">
  import Sparkles from '@lucide/svelte/icons/sparkles';
</script>

<div class="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center">
  <div
    class="flex size-11 items-center justify-center rounded-2xl bg-primary/10"
    aria-hidden="true"
  >
    <Sparkles class="size-5 text-primary" strokeWidth={1.75} />
  </div>
  <p class="mt-1 text-sm font-semibold text-foreground">Ask anything about your sources</p>
  <p class="max-w-[280px] text-xs leading-relaxed text-muted-foreground/70">
    Answers are grounded in this notebook's selected sources.
  </p>
</div>
