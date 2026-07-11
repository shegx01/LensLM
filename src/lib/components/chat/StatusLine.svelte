<!-- In-flight status line driven by `AnswerStage` (AC2). -->
<script lang="ts">
  import type { AnswerStage } from '$lib/chat/types.js';

  interface Props {
    stage: AnswerStage | null;
  }

  let { stage }: Props = $props();

  const label = $derived.by(() => {
    switch (stage) {
      case 'Retrieving':
        return 'Searching sources…';
      case 'Thinking':
        return 'Thinking…';
      case 'Answering':
        return 'Answering…';
      default:
        return null;
    }
  });
</script>

{#if label}
  <div class="flex items-center gap-1.5 px-4 pb-1 text-xs text-muted-foreground" aria-live="polite">
    <span class="flex gap-0.5" aria-hidden="true">
      <span class="size-1 animate-pulse rounded-full bg-current [animation-delay:0ms]"></span>
      <span class="size-1 animate-pulse rounded-full bg-current [animation-delay:150ms]"></span>
      <span class="size-1 animate-pulse rounded-full bg-current [animation-delay:300ms]"></span>
    </span>
    {label}
  </div>
{/if}
