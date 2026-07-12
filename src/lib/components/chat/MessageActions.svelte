<!-- Action row: save-to-notes, copy, regenerate, thumbs up/down (AC11, AC13-AC14,
     AC22, #24). No citation chips (#23) here. -->
<script lang="ts">
  import Bookmark from '@lucide/svelte/icons/bookmark';
  import Copy from '@lucide/svelte/icons/copy';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import ThumbsUp from '@lucide/svelte/icons/thumbs-up';
  import ThumbsDown from '@lucide/svelte/icons/thumbs-down';
  import Check from '@lucide/svelte/icons/check';
  import { cn } from '$lib/utils.js';
  import type { ChatFeedback } from '$lib/chat/types.js';

  interface Props {
    feedback: ChatFeedback;
    saved: boolean;
    disabled?: boolean;
    oncopy: () => void;
    onregenerate: () => void;
    onfeedback: (next: 'up' | 'down') => void;
    onsave: () => void;
  }

  let {
    feedback,
    saved,
    disabled = false,
    oncopy,
    onregenerate,
    onfeedback,
    onsave
  }: Props = $props();

  let copied = $state(false);
  let copyTimer: ReturnType<typeof setTimeout> | undefined;

  function handleCopy(): void {
    oncopy();
    copied = true;
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => (copied = false), 1500);
  }

  $effect(() => {
    return () => clearTimeout(copyTimer);
  });

  const iconBtn =
    'flex size-7 items-center justify-center rounded-md text-muted-foreground/70 transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-40';
</script>

<div class="mt-1 flex items-center gap-0.5" role="toolbar" aria-label="Message actions">
  <button
    type="button"
    class={cn(iconBtn, saved && 'text-primary hover:text-primary')}
    aria-label={saved ? 'Remove from notes' : 'Save to notes'}
    aria-pressed={saved}
    onclick={onsave}
  >
    <Bookmark class="size-3.5" strokeWidth={2} fill={saved ? 'currentColor' : 'none'} />
  </button>

  <button
    type="button"
    class={iconBtn}
    aria-label={copied ? 'Copied' : 'Copy answer'}
    onclick={handleCopy}
  >
    {#if copied}
      <Check class="size-3.5 text-primary" strokeWidth={2.25} />
    {:else}
      <Copy class="size-3.5" strokeWidth={2} />
    {/if}
  </button>

  <button
    type="button"
    class={iconBtn}
    aria-label="Regenerate answer"
    {disabled}
    onclick={onregenerate}
  >
    <RefreshCw class="size-3.5" strokeWidth={2} />
  </button>

  <button
    type="button"
    class={cn(iconBtn, feedback === 'up' && 'text-primary hover:text-primary')}
    aria-label="Good response"
    aria-pressed={feedback === 'up'}
    onclick={() => onfeedback('up')}
  >
    <ThumbsUp class="size-3.5" strokeWidth={2} fill={feedback === 'up' ? 'currentColor' : 'none'} />
  </button>

  <button
    type="button"
    class={cn(iconBtn, feedback === 'down' && 'text-destructive hover:text-destructive')}
    aria-label="Bad response"
    aria-pressed={feedback === 'down'}
    onclick={() => onfeedback('down')}
  >
    <ThumbsDown
      class="size-3.5"
      strokeWidth={2}
      fill={feedback === 'down' ? 'currentColor' : 'none'}
    />
  </button>
</div>
