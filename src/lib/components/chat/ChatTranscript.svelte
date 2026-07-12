<!-- Scrollable transcript. Autoscrolls to the bottom while `pinnedToBottom`;
     scrolling up unpins and surfaces a "Jump to latest" pill (AC19). A turn
     with zero assistant versions (reloaded cancelled/errored turn) renders the
     user bubble alone — no assistant slot, no pager, no action row. -->
<script lang="ts">
  import { tick } from 'svelte';
  import ArrowDown from '@lucide/svelte/icons/arrow-down';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import UserMessage from './UserMessage.svelte';
  import AssistantMessage from './AssistantMessage.svelte';
  import StatusLine from './StatusLine.svelte';
  import ThoughtsDisclosure from './ThoughtsDisclosure.svelte';
  import EmptyState from './EmptyState.svelte';
  import ErrorCard from './ErrorCard.svelte';
  import type { AnswerStage, Turn } from '$lib/chat/types.js';

  interface Props {
    turns: Turn[];
    streaming: boolean;
    stage: AnswerStage | null;
    thinkingBuffer: string;
    answerBuffer: string;
    currentTurnId: string | null;
    error: { kind: string; message: string } | null;
    pinnedToBottom: boolean;
    oncopy: (content: string) => void;
    onregenerate: (turnId: string) => void;
    onfeedback: (messageId: string, next: 'up' | 'down') => void;
    onretry: () => void;
    onunpin: () => void;
    onjumptolatest: () => void;
  }

  let {
    turns,
    streaming,
    stage,
    thinkingBuffer,
    answerBuffer,
    currentTurnId,
    error,
    pinnedToBottom,
    oncopy,
    onregenerate,
    onfeedback,
    onretry,
    onunpin,
    onjumptolatest
  }: Props = $props();

  let viewportRef = $state<HTMLElement | null>(null);
  const isEmpty = $derived(turns.length === 0 && !streaming);
  // Treat within Npx of bottom as "at bottom" for autoscroll pin/unpin.
  const PIN_THRESHOLD_PX = 48;

  function scrollToBottom(): void {
    if (!viewportRef) return;
    viewportRef.scrollTop = viewportRef.scrollHeight;
  }

  // Re-pin follows content growth (streaming deltas, new turns) — but only
  // while pinned; an unpinned user stays put even as new content arrives.
  $effect(() => {
    void turns.length;
    void answerBuffer;
    void thinkingBuffer;
    if (pinnedToBottom) {
      tick().then(scrollToBottom);
    }
  });

  function handleScroll(e: Event): void {
    const el = e.currentTarget as HTMLElement;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom > PIN_THRESHOLD_PX && pinnedToBottom) {
      onunpin();
    }
  }

  function handleJumpToLatest(): void {
    onjumptolatest();
    tick().then(scrollToBottom);
  }

  // The ScrollArea wrapper doesn't forward `onscroll` to the Viewport (only
  // `viewportRef` is exposed via bind:), so wire the listener imperatively.
  $effect(() => {
    const el = viewportRef;
    if (!el) return;
    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
  });
</script>

<div class="relative flex min-h-0 flex-1 flex-col">
  {#if isEmpty}
    <EmptyState />
  {:else}
    <ScrollArea bind:viewportRef class="min-h-0 flex-1">
      <div class="flex flex-col pb-2" role="log" aria-label="Chat transcript">
        {#each turns as turn (turn.turn_id)}
          <UserMessage content={turn.user.content} />

          {#if turn.versions.length > 0}
            <AssistantMessage
              versions={turn.versions}
              {oncopy}
              onregenerate={() => onregenerate(turn.turn_id)}
              {onfeedback}
              regenerateDisabled={streaming}
            />
          {/if}

          {#if streaming && currentTurnId === turn.turn_id}
            <StatusLine {stage} />
            {#if thinkingBuffer.length > 0}
              <ThoughtsDisclosure thinking={thinkingBuffer} />
            {/if}
            {#if answerBuffer.length > 0}
              <AssistantMessage
                versions={[
                  {
                    id: `${turn.turn_id}-streaming`,
                    notebook_id: turn.user.notebook_id,
                    turn_id: turn.turn_id,
                    role: 'assistant',
                    content: answerBuffer,
                    citations: null,
                    feedback: null,
                    tokens_used: null,
                    created_at: turn.user.created_at
                  }
                ]}
                oncopy={() => {}}
                onregenerate={() => {}}
                onfeedback={() => {}}
                regenerateDisabled={true}
                highlightCode={false}
              />
            {/if}
          {/if}
          {#if error && currentTurnId === turn.turn_id}
            <ErrorCard {error} {onretry} />
          {/if}
        {/each}
      </div>
    </ScrollArea>

    {#if !pinnedToBottom}
      <button
        type="button"
        onclick={handleJumpToLatest}
        aria-label="Jump to latest message"
        class="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-popover px-3 py-1.5 text-xs font-medium text-popover-foreground shadow-md transition-opacity hover:opacity-90"
      >
        <ArrowDown class="size-3" strokeWidth={2.5} />
        Jump to latest
      </button>
    {/if}
  {/if}
</div>
