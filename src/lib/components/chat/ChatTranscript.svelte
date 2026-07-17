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
  import ChatScrubber from './ChatScrubber.svelte';
  import StatusLine from './StatusLine.svelte';
  import ThoughtsDisclosure from './ThoughtsDisclosure.svelte';
  import EmptyState from './EmptyState.svelte';
  import ErrorCard from './ErrorCard.svelte';
  import { fade } from 'svelte/transition';
  import { prefersReducedMotion } from '$lib/motion/index.js';
  import type { AnswerStage, Turn } from '$lib/chat/types.js';

  interface Props {
    notebookId: string;
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
    /** Px reserved at the bottom for the floating composer overlay, so the last
     * turn rests above it and the jump pill clears it. Default 0. */
    bottomInset?: number;
  }

  let {
    notebookId,
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
    onjumptolatest,
    bottomInset = 0
  }: Props = $props();

  let viewportRef = $state<HTMLElement | null>(null);
  let activeTurnId = $state<string | null>(null);
  const isEmpty = $derived(turns.length === 0 && !streaming);
  // Treat within Npx of bottom as "at bottom" for autoscroll pin/unpin.
  const PIN_THRESHOLD_PX = 48;
  // A turn counts as "in view" for the scrubber once its top is within this many
  // px below the viewport top (i.e. it's the turn you're currently reading).
  const ACTIVE_TOP_BAND_PX = 96;

  function scrollToBottom(smooth = false): void {
    if (!viewportRef) return;
    // scrollTo isn't implemented in happy-dom (tests) — fall back to scrollTop.
    if (typeof viewportRef.scrollTo === 'function') {
      viewportRef.scrollTo({
        top: viewportRef.scrollHeight,
        behavior: smooth && !prefersReducedMotion() ? 'smooth' : 'auto'
      });
    } else {
      viewportRef.scrollTop = viewportRef.scrollHeight;
    }
  }

  /** Mark the last turn whose top has scrolled to/above the reading band. */
  function updateActiveTurn(): void {
    const vp = viewportRef;
    if (!vp) return;
    const vpTop = vp.getBoundingClientRect().top;
    let active: string | null = null;
    for (const el of vp.querySelectorAll<HTMLElement>('[data-turn-id]')) {
      if (el.getBoundingClientRect().top - vpTop <= ACTIVE_TOP_BAND_PX) {
        active = el.dataset.turnId ?? active;
      } else {
        break;
      }
    }
    activeTurnId = active ?? turns[0]?.turn_id ?? null;
  }

  function scrollToTurn(turnId: string): void {
    // Jumping to the last turn lands at the bottom and re-pins (like the pill),
    // rather than top-aligning it and flashing the "Jump to latest" affordance.
    if (turns[turns.length - 1]?.turn_id === turnId) {
      handleJumpToLatest();
      return;
    }
    const el = viewportRef?.querySelector<HTMLElement>(`[data-turn-id="${CSS.escape(turnId)}"]`);
    // scrollIntoView fires the scroll handler, which manages pin/unpin + active turn.
    el?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  // Re-pin follows content growth (streaming deltas, new turns) — but only
  // while pinned; an unpinned user stays put even as new content arrives.
  // Smooth-scroll on discrete events (a new turn arrives, the status line
  // appears) but stay instant for token deltas so the pin can't lag behind a
  // fast stream. First placement after mount is always instant — no animated
  // scroll when a notebook opens.
  // Sentinels (not prop reads) — the first effect run seeds them, and `hasSettled`
  // keeps that first placement instant regardless.
  let prevTurnsLen = -1;
  let prevStage: AnswerStage | null = null;
  let hasSettled = false;
  $effect(() => {
    const turnsLen = turns.length;
    const curStage = stage;
    void answerBuffer;
    void thinkingBuffer;
    const structural = turnsLen !== prevTurnsLen || (curStage !== null && prevStage === null);
    prevTurnsLen = turnsLen;
    prevStage = curStage;
    if (pinnedToBottom) {
      tick().then(() => scrollToBottom(hasSettled && structural));
    }
    tick().then(() => {
      updateActiveTurn();
      hasSettled = true;
    });
  });

  function handleScroll(e: Event): void {
    const el = e.currentTarget as HTMLElement;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom > PIN_THRESHOLD_PX && pinnedToBottom) {
      onunpin();
    }
    updateActiveTurn();
  }

  function handleJumpToLatest(): void {
    onjumptolatest();
    tick().then(() => scrollToBottom(true));
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
    <!-- Scroll-edge fade: content dissolves into the canvas as it scrolls up
         toward the floating top bar, instead of meeting a hard cut. -->
    <div
      class="pointer-events-none absolute inset-x-0 top-0 z-10 h-6 bg-gradient-to-b from-[var(--app-canvas)] to-transparent"
      aria-hidden="true"
    ></div>
    <ScrollArea bind:viewportRef scrollbarYClasses="hidden" class="min-h-0 flex-1">
      <!-- Optical gutter: message px-4 = 16px each side, then pr-2 adds 8px on the
           right only → 24px right (clears the right-edge scrubber lane) vs 16px
           left. The left rail floats with an 8px margin, so 16px + 8px reads the
           same as the flush right side's 24px — content sits centered between the
           two rails. Composer + EmptyState mirror this pl-4/pr-6 inset. -->
      <div
        class="flex flex-col pr-2 pb-2"
        role="log"
        aria-label="Chat transcript"
        style:padding-bottom={`${bottomInset + 8}px`}
      >
        {#each turns as turn (turn.turn_id)}
          <div data-turn-id={turn.turn_id}>
            <UserMessage content={turn.user.content} />

            {#if turn.versions.length > 0}
              <AssistantMessage
                {notebookId}
                versions={turn.versions}
                {oncopy}
                onregenerate={() => onregenerate(turn.turn_id)}
                {onfeedback}
                regenerateDisabled={streaming}
              />
            {/if}
          </div>

          {#if streaming && currentTurnId === turn.turn_id}
            <StatusLine {stage} />
            {#if thinkingBuffer.length > 0}
              <ThoughtsDisclosure thinking={thinkingBuffer} />
            {/if}
            {#if answerBuffer.length > 0}
              <AssistantMessage
                {notebookId}
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
                finalized={false}
              />
            {/if}
          {/if}
          {#if error && currentTurnId === turn.turn_id}
            <ErrorCard {error} {onretry} />
          {/if}
        {/each}
      </div>
    </ScrollArea>

    <ChatScrubber {turns} {activeTurnId} onjump={scrollToTurn} />

    {#if !pinnedToBottom}
      <button
        type="button"
        in:fade={{ duration: 150 }}
        onclick={handleJumpToLatest}
        aria-label="Jump to latest message"
        style:bottom={`${bottomInset + 12}px`}
        class="absolute left-1/2 z-30 flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-popover px-3 py-1.5 text-xs font-medium text-popover-foreground shadow-[var(--shadow-bar)] transition-transform hover:-translate-y-px active:scale-[0.96]"
      >
        <ArrowDown class="size-3" strokeWidth={2.5} />
        Jump to latest
      </button>
    {/if}
  {/if}
</div>
