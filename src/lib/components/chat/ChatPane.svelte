<!-- Root chat panel for a notebook. Hydrates on mount/notebook-change; lays out
     the transcript (fills height) above a fixed composer (AC12, AC17). -->
<script lang="ts">
  import ChatTranscript from './ChatTranscript.svelte';
  import ChatComposer from './ChatComposer.svelte';
  import {
    chatStore,
    hydrate,
    send,
    stop,
    regenerate,
    setFeedback,
    copyMessage,
    jumpToLatest,
    unpin
  } from '$lib/chat/chat-state.svelte.js';
  import { hydrate as hydrateNotes } from '$lib/notes/notes-state.svelte.js';

  interface Props {
    notebookId: string;
  }

  let { notebookId }: Props = $props();

  $effect(() => {
    void hydrate(notebookId);
    // Save-button state (MessageActions) needs saved-state up front, not just
    // when the Notes tab is opened.
    void hydrateNotes(notebookId);
  });

  const turns = $derived(chatStore.turns(notebookId));
  const streaming = $derived(chatStore.streaming(notebookId));
  const stage = $derived(chatStore.stage(notebookId));
  const thinkingBuffer = $derived(chatStore.thinkingBuffer(notebookId));
  const answerBuffer = $derived(chatStore.answerBuffer(notebookId));
  const pendingCitations = $derived(chatStore.pendingCitations(notebookId));
  const currentTurnId = $derived(chatStore.currentTurnId(notebookId));
  const ungroundedTurnId = $derived(chatStore.ungroundedTurnId(notebookId));
  const error = $derived(chatStore.error(notebookId));
  const reindexing = $derived(chatStore.reindexing(notebookId));
  const pinnedToBottom = $derived(chatStore.pinnedToBottom(notebookId));

  function handleSend(question: string): void {
    void send(notebookId, question);
  }

  function handleRetry(): void {
    // Re-ask under the SAME turn_id (no duplicate user row), not a new `send`.
    if (currentTurnId) void regenerate(notebookId, currentTurnId);
  }

  // Height of the floating composer overlay (fade + composer). Fed back to the
  // transcript as a bottom inset so the last turn can rest just above it and
  // never hides behind it, while mid-scroll content dissolves under the fade.
  let overlayHeight = $state(0);
</script>

<div class="relative flex min-h-0 flex-1 flex-col overflow-hidden">
  <ChatTranscript
    {notebookId}
    {turns}
    {streaming}
    {stage}
    {thinkingBuffer}
    {answerBuffer}
    {pendingCitations}
    {currentTurnId}
    {ungroundedTurnId}
    {error}
    {reindexing}
    {pinnedToBottom}
    bottomInset={overlayHeight}
    oncopy={(content) => void copyMessage(content)}
    onregenerate={(turnId) => void regenerate(notebookId, turnId)}
    onfeedback={(messageId, next) => void setFeedback(notebookId, messageId, next)}
    onretry={handleRetry}
    onunpin={() => unpin(notebookId)}
    onjumptolatest={() => jumpToLatest(notebookId)}
  />

  <!-- Floating composer overlay: the transcript fills the full height behind it;
       a transparent→canvas fade dissolves content as it scrolls under, and the
       composer floats on the canvas backing below the fade. pointer-events-none
       on the wrapper lets scroll/clicks pass through the fade to the transcript;
       the composer itself re-enables them. -->
  <div
    bind:clientHeight={overlayHeight}
    class="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex flex-col"
  >
    <div
      aria-hidden="true"
      class="h-10 bg-gradient-to-b from-transparent to-[var(--app-canvas)]"
    ></div>
    <div class="pointer-events-auto bg-[var(--app-canvas)]">
      <ChatComposer {streaming} onsend={handleSend} onstop={() => void stop(notebookId)} />
    </div>
  </div>
</div>
