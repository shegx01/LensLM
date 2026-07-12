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
  const currentTurnId = $derived(chatStore.currentTurnId(notebookId));
  const error = $derived(chatStore.error(notebookId));
  const pinnedToBottom = $derived(chatStore.pinnedToBottom(notebookId));

  function handleSend(question: string): void {
    void send(notebookId, question);
  }

  function handleRetry(): void {
    // Re-ask under the SAME turn_id (no duplicate user row), not a new `send`.
    if (currentTurnId) void regenerate(notebookId, currentTurnId);
  }
</script>

<div class="flex min-h-0 flex-1 flex-col overflow-hidden">
  <ChatTranscript
    {notebookId}
    {turns}
    {streaming}
    {stage}
    {thinkingBuffer}
    {answerBuffer}
    {currentTurnId}
    {error}
    {pinnedToBottom}
    oncopy={(content) => void copyMessage(content)}
    onregenerate={(turnId) => void regenerate(notebookId, turnId)}
    onfeedback={(messageId, next) => void setFeedback(notebookId, messageId, next)}
    onretry={handleRetry}
    onunpin={() => unpin(notebookId)}
    onjumptolatest={() => jumpToLatest(notebookId)}
  />

  <ChatComposer {streaming} onsend={handleSend} onstop={() => void stop(notebookId)} />
</div>
