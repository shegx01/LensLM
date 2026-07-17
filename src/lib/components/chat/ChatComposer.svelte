<!-- Auto-growing multiline composer. Enter sends, Shift+Enter inserts a newline;
     empty/whitespace-only input cannot send (AC9). While streaming, Send morphs
     into a Stop button wired to `stop()` (AC10). Native <textarea> — no
     Textarea primitive exists in ui/. Design ref: round-input.png (pill shape,
     green send button). -->
<script lang="ts">
  import ArrowUp from '@lucide/svelte/icons/arrow-up';
  import Square from '@lucide/svelte/icons/square';
  import { cn } from '$lib/utils.js';
  import { popIn } from '$lib/motion/index.js';

  interface Props {
    streaming: boolean;
    onsend: (question: string) => void;
    onstop: () => void;
  }

  let { streaming, onsend, onstop }: Props = $props();

  let value = $state('');
  let textareaRef = $state<HTMLTextAreaElement | null>(null);

  const canSend = $derived(value.trim().length > 0);
  const MAX_HEIGHT_PX = 200;

  function autoGrow(el: HTMLTextAreaElement): void {
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }

  function handleInput(e: Event): void {
    autoGrow(e.currentTarget as HTMLTextAreaElement);
  }

  function submit(): void {
    if (!canSend) return;
    onsend(value.trim());
    value = '';
    if (textareaRef) {
      textareaRef.style.height = 'auto';
    }
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }
</script>

<div class="shrink-0 px-4 pb-4 pt-2">
  <div class="composer flex items-end gap-2 rounded-3xl bg-card px-2 py-2">
    <textarea
      bind:this={textareaRef}
      bind:value
      rows="1"
      placeholder="Ask anything about your sources…"
      aria-label="Ask a question about your sources"
      disabled={streaming}
      oninput={handleInput}
      onkeydown={handleKeydown}
      class="min-h-[36px] flex-1 resize-none border-0 bg-transparent px-2 py-1.5 text-sm text-foreground placeholder:text-muted-foreground/60 outline-none disabled:opacity-60"
      style={`max-height: ${MAX_HEIGHT_PX}px`}
    ></textarea>

    {#if streaming}
      <button
        type="button"
        in:popIn
        aria-label="Stop generating"
        onclick={onstop}
        class="flex size-9 shrink-0 items-center justify-center rounded-full bg-foreground text-background transition-[transform,opacity] hover:opacity-90 active:scale-[0.94] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Square class="size-3.5" fill="currentColor" strokeWidth={0} />
      </button>
    {:else}
      <button
        type="button"
        in:popIn
        aria-label="Send question"
        disabled={!canSend}
        onclick={submit}
        class={cn(
          'flex size-9 shrink-0 items-center justify-center rounded-full transition-[transform,opacity]',
          'bg-primary text-primary-foreground hover:opacity-90 active:scale-[0.94] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
          'disabled:opacity-40 disabled:active:scale-100'
        )}
      >
        <ArrowUp class="size-4" strokeWidth={2.5} />
      </button>
    {/if}
  </div>
</div>

<style>
  /* Rests as a hairline-outlined card; on focus it lifts a touch and the brand
     ring blooms. Transform/box-shadow only (never `all`), so it stays smooth. */
  .composer {
    box-shadow:
      0 0 0 1px var(--border),
      0 1px 2px oklch(0.2 0.02 293 / 0.04);
    transform: translateY(0);
    transition:
      box-shadow 0.22s var(--ease-out, ease),
      transform 0.22s var(--ease-out, ease);
  }
  .composer:focus-within {
    transform: translateY(calc(-1px * var(--rail-motion, 1)));
    box-shadow:
      0 0 0 1px color-mix(in oklch, var(--ring) 45%, transparent),
      0 0 0 4px color-mix(in oklch, var(--ring) 16%, transparent),
      0 8px 22px oklch(0.2 0.02 293 / 0.08);
  }
</style>
