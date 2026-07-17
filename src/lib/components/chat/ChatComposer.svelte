<!-- Two-row composer. Top: auto-growing textarea (Enter sends, Shift+Enter newline;
     empty/whitespace can't send, AC9). Bottom tool row: add-source (opens the same
     AddSourcesModal as the sources rail), a sources-scope chip (count + popover over
     the existing per-source `selected` state that retrieval already grounds on), a
     voice affordance (disabled — STT is a separate feature), and Send (morphs to Stop
     while streaming, AC10). bits-ui Popover/Tooltip are styled with inline Tailwind —
     scoped <style> classes don't reach their forwarded elements. -->
<script lang="ts">
  import ArrowUp from '@lucide/svelte/icons/arrow-up';
  import Square from '@lucide/svelte/icons/square';
  import Plus from '@lucide/svelte/icons/plus';
  import Layers from '@lucide/svelte/icons/layers';
  import Mic from '@lucide/svelte/icons/mic';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import Check from '@lucide/svelte/icons/check';
  import { Popover } from 'bits-ui';
  import { cn } from '$lib/utils.js';
  import { popIn } from '$lib/motion/index.js';
  import { sourcesStore, toggleSelected } from '$lib/sources/index.js';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';
  import AddSourcesModal from '$lib/components/sources/AddSourcesModal.svelte';

  interface Props {
    streaming: boolean;
    onsend: (question: string) => void;
    onstop: () => void;
  }

  let { streaming, onsend, onstop }: Props = $props();

  let value = $state('');
  let textareaRef = $state<HTMLTextAreaElement | null>(null);
  let addOpen = $state(false);

  const canSend = $derived(value.trim().length > 0);
  const MAX_HEIGHT_PX = 200;

  const totalCount = $derived(sourcesStore.sources.length);
  const selectedCount = $derived(sourcesStore.sources.filter((s) => s.selected === 1).length);
  const scopeLabel = $derived.by(() => {
    if (totalCount === 0) return 'No sources yet';
    if (selectedCount === 0) return 'No sources selected';
    return `Grounded in ${selectedCount} source${selectedCount === 1 ? '' : 's'}`;
  });

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

<div class="shrink-0 pt-2 pr-6 pb-4 pl-4">
  <TooltipProvider>
    <div class="composer flex flex-col gap-1.5 rounded-3xl bg-card px-3 pt-2.5 pb-2">
      <textarea
        bind:this={textareaRef}
        bind:value
        rows="1"
        placeholder="Ask anything about your sources…"
        aria-label="Ask a question about your sources"
        disabled={streaming}
        oninput={handleInput}
        onkeydown={handleKeydown}
        class="min-h-[28px] resize-none border-0 bg-transparent px-1 pt-1 text-sm text-foreground placeholder:text-muted-foreground/60 outline-none disabled:opacity-60"
        style={`max-height: ${MAX_HEIGHT_PX}px`}
      ></textarea>

      <div class="flex items-center gap-1.5">
        <!-- Add source — opens the same modal the sources rail uses. -->
        <button
          type="button"
          onclick={() => (addOpen = true)}
          aria-label="Add source"
          class="inline-flex h-[30px] items-center gap-1.5 rounded-full pr-[11px] pl-[9px] text-xs font-medium text-muted-foreground transition-[background,color,transform] hover:bg-muted hover:text-foreground active:scale-[0.96] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <Plus class="size-[15px]" strokeWidth={2.25} />
          Add source
        </button>

        <!-- Sources-scope chip — count + popover over the existing `selected` state. -->
        <Popover.Root>
          <Popover.Trigger
            aria-label="Sources used for this question"
            class="inline-flex h-[30px] items-center gap-1.5 rounded-full bg-primary/10 px-[9px] text-xs font-semibold text-primary transition-[background,transform] hover:bg-primary/[0.16] active:scale-[0.96] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Layers class="size-[14px]" strokeWidth={2} />
            {scopeLabel}
            <ChevronDown class="size-3 opacity-70" strokeWidth={2.5} />
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Content
              side="top"
              align="start"
              sideOffset={8}
              class="z-50 w-[288px] rounded-xl border border-border bg-popover p-1.5 text-popover-foreground shadow-[var(--shadow-rail)] data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
            >
              <p
                class="px-2 pt-1 pb-1.5 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase"
              >
                Sources in this answer
              </p>
              {#if totalCount === 0}
                <div class="px-2 py-3 text-center">
                  <p class="text-xs text-muted-foreground">No sources yet.</p>
                  <button
                    type="button"
                    onclick={() => (addOpen = true)}
                    class="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary-foreground transition-transform active:scale-[0.97]"
                  >
                    <Plus class="size-3.5" strokeWidth={2.5} />
                    Add a source
                  </button>
                </div>
              {:else}
                <div class="no-scrollbar max-h-[264px] overflow-y-auto">
                  {#each sourcesStore.sources as source (source.id)}
                    <button
                      type="button"
                      onclick={() => void toggleSelected(source.id)}
                      aria-pressed={source.selected === 1}
                      class="flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-muted"
                    >
                      <span
                        class={cn(
                          'flex size-4 shrink-0 items-center justify-center rounded-[4px] border transition-all',
                          source.selected === 1
                            ? 'border-primary bg-primary'
                            : 'border-border hover:border-primary/60'
                        )}
                      >
                        {#if source.selected === 1}
                          <Check class="size-[9px] text-primary-foreground" strokeWidth={3} />
                        {/if}
                      </span>
                      <span class="truncate text-[13px] text-foreground">{source.title}</span>
                    </button>
                  {/each}
                </div>
              {/if}
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>

        <div class="ml-auto flex items-center gap-1.5">
          <!-- Voice — honestly disabled until on-device STT is scoped. -->
          <Tooltip>
            <TooltipTrigger>
              <button
                type="button"
                disabled
                aria-label="Dictate question (coming soon)"
                class="grid size-8 place-items-center rounded-full text-muted-foreground opacity-50"
              >
                <Mic class="size-[17px]" strokeWidth={2} />
              </button>
            </TooltipTrigger>
            <TooltipContent side="top">Voice input — coming soon</TooltipContent>
          </Tooltip>

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
    </div>
  </TooltipProvider>
</div>

<AddSourcesModal open={addOpen} onclose={() => (addOpen = false)} />

<style>
  /* Floats on a soft layered shadow (same family as the header pill / rail) —
     elevation, not a border. On focus it lifts a touch and the brand ring blooms
     over the float. Transform/box-shadow only (never `all`), so it stays smooth. */
  .composer {
    box-shadow: var(--shadow-bar);
    transform: translateY(0);
    transition:
      box-shadow 0.22s var(--ease-out, ease),
      transform 0.22s var(--ease-out, ease);
  }
  .composer:focus-within {
    transform: translateY(calc(-2px * var(--rail-motion, 1)));
    box-shadow:
      0 0 0 1px color-mix(in oklch, var(--ring) 40%, transparent),
      0 0 0 4px color-mix(in oklch, var(--ring) 15%, transparent),
      var(--shadow-bar),
      0 14px 32px oklch(0.2 0.02 293 / 0.1);
  }
</style>
