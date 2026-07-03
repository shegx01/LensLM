<script lang="ts">
  // NotebookCreateDialog — Step 4.6 of M3.
  // Parent owns open state; fields reset on re-open. On success: closes + resets.
  // On failure: shows inline error and keeps dialog open so the user can retry.

  import type { Component } from 'svelte';
  import BookOpen from '@lucide/svelte/icons/book-open';
  import Code2 from '@lucide/svelte/icons/code-2';
  import FileText from '@lucide/svelte/icons/file-text';
  import NotebookIcon from '@lucide/svelte/icons/notebook';
  import X from '@lucide/svelte/icons/x';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Loader2 from '@lucide/svelte/icons/loader-2';
  import AlertCircle from '@lucide/svelte/icons/alert-circle';

  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription
  } from '$lib/components/ui/dialog/index.js';
  import { cn } from '$lib/utils.js';
  import { createNotebookAction, notebookStore } from '$lib/notebooks/index.js';
  import type { FocusMode } from '$lib/notebooks/types.js';

  let {
    open = false,
    onOpenChange
  }: {
    open: boolean;
    onOpenChange: (v: boolean) => void;
  } = $props();

  let name = $state('');
  let description = $state('');
  let focusMode = $state<FocusMode>('research');
  let submitting = $state(false);
  let formError = $state<string | null>(null);

  let nameInputRef = $state<HTMLInputElement | null>(null);

  $effect(() => {
    if (open) {
      name = '';
      description = '';
      focusMode = 'research';
      formError = null;
      submitting = false;
      setTimeout(() => nameInputRef?.focus(), 0);
    }
  });

  const canSubmit = $derived(name.trim().length > 0 && !submitting);

  async function handleCreate(): Promise<void> {
    if (!canSubmit) return;
    submitting = true;
    formError = null;
    try {
      const created = await createNotebookAction(
        name.trim(),
        description.trim() || null,
        focusMode
      );
      if (!created) {
        formError = notebookStore.error ?? 'Could not create the notebook.';
        return;
      }
      onOpenChange(false);
    } finally {
      submitting = false;
    }
  }

  function handleCancel(): void {
    onOpenChange(false);
  }

  function handleKeydown(e: KeyboardEvent): void {
    // Textarea Enter inserts a newline; don't submit from there.
    if ((e.target as HTMLElement)?.tagName === 'TEXTAREA') return;
    if (e.key === 'Enter' && canSubmit) {
      e.preventDefault();
      void handleCreate();
    }
  }

  type FocusModeOption = {
    id: FocusMode;
    label: string;
    description: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Icon: Component<any>;
  };

  const FOCUS_MODES: FocusModeOption[] = [
    {
      id: 'research',
      label: 'Research',
      description: 'PDFs, papers, web pages and research docs',
      Icon: BookOpen
    },
    {
      id: 'coding',
      label: 'Coding',
      description: 'Code repos, API docs, specs and READMEs',
      Icon: Code2
    },
    {
      id: 'notes',
      label: 'Notes',
      description: 'Freeform notes, meeting transcripts, ideas',
      Icon: FileText
    }
  ];

  const LABEL_CLASS = 'text-[10px] font-bold tracking-[0.08em] uppercase text-muted-foreground';

  const FIELD_CLASS = cn(
    'w-full rounded-[10px] border border-border bg-surface-raised text-foreground',
    'text-sm outline-none transition-colors',
    'placeholder:text-muted-foreground',
    'focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40',
    'disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50'
  );
</script>

<Dialog
  {open}
  onOpenChange={(v) => {
    if (!submitting) onOpenChange(v);
  }}
>
  <DialogContent
    class="w-full max-w-md gap-0 overflow-hidden rounded-xl border-border bg-card p-0"
    showCloseButton={false}
  >
    <div class="px-[26px] pt-6 pb-[26px]">
      <DialogHeader class="mb-[22px] flex-row items-center justify-between gap-2 space-y-0">
        <div class="flex items-center gap-2.5">
          <div
            class="flex size-[30px] shrink-0 items-center justify-center rounded-[9px] bg-muted text-muted-foreground"
            aria-hidden="true"
          >
            <NotebookIcon class="size-3.5" />
          </div>
          <div class="flex flex-col">
            <DialogTitle class="text-base font-bold tracking-[-0.3px] text-foreground">
              New notebook
            </DialogTitle>
            <DialogDescription class="text-[11px] text-muted-foreground">
              Create a new knowledge space
            </DialogDescription>
          </div>
        </div>

        <button
          type="button"
          aria-label="Close"
          onclick={handleCancel}
          disabled={submitting}
          class={cn(
            'flex size-7 shrink-0 items-center justify-center rounded-full',
            'bg-muted text-muted-foreground transition-opacity hover:opacity-65',
            'cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
            'disabled:pointer-events-none disabled:opacity-50'
          )}
        >
          <X class="size-3" strokeWidth={2.5} />
        </button>
      </DialogHeader>

      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <form
        onsubmit={(e) => {
          e.preventDefault();
          handleCreate();
        }}
        onkeydown={handleKeydown}
      >
        <div class="mb-[18px] flex flex-col gap-[7px]">
          <label for="notebook-name" class={LABEL_CLASS}>Name</label>
          <input
            id="notebook-name"
            type="text"
            bind:this={nameInputRef}
            bind:value={name}
            placeholder="e.g. Q3 Earnings Research"
            autocomplete="off"
            aria-required="true"
            aria-invalid={name.length > 0 && name.trim().length === 0}
            disabled={submitting}
            class={cn(FIELD_CLASS, 'h-11 px-3.5')}
          />
        </div>

        <div class="mb-[18px] flex flex-col gap-[7px]">
          <label for="notebook-description" class={LABEL_CLASS}>Description</label>
          <textarea
            id="notebook-description"
            bind:value={description}
            placeholder="What's this notebook about? (optional)"
            disabled={submitting}
            rows={2}
            class={cn(FIELD_CLASS, 'resize-none px-3.5 py-2.5 leading-snug')}
          ></textarea>
        </div>

        <div class="mb-[22px] flex flex-col gap-[9px]">
          <span class={LABEL_CLASS}>Focus mode</span>
          <div class="flex flex-col gap-[5px]" role="radiogroup" aria-label="Focus mode">
            {#each FOCUS_MODES as mode (mode.id)}
              {@const selected = focusMode === mode.id}
              <button
                type="button"
                role="radio"
                aria-checked={selected}
                disabled={submitting}
                onclick={() => {
                  focusMode = mode.id;
                }}
                class={cn(
                  'flex items-center gap-3 rounded-[10px] border px-[13px] py-[11px] text-left',
                  'cursor-pointer transition-all duration-150 outline-none',
                  'focus-visible:ring-2 focus-visible:ring-ring/50',
                  'disabled:pointer-events-none disabled:opacity-50',
                  selected
                    ? 'border-primary bg-primary/10'
                    : 'border-border bg-transparent hover:bg-muted/50'
                )}
              >
                <div
                  class={cn(
                    'flex size-[34px] shrink-0 items-center justify-center rounded-lg',
                    selected ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
                  )}
                  aria-hidden="true"
                >
                  <mode.Icon class="size-[15px]" strokeWidth={1.8} />
                </div>

                <div class="min-w-0 flex-1">
                  <p class="text-[13px] leading-tight font-bold text-foreground">
                    {mode.label}
                  </p>
                  <p class="mt-0.5 text-[11px] leading-tight text-muted-foreground">
                    {mode.description}
                  </p>
                </div>

                <div
                  class={cn(
                    'ml-auto flex size-4 shrink-0 items-center justify-center rounded-full border-[1.5px]',
                    'transition-colors',
                    selected ? 'border-primary' : 'border-muted-foreground/40'
                  )}
                  aria-hidden="true"
                >
                  {#if selected}
                    <div class="size-2 rounded-full bg-primary"></div>
                  {/if}
                </div>
              </button>
            {/each}
          </div>
        </div>

        {#if formError}
          <div
            role="alert"
            class={cn(
              'mb-[18px] flex items-center gap-2 rounded-[10px] px-[13px] py-2.5',
              'border border-destructive/30 bg-destructive/10 text-sm text-destructive'
            )}
          >
            <AlertCircle class="size-4 shrink-0" />
            <span>{formError}</span>
          </div>
        {/if}

        <div class="flex gap-2">
          <button
            type="button"
            onclick={handleCancel}
            disabled={submitting}
            class={cn(
              'flex h-[42px] flex-1 items-center justify-center rounded-[10px]',
              'bg-muted text-[13px] font-semibold text-muted-foreground',
              'cursor-pointer transition-opacity hover:opacity-70 outline-none',
              'focus-visible:ring-2 focus-visible:ring-ring/50',
              'disabled:pointer-events-none disabled:opacity-50'
            )}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!canSubmit}
            onclick={handleCreate}
            aria-busy={submitting}
            class={cn(
              'flex h-[42px] flex-[2] items-center justify-center gap-1.5 rounded-[10px]',
              'bg-primary text-[13px] font-semibold text-primary-foreground',
              'cursor-pointer transition-all hover:opacity-90 outline-none',
              'focus-visible:ring-2 focus-visible:ring-ring/50',
              'disabled:pointer-events-none disabled:opacity-50'
            )}
          >
            {#if submitting}
              <Loader2 class="size-3.5 animate-spin" />
              Creating...
            {:else}
              Create notebook
              <ArrowRight class="size-3.5" strokeWidth={2.5} />
            {/if}
          </button>
        </div>
      </form>
    </div>
  </DialogContent>
</Dialog>
