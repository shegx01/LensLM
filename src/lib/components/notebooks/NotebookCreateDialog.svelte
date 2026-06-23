<script lang="ts">
  // NotebookCreateDialog — Step 4.6 of M3.
  //
  // Controlled by the parent via `open` + `onOpenChange`. The parent (AppShell /
  // NotebooksSidebar) owns the open state; the sidebar's "New notebook" button
  // triggers it.
  //
  // Focus mode defaults to "research" (plan §4.6 acceptance criteria).
  // Description field added per design brief even though the screenshot omits it.
  //
  // On success: calls createNotebookAction, closes dialog, and resets fields.
  // On failure: shows an inline error banner and keeps the dialog open.
  // Fields reset whenever the dialog re-opens (via $effect watching `open`).

  import type { Component } from 'svelte';
  import BookOpen from '@lucide/svelte/icons/book-open';
  import Code2 from '@lucide/svelte/icons/code-2';
  import FileText from '@lucide/svelte/icons/file-text';
  import NotebookIcon from '@lucide/svelte/icons/notebook';
  import CheckCircle2 from '@lucide/svelte/icons/check-circle-2';
  import AlertCircle from '@lucide/svelte/icons/alert-circle';

  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter
  } from '$lib/components/ui/dialog/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { cn } from '$lib/utils.js';
  import { createNotebookAction } from '$lib/notebooks/index.js';
  import type { FocusMode } from '$lib/notebooks/types.js';

  // ---------------------------------------------------------------------------
  // Props
  // ---------------------------------------------------------------------------

  let {
    open = false,
    onOpenChange
  }: {
    /** Whether the dialog is visible. Parent owns this state. */
    open: boolean;
    /** Called by the dialog when it wants to open or close itself. */
    onOpenChange: (v: boolean) => void;
  } = $props();

  // ---------------------------------------------------------------------------
  // Local form state
  // ---------------------------------------------------------------------------

  let name = $state('');
  let description = $state('');
  let focusMode = $state<FocusMode>('research');
  let submitting = $state(false);
  let formError = $state<string | null>(null);

  /** Ref to the name input for autofocus on open */
  let nameInputRef = $state<HTMLInputElement | null>(null);

  // Reset all fields when the dialog opens so re-open shows a blank form.
  $effect(() => {
    if (open) {
      name = '';
      description = '';
      focusMode = 'research';
      formError = null;
      submitting = false;
      // Autofocus the name field on open (microtask to allow DOM to settle)
      setTimeout(() => nameInputRef?.focus(), 0);
    }
  });

  // ---------------------------------------------------------------------------
  // Computed
  // ---------------------------------------------------------------------------

  const canSubmit = $derived(name.trim().length > 0 && !submitting);

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  async function handleCreate(): Promise<void> {
    if (!canSubmit) return;
    submitting = true;
    formError = null;
    try {
      await createNotebookAction(name.trim(), description.trim() || null, focusMode);
      onOpenChange(false);
    } catch (err) {
      formError = err instanceof Error ? err.message : String(err);
    } finally {
      submitting = false;
    }
  }

  function handleCancel(): void {
    onOpenChange(false);
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Enter' && canSubmit) {
      e.preventDefault();
      handleCreate();
    }
  }

  // ---------------------------------------------------------------------------
  // Focus mode card definitions
  // ---------------------------------------------------------------------------

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
</script>

<Dialog
  {open}
  onOpenChange={(v) => {
    if (!submitting) onOpenChange(v);
  }}
>
  <DialogContent class="w-full max-w-md gap-0 p-0 overflow-hidden" showCloseButton={true}>
    <!-- ── Header ─────────────────────────────────────────────────────────── -->
    <DialogHeader class="px-5 pt-5 pb-4 gap-1">
      <div class="flex items-center gap-2">
        <div
          class="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary"
        >
          <NotebookIcon class="size-4" />
        </div>
        <DialogTitle class="text-base font-semibold leading-tight">New notebook</DialogTitle>
      </div>
      <DialogDescription class="text-muted-foreground text-sm pl-9">
        Create a new knowledge space
      </DialogDescription>
    </DialogHeader>

    <!-- ── Body ──────────────────────────────────────────────────────────── -->
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <form
      class="flex flex-col gap-5 px-5 pb-2"
      onsubmit={(e) => {
        e.preventDefault();
        handleCreate();
      }}
      onkeydown={handleKeydown}
    >
      <!-- Name field -->
      <div class="flex flex-col gap-1.5">
        <label
          for="notebook-name"
          class="text-xs font-semibold tracking-widest text-muted-foreground uppercase"
        >
          Name
        </label>
        <Input
          id="notebook-name"
          bind:ref={nameInputRef}
          bind:value={name}
          placeholder="e.g. Q3 Earnings Research"
          autocomplete="off"
          aria-required="true"
          aria-invalid={name.length > 0 && name.trim().length === 0}
          disabled={submitting}
          class="h-9 text-sm"
        />
      </div>

      <!-- Description field -->
      <div class="flex flex-col gap-1.5">
        <label
          for="notebook-description"
          class="text-xs font-semibold tracking-widest text-muted-foreground uppercase"
        >
          Description
        </label>
        <textarea
          id="notebook-description"
          bind:value={description}
          placeholder="What's this notebook about? (optional)"
          disabled={submitting}
          rows={2}
          class={cn(
            'dark:bg-input/30 border-input focus-visible:border-ring focus-visible:ring-ring/50',
            'rounded-lg border bg-transparent px-2.5 py-2 text-sm resize-none',
            'w-full outline-none transition-colors focus-visible:ring-3',
            'placeholder:text-muted-foreground disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50'
          )}
        ></textarea>
      </div>

      <!-- Focus mode selection -->
      <div class="flex flex-col gap-2">
        <span class="text-xs font-semibold tracking-widest text-muted-foreground uppercase">
          Focus Mode
        </span>
        <div class="flex flex-col gap-2" role="radiogroup" aria-label="Focus mode">
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
                'flex items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-all',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
                'disabled:pointer-events-none disabled:opacity-50',
                selected
                  ? 'border-primary/40 bg-primary/8 text-foreground'
                  : 'border-border bg-transparent text-foreground hover:bg-muted/50'
              )}
            >
              <!-- Mode icon -->
              <div
                class={cn(
                  'flex size-8 shrink-0 items-center justify-center rounded-md',
                  selected ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
                )}
              >
                <mode.Icon class="size-4" />
              </div>

              <!-- Label + description -->
              <div class="min-w-0 flex-1">
                <p class={cn('text-sm font-medium leading-tight', selected && 'text-primary')}>
                  {mode.label}
                </p>
                <p class="text-muted-foreground mt-0.5 text-xs leading-tight">
                  {mode.description}
                </p>
              </div>

              <!-- Radio indicator -->
              <div
                class={cn(
                  'ml-auto flex size-4 shrink-0 items-center justify-center rounded-full border-2 transition-all',
                  selected
                    ? 'border-primary bg-primary'
                    : 'border-muted-foreground/40 bg-transparent'
                )}
                aria-hidden="true"
              >
                {#if selected}
                  <div class="size-1.5 rounded-full bg-primary-foreground"></div>
                {/if}
              </div>
            </button>
          {/each}
        </div>
      </div>

      <!-- Inline error banner -->
      {#if formError}
        <div
          role="alert"
          class="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
        >
          <AlertCircle class="size-4 shrink-0" />
          <span>{formError}</span>
        </div>
      {/if}
    </form>

    <!-- ── Footer ─────────────────────────────────────────────────────────── -->
    <DialogFooter class="px-5 py-4 mt-1">
      <Button variant="ghost" onclick={handleCancel} disabled={submitting} class="min-w-[72px]">
        Cancel
      </Button>
      <Button
        variant="default"
        disabled={!canSubmit}
        onclick={handleCreate}
        class="min-w-[148px] gap-1.5"
        aria-busy={submitting}
      >
        {#if submitting}
          <CheckCircle2 class="size-4 animate-spin" />
          Creating...
        {:else}
          Create notebook →
        {/if}
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
