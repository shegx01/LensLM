<!--
  NotebookSettingsSheet — per-notebook settings sheet (ADR B1). Mounts the shared
  <EmbeddingsSection mode="notebook" /> scoped to this notebook's embedding coordinate;
  toggled by `notebookStore.notebookSettingsOpen`.
-->
<script lang="ts">
  import { notebookStore } from '$lib/notebooks/index.js';
  import { Dialog, DialogContent } from '$lib/components/ui/dialog/index.js';
  import EmbeddingsSection from './EmbeddingsSection.svelte';

  const open = $derived(notebookStore.notebookSettingsOpen);
  const activeNotebook = $derived(notebookStore.activeNotebook);

  function close(): void {
    notebookStore.notebookSettingsOpen = false;
  }
</script>

<Dialog
  {open}
  onOpenChange={(v) => {
    if (!v) close();
  }}
>
  <DialogContent
    class="flex max-h-[calc(100vh-6rem)] w-[min(460px,calc(100vw-3rem))] max-w-none flex-col gap-0 overflow-hidden p-0 sm:max-w-none"
    aria-label={activeNotebook ? `${activeNotebook.title} settings` : 'Notebook settings'}
    data-notebook-settings-sheet
  >
    {#if activeNotebook}
      <div class="shrink-0 border-b border-border px-7 pt-6 pb-4">
        <p
          class="truncate text-[0.7rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70"
        >
          {activeNotebook.title}
        </p>
        <h1 class="mt-1 text-base font-bold tracking-[-0.3px] text-foreground">
          Notebook settings
        </h1>
      </div>
      <div class="flex-1 overflow-y-auto px-7 py-6">
        <EmbeddingsSection mode="notebook" notebookId={activeNotebook.id} onchange={close} />
      </div>
    {/if}
  </DialogContent>
</Dialog>
