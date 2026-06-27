<!--
  NotebookSettingsSheet — the per-notebook "{notebook} settings" surface
  (plan Step 11, ADR decision B1). A sheet opened from the NotebookTopBar pill's
  settings gear (now enabled). Mounts the SAME shared <EmbeddingsSection /> as the
  global Preferences shell, but in "notebook mode": it reads + writes THIS
  notebook's coordinate (get/set_notebook_embedding_model), shows the re-embed
  confirm dialog on a change to an indexed coordinate, and surfaces re-embed
  progress (amber-pulse). Disabled-while-in-flight is handled inside the section.

  Built as a shadcn <Dialog> (the proven overlay mount); toggled by
  `notebookStore.notebookSettingsOpen`. Guards against opening with no active
  notebook (the gear is hidden in that case, but the guard is defensive).

  Tokens only — light + dark + every accent ([[theming-light-dark-accent]]).
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
      <!-- Sheet header — the notebook name scopes the surface (vs the global default). -->
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
