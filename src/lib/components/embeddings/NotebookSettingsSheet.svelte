<!--
  NotebookSettingsSheet — per-notebook settings sheet (ADR B1). Hosts the shared
  <SettingsShell> (Dialog mode: no Back, no label — the sheet owns its own header)
  with live Embeddings + Retrieval sections; toggled by `notebookStore.notebookSettingsOpen`.
-->
<script lang="ts">
  import { notebookStore } from '$lib/notebooks/index.js';
  import { Dialog, DialogContent } from '$lib/components/ui/dialog/index.js';
  import SettingsShell, { type NavItem } from './SettingsShell.svelte';
  import EmbeddingsSection from './EmbeddingsSection.svelte';
  import RetrievalSection from './RetrievalSection.svelte';
  import Share2 from '@lucide/svelte/icons/share-2';
  import Network from '@lucide/svelte/icons/network';

  const open = $derived(notebookStore.notebookSettingsOpen);
  const activeNotebook = $derived(notebookStore.activeNotebook);

  const NAV: NavItem[] = [
    { id: 'embeddings', label: 'Embeddings', icon: Share2, stub: false },
    { id: 'retrieval', label: 'Retrieval', icon: Network, stub: false }
  ];

  let active = $state<string>('embeddings');

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
    class="flex max-h-[calc(100vh-6rem)] min-h-[420px] w-[min(760px,calc(100vw-3rem))] max-w-none flex-col gap-0 overflow-hidden p-0 sm:max-w-none"
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
      <div class="flex min-h-0 flex-1 overflow-hidden">
        <SettingsShell nav={NAV} bind:active>
          {#snippet content(active)}
            {#if active === 'embeddings'}
              <EmbeddingsSection mode="notebook" notebookId={activeNotebook.id} onchange={close} />
            {:else if active === 'retrieval'}
              <RetrievalSection notebookId={activeNotebook.id} />
            {/if}
          {/snippet}
        </SettingsShell>
      </div>
    {/if}
  </DialogContent>
</Dialog>
