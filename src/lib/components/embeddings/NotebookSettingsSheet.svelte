<!--
  NotebookSettingsSheet — per-notebook Embeddings + Retrieval, hosted in <SettingsShell>.
  The header owns Back, so the shell is given neither Back nor label.
-->
<script lang="ts">
  import { notebookStore } from '$lib/notebooks/index.js';
  import SettingsShell, { type NavItem } from './SettingsShell.svelte';
  import EmbeddingsSection from './EmbeddingsSection.svelte';
  import RetrievalSection from './RetrievalSection.svelte';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
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

  function onKeydown(e: KeyboardEvent): void {
    if (e.key !== 'Escape' || !open) return;
    // A nested overlay (e.g. the re-embed confirm dialog) preventDefaults its own Escape;
    // don't also tear down the whole settings surface in that case.
    if (e.defaultPrevented) return;
    close();
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#if open && activeNotebook}
  <section
    class="flex h-full min-h-0 flex-1 flex-col overflow-hidden bg-background"
    aria-label={`${activeNotebook.title} settings`}
    data-notebook-settings-sheet
  >
    <div class="flex shrink-0 items-center gap-3 border-b border-border/70 px-6 pt-5 pb-4">
      <button
        type="button"
        onclick={close}
        class="flex size-9 shrink-0 items-center justify-center rounded-[11px] text-muted-foreground transition-[color,background-color,transform] duration-150 hover:bg-foreground/[0.05] hover:text-foreground active:scale-[0.96] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label="Back to notebook"
      >
        <ArrowLeft class="size-[18px]" strokeWidth={2} />
      </button>
      <div class="min-w-0">
        <p
          class="truncate text-[0.62rem] font-bold uppercase tracking-[0.1em] text-muted-foreground/60"
        >
          {activeNotebook.title}
        </p>
        <h1 class="mt-0.5 text-[0.95rem] font-bold tracking-[-0.3px] text-balance text-foreground">
          Notebook settings
        </h1>
      </div>
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
  </section>
{/if}
