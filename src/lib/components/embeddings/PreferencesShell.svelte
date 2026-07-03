<!--
  PreferencesShell — in-place global Settings view (ADR B1, no overlay/portal).
  Renders in the main content region when `notebookStore.settingsOpen` is true;
  mounts <EmbeddingsSection mode="global" /> for the app-wide embedding default.
-->
<script lang="ts">
  import { notebookStore } from '$lib/notebooks/index.js';
  import { cn } from '$lib/utils.js';
  import EmbeddingsSection from './EmbeddingsSection.svelte';
  import GeneralSection from './GeneralSection.svelte';
  import IngestionSection from './IngestionSection.svelte';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import Cpu from '@lucide/svelte/icons/cpu';
  import Share2 from '@lucide/svelte/icons/share-2';
  import Download from '@lucide/svelte/icons/download';
  import HardDrive from '@lucide/svelte/icons/hard-drive';
  import Shield from '@lucide/svelte/icons/shield';
  import Keyboard from '@lucide/svelte/icons/keyboard';
  import Info from '@lucide/svelte/icons/info';

  const open = $derived(notebookStore.settingsOpen);

  type SectionId =
    | 'general'
    | 'ai'
    | 'embeddings'
    | 'ingestion'
    | 'storage'
    | 'privacy'
    | 'shortcuts'
    | 'about';

  // `embeddings` and `ingestion` are live; the rest are coming-soon stubs.
  const NAV: { id: SectionId; label: string; icon: typeof Settings2; stub: boolean }[] = [
    { id: 'general', label: 'General', icon: Settings2, stub: false },
    { id: 'ai', label: 'AI Model', icon: Cpu, stub: true },
    { id: 'embeddings', label: 'Embeddings', icon: Share2, stub: false },
    { id: 'ingestion', label: 'Ingestion', icon: Download, stub: false },
    { id: 'storage', label: 'Storage', icon: HardDrive, stub: true },
    { id: 'privacy', label: 'Privacy', icon: Shield, stub: true },
    { id: 'shortcuts', label: 'Shortcuts', icon: Keyboard, stub: true },
    { id: 'about', label: 'About', icon: Info, stub: true }
  ];

  let active = $state<SectionId>('embeddings');

  function close(): void {
    notebookStore.settingsOpen = false;
  }
</script>

{#if open}
  <section
    class="flex h-full min-h-0 flex-1 overflow-hidden bg-background"
    aria-label="Preferences"
    data-preferences-shell
  >
    <nav
      class="flex w-[220px] shrink-0 flex-col gap-px overflow-y-auto border-r border-border px-2.5 py-3.5"
      aria-label="Preferences sections"
    >
      <button
        type="button"
        onclick={close}
        class={cn(
          'mb-2.5 flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-left text-[0.78rem] font-semibold text-muted-foreground transition-colors',
          'hover:bg-muted/50 hover:text-foreground',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
        )}
      >
        <ArrowLeft class="size-3.5 shrink-0" aria-hidden="true" />
        <span>Back</span>
      </button>

      <p
        class="px-2.5 pb-2.5 text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70"
      >
        Preferences
      </p>

      {#each NAV as item (item.id)}
        {@const isActive = active === item.id}
        <button
          type="button"
          aria-current={isActive ? 'page' : undefined}
          aria-disabled={item.stub}
          onclick={() => {
            if (!item.stub) active = item.id;
          }}
          class={cn(
            'flex h-[34px] items-center gap-2.5 rounded-lg px-2.5 text-left text-[0.78rem] font-semibold transition-colors',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            item.id === 'about' && 'mt-auto',
            isActive
              ? 'bg-primary/10 text-primary'
              : item.stub
                ? 'cursor-default text-muted-foreground/40'
                : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
          )}
        >
          <item.icon class="size-3.5 shrink-0" aria-hidden="true" />
          <span class="flex-1 truncate">{item.label}</span>
          {#if item.stub}
            <span class="text-[0.6rem] font-medium text-muted-foreground/40">Soon</span>
          {/if}
        </button>
      {/each}
    </nav>

    <div class="flex-1 overflow-y-auto px-10 py-8">
      {#if active === 'general'}
        <GeneralSection />
      {:else if active === 'embeddings'}
        <EmbeddingsSection mode="global" />
      {:else if active === 'ingestion'}
        <IngestionSection />
      {/if}
    </div>
  </section>
{/if}
