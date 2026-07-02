<!--
  PreferencesShell — the global Settings surface (plan Step 10, ADR B1).

  IN-PLACE Preferences view (matching the design's settings-inplace.png /
  Lens.dc.html `showSettingsPage`). NOT a floating modal: it renders inside the
  normal content region (right of the left notebook sidebar) like a route/view
  swap — no backdrop, no centered card, no portal. AppShell mounts this in the
  main+right grid span when `notebookStore.settingsOpen` is true, hiding the
  notebook top-bar / content / sources rail; the left notebook sidebar stays.

  Layout: a 220px Preferences nav column (a "← Back" affordance at top, then a
  "PREFERENCES" label, then nav items General · AI Model · Embeddings · Storage
  · Privacy · Shortcuts · About) and a section content pane to its right. Only
  EMBEDDINGS is live in 4b-B; every other section is a "coming soon" stub. The
  live Embeddings section mounts the shared <EmbeddingsSection mode="global" />
  so the app-wide default (config) is set here — new notebooks adopt it.

  "← Back" closes the view (sets `notebookStore.settingsOpen = false`), returning
  to the notebook view. Opened by the AccountFooter "Settings" button.

  Tokens only — light + dark + every accent ([[theming-light-dark-accent]]).
-->
<script lang="ts">
  import { notebookStore } from '$lib/notebooks/index.js';
  import { cn } from '$lib/utils.js';
  import EmbeddingsSection from './EmbeddingsSection.svelte';
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

  // Left-nav order matches the design (Lens.dc.html Preferences nav). `embeddings`
  // and `ingestion` are live (`stub: false`); the rest are coming-soon stubs.
  // `About` sits at the bottom in the design (pushed down by a spacer).
  const NAV: { id: SectionId; label: string; icon: typeof Settings2; stub: boolean }[] = [
    { id: 'general', label: 'General', icon: Settings2, stub: true },
    { id: 'ai', label: 'AI Model', icon: Cpu, stub: true },
    { id: 'embeddings', label: 'Embeddings', icon: Share2, stub: false },
    { id: 'ingestion', label: 'Ingestion', icon: Download, stub: false },
    { id: 'storage', label: 'Storage', icon: HardDrive, stub: true },
    { id: 'privacy', label: 'Privacy', icon: Shield, stub: true },
    { id: 'shortcuts', label: 'Shortcuts', icon: Keyboard, stub: true },
    { id: 'about', label: 'About', icon: Info, stub: true }
  ];

  // Open straight to the only live section (4b-B ships Embeddings only).
  let active = $state<SectionId>('embeddings');

  function close(): void {
    notebookStore.settingsOpen = false;
  }
</script>

{#if open}
  <!-- In-place Preferences view: fills the content region (no overlay). The
       outer row is a normal flex container; the top edge stays a Tauri drag
       region so the window can still be moved from the empty top band. -->
  <section
    class="flex h-full min-h-0 flex-1 overflow-hidden bg-background"
    aria-label="Preferences"
    data-preferences-shell
  >
    <!-- ── Preferences nav column ── -->
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

    <!-- ── Section content pane ── -->
    <div class="flex-1 overflow-y-auto px-10 py-8">
      {#if active === 'embeddings'}
        <EmbeddingsSection mode="global" />
      {:else if active === 'ingestion'}
        <IngestionSection />
      {/if}
    </div>
  </section>
{/if}
