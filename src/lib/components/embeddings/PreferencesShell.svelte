<!--
  PreferencesShell — the global Settings surface (plan Step 10, ADR B1).

  A centered Preferences modal (matching the design's settings-page.png) with a
  left-nav (PREFERENCES: General · AI Model · Embeddings · Storage · Privacy ·
  Shortcuts · About). Only EMBEDDINGS is live in 4b-B; every other section is a
  "coming soon" stub. The live Embeddings section mounts the shared
  <EmbeddingsSection mode="global" /> so the app-wide default (config) is set
  here — new notebooks adopt it.

  Built as a shadcn <Dialog> (same proven overlay mount as the dev inspector +
  TrashView) so Escape / focus-trap / portal / aria-modal come free. Opened by
  the now-enabled AccountFooter "Settings" button via `notebookStore.settingsOpen`.

  Tokens only — light + dark + every accent ([[theming-light-dark-accent]]).
-->
<script lang="ts">
  import { notebookStore } from '$lib/notebooks/index.js';
  import { Dialog, DialogContent } from '$lib/components/ui/dialog/index.js';
  import { cn } from '$lib/utils.js';
  import EmbeddingsSection from './EmbeddingsSection.svelte';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import Cpu from '@lucide/svelte/icons/cpu';
  import Share2 from '@lucide/svelte/icons/share-2';
  import HardDrive from '@lucide/svelte/icons/hard-drive';
  import Shield from '@lucide/svelte/icons/shield';
  import Keyboard from '@lucide/svelte/icons/keyboard';
  import Info from '@lucide/svelte/icons/info';

  const open = $derived(notebookStore.settingsOpen);

  type SectionId = 'general' | 'ai' | 'embeddings' | 'storage' | 'privacy' | 'shortcuts' | 'about';

  // Left-nav order matches the design (Lens.dc.html Preferences nav). Only
  // `embeddings` is live (`stub: false`); the rest are coming-soon stubs.
  const NAV: { id: SectionId; label: string; icon: typeof Settings2; stub: boolean }[] = [
    { id: 'general', label: 'General', icon: Settings2, stub: true },
    { id: 'ai', label: 'AI Model', icon: Cpu, stub: true },
    { id: 'embeddings', label: 'Embeddings', icon: Share2, stub: false },
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

<Dialog
  {open}
  onOpenChange={(v) => {
    if (!v) close();
  }}
>
  <DialogContent
    class="flex h-[min(560px,calc(100vh-6rem))] w-[min(840px,calc(100vw-4rem))] max-w-none flex-row gap-0 overflow-hidden p-0 sm:max-w-none"
    aria-label="Preferences"
    data-preferences-shell
  >
    <!-- ── Left-nav ── -->
    <nav
      class="flex w-[220px] shrink-0 flex-col gap-px overflow-y-auto border-r border-border bg-card/40 px-2.5 py-4"
      aria-label="Preferences sections"
    >
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

    <!-- ── Section content ── -->
    <div class="flex-1 overflow-y-auto px-10 py-8">
      {#if active === 'embeddings'}
        <EmbeddingsSection mode="global" />
      {/if}
    </div>
  </DialogContent>
</Dialog>
