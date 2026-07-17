<!--
  PreferencesShell — in-place global Settings view (ADR B1, no overlay/portal).
  Renders in the main content region when `notebookStore.settingsOpen` is true;
  mounts <EmbeddingsSection mode="global" /> for the app-wide embedding default.
-->
<script lang="ts">
  import { notebookStore } from '$lib/notebooks/index.js';
  import EmbeddingsSection from './EmbeddingsSection.svelte';
  import GeneralSection from './GeneralSection.svelte';
  import IngestionSection from './IngestionSection.svelte';
  import TtsConfigPanel from '../onboarding/TtsConfigPanel.svelte';
  import SettingsShell, { type NavItem } from './SettingsShell.svelte';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import Cpu from '@lucide/svelte/icons/cpu';
  import Share2 from '@lucide/svelte/icons/share-2';
  import Download from '@lucide/svelte/icons/download';
  import Volume2 from '@lucide/svelte/icons/volume-2';
  import HardDrive from '@lucide/svelte/icons/hard-drive';
  import Shield from '@lucide/svelte/icons/shield';
  import Keyboard from '@lucide/svelte/icons/keyboard';
  import Info from '@lucide/svelte/icons/info';

  const open = $derived(notebookStore.settingsOpen);

  // `embeddings`, `ingestion` and `text_to_speech` are live; the rest are coming-soon stubs.
  const NAV: NavItem[] = [
    { id: 'general', label: 'General', icon: Settings2, stub: false },
    { id: 'ai', label: 'AI Model', icon: Cpu, stub: true },
    { id: 'embeddings', label: 'Embeddings', icon: Share2, stub: false },
    { id: 'ingestion', label: 'Ingestion', icon: Download, stub: false },
    { id: 'text_to_speech', label: 'Text-to-Speech', icon: Volume2, stub: false },
    { id: 'storage', label: 'Storage', icon: HardDrive, stub: true },
    { id: 'privacy', label: 'Privacy', icon: Shield, stub: true },
    { id: 'shortcuts', label: 'Shortcuts', icon: Keyboard, stub: true },
    { id: 'about', label: 'About', icon: Info, stub: true }
  ];

  let active = $state<string>('embeddings');

  function close(): void {
    notebookStore.settingsOpen = false;
  }
</script>

{#if open}
  <section
    class="flex h-full min-h-0 flex-1 overflow-hidden"
    aria-label="Preferences"
    data-preferences-shell
  >
    <SettingsShell nav={NAV} bind:active onBack={close} label="Preferences">
      {#snippet content(active)}
        {#if active === 'general'}
          <GeneralSection />
        {:else if active === 'embeddings'}
          <EmbeddingsSection mode="global" />
        {:else if active === 'ingestion'}
          <IngestionSection />
        {:else if active === 'text_to_speech'}
          <TtsConfigPanel oncheck={async () => {}} oncollapse={() => {}} />
        {/if}
      {/snippet}
    </SettingsShell>
  </section>
{/if}
