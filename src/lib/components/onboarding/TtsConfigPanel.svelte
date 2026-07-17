<script lang="ts">
  import { cn } from '$lib/utils.js';
  import type { TtsEngineCatalogEntry } from '$lib/onboarding/system-check.js';
  import LocalTtsForm from './LocalTtsForm.svelte';
  import CloudTtsForm from './CloudTtsForm.svelte';

  // Parent-owned so a Cloud key save (CloudTtsForm.refreshCatalog) can re-derive
  // engine availability for both halves; each child fetches it once on mount.
  let catalog = $state<TtsEngineCatalogEntry[]>([]);

  type TtsTab = 'local' | 'cloud';
  let activeTab = $state<TtsTab>('local');
</script>

<div class="pt-3">
  <div
    class="bg-muted flex w-full items-center rounded-lg p-0.5"
    role="tablist"
    aria-label="Text-to-speech provider type"
  >
    <button
      role="tab"
      aria-selected={activeTab === 'local'}
      aria-controls="tts-panel-local"
      id="tts-tab-local"
      class={cn(
        'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        activeTab === 'local'
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
      onclick={() => (activeTab = 'local')}
    >
      Local
    </button>
    <button
      role="tab"
      aria-selected={activeTab === 'cloud'}
      aria-controls="tts-panel-cloud"
      id="tts-tab-cloud"
      class={cn(
        'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        activeTab === 'cloud'
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
      onclick={() => (activeTab = 'cloud')}
    >
      Cloud
    </button>
  </div>

  <!-- Both forms stay mounted; visibility toggles via the `hidden` class so a
       tab switch never drops typed-but-unsaved state or in-flight downloads. -->
  <LocalTtsForm bind:catalog active={activeTab === 'local'} />
  <CloudTtsForm bind:catalog active={activeTab === 'cloud'} />
</div>
