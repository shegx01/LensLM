<!--
  TranscriptionSection — engine picker for source transcription (mirrors
  TtsConfigPanel's two-column master-detail shell). Selection is local state
  seeded from the persisted asr.backend, not derived from it (AC 2.2): this is
  what lets the Apple row be selected without ever writing an unusable backend.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { cn } from '$lib/utils.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    ASR_ENGINE_CATALOG,
    asrBackendToken,
    asrEngineIdFromBackend,
    type AsrEngineId
  } from '$lib/asr/catalog.js';
  import { whisperModelDownloaded } from '$lib/asr/ipc.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import TranscriptionApplePane from './TranscriptionApplePane.svelte';
  import TranscriptionWhisperPane from './TranscriptionWhisperPane.svelte';
  import TranscriptionCloudPane from './TranscriptionCloudPane.svelte';
  import TranscriptionLanguageBlock from './TranscriptionLanguageBlock.svelte';

  let selectedEngine = $state<AsrEngineId>('automatic');
  let appleAvailable = $state(false);
  let whisperPresent = $state(false);
  let ready = $state(false);

  const persistedEngine = $derived(asrEngineIdFromBackend(appConfigStore.asr?.backend ?? ''));
  const selectedEntry = $derived(ASR_ENGINE_CATALOG.find((e) => e.id === selectedEngine));

  const automaticUsable = $derived(appleAvailable || whisperPresent);

  // The engine actually powering transcription — persisted AND usable. Drives the
  // language block's capability notice, which must describe the real engine, not the row.
  const activeEngine = $derived(isUsable(persistedEngine) ? persistedEngine : null);

  function isUsable(id: AsrEngineId): boolean {
    switch (id) {
      case 'automatic':
        return automaticUsable;
      case 'apple_native':
        return appleAvailable;
      case 'local_whisper':
        return whisperPresent;
      case 'cloud':
        return (
          !!appConfigStore.asr?.cloud_base_url &&
          !!appConfigStore.asr?.cloud_model &&
          !!appConfigStore.asr?.cloud_api_key &&
          appConfigStore.audioCloudConsent
        );
    }
  }

  type Pill = { text: string; tone: 'active' | 'setup' };

  function rowPill(id: AsrEngineId): Pill | null {
    if (id !== persistedEngine) return null;
    return isUsable(id)
      ? { text: 'Active', tone: 'active' }
      : { text: 'Needs setup', tone: 'setup' };
  }

  async function probeAppleAvailable(): Promise<boolean> {
    if (!isTauri()) return false;
    try {
      return await invoke<boolean>('asr_apple_native_available');
    } catch {
      return false;
    }
  }

  onMount(async () => {
    await ensureLoaded();
    selectedEngine = asrEngineIdFromBackend(appConfigStore.asr?.backend ?? '');
    const model = appConfigStore.asr?.whisper_model ?? '';
    const [apple, whisper] = await Promise.all([
      probeAppleAvailable(),
      model ? whisperModelDownloaded(model).catch(() => false) : Promise.resolve(false)
    ]);
    appleAvailable = apple;
    whisperPresent = whisper;
    ready = true;
  });

  /** Only writes `asr.backend` for rows already usable at select time (Automatic's `""`
   *  is always safe). Apple never writes while unavailable (R11); a row that is still
   *  blocked activates later from the pane that clears the blocker. */
  function pickEngine(id: AsrEngineId): void {
    if (id === selectedEngine) return;
    selectedEngine = id;
    if (id !== 'automatic' && !isUsable(id)) return;
    void persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, backend: asrBackendToken(id) } }));
  }

  function handleWhisperPresenceChange(modelId: string, downloaded: boolean): void {
    if (modelId !== appConfigStore.asr?.whisper_model) return;
    whisperPresent = downloaded;
    if (downloaded && selectedEngine === 'local_whisper') {
      void persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, backend: 'local_whisper' } }));
    }
  }
</script>

<section class="flex flex-col" aria-label="Transcription settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Transcription</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Choose how audio and video sources are transcribed to text.
  </p>

  {#if appConfigStore.loadError}
    <p class="mt-6 text-[0.8rem] text-destructive" role="alert">{appConfigStore.loadError}</p>
  {:else if ready}
    <div
      class="mt-6 grid grid-cols-1 items-start gap-3.5 md:grid-cols-[minmax(200px,0.85fr)_1.15fr]"
    >
      <div
        class="no-scrollbar flex max-h-[420px] flex-col gap-1.5 overflow-y-auto"
        role="radiogroup"
        aria-label="Transcription engine"
      >
        {#each ASR_ENGINE_CATALOG as e (e.id)}
          {@const checked = e.id === selectedEngine}
          {@const pill = rowPill(e.id)}
          {@const unavailable = e.id === 'apple_native' && !appleAvailable}
          <button
            type="button"
            role="radio"
            aria-checked={checked}
            onclick={() => pickEngine(e.id)}
            class={cn(
              'flex w-full items-center gap-2.5 rounded-[10px] border px-3 py-2.5 text-left transition-[background-color,border-color,transform] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:scale-[0.98]',
              checked ? 'border-primary/40 bg-primary/10' : 'border-transparent hover:bg-muted'
            )}
          >
            <span class="min-w-0 flex-1">
              <span class="flex items-center gap-2 text-[0.8rem] font-bold text-foreground">
                <span
                  class={cn(
                    'size-[7px] shrink-0 rounded-full',
                    e.id === persistedEngine ? 'bg-primary' : 'bg-muted-foreground/50'
                  )}
                  aria-hidden="true"
                ></span>
                <span class="truncate">{e.label}</span>
                {#if pill}
                  <span
                    class={cn(
                      'shrink-0 rounded-full px-1.5 py-px text-[0.58rem] font-bold uppercase tracking-[0.05em]',
                      pill.tone === 'active' && 'bg-primary text-primary-foreground',
                      pill.tone === 'setup' && 'bg-muted-foreground/15 text-muted-foreground'
                    )}
                  >
                    {pill.text}
                  </span>
                {/if}
              </span>
              <span class="mt-px block truncate text-[0.68rem] text-muted-foreground">
                {unavailable && e.unavailableReason ? e.unavailableReason : e.description}
              </span>
            </span>
          </button>
        {/each}
      </div>

      <div class="rounded-xl border border-border bg-card p-[18px]">
        <div class="min-w-0">
          <div class="truncate text-[0.95rem] font-extrabold text-foreground">
            {selectedEntry?.label}
          </div>
          <div class="text-[0.7rem] text-muted-foreground">{selectedEntry?.description}</div>
        </div>

        <div class="mt-4">
          {#if selectedEngine === 'automatic' && !automaticUsable}
            <div class="rounded-[10px] border border-border bg-muted/40 p-3">
              <p class="text-[0.72rem] text-muted-foreground">
                Apple transcription isn't available on this device and no Local Whisper model is
                downloaded yet. Download a model to enable Automatic.
              </p>
              <Button
                type="button"
                size="sm"
                class="mt-2"
                onclick={() => pickEngine('local_whisper')}
              >
                Set up Local Whisper
              </Button>
            </div>
          {:else if selectedEngine === 'apple_native'}
            <TranscriptionApplePane available={appleAvailable} />
          {:else if selectedEngine === 'local_whisper'}
            <TranscriptionWhisperPane onPresenceChange={handleWhisperPresenceChange} />
          {:else if selectedEngine === 'cloud'}
            <TranscriptionCloudPane />
          {/if}
        </div>
      </div>
    </div>

    <TranscriptionLanguageBlock {activeEngine} />
  {/if}
</section>
