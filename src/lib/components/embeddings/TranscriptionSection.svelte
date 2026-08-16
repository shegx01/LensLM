<!--
  TranscriptionSection — engine picker for source transcription (mirrors
  TtsConfigPanel's two-column master-detail shell). Selection is local state
  seeded from the persisted asr.backend, not derived from it: this is what
  lets the Apple row be selected without ever writing an unusable backend.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { cn } from '$lib/utils.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    ASR_ENGINE_CATALOG,
    appleAsrUnavailableReason,
    asrBackendLabel,
    asrBackendToken,
    asrEngineIdFromBackend,
    isTransportSafeBaseUrl,
    type AsrEngineId
  } from '$lib/asr/catalog.js';
  import {
    appleAsrAvailability,
    whisperModelDownloaded,
    resolveAsrBackend,
    type AppleAsrAvailability,
    type AsrBackend
  } from '$lib/asr/ipc.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import { toLensError } from '$lib/sources/lens-error.js';
  import TranscriptionApplePane from './TranscriptionApplePane.svelte';
  import TranscriptionWhisperPane from './TranscriptionWhisperPane.svelte';
  import TranscriptionCloudPane from './TranscriptionCloudPane.svelte';
  import TranscriptionLanguageBlock from './TranscriptionLanguageBlock.svelte';

  let selectedEngine = $state<AsrEngineId>('automatic');
  let appleAvailability = $state<AppleAsrAvailability | null>(null);
  let whisperPresent = $state(false);
  let ready = $state(false);
  let error = $state<string | null>(null);

  /** The router's real answer for "what runs right now" — never a client guess. */
  type ResolvedBackend =
    | { status: 'loading' }
    | { status: 'ready'; backend: AsrBackend }
    | { status: 'error' };
  let resolvedBackend = $state<ResolvedBackend>({ status: 'loading' });

  const persistedEngine = $derived(asrEngineIdFromBackend(appConfigStore.asr?.backend ?? ''));
  const selectedEntry = $derived(ASR_ENGINE_CATALOG.find((e) => e.id === selectedEngine));

  const appleAvailable = $derived(appleAvailability === 'available');
  const appleReason = $derived(appleAsrUnavailableReason(appleAvailability));
  const automaticUsable = $derived(appleAvailable || whisperPresent);

  // Drives the language block's capability notice, sourced from resolveAsrBackend
  // (the same seam transcribe() uses) rather than a persisted/usable guess.
  const activeEngine = $derived(
    resolvedBackend.status === 'ready' ? resolvedBackend.backend : null
  );

  function isUsable(id: AsrEngineId): boolean {
    switch (id) {
      case 'automatic':
        return automaticUsable;
      case 'apple_native':
        return appleAvailable;
      case 'local_whisper':
        return whisperPresent;
      case 'cloud': {
        const baseUrl = appConfigStore.asr?.cloud_base_url.trim() ?? '';
        return (
          !!appConfigStore.asr?.cloud_provider &&
          isTransportSafeBaseUrl(baseUrl) &&
          !!appConfigStore.asr?.cloud_model.trim() &&
          !!appConfigStore.asr?.cloud_api_key.trim() &&
          appConfigStore.audioCloudConsent
        );
      }
    }
  }

  type Pill = { text: string; tone: 'active' | 'setup' };

  function rowPill(id: AsrEngineId): Pill | null {
    if (id !== persistedEngine) return null;
    return isUsable(id)
      ? { text: 'Active', tone: 'active' }
      : { text: 'Needs setup', tone: 'setup' };
  }

  async function probeAppleAvailability(): Promise<AppleAsrAvailability | null> {
    try {
      return await appleAsrAvailability();
    } catch {
      return null;
    }
  }

  async function refreshResolvedBackend(): Promise<void> {
    try {
      const backend = await resolveAsrBackend();
      resolvedBackend = { status: 'ready', backend };
    } catch {
      resolvedBackend = { status: 'error' };
    }
  }

  onMount(async () => {
    await ensureLoaded();
    selectedEngine = asrEngineIdFromBackend(appConfigStore.asr?.backend ?? '');
    const model = appConfigStore.asr?.whisper_model ?? '';
    const [apple, whisper] = await Promise.all([
      probeAppleAvailability(),
      model ? whisperModelDownloaded(model).catch(() => false) : Promise.resolve(false)
    ]);
    appleAvailability = apple;
    whisperPresent = whisper;
    await refreshResolvedBackend();
    ready = true;
  });

  async function persistBackend(backend: string): Promise<void> {
    error = null;
    try {
      await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, backend } }));
      await refreshResolvedBackend();
    } catch (err) {
      error = toLensError(err).message;
    }
  }

  /** Only writes `asr.backend` for rows already usable at select time (Automatic's `""`
   *  is always safe). Apple never writes while unavailable; a row that is still
   *  blocked activates later from the pane that clears the blocker. */
  function pickEngine(id: AsrEngineId): void {
    if (id === selectedEngine) return;
    selectedEngine = id;
    if (id !== 'automatic' && !isUsable(id)) return;
    void persistBackend(asrBackendToken(id));
  }

  function handleWhisperPresenceChange(modelId: string, downloaded: boolean): void {
    if (modelId !== appConfigStore.asr?.whisper_model) return;
    whisperPresent = downloaded;
    if (downloaded && selectedEngine === 'local_whisper') {
      void persistBackend('local_whisper'); // already refreshes resolvedBackend
    } else {
      void refreshResolvedBackend();
    }
  }

  const UNKNOWN_RESOLUTION = "Couldn't determine which engine Automatic will use right now.";

  const automaticStatusText = $derived.by(() => {
    if (!automaticUsable) {
      return "Apple transcription isn't available on this device and no Local Whisper model is downloaded yet. Download a model to enable Automatic.";
    }
    const resolved = resolvedBackend;
    if (resolved.status === 'loading') return 'Determining which engine Automatic will use…';
    if (resolved.status === 'error') return UNKNOWN_RESOLUTION;
    const label = asrBackendLabel(resolved.backend);
    return label === null ? UNKNOWN_RESOLUTION : `Automatic currently resolves to ${label}.`;
  });
</script>

<section class="flex flex-col" aria-label="Transcription settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Transcription</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Choose how audio and video sources are transcribed to text.
  </p>

  {#if error}
    <p class="mt-6 text-[0.8rem] text-destructive" role="alert">{error}</p>
  {/if}

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
          {@const unavailableNote = e.id === 'apple_native' ? appleReason : null}
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
                {unavailableNote ?? e.description}
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
          {#if selectedEngine === 'automatic'}
            <div class="rounded-[10px] border border-border bg-muted/40 p-3">
              <p class="text-[0.72rem] text-muted-foreground">
                {automaticStatusText}
              </p>
              {#if !automaticUsable}
                <Button
                  type="button"
                  size="sm"
                  class="mt-2"
                  onclick={() => pickEngine('local_whisper')}
                >
                  Set up Local Whisper
                </Button>
              {/if}
            </div>
          {:else if selectedEngine === 'apple_native'}
            <TranscriptionApplePane available={appleAvailable} />
          {:else if selectedEngine === 'local_whisper'}
            <TranscriptionWhisperPane onPresenceChange={handleWhisperPresenceChange} />
          {:else if selectedEngine === 'cloud'}
            <TranscriptionCloudPane onRouterInputChange={() => void refreshResolvedBackend()} />
          {/if}
        </div>
      </div>
    </div>

    <TranscriptionLanguageBlock
      {activeEngine}
      onRouterInputChange={() => void refreshResolvedBackend()}
    />
  {/if}
</section>
