<!--
  TtsConfigPanel — active-engine selector for audio overviews (mirrors the AI Model
  pane). The row matching AppConfig.tts.backend gets the "Active" pill; the selected
  engine's setup + voices render below via LocalTtsForm / CloudTtsForm.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { cn } from '$lib/utils.js';
  import {
    ttsEngineCatalog,
    ttsBackendId,
    type TtsEngineCatalogEntry,
    type TtsEngineId
  } from '$lib/onboarding/system-check.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import LocalTtsForm from './LocalTtsForm.svelte';
  import CloudTtsForm from './CloudTtsForm.svelte';

  type LocalEngineId = Exclude<TtsEngineId, 'cloud'>;

  // Parent-owned so a Cloud key save (CloudTtsForm.refreshCatalog) re-derives engine
  // availability for the list too.
  let catalog = $state<TtsEngineCatalogEntry[]>([]);

  // The row highlighted in the engine list. `selectedLocalEngine` tracks which local
  // engine LocalTtsForm shows — it only changes when a local row is picked, so picking
  // Cloud leaves LocalTtsForm on its last engine.
  let selectedEngine = $state<TtsEngineId>('orpheus');
  let selectedLocalEngine = $state<LocalEngineId>('orpheus');
  // The persisted backend actually powering audio overviews — drives the "Active" pill.
  let activeEngine = $state<TtsEngineId | null>(null);
  let loaded = $state(false);
  let localForm = $state<{ activate: () => void } | undefined>();

  const selectedCapability = $derived(
    catalog.find((e) => e.id === selectedEngine)?.language_capability_label ?? ''
  );

  function engineLabel(id: TtsEngineId): string {
    if (id === 'orpheus') return 'Orpheus';
    if (id === 'qwen3_local') return 'Qwen3-TTS';
    return 'Cloud';
  }

  /** A child persist activated `id` — move the "Active" pill directly (no config re-read). */
  function onActivated(id: TtsEngineId): void {
    activeEngine = id;
  }

  onMount(async () => {
    if (!isTauri()) {
      loaded = true;
      return;
    }
    try {
      catalog = (await ttsEngineCatalog()) ?? [];
    } catch {
      catalog = [];
    }
    try {
      const cfg = await invoke<AppConfig>('get_config');
      activeEngine = ttsBackendId(cfg.tts.backend);
      // A locked active engine (e.g. a stale qwen3_local config on non-Apple-Silicon)
      // must not become the selected row — that would show a doomed download under a
      // header naming the wrong engine. Fall back to the default local engine; the
      // "Active" pill still marks the real backend via rowPill.
      const lockedActive = activeEngine !== 'cloud' && isLockedId(activeEngine);
      selectedEngine = lockedActive ? selectedLocalEngine : activeEngine;
      if (activeEngine !== 'cloud' && !lockedActive) selectedLocalEngine = activeEngine;
    } catch {
      // Non-fatal: default to Orpheus selected, nothing active.
    }
    loaded = true;
  });

  /** True when a row can't be selected on this machine (a local engine gated by
   *  hardware, e.g. Qwen off Apple Silicon). A needs-key engine stays selectable so
   *  the user can add a key. */
  function isLocked(e: TtsEngineCatalogEntry): boolean {
    return !e.available && !e.needs_key;
  }

  function isLockedId(id: TtsEngineId): boolean {
    const e = catalog.find((c) => c.id === id);
    return e ? isLocked(e) : false;
  }

  function pickEngine(e: TtsEngineCatalogEntry): void {
    if (isLocked(e) || e.id === selectedEngine) return;
    selectedEngine = e.id;
    if (e.id === 'cloud') return;
    if (e.id === selectedLocalEngine) {
      // The `engine` prop won't change, so LocalTtsForm's load effect can't observe
      // the re-pick — activate the (already-loaded) engine imperatively instead.
      localForm?.activate();
    } else {
      selectedLocalEngine = e.id;
    }
  }

  type Pill = { text: string; tone: 'active' | 'setup' | 'off' };

  // Derived from config (Active) + catalog (locked / needs-key) only — no disk probe.
  // The selected engine's real setup state (download / ready) shows in the form below.
  function rowPill(e: TtsEngineCatalogEntry): Pill | null {
    if (e.id === activeEngine) return { text: 'Active', tone: 'active' };
    if (isLocked(e)) return { text: 'Unavailable', tone: 'off' };
    if (!e.available && e.needs_key) return { text: 'Needs key', tone: 'setup' };
    return null;
  }
</script>

<section class="flex flex-col" aria-label="Text-to-speech settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Text-to-Speech</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Choose the voice engine and speakers for audio overviews. The selected engine is the one used.
  </p>

  {#if loaded}
    <div
      class="mt-6 grid grid-cols-1 items-start gap-3.5 md:grid-cols-[minmax(200px,0.85fr)_1.15fr]"
    >
      <!-- Left rail: the engine list. Selecting a ready row makes that engine active. -->
      <div
        class="no-scrollbar flex max-h-[420px] flex-col gap-1.5 overflow-y-auto"
        role="radiogroup"
        aria-label="Text-to-speech engine"
      >
        {#each catalog as e (e.id)}
          {@const checked = e.id === selectedEngine}
          {@const locked = isLocked(e)}
          {@const pill = rowPill(e)}
          <button
            type="button"
            role="radio"
            aria-checked={checked}
            aria-disabled={locked}
            disabled={locked}
            onclick={() => pickEngine(e)}
            class={cn(
              'flex w-full items-center gap-2.5 rounded-[10px] border px-3 py-2.5 text-left transition-[background-color,border-color,transform] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              !locked && 'active:scale-[0.98]',
              checked ? 'border-primary/40 bg-primary/10' : 'border-transparent hover:bg-muted',
              locked && 'cursor-not-allowed opacity-60'
            )}
          >
            <span class="min-w-0 flex-1">
              <span class="flex items-center gap-2 text-[0.8rem] font-bold text-foreground">
                <span
                  class={cn(
                    'size-[7px] shrink-0 rounded-full',
                    e.id === activeEngine ? 'bg-primary' : 'bg-muted-foreground/50'
                  )}
                  aria-hidden="true"
                ></span>
                <span class="truncate">{engineLabel(e.id)}</span>
                {#if pill}
                  <span
                    class={cn(
                      'shrink-0 rounded-full px-1.5 py-px text-[0.58rem] font-bold uppercase tracking-[0.05em]',
                      pill.tone === 'active' && 'bg-primary text-primary-foreground',
                      pill.tone === 'setup' && 'bg-muted-foreground/15 text-muted-foreground',
                      pill.tone === 'off' && 'bg-muted-foreground/10 text-muted-foreground/60'
                    )}
                  >
                    {pill.text}
                  </span>
                {/if}
              </span>
              <span class="mt-px block truncate text-[0.68rem] text-muted-foreground">
                {locked && e.unavailable_reason
                  ? e.unavailable_reason
                  : e.language_capability_label}
              </span>
            </span>
          </button>
        {/each}
      </div>

      <!-- Right detail panel for the selected engine. Both child forms stay mounted
           (visibility via `active`) so switching engines never drops typed-but-unsaved
           state or an in-flight download. -->
      <div class="rounded-xl border border-border bg-card p-[18px]">
        <div class="min-w-0">
          <div class="truncate text-[0.95rem] font-extrabold text-foreground">
            {engineLabel(selectedEngine)}
          </div>
          <div class="text-[0.7rem] text-muted-foreground">{selectedCapability}</div>
        </div>

        <div class="mt-4">
          <LocalTtsForm
            bind:this={localForm}
            bind:catalog
            engine={selectedLocalEngine}
            active={selectedEngine !== 'cloud'}
            onactivated={onActivated}
          />
          <CloudTtsForm
            bind:catalog
            active={selectedEngine === 'cloud'}
            onactivated={onActivated}
          />
        </div>
      </div>
    </div>
  {/if}
</section>
