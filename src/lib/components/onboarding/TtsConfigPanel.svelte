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
  // Cloud leaves LocalTtsForm on its last engine (hidden, state preserved).
  let selectedEngine = $state<TtsEngineId>('orpheus');
  let selectedLocalEngine = $state<LocalEngineId>('orpheus');
  // The persisted backend actually powering audio overviews — drives the "Active" pill.
  let activeEngine = $state<TtsEngineId | null>(null);
  let loaded = $state(false);
  let localForm = $state<{ activate: () => void } | undefined>();

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
      selectedEngine = activeEngine;
      // Never seed the local form with a locked engine (e.g. a stale qwen3_local
      // config on non-Apple-Silicon hardware) — it would show a doomed download.
      if (activeEngine !== 'cloud' && !isLockedId(activeEngine)) selectedLocalEngine = activeEngine;
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
    <div class="mt-6 flex flex-col gap-4">
      <div class="flex flex-col gap-2">
        <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
          Engine
        </p>
        <div class="flex flex-col gap-1.5" role="radiogroup" aria-label="Text-to-speech engine">
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
                'flex items-center gap-2.5 rounded-[9px] border px-3 py-2.5 text-left transition-[background-color,border-color,transform] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                !locked && 'active:scale-[0.985]',
                checked ? 'border-primary/55 bg-primary/8' : 'border-border bg-card hover:bg-muted',
                locked && 'cursor-not-allowed opacity-60'
              )}
            >
              <span
                class={cn(
                  'grid size-[15px] shrink-0 place-items-center rounded-full border-2',
                  checked ? 'border-primary' : 'border-muted-foreground'
                )}
                aria-hidden="true"
              >
                {#if checked}
                  <span class="size-[7px] rounded-full bg-primary"></span>
                {/if}
              </span>

              <span class="min-w-0 flex-1">
                <span class="block truncate text-[0.8rem] font-semibold text-foreground">
                  {engineLabel(e.id)}
                </span>
                <span class="mt-1 flex flex-wrap items-center gap-1.5">
                  <span
                    class="rounded-full bg-muted px-1.5 py-px text-[0.58rem] font-bold tracking-[0.02em] text-muted-foreground"
                  >
                    {e.language_capability_label}
                  </span>
                  {#if locked && e.unavailable_reason}
                    <span class="text-[0.62rem] text-muted-foreground">{e.unavailable_reason}</span>
                  {/if}
                </span>
              </span>

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
            </button>
          {/each}
        </div>
      </div>

      <!-- Both forms stay mounted; visibility toggles via `active` so switching engines
           never drops typed-but-unsaved state or an in-flight download. -->
      <LocalTtsForm
        bind:this={localForm}
        bind:catalog
        engine={selectedLocalEngine}
        active={selectedEngine !== 'cloud'}
        onactivated={onActivated}
      />
      <CloudTtsForm bind:catalog active={selectedEngine === 'cloud'} onactivated={onActivated} />
    </div>
  {/if}
</section>
