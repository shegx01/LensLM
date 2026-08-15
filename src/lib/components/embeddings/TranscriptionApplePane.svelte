<!--
  TranscriptionApplePane — confidence-threshold presets for Apple on-device ASR.
  Writing `apple_min_confidence` is allowed even when `available` is false (unlike
  `asr.backend`, this field has no degradation path to break — lib.rs clamps it
  regardless), so the machine is pre-configured for when the bridge is present.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { cn } from '$lib/utils.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';

  let { available }: { available: boolean } = $props();

  type PresetId = 'strict' | 'balanced' | 'lenient';

  interface Preset {
    id: PresetId;
    label: string;
    value: number;
    description: string;
  }

  // Balanced (0.5) matches lens-core/src/config.rs default_apple_min_confidence().
  const PRESETS: Preset[] = [
    {
      id: 'strict',
      label: 'Strict',
      value: 0.7,
      description: 'Re-transcribes the whole clip on Local Whisper more readily.'
    },
    { id: 'balanced', label: 'Balanced', value: 0.5, description: 'The recommended default.' },
    {
      id: 'lenient',
      label: 'Lenient',
      value: 0.3,
      description: 'Only falls back to Whisper when Apple is barely confident at all.'
    }
  ];

  function nearestPreset(value: number): PresetId {
    return PRESETS.reduce((closest, p) =>
      Math.abs(p.value - value) < Math.abs(closest.value - value) ? p : closest
    ).id;
  }

  let selected = $state<PresetId>('balanced');
  let ready = $state(false);

  onMount(async () => {
    await ensureLoaded();
    selected = nearestPreset(appConfigStore.asr?.apple_min_confidence ?? 0.5);
    ready = true;
  });

  function pick(preset: Preset): void {
    selected = preset.id;
    void persist((cfg) => ({
      ...cfg,
      asr: { ...cfg.asr, apple_min_confidence: preset.value }
    }));
  }
</script>

<div class="flex flex-col gap-3">
  <p class="text-[0.68rem] leading-relaxed text-muted-foreground">
    If Apple's confidence for the whole clip falls below this threshold, the clip is re-transcribed
    on Local Whisper. Needs a Whisper model already downloaded — otherwise the low-confidence Apple
    result is kept as-is.
  </p>

  {#if !available}
    <p class="text-[0.72rem] text-muted-foreground">
      This threshold takes effect once Apple on-device transcription is available on this device.
    </p>
  {/if}

  {#if ready}
    <div role="radiogroup" aria-label="Confidence threshold" class="flex flex-col gap-2">
      {#each PRESETS as preset (preset.id)}
        {@const checked = selected === preset.id}
        <button
          type="button"
          role="radio"
          aria-checked={checked}
          onclick={() => pick(preset)}
          class={cn(
            'flex flex-col items-start gap-0.5 rounded-[10px] border px-4 py-3 text-left transition-colors',
            checked ? 'border-primary/45 bg-primary/5' : 'border-border bg-card hover:bg-muted'
          )}
        >
          <span class="text-[0.8rem] font-bold text-foreground">{preset.label}</span>
          <span class="text-[0.68rem] text-muted-foreground">{preset.description}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>
