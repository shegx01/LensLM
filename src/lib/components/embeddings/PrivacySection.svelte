<!--
  PrivacySection — the "Privacy" panel inside the global Preferences view. A
  transparency mirror: a read-only cloud-egress list (LLM/TTS/ASR) derived from
  existing config, plus the two existing consent toggles. No new AppConfig fields.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';
  import { providerDescriptors } from '$lib/models/providers.js';

  interface EgressRow {
    label: string;
    active: boolean;
    detail: string;
  }

  let egressRows = $state<EgressRow[]>([]);
  let textConsent = $state(false);
  let audioConsent = $state(false);
  let saving = $state(false);
  let saveError = $state<string | null>(null);

  const anyActive = $derived(egressRows.some((row) => row.active));

  /** Cloud LLM egress: active when the pinned chat model belongs to a non-local provider. */
  function llmEgress(cfg: AppConfig): EgressRow {
    const pin = cfg.enrichment?.chat_model;
    const descriptor = pin ? providerDescriptors().find((d) => d.id === pin.provider) : undefined;
    const active = descriptor != null && descriptor.kind !== 'local';
    const entry = pin ? cfg.models?.find((m) => m.provider === pin.provider) : undefined;
    return {
      label: 'Chat & notes model',
      active,
      detail: active ? entry?.base_url?.trim() || descriptor?.name || 'Cloud' : 'Local (Ollama)'
    };
  }

  /** Cloud TTS egress: `tts.backend` is the tagged `{ cloud: CloudTtsKind }` variant. */
  function ttsEgress(cfg: AppConfig): EgressRow {
    const backend = cfg.tts?.backend;
    const active = typeof backend === 'object' && backend !== null && 'cloud' in backend;
    return {
      label: 'Text-to-speech',
      active,
      detail: active ? cfg.tts?.cloud?.base_url?.trim() || 'Cloud' : 'Local'
    };
  }

  /** Cloud ASR egress: `asr.backend` is a plain string; `"cloud"` is the explicit override. */
  function asrEgress(cfg: AppConfig): EgressRow {
    const active = cfg.asr?.backend === 'cloud';
    return {
      label: 'Speech-to-text',
      active,
      detail: active ? cfg.asr?.cloud_base_url?.trim() || 'Cloud' : 'Local'
    };
  }

  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      egressRows = [llmEgress(cfg), ttsEgress(cfg), asrEgress(cfg)];
      textConsent = cfg.enrichment?.cloud_consent ?? false;
      audioConsent = cfg.audio_cloud_consent ?? false;
    } catch {
      // Non-fatal: leave defaults (all local, consent off).
    }
  });

  async function handleTextConsent(checked: boolean): Promise<void> {
    textConsent = checked;
    saving = true;
    saveError = null;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        enrichment: { ...cfg.enrichment, cloud_consent: checked }
      }));
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save setting.';
      // Revert the optimistic update on failure.
      textConsent = !checked;
    } finally {
      saving = false;
    }
  }

  async function handleAudioConsent(checked: boolean): Promise<void> {
    audioConsent = checked;
    saving = true;
    saveError = null;
    try {
      await updateConfig((cfg) => ({ ...cfg, audio_cloud_consent: checked }));
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save setting.';
      // Revert the optimistic update on failure.
      audioConsent = !checked;
    } finally {
      saving = false;
    }
  }
</script>

<section class="flex flex-col" aria-label="Privacy settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Privacy</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Where your data can leave this device, and your consent for cloud features.
  </p>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Data leaving this device
    </p>

    {#if anyActive}
      <div class="mt-3 flex flex-col gap-2">
        {#each egressRows as row (row.label)}
          <div
            class="flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
          >
            <span class="min-w-0 flex-1">
              <span class="block text-[0.78rem] font-bold text-foreground">{row.label}</span>
              <span class="mt-0.5 block truncate text-[0.68rem] text-muted-foreground"
                >{row.detail}</span
              >
            </span>
            <span
              class="shrink-0 rounded-full px-2 py-0.5 text-[0.6rem] font-bold uppercase tracking-[0.05em] {row.active
                ? 'bg-primary/15 text-primary'
                : 'bg-muted text-muted-foreground'}"
            >
              {row.active ? 'Cloud' : 'Local'}
            </span>
          </div>
        {/each}
      </div>
    {:else}
      <p
        class="mt-3 rounded-[10px] bg-muted px-3.5 py-3 text-[0.75rem] text-muted-foreground"
        role="status"
      >
        No data leaves this device.
      </p>
    {/if}
  </div>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Cloud consent
    </p>

    <label
      class="mt-3 flex cursor-pointer items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5 transition-colors hover:border-border/80"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">Allow cloud text models</span>
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          Lets chat, notes, and other enrichment tasks use a cloud model when one is pinned. Leave
          off to stay fully on-device.
        </span>
      </span>
      <Switch
        checked={textConsent}
        disabled={saving}
        aria-label="Allow cloud text models"
        onCheckedChange={handleTextConsent}
      />
    </label>

    <label
      class="mt-3 flex cursor-pointer items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5 transition-colors hover:border-border/80"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">Allow cloud audio</span>
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          Lets text-to-speech and speech-to-text use a cloud provider when one is configured.
        </span>
      </span>
      <Switch
        checked={audioConsent}
        disabled={saving}
        aria-label="Allow cloud audio"
        onCheckedChange={handleAudioConsent}
      />
    </label>
  </div>

  {#if saveError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{saveError}</p>
  {/if}
</section>
