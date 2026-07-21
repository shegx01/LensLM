<script lang="ts">
  import { onMount } from 'svelte';
  import { fade } from 'svelte/transition';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Input } from '$lib/components/ui/input/index.js';
  import ApiKeyField from '$lib/components/llm/ApiKeyField.svelte';
  import { cn } from '$lib/utils.js';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import {
    saveTtsProvider,
    ttsEngineCatalog,
    ttsBackendId,
    type TtsVoice,
    type TtsEngineCatalogEntry,
    type TtsEngineId
  } from '$lib/onboarding/system-check.js';
  import { prefersReducedMotion } from '$lib/motion/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';

  // Bound (not a plain prop) because refreshCatalog() reassigns `catalog` after a
  // key save, which a child can't propagate through a plain prop in Svelte 5.
  // See TtsConfigPanel for the ownership rationale.
  let {
    catalog = $bindable(),
    active,
    onactivated
  }: {
    catalog: TtsEngineCatalogEntry[];
    active: boolean;
    onactivated?: (id: TtsEngineId) => void;
  } = $props();

  const CUSTOM_VOICE = '__custom__';

  let cloudApiKey = $state('');
  // The real, currently-persisted key. Never bound to an input — only resent on
  // saves that touch other fields (base URL/voices) so masking never writes a
  // blank over a real key (mirrors the #194 Cloud-key-wipe regression fix).
  let savedCloudApiKey = $state('');
  let cloudError = $state<string | null>(null);

  // Saved key is masked; Save re-enables only after the user enters a fresh key.
  let hasSavedKey = $state(false);
  let editingKey = $state(false);

  let cloudBaseUrl = $state('https://api.openai.com');
  let cloudHostPreset = $state('');
  let cloudGuestPreset = $state('');
  // Snippet parameters are read-only, so the free-text custom voice ids live in
  // per-role $state objects and the snippet binds to their `.custom` member.
  const host = $state({ custom: '' });
  const guest = $state({ custom: '' });

  const cloudEntry = $derived(catalog.find((e) => e.id === 'cloud') ?? null);
  const cloudVoices = $derived(cloudEntry?.preset_voices ?? []);
  const cloudMaleVoices = $derived(cloudVoices.filter((v) => v.gender === 'male'));
  const cloudFemaleVoices = $derived(cloudVoices.filter((v) => v.gender === 'female'));

  // The resolved host/guest voice id to persist: the curated pick, unless the
  // user chose the free-text escape hatch (or no curated voices exist yet).
  const cloudHostVoice = $derived(
    cloudMaleVoices.length > 0 && cloudHostPreset !== CUSTOM_VOICE
      ? cloudHostPreset
      : host.custom.trim()
  );
  const cloudGuestVoice = $derived(
    cloudFemaleVoices.length > 0 && cloudGuestPreset !== CUSTOM_VOICE
      ? cloudGuestPreset
      : guest.custom.trim()
  );

  /** Splits a saved voice id into {preset, custom}: a known curated id selects
   *  itself; anything else (or no curated list yet) falls back to free-text. */
  function classifyCloudVoice(
    saved: string,
    presets: TtsVoice[]
  ): { preset: string; custom: string } {
    // Nothing saved yet: leave `preset` empty so the caller's `|| presets[0]?.id`
    // fallback picks the default — CUSTOM_VOICE here would short-circuit that `||`.
    if (!saved) return { preset: '', custom: '' };
    if (presets.some((v) => v.id === saved)) return { preset: saved, custom: '' };
    return { preset: presets.length > 0 ? CUSTOM_VOICE : '', custom: saved };
  }

  /** Crossfade duration, collapsed to 0 under the shared reduced-motion rule
   *  (which also honours the app's `data-motion` Animations preference). */
  function motionMs(ms: number): number {
    return prefersReducedMotion() ? 0 : ms;
  }

  onMount(async () => {
    if (!isTauri()) return;
    // The panel passes a populated catalog; only self-fetch when mounted standalone.
    if (catalog.length === 0) {
      try {
        catalog = (await ttsEngineCatalog()) ?? [];
      } catch {
        catalog = [];
      }
    }
    try {
      const cfg = await invoke<AppConfig>('get_config');
      if (cfg.tts?.cloud && cfg.tts.cloud.api_key.trim() !== '') {
        hasSavedKey = true;
        savedCloudApiKey = cfg.tts.cloud.api_key;
        cloudApiKey = '';
      }
      cloudBaseUrl = cfg.tts?.cloud?.base_url?.trim() || 'https://api.openai.com';
      // Voices are shared across engines; only trust them as Cloud picks when
      // Cloud is the currently-active backend (otherwise they belong to whatever
      // local engine is active and would be nonsense cloud voice ids).
      const backendIsCloud = ttsBackendId(cfg.tts.backend) === 'cloud';
      const savedHost =
        backendIsCloud && typeof cfg.voices?.host === 'string' ? cfg.voices.host : '';
      const savedGuest =
        backendIsCloud && typeof cfg.voices?.guest === 'string' ? cfg.voices.guest : '';
      // Read the just-fetched catalog directly — the `cloudMaleVoices` derived can
      // be stale in this async continuation right after reassigning its source.
      const presets = (catalog.find((e) => e.id === 'cloud')?.preset_voices ?? []).slice();
      const maleVoices = presets.filter((v) => v.gender === 'male');
      const femaleVoices = presets.filter((v) => v.gender === 'female');
      const hostClass = classifyCloudVoice(savedHost, maleVoices);
      cloudHostPreset = hostClass.preset || maleVoices[0]?.id || '';
      host.custom = hostClass.custom;
      const guestClass = classifyCloudVoice(savedGuest, femaleVoices);
      cloudGuestPreset = guestClass.preset || femaleVoices[0]?.id || '';
      guest.custom = guestClass.custom;
    } catch {
      // Non-fatal: fall back to the default empty Cloud form.
    }
  });

  /** Re-fetch the catalog so Cloud's backend-derived `available` reflects the
   *  just-saved key immediately — without this the user saves a valid key but
   *  Cloud stays reported unselectable until app restart. */
  async function refreshCatalog(): Promise<void> {
    try {
      catalog = (await ttsEngineCatalog()) ?? [];
    } catch {
      // Keep the previous catalog on a transient re-fetch failure.
    }
  }

  /** Reactive Cloud persist (mirrors persistLocalTts); no Save button. */
  async function persistCloud(): Promise<void> {
    cloudError = null;
    // Don't activate an unusable cloud backend: require a base URL and either a
    // saved key or a freshly-typed one (this is the guard the old Save gate gave).
    if (!cloudBaseUrl.trim() || (!hasSavedKey && !cloudApiKey.trim())) return;
    try {
      // Only a freshly-typed key replaces the stored one; otherwise resend the
      // saved key — a blank field can never overwrite a real key.
      const apiKey =
        editingKey && cloudApiKey.trim()
          ? cloudApiKey
          : hasSavedKey
            ? savedCloudApiKey
            : cloudApiKey;
      await saveTtsProvider({
        provider: 'cloud',
        apiKey,
        baseUrl: cloudBaseUrl,
        hostVoice: cloudHostVoice,
        guestVoice: cloudGuestVoice
      });
      savedCloudApiKey = apiKey;
      hasSavedKey = apiKey.trim() !== '';
      editingKey = false;
      cloudApiKey = '';
      await refreshCatalog();
      onactivated?.('cloud');
    } catch (err) {
      cloudError = err instanceof Error ? err.message : 'Could not save configuration.';
    }
  }

  /** API-key field's commit (blur) handler: persists a freshly-typed key (first-time
   *  entry or an explicit replace), but blurring an emptied "replace" field
   *  re-masks instead of persisting — never wipes a real saved key with blank. */
  function handleKeyCommit(): void {
    if (editingKey && !cloudApiKey.trim()) {
      editingKey = false;
      return;
    }
    if (editingKey || (!hasSavedKey && cloudApiKey.trim())) {
      void persistCloud();
    }
  }
</script>

{#snippet cloudVoicePicker(opts: {
  id: string;
  label: string;
  voices: TtsVoice[];
  preset: string;
  onpreset: (v: string) => void;
  field: { custom: string };
  customPlaceholder: string;
  fallbackPlaceholder: string;
})}
  <div class="flex flex-col gap-1.5">
    <label for={opts.id} class="text-[0.72rem] font-bold text-foreground">
      {opts.label}
    </label>
    {#if opts.voices.length > 0}
      <Select
        type="single"
        value={opts.preset}
        onValueChange={(v) => {
          if (v) opts.onpreset(v);
        }}
        items={[
          ...opts.voices.map((voice) => ({ value: voice.id, label: voice.name })),
          { value: CUSTOM_VOICE, label: 'Custom voice ID…' }
        ]}
      >
        <SelectTrigger id={opts.id} class="w-full">
          <SelectValue placeholder="Select a voice" />
        </SelectTrigger>
        <SelectContent
          class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
        >
          {#each opts.voices as voice (voice.id)}
            <SelectItem value={voice.id} label={voice.name}>{voice.name}</SelectItem>
          {/each}
          <SelectItem value={CUSTOM_VOICE} label="Custom voice ID…">Custom voice ID…</SelectItem>
        </SelectContent>
      </Select>
      {#if opts.preset === CUSTOM_VOICE}
        <Input
          id={`${opts.id}-custom`}
          type="text"
          bind:value={opts.field.custom}
          placeholder={opts.customPlaceholder}
          autocomplete="off"
          onblur={() => void persistCloud()}
        />
      {/if}
    {:else}
      <Input
        id={opts.id}
        type="text"
        bind:value={opts.field.custom}
        placeholder={opts.fallbackPlaceholder}
        autocomplete="off"
        onblur={() => void persistCloud()}
      />
    {/if}
  </div>
{/snippet}

<div
  role="group"
  aria-label="Cloud voice engine setup"
  class={cn('flex flex-col gap-4', !active && 'hidden')}
>
  {#if cloudEntry}
    {#if !cloudEntry.available}
      <p
        transition:fade={{ duration: motionMs(160) }}
        class="flex items-center gap-2 rounded-[10px] bg-destructive/10 px-3.5 py-3 text-[0.72rem] text-destructive ring-1 ring-destructive/30"
        role="status"
      >
        <CircleAlert class="size-3.5 shrink-0" aria-hidden="true" />
        {cloudEntry.unavailable_reason ?? 'Cloud is unavailable.'} Add an API key below to enable it.
      </p>
    {:else}
      <p
        transition:fade={{ duration: motionMs(160) }}
        class="flex items-center gap-2 rounded-[10px] bg-primary/10 px-3.5 py-3 text-[0.72rem] text-primary ring-1 ring-primary/30"
        role="status"
      >
        <CircleCheck class="size-3.5 shrink-0" aria-hidden="true" />
        Cloud is available
      </p>
    {/if}
  {/if}

  <div>
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Connection
    </p>
    <p class="mt-2 text-pretty text-[0.72rem] leading-relaxed text-muted-foreground">
      Connect any endpoint that implements OpenAI's speech API (POST /v1/audio/speech) — OpenAI
      itself, hosted providers like Groq or DeepInfra, or a self-hosted server such as LocalAI.
    </p>

    <div class="mt-3">
      <ApiKeyField
        id="tts-cloud-key"
        bind:value={cloudApiKey}
        bind:editing={editingKey}
        {hasSavedKey}
        oncommit={handleKeyCommit}
      />
    </div>

    <div class="mt-3 flex flex-col gap-1.5">
      <label for="tts-cloud-base-url" class="text-[0.72rem] font-bold text-foreground">
        Base URL
      </label>
      <Input
        id="tts-cloud-base-url"
        type="text"
        bind:value={cloudBaseUrl}
        placeholder="https://api.openai.com"
        autocomplete="off"
        onblur={() => void persistCloud()}
      />
      <p class="text-[0.68rem] leading-relaxed text-muted-foreground">
        API root only — no trailing <code
          class="rounded bg-muted px-1 py-px font-mono text-[0.62rem]">/v1</code
        >; it's appended automatically.
      </p>
    </div>
  </div>

  <div>
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Voices
    </p>

    <div class="mt-3 flex flex-col gap-3">
      {@render cloudVoicePicker({
        id: 'tts-cloud-host-voice',
        label: 'Host speaker',
        voices: cloudMaleVoices,
        preset: cloudHostPreset,
        onpreset: (v) => {
          cloudHostPreset = v;
          void persistCloud();
        },
        field: host,
        customPlaceholder: 'e.g. alloy',
        fallbackPlaceholder: 'Voice ID (e.g. alloy)'
      })}

      {@render cloudVoicePicker({
        id: 'tts-cloud-guest-voice',
        label: 'Guest speaker',
        voices: cloudFemaleVoices,
        preset: cloudGuestPreset,
        onpreset: (v) => {
          cloudGuestPreset = v;
          void persistCloud();
        },
        field: guest,
        customPlaceholder: 'e.g. onyx',
        fallbackPlaceholder: 'Voice ID (e.g. onyx)'
      })}

      <p class="text-[0.68rem] leading-relaxed text-muted-foreground">
        Curated voices are OpenAI's. Using another provider? Enter its own voice IDs.
      </p>
    </div>
  </div>

  {#if cloudError}
    <p class="text-[0.72rem] text-destructive" role="alert">{cloudError}</p>
  {/if}
</div>
