<!-- Global Language + translate block for the Transcription panel. One control set for
     every engine, with a tri-state notice for how the active engine handles `translate`
     — the only field that diverges (language is honoured by all four). -->
<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import Info from '@lucide/svelte/icons/info';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { cn } from '$lib/utils.js';
  import {
    ASR_LANGUAGE_OPTIONS,
    asrCapability,
    appleTranslateRerouteNotice,
    whisperTranslateNotice,
    isOtherAsrLanguage,
    type AsrLanguageValue
  } from '$lib/asr/catalog.js';
  import { whisperModelDownloaded, type AsrBackend } from '$lib/asr/ipc.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import { toLensError } from '$lib/sources/lens-error.js';
  import type { AsrLang } from '$lib/theme/types.js';

  /** `activeEngine` is the router's resolved backend, so it is never the `automatic` UI id.
   *  `onRouterInputChange` fires after each write here, both of which the router reads. */
  let {
    activeEngine,
    onRouterInputChange
  }: { activeEngine: AsrBackend | null; onRouterInputChange?: () => void } = $props();

  const AUTO_TOKEN = '__auto__';
  const OTHER_TOKEN = '__other__';

  onMount(() => {
    void ensureLoaded();
  });

  const language = $derived<AsrLanguageValue>(appConfigStore.asr?.language ?? null);
  const translate = $derived(appConfigStore.asr?.translate ?? false);

  const selectToken = $derived(
    language === null ? AUTO_TOKEN : isOtherAsrLanguage(language) ? OTHER_TOKEN : language
  );

  // Distinct from `selectToken`: picking "Other…" doesn't persist by itself (it has
  // no code yet), so the dropdown's shown value must survive independently of config
  // until the free-text field commits — otherwise it snaps back the instant it renders.
  let uiToken = $state(untrack(() => selectToken));
  $effect(() => {
    uiToken = selectToken;
  });
  const showOtherInput = $derived(uiToken === OTHER_TOKEN);

  let otherCode = $state('');
  $effect(() => {
    if (isOtherAsrLanguage(language)) otherCode = language.Other;
  });

  let error = $state<string | null>(null);

  const selectItems = [
    ...ASR_LANGUAGE_OPTIONS.map((option) => ({
      value: option.value ?? AUTO_TOKEN,
      label: option.label
    })),
    { value: OTHER_TOKEN, label: 'Other (language code)…' }
  ];

  async function persistLanguage(value: AsrLanguageValue): Promise<void> {
    try {
      error = null;
      await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, language: value } }));
      onRouterInputChange?.();
    } catch (err) {
      error = toLensError(err).message;
    }
  }

  function handleLanguageSelect(token: string): void {
    uiToken = token;
    if (token === OTHER_TOKEN) return;
    void persistLanguage(token === AUTO_TOKEN ? null : (token as AsrLang));
  }

  function handleOtherBlur(): void {
    const code = otherCode.trim();
    if (!code) return;
    void persistLanguage({ Other: code });
  }

  async function handleTranslateChange(checked: boolean): Promise<void> {
    try {
      error = null;
      await persist((cfg) => ({ ...cfg, asr: { ...cfg.asr, translate: checked } }));
      onRouterInputChange?.();
    } catch (err) {
      error = toLensError(err).message;
    }
  }

  // Only the cases that end up executing on Local Whisper need a live disk probe —
  // it directly (local_whisper) or via the translate reroute (apple_native). Every
  // other capability reads statically from ASR_CAPABILITY_MATRIX.
  let whisperPresent = $state<boolean | null>(null);
  $effect(() => {
    const engine = activeEngine;
    const translateOn = translate;
    const modelId = appConfigStore.asr?.whisper_model ?? 'base';
    if ((engine !== 'apple_native' && engine !== 'local_whisper') || !translateOn) {
      whisperPresent = null;
      return;
    }
    let cancelled = false;
    void whisperModelDownloaded(modelId).then((present) => {
      if (!cancelled) whisperPresent = present;
    });
    return () => {
      cancelled = true;
    };
  });

  interface CapabilityNotice {
    tone: 'info' | 'warning';
    text: string;
  }

  function computeNotice(
    engine: AsrBackend | null,
    translateOn: boolean,
    whisperOn: boolean | null
  ): CapabilityNotice | null {
    if (engine === null) return null;
    const mode = asrCapability(engine, 'translate');
    if (mode === null) return null;
    if (mode === 'reroutes') {
      if (!translateOn) {
        return { tone: 'info', text: appleTranslateRerouteNotice(true) };
      }
      return { tone: 'warning', text: appleTranslateRerouteNotice(whisperOn ?? false) };
    }
    if (mode === 'ignored') {
      return {
        tone: 'info',
        text: 'This engine ignores translate — audio is transcribed in its spoken language regardless.'
      };
    }
    return translateOn && whisperOn === false
      ? { tone: 'warning', text: whisperTranslateNotice(false) }
      : { tone: 'info', text: whisperTranslateNotice(true) };
  }

  const capabilityNotice = $derived(computeNotice(activeEngine, translate, whisperPresent));
</script>

<section class="flex flex-col gap-4" aria-label="Transcription language">
  <div>
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Language
    </p>
    <p class="mt-1 text-[0.68rem] leading-relaxed text-muted-foreground">
      Applies to every transcription engine.
    </p>

    <div class="mt-2 flex flex-col gap-1.5">
      <label for="asr-language" class="text-[0.72rem] font-bold text-foreground">
        Spoken language
      </label>
      <Select
        type="single"
        value={uiToken}
        onValueChange={(v) => {
          if (v) handleLanguageSelect(v);
        }}
        items={selectItems}
      >
        <SelectTrigger id="asr-language" class="w-full">
          <SelectValue placeholder="Auto-detect" />
        </SelectTrigger>
        <SelectContent
          class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
        >
          {#each selectItems as item (item.value)}
            <SelectItem value={item.value} label={item.label}>{item.label}</SelectItem>
          {/each}
        </SelectContent>
      </Select>
      {#if showOtherInput}
        <Input
          id="asr-language-other"
          type="text"
          bind:value={otherCode}
          placeholder="Language code, e.g. ar"
          aria-label="Language code"
          autocomplete="off"
          onblur={handleOtherBlur}
        />
      {/if}
    </div>
  </div>

  {#if error}
    <p class="text-[0.72rem] text-destructive" role="alert">{error}</p>
  {/if}

  <label
    class="flex cursor-pointer items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5 transition-colors hover:border-border/80"
  >
    <span class="min-w-0 flex-1">
      <span class="block text-[0.78rem] font-bold text-foreground">Translate to English</span>
      <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
        Ask the engine to translate spoken audio into English instead of transcribing it verbatim.
      </span>
    </span>
    <Switch
      checked={translate}
      onCheckedChange={handleTranslateChange}
      aria-label="Translate to English"
    />
  </label>

  {#if capabilityNotice}
    <div
      class={cn(
        'flex items-start gap-2 rounded-lg border px-3 py-2.5 text-xs',
        capabilityNotice.tone === 'warning'
          ? 'border-destructive/25 bg-destructive/5 text-destructive'
          : 'border-border bg-muted/40 text-muted-foreground'
      )}
      role={capabilityNotice.tone === 'warning' ? 'alert' : 'status'}
    >
      {#if capabilityNotice.tone === 'warning'}
        <TriangleAlert class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      {:else}
        <Info class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      {/if}
      <p>{capabilityNotice.text}</p>
    </div>
  {/if}
</section>
