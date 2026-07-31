<!-- Audio Overview setup modal (#29 redesign). Hands a resolved OverviewSetup to the
     caller (which owns the generate lifecycle). Format suggests a length that the user
     can override; once overridden, switching format no longer moves it. The Language
     picker hides when the active TTS engine is single-language. -->
<script lang="ts">
  import type { Component } from 'svelte';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import WandSparkles from '@lucide/svelte/icons/wand-sparkles';
  import Telescope from '@lucide/svelte/icons/telescope';
  import Zap from '@lucide/svelte/icons/zap';
  import Scale from '@lucide/svelte/icons/scale';
  import Swords from '@lucide/svelte/icons/swords';
  import Languages from '@lucide/svelte/icons/languages';
  import Check from '@lucide/svelte/icons/check';
  import FileText from '@lucide/svelte/icons/file-text';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription
  } from '$lib/components/ui/dialog/index.js';
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';
  import { overviewLanguageOptions } from '$lib/onboarding/system-check.js';
  import { suggestOverviewFocus } from '$lib/sources/audio-ipc.js';
  import type { Length, OverviewFormat, OverviewSetup } from '$lib/sources/audio-ipc.js';

  let {
    open = $bindable(false),
    selectedCount = 0,
    notebookId = null,
    onGenerate
  }: {
    open?: boolean;
    selectedCount?: number;
    notebookId?: string | null;
    onGenerate: (setup: OverviewSetup) => void;
  } = $props();

  type FormatOption = {
    value: OverviewFormat;
    label: string;
    tag: string;
    desc: string;
    defaultLength: Length;
    icon: Component;
  };

  const FORMATS: FormatOption[] = [
    {
      value: 'deep_dive',
      label: 'Deep dive',
      tag: 'Thorough & exploratory',
      desc: 'A longer two-host conversation that explores the sources thoroughly, with analysis and back-and-forth.',
      defaultLength: 'medium',
      icon: Telescope
    },
    {
      value: 'brief',
      label: 'Brief',
      tag: 'Short & high-signal',
      desc: 'A short, high-signal rundown that gets straight to the essentials.',
      defaultLength: 'short',
      icon: Zap
    },
    {
      value: 'critique',
      label: 'Critique',
      tag: 'Strengths & gaps',
      desc: 'A critical evaluation — weighing strengths, weaknesses, gaps and open questions rather than just summarizing.',
      defaultLength: 'medium',
      icon: Scale
    },
    {
      value: 'debate',
      label: 'Debate',
      tag: 'Opposing positions',
      desc: 'Two hosts argue opposing positions, pressure-testing each claim from the sources.',
      defaultLength: 'long',
      icon: Swords
    }
  ];

  const LENGTHS: { value: Length; label: string; hint: string }[] = [
    { value: 'short', label: 'Short', hint: '~5 min' },
    { value: 'medium', label: 'Medium', hint: '~10 min' },
    { value: 'long', label: 'Long', hint: '~15 min' }
  ];

  // SYNC-CHECK: keys mirror lens-core tts::catalog::Lang (serde snake_case).
  const LANG_LABELS: Record<string, string> = {
    english: 'English',
    chinese: 'Chinese',
    german: 'German',
    italian: 'Italian',
    portuguese: 'Portuguese',
    spanish: 'Spanish',
    japanese: 'Japanese',
    korean: 'Korean',
    french: 'French',
    russian: 'Russian',
    dutch: 'Dutch',
    arabic: 'Arabic',
    hindi: 'Hindi'
  };

  let format = $state<OverviewFormat>('deep_dive');
  let length = $state<Length>('medium');
  /** Once the user picks a length, switching format no longer moves it (no "jumping"). */
  let lengthTouched = $state(false);
  /** '' = auto / source language (omitted from the request). */
  let language = $state('');
  let focus = $state('');
  let langOptions = $state<string[]>([]);
  let suggesting = $state(false);
  let suggestError = $state<string | null>(null);

  const selectedFormat = $derived(FORMATS.find((f) => f.value === format) ?? FORMATS[0]);
  const scopeLabel = $derived(`${selectedCount} source${selectedCount === 1 ? '' : 's'}`);
  const lengthIsSuggested = $derived(!lengthTouched && length === selectedFormat.defaultLength);

  const languageItems = $derived([
    { value: '', label: 'Auto — match sources' },
    ...langOptions.map((id) => {
      const label = LANG_LABELS[id] ?? id;
      return { value: label, label };
    })
  ]);

  // Reset to a clean setup and (re)load the active engine's languages each open.
  $effect(() => {
    if (!open) return;
    format = 'deep_dive';
    length = 'medium';
    lengthTouched = false;
    language = '';
    focus = '';
    suggesting = false;
    suggestError = null;
    void (async () => {
      try {
        langOptions = await overviewLanguageOptions();
      } catch (err) {
        console.error('AudioOverviewSetup: language load failed', err);
        langOptions = [];
      }
    })();
  });

  function pickFormat(f: FormatOption): void {
    format = f.value;
    // The default is only a suggestion — honour an explicit length the user already set.
    if (!lengthTouched) length = f.defaultLength;
  }

  function pickLength(value: Length): void {
    length = value;
    lengthTouched = true;
  }

  async function suggest(): Promise<void> {
    if (!notebookId || suggesting) return;
    suggesting = true;
    suggestError = null;
    try {
      const phrase = await suggestOverviewFocus(notebookId);
      const trimmed = phrase.trim();
      if (trimmed) focus = trimmed;
      else suggestError = 'No suggestion came back — add your own focus below.';
    } catch (err) {
      console.error('AudioOverviewSetup: suggest focus failed', err);
      suggestError = "Couldn't reach the model. Add your own focus, or check it in Settings.";
    } finally {
      suggesting = false;
    }
  }

  function onFocusInput(): void {
    if (suggestError) suggestError = null;
  }

  function generate(): void {
    const trimmed = focus.trim();
    onGenerate({
      format,
      length,
      language: language || undefined,
      focus: trimmed || undefined
    });
    open = false;
  }

  function cancel(): void {
    open = false;
  }
</script>

<Dialog {open} onOpenChange={(v) => (open = v)}>
  <DialogContent
    showCloseButton={true}
    class="flex max-h-[85vh] w-full flex-col gap-0 overflow-hidden rounded-2xl border-border bg-card p-0 sm:max-w-3xl"
  >
    <DialogHeader class="flex-row items-center gap-3 px-6 py-4 space-y-0 text-left shrink-0">
      <div class="header-icon" aria-hidden="true">
        <Sparkles class="size-[19px]" strokeWidth={2} />
      </div>
      <div class="flex min-w-0 flex-col gap-1">
        <DialogTitle class="text-base font-bold leading-none text-foreground">
          Audio Overview
        </DialogTitle>
        <DialogDescription class="text-[0.75rem] leading-tight text-muted-foreground">
          Two AI hosts discuss your sources — grounded in {scopeLabel}
        </DialogDescription>
      </div>
    </DialogHeader>

    <div class="flex min-h-0 flex-1 overflow-hidden">
      <div class="format-rail no-scrollbar">
        <p class="section-label">Format</p>
        <div class="flex flex-col gap-1.5">
          {#each FORMATS as opt (opt.value)}
            <button
              type="button"
              class="fmt-row"
              data-active={format === opt.value}
              aria-label={opt.label}
              aria-pressed={format === opt.value}
              onclick={() => pickFormat(opt)}
            >
              <span class="fmt-row-icon" aria-hidden="true">
                <opt.icon class="size-[18px]" strokeWidth={1.9} />
              </span>
              <span class="fmt-row-text">
                <span class="fmt-row-label">{opt.label}</span>
                <span class="fmt-row-tag">{opt.tag}</span>
              </span>
              <span class="fmt-row-check" aria-hidden="true">
                {#if format === opt.value}<Check class="size-3" strokeWidth={3} />{/if}
              </span>
            </button>
          {/each}
        </div>
      </div>

      <div class="config-col">
        <div class="config-pane no-scrollbar">
          <div class="fmt-desc-card">
            <span class="fmt-desc-icon" aria-hidden="true">
              <selectedFormat.icon class="size-4" strokeWidth={1.9} />
            </span>
            <p>{selectedFormat.desc}</p>
          </div>

          <section>
            <p class="section-label">
              Length
              {#if lengthIsSuggested}
                <span class="suggested-chip">Suggested</span>
              {/if}
            </p>
            <div class="len-group">
              {#each LENGTHS as opt (opt.value)}
                <button
                  type="button"
                  class="len-btn"
                  data-active={length === opt.value}
                  aria-label={opt.label}
                  aria-pressed={length === opt.value}
                  onclick={() => pickLength(opt.value)}
                >
                  <span class="len-label">{opt.label}</span>
                  <span class="len-hint">{opt.hint}</span>
                </button>
              {/each}
            </div>
          </section>

          {#if langOptions.length > 0}
            <section>
              <p id="overview-language-label" class="section-label">
                <Languages class="size-3.5 text-muted-foreground" strokeWidth={2} />
                Language
              </p>
              <Select
                type="single"
                value={language}
                onValueChange={(v) => (language = v ?? '')}
                items={languageItems}
              >
                <SelectTrigger
                  id="overview-language"
                  aria-labelledby="overview-language-label"
                  class="h-10 w-full rounded-xl"
                >
                  <SelectValue placeholder="Auto — match sources" />
                </SelectTrigger>
                <SelectContent
                  class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
                >
                  {#each languageItems as opt (opt.value)}
                    <SelectItem value={opt.value} label={opt.label}>{opt.label}</SelectItem>
                  {/each}
                </SelectContent>
              </Select>
            </section>
          {/if}

          <section>
            <div class="mb-2 flex items-center justify-between">
              <label for="overview-focus" class="section-label mb-0">
                Focus <span class="font-medium normal-case tracking-normal text-muted-foreground/70"
                  >· optional</span
                >
              </label>
              <button
                type="button"
                class="suggest-btn"
                disabled={suggesting || !notebookId}
                aria-busy={suggesting}
                onclick={suggest}
              >
                {#if suggesting}
                  <span class="suggest-spinner" aria-hidden="true"></span>
                {:else}
                  <WandSparkles class="size-3" strokeWidth={2} />
                {/if}
                {suggesting ? 'Suggesting…' : 'Suggest'}
              </button>
            </div>
            <textarea
              id="overview-focus"
              bind:value={focus}
              oninput={onFocusInput}
              placeholder="What the hosts should focus on — or tap Suggest for topics drawn from your sources."
              rows="5"
              class="focus-area"
            ></textarea>
            {#if suggestError}
              <p
                class="mt-1.5 flex items-center gap-1.5 text-[0.7rem] text-destructive"
                role="alert"
              >
                <TriangleAlert class="size-3 shrink-0" strokeWidth={2} />
                {suggestError}
              </p>
            {/if}
          </section>
        </div>

        <div class="config-footer">
          <span class="scope-pill">
            <FileText class="size-3.5" strokeWidth={2} />
            {scopeLabel}
          </span>
          <div class="flex items-center gap-2">
            <button type="button" class="ghost-btn" onclick={cancel}>Cancel</button>
            <button type="button" class="generate-btn" onclick={generate}>
              <Sparkles class="size-[14px]" strokeWidth={2} />
              Generate
            </button>
          </div>
        </div>
      </div>
    </div>
  </DialogContent>
</Dialog>

<style>
  .header-icon {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    flex: none;
    border-radius: 11px;
    color: var(--primary-foreground);
    background: linear-gradient(
      135deg,
      var(--primary),
      color-mix(in oklch, var(--primary) 78%, black)
    );
    box-shadow: 0 4px 12px -4px color-mix(in oklch, var(--primary) 55%, transparent);
  }

  .section-label {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 9px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.055em;
    text-transform: uppercase;
    color: var(--muted-foreground);
  }

  .suggested-chip {
    text-transform: none;
    letter-spacing: 0;
    font-size: 0.62rem;
    font-weight: 600;
    padding: 1px 7px;
    border-radius: 999px;
    color: var(--primary);
    background: color-mix(in oklch, var(--primary) 13%, transparent);
  }

  /* Two-pane body: a format chooser rail on the left, the run config on the right. */
  .format-rail {
    flex: none;
    width: 15.5rem;
    padding: 18px 16px;
    background: color-mix(in oklch, var(--muted) 55%, transparent);
    overflow-y: auto;
  }
  /* Right column: scrolling config above, a pinned footer below — so the footer never
     spans the format rail, letting the rail run the full height of the modal. */
  .config-col {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
  }
  .config-pane {
    display: flex;
    flex-direction: column;
    gap: 18px;
    flex: 1;
    min-height: 0;
    padding: 18px 22px 18px;
    overflow-y: auto;
  }
  .config-footer {
    display: flex;
    flex: none;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 22px 16px;
  }

  /* Format rows — horizontal cards (icon · label + tag · check). Selected reads as an
     accent-filled row, so the chosen format is unmistakable at a glance. */
  .fmt-row {
    display: flex;
    align-items: center;
    gap: 11px;
    width: 100%;
    padding: 10px 11px;
    border-radius: 12px;
    border: 1.5px solid transparent;
    background: transparent;
    cursor: pointer;
    text-align: left;
    transition:
      border-color 0.16s var(--ease-out, ease),
      background 0.16s var(--ease-out, ease),
      box-shadow 0.16s var(--ease-out, ease);
  }
  .fmt-row:hover:not([data-active='true']) {
    background: color-mix(in oklch, var(--foreground) 5%, transparent);
  }
  .fmt-row[data-active='true'] {
    border-color: var(--primary);
    background: color-mix(in oklch, var(--primary) 11%, var(--card));
    box-shadow: var(--shadow-tile);
  }
  .fmt-row:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }
  .fmt-row-icon {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    flex: none;
    border-radius: 10px;
    color: var(--muted-foreground);
    background: color-mix(in oklch, var(--foreground) 6%, transparent);
    transition:
      color 0.16s var(--ease-out, ease),
      background 0.16s var(--ease-out, ease);
  }
  .fmt-row[data-active='true'] .fmt-row-icon {
    color: var(--primary);
    background: color-mix(in oklch, var(--primary) 16%, transparent);
  }
  .fmt-row-text {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
    flex: 1;
  }
  .fmt-row-label {
    font-size: 0.82rem;
    font-weight: 650;
    color: var(--foreground);
  }
  .fmt-row-tag {
    font-size: 0.66rem;
    line-height: 1.25;
    color: var(--muted-foreground);
  }
  .fmt-row-check {
    display: grid;
    place-items: center;
    width: 18px;
    height: 18px;
    flex: none;
    border-radius: 999px;
    color: var(--primary-foreground);
    background: var(--primary);
    opacity: 0;
    transform: scale(0.6);
    transition:
      opacity 0.16s var(--ease-out, ease),
      transform 0.16s var(--ease-spring, ease);
  }
  .fmt-row[data-active='true'] .fmt-row-check {
    opacity: 1;
    transform: scale(1);
  }

  /* Selected-format blurb at the top of the config pane. */
  .fmt-desc-card {
    display: flex;
    gap: 11px;
    padding: 13px 15px;
    border-radius: 12px;
    background: color-mix(in oklch, var(--primary) 7%, transparent);
  }
  .fmt-desc-icon {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    flex: none;
    border-radius: 8px;
    color: var(--primary);
    background: color-mix(in oklch, var(--primary) 14%, transparent);
  }
  .fmt-desc-card p {
    font-size: 0.75rem;
    line-height: 1.5;
    color: var(--foreground);
  }

  /* Length — segmented control; each cell shows a label + rough duration hint. */
  .len-group {
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: 1fr;
    gap: 4px;
    padding: 4px;
    border-radius: 12px;
    background: var(--muted);
  }
  .len-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
    padding: 6px 4px;
    border: 0;
    border-radius: 9px;
    background: transparent;
    cursor: pointer;
    color: var(--muted-foreground);
    transition:
      color 0.16s var(--ease-out, ease),
      background 0.16s var(--ease-out, ease),
      box-shadow 0.16s var(--ease-out, ease);
  }
  .len-btn[data-active='true'] {
    background: var(--card);
    box-shadow: var(--shadow-tile);
  }
  .len-btn:not([data-active='true']):hover {
    color: var(--foreground);
  }
  .len-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }
  .len-label {
    font-size: 0.76rem;
    font-weight: 650;
  }
  .len-btn[data-active='true'] .len-label {
    color: var(--card-foreground);
  }
  .len-hint {
    font-size: 0.62rem;
    color: var(--muted-foreground);
    font-variant-numeric: tabular-nums;
  }

  .suggest-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 26px;
    padding: 0 11px;
    border: 0;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 650;
    cursor: pointer;
    color: var(--primary);
    background: color-mix(in oklch, var(--primary) 12%, transparent);
    transition:
      background 0.16s var(--ease-out, ease),
      opacity 0.16s var(--ease-out, ease);
  }
  .suggest-btn:hover:not(:disabled) {
    background: color-mix(in oklch, var(--primary) 20%, transparent);
  }
  .suggest-btn:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }
  .suggest-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }

  .focus-area {
    width: 100%;
    resize: none;
    border-radius: 12px;
    border: 1.5px solid var(--border);
    background: var(--background);
    padding: 10px 12px;
    font-size: 0.78rem;
    line-height: 1.55;
    color: var(--foreground);
    outline: none;
    transition:
      border-color 0.16s var(--ease-out, ease),
      box-shadow 0.16s var(--ease-out, ease);
  }
  .focus-area::placeholder {
    color: color-mix(in oklch, var(--muted-foreground) 70%, transparent);
  }
  .focus-area:focus-visible {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--primary) 18%, transparent);
  }

  .generate-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 33px;
    padding: 0 15px;
    border: 0;
    border-radius: 9px;
    font-size: 0.8rem;
    font-weight: 650;
    cursor: pointer;
    color: var(--primary-foreground);
    background: var(--primary);
    box-shadow: 0 4px 12px -4px color-mix(in oklch, var(--primary) 55%, transparent);
    transition:
      opacity 0.16s var(--ease-out, ease),
      transform 0.16s var(--ease-out, ease);
  }
  .generate-btn:hover {
    opacity: 0.94;
  }
  .generate-btn:active {
    transform: scale(calc(1 - 0.03 * var(--rail-motion, 1)));
  }
  .generate-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }

  .ghost-btn {
    height: 33px;
    padding: 0 14px;
    border: 1px solid var(--border);
    border-radius: 9px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    color: var(--muted-foreground);
    background: var(--card);
    transition:
      color 0.16s var(--ease-out, ease),
      background 0.16s var(--ease-out, ease);
  }
  .ghost-btn:hover {
    color: var(--foreground);
    background: color-mix(in oklch, var(--foreground) 5%, var(--card));
  }
  .ghost-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }

  .scope-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 28px;
    padding: 0 12px 0 10px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    color: var(--muted-foreground);
    background: color-mix(in oklch, var(--foreground) 6%, transparent);
  }

  .suggest-spinner {
    width: 0.7rem;
    height: 0.7rem;
    border-radius: 50%;
    border: 2px solid color-mix(in oklch, var(--primary) 30%, transparent);
    border-top-color: var(--primary);
    animation: suggest-spin 0.7s linear infinite;
  }
  @keyframes suggest-spin {
    to {
      transform: rotate(360deg);
    }
  }
  /* Calm mode / reduced-motion freezes the spinner (still visible as a ring),
     matching the app's --rail-motion gate. */
  :global(:root[data-motion='off']) .suggest-spinner {
    animation: none;
  }
  @media (prefers-reduced-motion: reduce) {
    :global(:root:not([data-motion='on'])) .suggest-spinner {
      animation: none;
    }
  }
</style>
