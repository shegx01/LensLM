<!-- Audio Overview setup modal (#29 redesign). Collects Format / Length / Language /
     Focus, then hands a resolved OverviewSetup back to the caller which owns the
     generate lifecycle. Format sets a default length that stays adjustable; the
     Language picker is hidden when the active TTS engine is single-language. -->
<script lang="ts">
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import Headphones from '@lucide/svelte/icons/headphones';
  import Wand from '@lucide/svelte/icons/wand-sparkles';
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter
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
    desc: string;
    defaultLength: Length;
  };

  const FORMATS: FormatOption[] = [
    {
      value: 'deep_dive',
      label: 'Deep dive',
      desc: 'A longer two-host conversation that explores the sources thoroughly, with analysis and back-and-forth.',
      defaultLength: 'medium'
    },
    {
      value: 'brief',
      label: 'Brief',
      desc: 'A short, high-signal rundown that gets to the essentials fast.',
      defaultLength: 'short'
    },
    {
      value: 'critique',
      label: 'Critique',
      desc: 'A critical evaluation — weighing strengths, weaknesses, gaps and open questions rather than just summarizing.',
      defaultLength: 'medium'
    },
    {
      value: 'debate',
      label: 'Debate',
      desc: 'Two hosts argue opposing positions, pressure-testing each claim from the sources.',
      defaultLength: 'long'
    }
  ];

  const LENGTHS: { value: Length; label: string }[] = [
    { value: 'short', label: 'Short' },
    { value: 'medium', label: 'Medium' },
    { value: 'long', label: 'Long' }
  ];

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
  /** '' = auto / source language (omitted from the request). */
  let language = $state('');
  let focus = $state('');
  let langOptions = $state<string[]>([]);
  let suggesting = $state(false);

  const selectedFormat = $derived(FORMATS.find((f) => f.value === format) ?? FORMATS[0]);
  const scopeLabel = $derived(`${selectedCount} source${selectedCount === 1 ? '' : 's'}`);

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
    language = '';
    focus = '';
    suggesting = false;
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
    length = f.defaultLength;
  }

  async function suggest(): Promise<void> {
    if (!notebookId || suggesting) return;
    suggesting = true;
    try {
      const phrase = await suggestOverviewFocus(notebookId);
      if (phrase.trim()) focus = phrase.trim();
    } catch (err) {
      console.error('AudioOverviewSetup: suggest focus failed', err);
    } finally {
      suggesting = false;
    }
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
</script>

<Dialog {open} onOpenChange={(v) => (open = v)}>
  <DialogContent
    showCloseButton={true}
    class="flex max-h-[86vh] w-full max-w-md flex-col gap-0 overflow-hidden rounded-xl border-border bg-card p-0"
  >
    <DialogHeader
      class="flex-row items-center gap-2.5 border-b border-border px-5 py-4 space-y-0 text-left"
    >
      <div
        class="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary"
        aria-hidden="true"
      >
        <Headphones class="size-4" strokeWidth={2} />
      </div>
      <div class="flex min-w-0 flex-col">
        <DialogTitle class="text-sm font-bold text-foreground">Generate Audio Overview</DialogTitle>
        <DialogDescription class="text-[11px] text-muted-foreground">
          Grounded in {scopeLabel}
        </DialogDescription>
      </div>
    </DialogHeader>

    <div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
      <fieldset class="mb-5">
        <legend class="mb-2 text-[0.7rem] font-bold uppercase tracking-wide text-muted-foreground">
          Format
        </legend>
        <div class="flex flex-wrap gap-1.5">
          {#each FORMATS as opt (opt.value)}
            <button
              type="button"
              class="pill"
              data-active={format === opt.value}
              aria-pressed={format === opt.value}
              onclick={() => pickFormat(opt)}
            >
              {opt.label}
            </button>
          {/each}
        </div>
        <p class="mt-2 text-[0.72rem] leading-relaxed text-muted-foreground">
          {selectedFormat.desc}
        </p>
      </fieldset>

      <fieldset class="mb-5">
        <legend class="mb-2 text-[0.7rem] font-bold uppercase tracking-wide text-muted-foreground">
          Length
        </legend>
        <div class="flex flex-wrap gap-1.5">
          {#each LENGTHS as opt (opt.value)}
            <button
              type="button"
              class="pill flex-1"
              data-active={length === opt.value}
              aria-pressed={length === opt.value}
              onclick={() => (length = opt.value)}
            >
              {opt.label}
            </button>
          {/each}
        </div>
      </fieldset>

      {#if langOptions.length > 0}
        <div class="mb-5 flex flex-col gap-2">
          <span class="text-[0.7rem] font-bold uppercase tracking-wide text-muted-foreground">
            Language
          </span>
          <Select
            type="single"
            value={language}
            onValueChange={(v) => (language = v ?? '')}
            items={languageItems}
          >
            <SelectTrigger class="w-full">
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
        </div>
      {/if}

      <div class="flex flex-col gap-2">
        <div class="flex items-center justify-between">
          <span class="text-[0.7rem] font-bold uppercase tracking-wide text-muted-foreground">
            Focus <span class="font-medium normal-case tracking-normal text-muted-foreground/70"
              >· optional</span
            >
          </span>
          <button
            type="button"
            class="inline-flex items-center gap-1.5 rounded-full bg-primary/12 px-2.5 py-1 text-[0.7rem] font-semibold text-primary transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={suggesting || !notebookId}
            onclick={suggest}
          >
            {#if suggesting}
              <span class="suggest-spinner" aria-hidden="true"></span>
            {:else}
              <Wand class="size-3" strokeWidth={2} />
            {/if}
            Suggest
          </button>
        </div>
        <textarea
          bind:value={focus}
          placeholder="e.g. keep it executive-level, lead with the numbers"
          rows="3"
          class="w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-xs leading-relaxed text-foreground outline-none placeholder:text-muted-foreground/60 focus-visible:ring-2 focus-visible:ring-ring"
        ></textarea>
      </div>
    </div>

    <DialogFooter
      class="flex-row items-center justify-between gap-2 border-t border-border px-5 py-3.5 space-x-0"
    >
      <span class="text-[0.7rem] text-muted-foreground">{scopeLabel}</span>
      <button
        type="button"
        class="press inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-[opacity,transform] hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onclick={generate}
      >
        <Sparkles class="size-[13px]" strokeWidth={2} />
        Generate
      </button>
    </DialogFooter>
  </DialogContent>
</Dialog>

<style>
  .pill {
    min-width: 4.5rem;
    height: 2.25rem;
    padding: 0 0.85rem;
    border-radius: 0.55rem;
    border: 1px solid var(--border);
    background: var(--muted);
    color: var(--muted-foreground);
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    transition:
      background 0.15s var(--ease-out, ease),
      color 0.15s var(--ease-out, ease),
      border-color 0.15s var(--ease-out, ease);
  }
  .pill:hover:not([data-active='true']) {
    color: var(--foreground);
    border-color: color-mix(in oklch, var(--foreground) 20%, transparent);
  }
  .pill[data-active='true'] {
    background: var(--primary);
    border-color: var(--primary);
    color: var(--primary-foreground);
  }
  .pill:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }
  .press:active {
    transform: scale(calc(1 - 0.03 * var(--rail-motion, 1)));
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
