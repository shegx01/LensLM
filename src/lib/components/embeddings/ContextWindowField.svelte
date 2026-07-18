<!--
  ContextWindowField — universal context-window selector (presets + custom tokens).
  Persists to ModelConfig.context (token budgeting), so it is always user-editable.
  When the catalog exposes a model limit it is shown as an advisory hint / default,
  never a hard cap (Ollama has no catalog limit and stands on presets + custom alone).
-->
<script lang="ts">
  import { cn } from '$lib/utils.js';

  let {
    value = $bindable(8192),
    hint = null,
    onchange
  }: {
    value?: number;
    /** Advisory catalog limit line, e.g. "128K context (catalog limit)". */
    hint?: string | null;
    onchange?: (tokens: number) => void;
  } = $props();

  const PRESETS = [
    { label: '4K', value: 4096 },
    { label: '8K', value: 8192 },
    { label: '16K', value: 16384 },
    { label: '32K', value: 32768 },
    { label: '128K', value: 131072 }
  ] as const;

  function pick(tokens: number): void {
    value = tokens;
    onchange?.(tokens);
  }

  function onCustom(e: Event): void {
    const v = parseInt((e.currentTarget as HTMLInputElement).value, 10);
    if (Number.isFinite(v) && v >= 256) {
      value = v;
      onchange?.(v);
    }
  }
</script>

<div class="flex flex-col gap-1.5">
  <span class="text-[0.68rem] font-medium text-muted-foreground">Context window</span>
  <div class="flex gap-1" role="group" aria-label="Context window size">
    {#each PRESETS as opt (opt.value)}
      <button
        type="button"
        onclick={() => pick(opt.value)}
        aria-pressed={value === opt.value}
        class={cn(
          'flex-1 rounded-md border px-2 py-1.5 text-[0.75rem] font-medium transition-colors',
          value === opt.value
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border bg-transparent text-muted-foreground hover:bg-muted hover:text-foreground'
        )}
      >
        {opt.label}
      </button>
    {/each}
  </div>
  <div class="mt-1 flex items-center gap-2">
    <input
      id="ai-context-custom"
      type="number"
      min="256"
      step="256"
      {value}
      oninput={onCustom}
      aria-label="Custom context window in tokens"
      class="h-8 w-full min-w-0 rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
    />
    <span class="shrink-0 text-[0.72rem] text-muted-foreground">tokens (custom)</span>
  </div>
  {#if hint}
    <p class="text-[0.72rem] leading-relaxed text-muted-foreground">{hint}</p>
  {/if}
</div>
