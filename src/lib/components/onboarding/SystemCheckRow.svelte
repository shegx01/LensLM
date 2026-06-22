<script lang="ts">
  import Check from '@lucide/svelte/icons/check';
  import X from '@lucide/svelte/icons/x';
  import Clock from '@lucide/svelte/icons/clock';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import { Card } from '$lib/components/ui/card/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import type { CheckResult, CheckAction } from '$lib/onboarding/system-check.js';
  import LlmConfigPanel from './LlmConfigPanel.svelte';
  import EmbeddingConfigPanel from './EmbeddingConfigPanel.svelte';
  import TtsConfigPanel from './TtsConfigPanel.svelte';

  let {
    result,
    onaction,
    oncheck
  }: {
    result: CheckResult;
    onaction?: (action: CheckAction) => void;
    /** Re-run the parent system check (for config panel's Save / Test). */
    oncheck?: () => Promise<void>;
  } = $props();

  // Status → icon-badge treatment. Pending is DELIBERATELY distinct from Pass
  // (plan change #13, HARD GATE): a muted/neutral clock on a muted surface with
  // muted-foreground text — never the green `text-primary`/`bg-primary` of Pass.
  const STATUS = {
    pass: {
      icon: Check,
      badgeClass: 'bg-primary/15 text-primary',
      labelClass: ''
    },
    fail: {
      icon: X,
      badgeClass: 'bg-destructive/15 text-destructive',
      labelClass: 'text-destructive'
    },
    pending: {
      icon: Clock,
      badgeClass: 'bg-muted text-muted-foreground',
      labelClass: ''
    }
  } as const;

  // Neutral fallback for any status outside the known union.
  const FALLBACK_VIEW = STATUS.pending;

  const view = $derived(STATUS[result.status] ?? FALLBACK_VIEW);
  const StatusIcon = $derived(view.icon);

  const ACTION_LABEL: Record<CheckAction, string> = {
    configure: 'Configure',
    choose: 'Choose',
    retry: 'Retry'
  };

  // Expandable rows:
  //   llm_runtime  + configure → LlmConfigPanel
  //   embedding_model + choose → EmbeddingConfigPanel
  //   text_to_speech  + choose → TtsConfigPanel
  const isExpandable = $derived(
    (result.id === 'llm_runtime' && result.action === 'configure') ||
      (result.id === 'embedding_model' && result.action === 'choose') ||
      (result.id === 'text_to_speech' && result.action === 'choose')
  );

  // `retry` is always live. Expandable rows are available (they expand inline).
  // All other configure/choose actions are disabled (Settings not built yet).
  const isAvailable = (action: CheckAction): boolean => action === 'retry' || isExpandable;

  let expanded = $state(false);

  function toggleExpanded(): void {
    expanded = !expanded;
  }

  const needsEmphasis = $derived(result.status === 'fail' || result.action !== null);

  const cardClass = $derived(
    cn(
      expanded ? 'flex-col items-stretch gap-0 px-4 py-3' : 'flex-row items-center gap-3 px-4 py-3',
      needsEmphasis && 'ring-foreground/20'
    )
  );
</script>

<Card size="sm" class={cardClass}>
  <!-- Row header: always visible -->
  <div class={cn('flex items-center gap-3', expanded && 'w-full')}>
    <span
      class={cn(
        'flex size-8 shrink-0 items-center justify-center rounded-full [&_svg]:size-4',
        view.badgeClass
      )}
      aria-hidden="true"
    >
      <StatusIcon />
    </span>

    <div class="min-w-0 flex-1">
      <p class={cn('truncate text-sm font-bold', view.labelClass)}>{result.label}</p>
      <p class="text-muted-foreground truncate text-[0.8rem]">{result.detail}</p>
    </div>

    {#if result.action}
      {@const action = result.action}
      {@const available = isAvailable(action)}
      <Button
        variant="outline"
        size="sm"
        disabled={!available}
        title={available ? undefined : 'Available in Settings'}
        aria-expanded={isExpandable ? expanded : undefined}
        onclick={() => {
          if (!available) return;
          if (isExpandable) {
            toggleExpanded();
          } else {
            onaction?.(action);
          }
        }}
      >
        {ACTION_LABEL[action]}
        {#if action === 'retry'}
          <RefreshCw />
        {:else if isExpandable}
          {#if expanded}
            <ChevronUp />
          {:else}
            <ChevronDown />
          {/if}
        {:else}
          <ChevronDown />
        {/if}
      </Button>
    {/if}
  </div>

  <!-- Inline expansion panels -->
  {#if isExpandable && expanded}
    {#if result.id === 'llm_runtime'}
      <LlmConfigPanel
        oncheck={oncheck ?? (() => Promise.resolve())}
        oncollapse={() => (expanded = false)}
      />
    {:else if result.id === 'embedding_model'}
      <EmbeddingConfigPanel
        oncheck={oncheck ?? (() => Promise.resolve())}
        oncollapse={() => (expanded = false)}
      />
    {:else if result.id === 'text_to_speech'}
      <TtsConfigPanel
        oncheck={oncheck ?? (() => Promise.resolve())}
        oncollapse={() => (expanded = false)}
      />
    {/if}
  {/if}
</Card>
