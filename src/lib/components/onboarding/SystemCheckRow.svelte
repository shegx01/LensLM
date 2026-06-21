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

  let {
    result,
    onaction,
    oncheck
  }: {
    result: CheckResult;
    onaction?: (action: CheckAction) => void;
    /** Re-run the parent system check (for LLM configure panel's Save). */
    oncheck?: () => Promise<void>;
  } = $props();

  // Status → icon-badge treatment. Pending is DELIBERATELY distinct from Pass
  // (plan change #13, HARD GATE): a muted/neutral clock on a muted surface with
  // muted-foreground text — never the green `text-primary`/`bg-primary` of Pass.
  // This is load-bearing for the honesty thesis (pre-mortem #2): a future flip
  // to green must be visually and test-detectable.
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

  // Neutral fallback for any status outside the known union (defensive against a
  // future/garbled IPC value): reuse the muted Pending treatment — NEVER the
  // Pass green — and avoid an `undefined`-class crash. Sharing the Pending view
  // is deliberate: an unknown status reads as "not affirmatively healthy".
  const FALLBACK_VIEW = STATUS.pending;

  const view = $derived(STATUS[result.status] ?? FALLBACK_VIEW);
  const StatusIcon = $derived(view.icon);

  // Action button copy + icon (Configure/Choose carry a chevron like the design;
  // Retry carries a refresh glyph).
  const ACTION_LABEL: Record<CheckAction, string> = {
    configure: 'Configure',
    choose: 'Choose',
    retry: 'Retry'
  };

  // The `configure` action on the `llm_runtime` row is the ONLY expandable
  // affordance in M1. All other actions (choose/retry) use the original
  // behaviour. `choose` stays disabled (Available in Settings). `retry` is live.
  const isExpandable = $derived(result.id === 'llm_runtime' && result.action === 'configure');

  // `configure`/`choose` open Settings, which is not built until a later
  // milestone. For non-llm_runtime rows we render them DISABLED with an
  // explanatory tooltip rather than shipping a button that silently does nothing.
  // `retry` is live (it re-runs the check via the parent's `onaction`).
  const isAvailable = (action: CheckAction): boolean =>
    action === 'retry' || (action === 'configure' && isExpandable);

  // Inline expand state — only meaningful when `isExpandable` is true.
  let expanded = $state(false);

  function toggleExpanded(): void {
    expanded = !expanded;
  }

  // Attention rows (fail status OR a row with an action affordance) get a
  // subtly stronger ring to match the design mock — token-based only.
  // Pass/Pending rows keep the Card default (ring-foreground/10).
  const needsEmphasis = $derived(result.status === 'fail' || result.action !== null);

  // The card switches from flex-row to flex-col when the panel is open so the
  // expansion renders below the row header.
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

  <!-- Inline LLM config panel — only renders when expanded -->
  {#if isExpandable && expanded}
    <LlmConfigPanel
      oncheck={oncheck ?? (() => Promise.resolve())}
      oncollapse={() => (expanded = false)}
    />
  {/if}
</Card>
