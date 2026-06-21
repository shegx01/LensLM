<script lang="ts">
  import Check from '@lucide/svelte/icons/check';
  import X from '@lucide/svelte/icons/x';
  import Clock from '@lucide/svelte/icons/clock';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import RefreshCw from '@lucide/svelte/icons/refresh-cw';
  import { Card } from '$lib/components/ui/card/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import type { CheckResult, CheckAction } from '$lib/onboarding/system-check.js';

  let { result, onaction }: { result: CheckResult; onaction?: (action: CheckAction) => void } =
    $props();

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

  // `configure`/`choose` open Settings, which is not built until a later
  // milestone. We render them DISABLED with an explanatory tooltip rather than
  // shipping a button that silently does nothing. `retry` is live (it re-runs
  // the check via the parent's `onaction`).
  const isAvailable = (action: CheckAction): boolean => action === 'retry';

  // Attention rows (fail status OR a row with an action affordance) get a
  // subtly stronger ring to match the design mock — token-based only.
  // Pass/Pending rows keep the Card default (ring-foreground/10).
  const needsEmphasis = $derived(result.status === 'fail' || result.action !== null);
</script>

<Card
  size="sm"
  class={cn('flex-row items-center gap-3 px-4 py-3', needsEmphasis && 'ring-foreground/20')}
>
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
      onclick={() => available && onaction?.(action)}
    >
      {ACTION_LABEL[action]}
      {#if action === 'retry'}
        <RefreshCw />
      {:else}
        <ChevronDown />
      {/if}
    </Button>
  {/if}
</Card>
