<script lang="ts">
  import Check from '@lucide/svelte/icons/check';
  import X from '@lucide/svelte/icons/x';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import { Card } from '$lib/components/ui/card/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import { expandFade } from '$lib/motion/index.js';
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

  // Status → icon-badge treatment. Each row is a binary readiness gate: Pass
  // (green primary) or Fail (destructive). An unknown status falls back to the
  // Fail treatment so an unexpected value can never masquerade as a Pass.
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
    }
  } as const;

  // Safe fallback for any status outside the known union — never the Pass view.
  const FALLBACK_VIEW = STATUS.fail;

  const view = $derived(STATUS[result.status] ?? FALLBACK_VIEW);
  const StatusIcon = $derived(view.icon);

  const ACTION_LABEL: Record<CheckAction, string> = {
    configure: 'Configure',
    choose: 'Choose'
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

  // Expandable rows are available (they expand inline). All other
  // configure/choose actions are disabled (Settings not built yet).
  const available = $derived(isExpandable);

  let expanded = $state(false);

  function toggleExpanded(): void {
    expanded = !expanded;
  }

  // Uniform border on every row (the Card default ring). Per design, a failed
  // or actionable row is differentiated ONLY by its icon badge + label color,
  // never by a heavier border — so no per-row ring override here.
  //
  // Always column-stretch so the header row's layout is IDENTICAL whether
  // collapsed or expanded — clicking the action button only reveals the panel
  // below, it never reflows the header (no button "jump"). gap-0: the panels
  // bring their own top border/padding.
  const cardClass = 'flex-col items-stretch gap-0 px-4 py-3';
</script>

<Card size="sm" class={cardClass}>
  <!-- Row header: always visible; w-full + identical layout in both states -->
  <div class="flex w-full items-center gap-3">
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
      <Button
        variant="outline"
        size="sm"
        class="shrink-0"
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
        {#if isExpandable && expanded}
          <ChevronUp />
        {:else}
          <ChevronDown />
        {/if}
      </Button>
    {/if}
  </div>

  <!-- Inline expansion panels: lazy-mounted, so collapsed content stays out of
       the DOM + keyboard focus order (a11y). One coordinated height+opacity
       tween (expandFade) animates open AND close — no competing animations. -->
  {#if isExpandable && expanded}
    <div transition:expandFade>
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
    </div>
  {/if}
</Card>
