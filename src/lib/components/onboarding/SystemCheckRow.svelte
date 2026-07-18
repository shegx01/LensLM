<script lang="ts">
  import Check from '@lucide/svelte/icons/check';
  import X from '@lucide/svelte/icons/x';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import { Button } from '$lib/components/ui/button/index.js';
  import { expandFade } from '$lib/motion/index.js';
  import type { CheckResult, CheckAction, SaveApi } from '$lib/onboarding/system-check.js';
  import SystemCheckTile from './SystemCheckTile.svelte';
  import EmbeddingConfigPanel from './EmbeddingConfigPanel.svelte';
  import OnboardingLlmPicker from './OnboardingLlmPicker.svelte';

  let {
    result,
    onaction,
    oncheck,
    onready
  }: {
    result: CheckResult;
    onaction?: (action: CheckAction) => void;
    /** Re-run the parent system check (for the config panel's Save). */
    oncheck?: () => Promise<void>;
    /** Forwarded from the LLM picker so the step footer can drive Save & continue. */
    onready?: (api: SaveApi) => void;
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

  // Only embedding_model keeps the Choose→expand collapsible; llm_runtime renders
  // the always-visible picker directly (no Configure toggle).
  const isExpandable = $derived(result.id === 'embedding_model' && result.action === 'choose');

  let expanded = $state(false);

  function toggleExpanded(): void {
    expanded = !expanded;
  }
</script>

{#if result.id === 'llm_runtime'}
  <OnboardingLlmPicker {onready} />
{:else}
  <SystemCheckTile
    icon={StatusIcon}
    badgeClass={view.badgeClass}
    title={result.label}
    subtitle={result.detail}
    titleClass={view.labelClass}
  >
    {#snippet status()}
      {#if result.action}
        {@const action = result.action}
        <Button
          variant="outline"
          size="sm"
          class="shrink-0"
          disabled={!isExpandable}
          title={isExpandable ? undefined : 'Available in Settings'}
          aria-expanded={isExpandable ? expanded : undefined}
          aria-label={`${ACTION_LABEL[action]} ${result.label}`}
          onclick={() => {
            if (!isExpandable) return;
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
    {/snippet}

    <!-- Inline expansion panel: lazy-mounted, so collapsed content stays out of
         the DOM + keyboard focus order (a11y). One coordinated height+opacity
         tween (expandFade) animates open AND close. -->
    {#if isExpandable && expanded}
      <div transition:expandFade>
        <EmbeddingConfigPanel
          oncheck={oncheck ?? (() => Promise.resolve())}
          oncollapse={() => (expanded = false)}
        />
      </div>
    {/if}
  </SystemCheckTile>
{/if}
