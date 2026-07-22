<script lang="ts">
  import type { CheckResult, SaveApi } from '$lib/onboarding/system-check.js';
  import OnboardingLlmPicker from './OnboardingLlmPicker.svelte';
  import OnboardingEmbeddingPicker from './OnboardingEmbeddingPicker.svelte';

  let {
    result,
    oncheck,
    onready
  }: {
    result: CheckResult;
    /** Re-run the parent system check after the embedding picker persists a default. */
    oncheck?: () => Promise<void>;
    /** Forwarded from the LLM picker so the step footer can drive Save & continue. */
    onready?: (api: SaveApi) => void;
  } = $props();
</script>

<!-- Both readiness gates render an always-visible, purpose-built picker inline —
     no status-badge tile, no Choose/expand step. The row id selects which. -->
{#if result.id === 'llm_runtime'}
  <OnboardingLlmPicker {onready} />
{:else}
  <OnboardingEmbeddingPicker oncheck={oncheck ?? (() => Promise.resolve())} />
{/if}
