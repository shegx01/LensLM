<!--
  EnrichmentOverride — collapsed "Advanced: separate enrichment model" disclosure.
  Unset/collapsed ⇒ enrichment follows the chat entry (the Explicit routing pin);
  set ⇒ pins BOTH coref_model and map_model to the same model. Clearing writes null
  for both. Emits the chosen TaskModel (or null); the panel owns persistence.
-->
<script lang="ts">
  import { untrack } from 'svelte';
  import { cn } from '$lib/utils.js';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import { SELECT_CLASS } from '$lib/components/onboarding/styles.js';
  import type { TaskModel } from '$lib/theme/types.js';
  import type { ModelOption } from '$lib/models/catalog.js';

  let {
    value = null,
    options = [],
    providerId,
    onchange
  }: {
    value?: TaskModel | null;
    options?: ModelOption[];
    providerId: string;
    onchange: (model: TaskModel | null) => void;
  } = $props();

  // Start expanded only if a pin already exists; ignore later prop changes.
  let open = $state(untrack(() => value !== null));

  function onSelect(e: Event): void {
    const model = (e.currentTarget as HTMLSelectElement).value;
    onchange(model ? { provider: providerId, model } : null);
  }
</script>

<div class="mt-4 border-t border-border pt-4">
  <button
    type="button"
    aria-expanded={open}
    onclick={() => (open = !open)}
    class="flex w-full items-center justify-between gap-2 text-left text-[0.78rem] font-semibold text-foreground"
  >
    <span>Advanced: separate enrichment model</span>
    <ChevronDown
      class={cn('size-4 transition-transform', open && 'rotate-180')}
      aria-hidden="true"
    />
  </button>

  {#if open}
    <div class="mt-3 flex flex-col gap-1.5">
      <label for="ai-enrichment-model" class="text-[0.68rem] font-medium text-muted-foreground">
        Enrichment model
      </label>
      <select
        id="ai-enrichment-model"
        value={value?.model ?? ''}
        onchange={onSelect}
        class={SELECT_CLASS}
      >
        <option value="">Use the chat model</option>
        {#each options as opt (opt.id)}
          <option value={opt.id}>{opt.label}</option>
        {/each}
      </select>
      <p class="text-[0.72rem] leading-relaxed text-muted-foreground">
        Pins coreference and structural mapping to a dedicated model. Leave on “Use the chat model”
        to let enrichment follow your chat selection.
      </p>
    </div>
  {/if}
</div>
