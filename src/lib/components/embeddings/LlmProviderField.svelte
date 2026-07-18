<!--
  LlmProviderField — Settings-local chat-provider selector: a Local/Cloud segmented
  toggle plus, in Cloud mode, the full `CLOUD_PROVIDERS` list (including the custom
  OpenAI-compatible endpoint). Emits the canonical provider id the rest of the panel
  persists. Settings-only — onboarding uses a curated 3-preset picker instead.
-->
<script lang="ts">
  import { cn } from '$lib/utils.js';
  import { SELECT_CLASS } from '$lib/components/onboarding/styles.js';
  import { CLOUD_PROVIDERS, CLOUD_PROVIDER_IDS } from '$lib/onboarding/cloud-providers.js';

  let {
    kind,
    providerId,
    onchange
  }: {
    kind: 'local' | 'cloud';
    /** Canonical cloud provider id when `kind === 'cloud'`. */
    providerId: string;
    onchange: (sel: { kind: 'local' | 'cloud'; providerId: string }) => void;
  } = $props();

  const GROUPS = [
    { key: 'popular' as const, label: 'Popular' },
    { key: 'all' as const, label: 'All' }
  ] as const;

  const groups = $derived(
    GROUPS.map((g) => ({
      ...g,
      items: CLOUD_PROVIDERS.filter((p) => p.group === g.key)
    })).filter((g) => g.items.length > 0)
  );

  function pickLocal(): void {
    if (kind === 'local') return;
    onchange({ kind: 'local', providerId: 'ollama' });
  }

  function pickCloud(): void {
    if (kind === 'cloud') return;
    // Reuse the incoming id only when it is a real cloud provider; the default local
    // state carries providerId='ollama', which must not pin the cloud entry to a local id.
    const cloudId = CLOUD_PROVIDER_IDS.includes(providerId) ? providerId : CLOUD_PROVIDERS[0].id;
    onchange({ kind: 'cloud', providerId: cloudId });
  }
</script>

<div class="flex flex-col gap-2">
  <div
    class="flex w-full items-center rounded-lg bg-muted p-0.5"
    role="tablist"
    aria-label="Provider type"
  >
    <button
      role="tab"
      type="button"
      aria-selected={kind === 'local'}
      onclick={pickLocal}
      class={cn(
        'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        kind === 'local'
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
    >
      Local
    </button>
    <button
      role="tab"
      type="button"
      aria-selected={kind === 'cloud'}
      onclick={pickCloud}
      class={cn(
        'flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
        kind === 'cloud'
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
    >
      Cloud API
    </button>
  </div>

  {#if kind === 'cloud'}
    <select
      id="ai-cloud-provider"
      aria-label="Cloud provider"
      value={providerId}
      onchange={(e) => onchange({ kind: 'cloud', providerId: e.currentTarget.value })}
      class={SELECT_CLASS}
    >
      {#each groups as group (group.key)}
        <optgroup label={group.label}>
          {#each group.items as p (p.id)}
            <option value={p.id}>{p.name}</option>
          {/each}
        </optgroup>
      {/each}
    </select>
  {/if}
</div>
