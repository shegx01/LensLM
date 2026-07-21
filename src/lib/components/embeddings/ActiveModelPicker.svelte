<!--
  ActiveModelPicker — the explicit "which configured model answers chat, notes, and the
  audio overview" control (#29/#233). Before this, the pin was implicit: an unset
  enrichment.chat_model fell through to routing, which could fail silently and only
  surface as "no chat model configured" after a restart. Reactive persist via
  set_active_chat_model — no Save button. The picker only ever pins an explicit choice;
  it never clears the pin (that would reintroduce the implicit-routing failure state).
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import { SELECT_CLASS } from '$lib/components/onboarding/styles.js';
  import { setActiveChatModel } from '$lib/models/catalog.js';
  import { activeModelStore, refreshActiveModel } from '$lib/models/active-model.svelte.js';

  const KEY_SEP = ' ';

  let pending = $state(false);
  let error = $state<string | null>(null);

  const candidates = $derived(activeModelStore.candidates);
  const active = $derived(activeModelStore.active);

  function keyOf(provider: string, model: string): string {
    return `${provider}${KEY_SEP}${model}`;
  }

  // '' when unset so the disabled "none selected" placeholder is shown, never a silent
  // fallback to the first option.
  const selectedValue = $derived(active ? keyOf(active.provider, active.model) : '');

  const activeMatch = $derived(
    active
      ? candidates.find((c) => c.provider === active.provider && c.model === active.model)
      : null
  );

  // Set but resolving to no candidate (removed from models[] since it was pinned): rendered
  // as its own disabled option so the select keeps showing it rather than falling back.
  const staleActive = $derived(active && !activeMatch ? active : null);

  // `null` when unset; otherwise whether the pin still resolves, with its reason.
  const activeStatus = $derived.by(() => {
    if (!active) return null;
    if (!activeMatch) return { available: false, reason: 'no longer configured' };
    return { available: activeMatch.available, reason: activeMatch.reason };
  });

  const unavailableCandidates = $derived(candidates.filter((c) => !c.available));

  // Every value the <select> can hold. A native <select> silently shows its first option
  // when its value matches none — so the bound value must always be in this set.
  const optionValues = $derived(
    new Set<string>([
      ...(active === null ? [''] : []),
      ...candidates.map((c) => keyOf(c.provider, c.model)),
      ...(staleActive ? [keyOf(staleActive.provider, staleActive.model)] : [])
    ])
  );

  const boundValue = $derived.by(() => {
    if (!optionValues.has(selectedValue)) {
      throw new Error(`ActiveModelPicker: selected value "${selectedValue}" matches no option`);
    }
    return selectedValue;
  });

  onMount(() => {
    void refreshActiveModel();
  });

  async function onSelectChange(e: Event): Promise<void> {
    const value = (e.currentTarget as HTMLSelectElement).value;
    if (value === selectedValue) return;
    error = null;
    pending = true;
    try {
      const target = candidates.find((c) => keyOf(c.provider, c.model) === value);
      if (!target || !target.available) return;
      await setActiveChatModel(target.provider, target.model);
      await refreshActiveModel();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Could not update the active model.';
    } finally {
      pending = false;
    }
  }
</script>

<div class="flex flex-col gap-2 rounded-xl p-3 shadow-xs ring-1 ring-primary/30 bg-primary/5">
  <label
    for="active-model-select"
    class="text-[0.68rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70"
  >
    Active model
  </label>
  <p class="text-[0.75rem] text-muted-foreground">
    The one model chat, notes, and the audio overview all resolve to.
  </p>

  {#if candidates.length === 0}
    <p class="text-[0.78rem] text-muted-foreground" role="status">
      No models configured yet — set one up below, then pick it here.
    </p>
  {:else}
    <select
      id="active-model-select"
      aria-label="Active model"
      value={boundValue}
      onchange={(e) => void onSelectChange(e)}
      disabled={pending}
      class={SELECT_CLASS}
    >
      {#if active === null}
        <option value="" disabled hidden>None selected — pick a model</option>
      {/if}
      {#each candidates as c (keyOf(c.provider, c.model))}
        <option
          value={keyOf(c.provider, c.model)}
          disabled={!c.available}
          title={c.available ? undefined : (c.reason ?? undefined)}
        >
          {c.label}
        </option>
      {/each}
      {#if staleActive}
        <option value={keyOf(staleActive.provider, staleActive.model)} disabled>
          {staleActive.provider} · {staleActive.model} — no longer available
        </option>
      {/if}
    </select>

    {#if unavailableCandidates.length > 0}
      <ul class="flex flex-col gap-0.5 text-[0.7rem] text-muted-foreground">
        {#each unavailableCandidates as c (keyOf(c.provider, c.model))}
          <li>{c.label} — {c.reason}</li>
        {/each}
      </ul>
    {/if}
  {/if}

  {#if active === null}
    <p class="flex items-center gap-1.5 text-[0.75rem] text-destructive" role="status">
      <CircleAlert class="size-3.5 shrink-0" aria-hidden="true" />
      Without a pin, chat can fail with "no chat model configured" after restart.
    </p>
  {:else if activeStatus && !activeStatus.available}
    <p class="flex items-center gap-1.5 text-[0.75rem] text-destructive" role="status">
      <CircleAlert class="size-3.5 shrink-0" aria-hidden="true" />
      The active model is unavailable ({activeStatus.reason}). Choose another.
    </p>
  {/if}

  {#if error}
    <p class="text-[0.75rem] text-destructive" role="alert">{error}</p>
  {/if}
</div>
