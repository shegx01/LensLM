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
  import {
    Select,
    SelectTrigger,
    SelectValue,
    SelectContent,
    SelectItem
  } from '$lib/components/ui/select/index.js';
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

  // Every value the Select can hold. The bound value must always be in this set, or it
  // resolves to no item (showing the placeholder) instead of the intended pin.
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

  // The item set bits-ui resolves the trigger label from — candidates plus the stale pin,
  // each carrying its disabled state so unavailable picks stay selectable-disabled.
  const selectItems = $derived([
    ...candidates.map((c) => ({
      value: keyOf(c.provider, c.model),
      label: c.label,
      disabled: !c.available
    })),
    ...(staleActive
      ? [
          {
            value: keyOf(staleActive.provider, staleActive.model),
            label: `${staleActive.provider} · ${staleActive.model} — no longer available`,
            disabled: true
          }
        ]
      : [])
  ]);

  onMount(() => {
    void refreshActiveModel();
  });

  async function onSelectChange(value: string): Promise<void> {
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
    <Select
      type="single"
      value={boundValue}
      onValueChange={(v) => {
        if (v) void onSelectChange(v);
      }}
      disabled={pending}
      items={selectItems}
    >
      <SelectTrigger id="active-model-select" class="w-full" aria-label="Active model">
        <SelectValue placeholder="None selected — pick a model" />
      </SelectTrigger>
      <SelectContent
        class="origin-(--bits-select-content-transform-origin) duration-200 ease-[cubic-bezier(0.23,1,0.32,1)]"
      >
        {#each candidates as c (keyOf(c.provider, c.model))}
          <SelectItem value={keyOf(c.provider, c.model)} label={c.label} disabled={!c.available}>
            {c.label}
          </SelectItem>
        {/each}
        {#if staleActive}
          <SelectItem value={keyOf(staleActive.provider, staleActive.model)} disabled>
            {staleActive.provider} · {staleActive.model} — no longer available
          </SelectItem>
        {/if}
      </SelectContent>
    </Select>

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
