<script lang="ts" module>
  import type { ActionId } from '$lib/shortcuts/registry.js';

  /** One editable key in a display row; paired rows (Seek, Skip, Speed) carry two. */
  export interface RowChip {
    id: ActionId;
    action: string;
    token: string;
    overridden: boolean;
    remappable: boolean;
  }
</script>

<script lang="ts">
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import {
    render as renderBinding,
    toToken,
    type Binding,
    type Platform
  } from '$lib/shortcuts/binding.js';

  interface Props {
    label: string;
    description: string;
    chips: readonly RowChip[];
    platform: Platform;
    armedId: ActionId | null;
    candidate: Binding | null;
    message: string | null;
    onarm: (id: ActionId) => void;
    oncapture: (event: KeyboardEvent) => void;
    ondisarm: () => void;
    onreset: (id: ActionId) => void;
  }

  let {
    label,
    description,
    chips,
    platform,
    armedId,
    candidate,
    message,
    onarm,
    oncapture,
    ondisarm,
    onreset
  }: Props = $props();

  const KBD =
    'inline-block rounded bg-muted px-1.5 py-0.5 font-[inherit] text-[0.7rem] font-semibold text-foreground';

  const refs = $state<Record<string, HTMLButtonElement | null>>({});

  const rowArmed = $derived(chips.some((chip) => chip.id === armedId));
  const reserved = $derived(chips.every((chip) => !chip.remappable));

  // Without focus the capture listener is element-scoped on an unfocused element, so
  // keystrokes bypass it and reach AppShell's window listener while the row still
  // looks armed.
  $effect(() => {
    if (armedId === null) return;
    const el = refs[armedId];
    if (el && document.activeElement !== el) el.focus();
  });
</script>

<div
  data-shortcut-row
  class="flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5 {rowArmed
    ? 'ring-2 ring-ring'
    : ''}"
>
  <span class="min-w-0 flex-1">
    <span class="block text-[0.78rem] font-bold text-foreground">{label}</span>
    <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">{description}</span>
    {#if message}
      <span class="mt-1 block text-[0.68rem] text-destructive" role="alert">{message}</span>
    {/if}
  </span>

  <span class="flex shrink-0 items-center gap-1.5">
    {#if reserved}
      <span
        class="text-[0.62rem] font-semibold uppercase tracking-[0.06em] text-muted-foreground/70"
      >
        Conventional
      </span>
    {/if}
    {#each chips as chip, i (chip.id)}
      {#if i > 0}<span class="text-[0.68rem] text-muted-foreground/50">/</span>{/if}
      {#if chip.remappable}
        {@const armed = chip.id === armedId}
        <button
          type="button"
          bind:this={refs[chip.id]}
          aria-label={`Change shortcut for ${chip.action}`}
          aria-pressed={armed}
          onclick={() => onarm(chip.id)}
          onkeydown={(event) => {
            if (armed) oncapture(event);
          }}
          onblur={() => {
            if (armed) ondisarm();
          }}
          class="rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <kbd class="{KBD} {armed ? 'border border-primary' : ''}">
            {#if !armed}
              {renderBinding(chip.token, platform)}
            {:else if candidate}
              {renderBinding(toToken(candidate), platform)}
            {:else}
              Press a key…
            {/if}
          </kbd>
        </button>
        {#if chip.overridden}
          <button
            type="button"
            aria-label={`Reset ${chip.action} to default`}
            onclick={() => onreset(chip.id)}
            class="text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <RotateCcw class="size-3" />
          </button>
        {/if}
      {:else}
        <kbd class={KBD}>{renderBinding(chip.token, platform)}</kbd>
      {/if}
    {/each}
  </span>
</div>
