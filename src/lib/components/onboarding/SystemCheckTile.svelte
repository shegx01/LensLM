<!--
  SystemCheckTile — single-sourced chrome consumed by BOTH the embedding row
  and OnboardingLlmPicker so the step reads as one system even though the two
  rows differ in interaction.
-->
<script lang="ts">
  import type { Component, Snippet } from 'svelte';
  import { cn } from '$lib/utils.js';

  let {
    icon,
    badgeClass = 'bg-primary/15 text-primary',
    title,
    subtitle,
    titleClass = '',
    status,
    children,
    class: className
  }: {
    icon: Component;
    /** Icon-badge tint (token-only classes). */
    badgeClass?: string;
    title: string;
    subtitle: string;
    /** Extra title classes (e.g. destructive tint on a failing row). */
    titleClass?: string;
    /** Right-aligned header slot: status pill or action control. */
    status?: Snippet;
    /** Tile body below the header (field group / expansion panel). */
    children?: Snippet;
    class?: string;
  } = $props();

  const Icon = $derived(icon);
</script>

<div
  class={cn(
    'bg-card text-card-foreground ring-foreground/10 flex flex-col gap-0 rounded-[13px] px-4 py-3 shadow-[var(--shadow-tile)] ring-1',
    className
  )}
>
  <div class="flex w-full items-center gap-3">
    <span
      class={cn(
        'flex size-8 shrink-0 items-center justify-center rounded-[10px] [&_svg]:size-4',
        badgeClass
      )}
      aria-hidden="true"
    >
      <Icon />
    </span>

    <div class="min-w-0 flex-1">
      <p class={cn('text-foreground truncate text-sm font-bold', titleClass)}>{title}</p>
      <p class="text-muted-foreground truncate text-[0.8rem]">{subtitle}</p>
    </div>

    {#if status}
      {@render status()}
    {/if}
  </div>

  {#if children}
    {@render children()}
  {/if}
</div>
