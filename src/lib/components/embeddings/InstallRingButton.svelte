<!--
  Circular download button with an indeterminate progress ring, shared by the
  Settings and onboarding embedding pickers (fastembed reports no byte progress).
-->
<script lang="ts">
  import Download from '@lucide/svelte/icons/download';
  import { Button } from '$lib/components/ui/button/index.js';
  import { cn } from '$lib/utils.js';
  import { prefersReducedMotion } from '$lib/motion/index.js';

  let {
    installing,
    label,
    onclick
  }: {
    installing: boolean;
    label: string;
    onclick: () => void;
  } = $props();

  const reduceMotion = $derived(prefersReducedMotion());
</script>

<span class="relative inline-flex size-11 shrink-0 items-center justify-center">
  {#if installing}
    <span
      class={cn('absolute inset-0 rounded-full', !reduceMotion && 'animate-install-ring')}
      style={`border: 2.5px solid color-mix(in oklch, var(--primary) ${reduceMotion ? '45' : '20'}%, transparent); border-top-color: ${reduceMotion ? 'color-mix(in oklch, var(--primary) 45%, transparent)' : 'var(--primary)'};`}
      aria-hidden="true"
    ></span>
  {/if}
  <Button
    type="button"
    size="icon"
    {onclick}
    disabled={installing}
    aria-label={label}
    class="relative size-9 rounded-full transition-transform duration-150 active:scale-[0.97]"
  >
    <Download class="size-4" />
  </Button>
</span>

<style>
  /* Indeterminate install ring: gated by --rail-motion so a runtime "reduce
     motion" toggle also stalls it, in addition to the reduceMotion JS check
     that skips the class entirely. */
  @keyframes install-ring-spin {
    to {
      transform: rotate(360deg);
    }
  }
  .animate-install-ring {
    animation: install-ring-spin calc(0.9s / max(var(--rail-motion, 1), 0.0001)) linear infinite;
  }
</style>
