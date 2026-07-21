<!--
  ProviderLogo — fixed-size rounded chip showing a provider's bundled brand mark,
  or a monogram fallback when none is vendored (e.g. openai-compatible).
-->
<script lang="ts">
  import { providerLogo, providerMonogram } from '$lib/models/provider-logos.js';

  let { id, name, size = 24 }: { id: string; name: string; size?: number } = $props();

  const svg = $derived(providerLogo(id));
</script>

<div
  class="flex shrink-0 items-center justify-center rounded-md border border-border bg-muted"
  style:width="{size}px"
  style:height="{size}px"
>
  {#if svg}
    <!-- Build-vendored + sanitized at fetch time (scripts/fetch-provider-logos.sh) — no user input. -->
    <div class="text-foreground/70" style:width="{size * 0.6}px" style:height="{size * 0.6}px">
      {@html svg}
    </div>
  {:else}
    <span class="font-semibold text-foreground/70" style:font-size="{size * 0.45}px">
      {providerMonogram(name)}
    </span>
  {/if}
</div>

<style>
  /* Injected SVGs have no width/height attrs (stripped at fetch time) — fill the wrapper. */
  div :global(svg) {
    width: 100%;
    height: 100%;
  }
</style>
