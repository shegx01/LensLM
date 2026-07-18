<!-- Centered empty state (AC20): no messages yet. Branded Aperture hero that
     ties the chat back to the rail's wordmark; staggered reveal on mount. -->
<script lang="ts">
  import Aperture from '@lucide/svelte/icons/aperture';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import { fadeRise } from '$lib/motion/index.js';
  import { notebookStore } from '$lib/notebooks/index.js';
  import { chatProviderStore } from '$lib/models/chat-provider.svelte.js';

  const hasProvider = $derived(chatProviderStore.available);
</script>

<div class="flex flex-1 flex-col items-center justify-center gap-2.5 pr-6 pl-4 text-center">
  <div class="hero" use:fadeRise={{ y: 10, delay: 0 }} aria-hidden="true">
    <span class="halo"></span>
    <span class="aperture">
      <Aperture class="size-6" strokeWidth={1.75} />
    </span>
  </div>
  <p
    class="mt-1 text-[15px] font-semibold tracking-[-0.01em] text-foreground [text-wrap:balance]"
    use:fadeRise={{ y: 8, delay: 0.06 }}
  >
    Ask anything about your sources
  </p>
  <p
    class="max-w-[300px] text-xs leading-relaxed text-muted-foreground/70 [text-wrap:pretty]"
    use:fadeRise={{ y: 8, delay: 0.12 }}
  >
    Answers are grounded in this notebook's selected sources.
  </p>
  {#if !hasProvider}
    <!-- AC-11: no usable chat model → CTA deep-linking to Settings → AI Model. -->
    <button
      type="button"
      onclick={() => notebookStore.openSettings('ai')}
      use:fadeRise={{ y: 8, delay: 0.18 }}
      class="mt-1.5 inline-flex items-center gap-1.5 rounded-full bg-primary px-3.5 py-2 text-xs font-semibold text-primary-foreground transition-[transform,opacity] hover:opacity-90 active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <Settings2 class="size-3.5" strokeWidth={2.25} />
      Set up a model in Settings
    </button>
  {/if}
</div>

<style>
  .hero {
    position: relative;
    display: grid;
    place-items: center;
    width: 52px;
    height: 52px;
    border-radius: 16px;
    color: var(--primary-foreground);
    background: var(--primary);
    box-shadow:
      inset 0 1px 0 oklch(1 0 0 / 0.18),
      0 8px 22px color-mix(in oklch, var(--primary) 30%, transparent);
  }
  /* Soft primary halo that breathes outward — calm on reduce-motion (rail-motion=0). */
  .halo {
    position: absolute;
    inset: 0;
    border-radius: inherit;
    box-shadow: 0 0 0 0 color-mix(in oklch, var(--primary) 45%, transparent);
    animation: heroHalo calc(3s / max(var(--rail-motion, 1), 0.0001)) var(--ease-out, ease) infinite;
  }
  .aperture {
    position: relative;
    display: inline-flex;
    line-height: 0;
    animation: heroSpin calc(24s / max(var(--rail-motion, 1), 0.0001)) linear infinite;
  }
  @keyframes heroHalo {
    0% {
      box-shadow: 0 0 0 0 color-mix(in oklch, var(--primary) 40%, transparent);
    }
    70%,
    100% {
      box-shadow: 0 0 0 14px color-mix(in oklch, var(--primary) 0%, transparent);
    }
  }
  @keyframes heroSpin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
