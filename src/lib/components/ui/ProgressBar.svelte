<!-- Shared determinate/indeterminate progress track. Reduced motion swaps the animated
     slide for a static striped fill rather than slowing it: at --rail-motion:0 the slide's
     period grows to ~11000s and parks off-screen (frozen), so it must not merely slow. -->
<script lang="ts">
  let { value = null }: { value?: number | null } = $props();

  const clamped = $derived(
    value === null || value === undefined ? null : Math.min(100, Math.max(0, value))
  );
</script>

<div
  class="progress-track"
  role="progressbar"
  aria-valuemin={0}
  aria-valuemax={100}
  aria-valuenow={clamped ?? undefined}
>
  {#if clamped !== null}
    <div class="progress-fill" style:width="{clamped}%"></div>
  {:else}
    <div class="progress-fill indeterminate-slide" aria-hidden="true"></div>
    <div class="progress-fill indeterminate-static" aria-hidden="true"></div>
  {/if}
</div>

<style>
  .progress-track {
    position: relative;
    height: 5px;
    border-radius: 999px;
    background: var(--muted);
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    border-radius: 999px;
    background: var(--primary);
    transition: width 0.3s var(--ease-out, ease);
  }
  .indeterminate-slide {
    position: absolute;
    top: 0;
    width: 40%;
    left: -40%;
    animation: progress-bar-slide calc(1.1s / max(var(--rail-motion, 1), 0.0001)) ease-in-out
      infinite;
  }
  @keyframes progress-bar-slide {
    0% {
      left: -40%;
    }
    100% {
      left: 100%;
    }
  }

  /* Static fallback for the indeterminate state — full-width diagonal stripes at
     reduced opacity, so "in progress, duration unknown" stays visually distinct
     from both the idle track and a determinate fill. */
  .indeterminate-static {
    display: none;
    position: absolute;
    inset: 0;
    width: 100%;
    opacity: 0.55;
    background: repeating-linear-gradient(
      45deg,
      var(--primary) 0 6px,
      color-mix(in oklch, var(--primary) 40%, transparent) 6px 12px
    );
  }

  @media (prefers-reduced-motion: reduce) {
    :global(:root:not([data-motion='on'])) .indeterminate-slide {
      display: none;
    }
    :global(:root:not([data-motion='on'])) .indeterminate-static {
      display: block;
    }
  }
  :global([data-motion='off']) .indeterminate-slide {
    display: none;
  }
  :global([data-motion='off']) .indeterminate-static {
    display: block;
  }
</style>
