<!-- In-flight status line driven by `AnswerStage` (AC2). -->
<script lang="ts">
  import type { AnswerStage } from '$lib/chat/types.js';

  interface Props {
    stage: AnswerStage | null;
  }

  let { stage }: Props = $props();

  const label = $derived.by(() => {
    switch (stage) {
      case 'Retrieving':
        return 'Searching sources…';
      case 'Thinking':
        return 'Thinking…';
      case 'Answering':
        return 'Answering…';
      default:
        return null;
    }
  });
</script>

{#if label}
  <div class="flex items-center gap-2 px-4 pb-1 text-xs text-muted-foreground" aria-live="polite">
    <span class="dots" aria-hidden="true">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </span>
    <span class="shimmer">{label}</span>
  </div>
{/if}

<style>
  .dots {
    display: inline-flex;
    align-items: center;
    gap: 3px;
  }
  .dot {
    width: 5px;
    height: 5px;
    border-radius: 999px;
    background: var(--primary);
    animation: dotWave calc(1.1s / max(var(--rail-motion, 1), 0.0001)) var(--ease-out, ease)
      infinite;
  }
  .dot:nth-child(2) {
    animation-delay: calc(0.13s * var(--rail-motion, 1));
  }
  .dot:nth-child(3) {
    animation-delay: calc(0.26s * var(--rail-motion, 1));
  }
  @keyframes dotWave {
    0%,
    100% {
      opacity: 0.3;
      transform: translateY(0) scale(0.85);
    }
    40% {
      opacity: 1;
      transform: translateY(-2px) scale(1);
    }
  }
  /* Text shimmer sweeps the brand primary across the muted label. Colors are
     explicit tokens (not currentColor — the clip makes the text transparent). */
  .shimmer {
    background: linear-gradient(
      100deg,
      var(--muted-foreground) 0%,
      var(--muted-foreground) 38%,
      var(--primary) 50%,
      var(--muted-foreground) 62%,
      var(--muted-foreground) 100%
    );
    background-size: 220% 100%;
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
    animation: shimmerSweep calc(2.4s / max(var(--rail-motion, 1), 0.0001)) linear infinite;
  }
  @keyframes shimmerSweep {
    from {
      background-position: 120% 0;
    }
    to {
      background-position: -120% 0;
    }
  }
</style>
