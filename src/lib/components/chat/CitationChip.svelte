<!-- A single numbered source chip. A real <button> (not the Badge element, which is
     span/a only) so live chips have click + keyboard + disabled semantics. Styled
     with badgeVariants for token-only theming (light/dark/accent). Stale chips
     (source no longer in the store) render disabled + dimmed and do not activate. -->
<script lang="ts">
  import { cn } from '$lib/utils.js';
  import { badgeVariants } from '$lib/components/ui/badge/index.js';

  interface Props {
    n: number;
    label: string;
    live: boolean;
    onactivate?: () => void;
  }

  let { n, label, live, onactivate }: Props = $props();
</script>

<button
  type="button"
  disabled={!live}
  aria-disabled={!live}
  aria-label={live ? `Source ${n}: ${label}` : `Source ${n}: ${label} (unavailable)`}
  title={live ? undefined : 'Source no longer available'}
  onclick={live ? onactivate : undefined}
  class={cn(
    badgeVariants({ variant: 'secondary' }),
    'max-w-[12rem] cursor-pointer gap-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    !live && 'cursor-default opacity-60'
  )}
>
  <span class="tabular-nums font-semibold">{n}</span>
  <span class="truncate">{label}</span>
</button>
