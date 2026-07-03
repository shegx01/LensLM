<script lang="ts">
  import { setMode, userPrefersMode } from 'mode-watcher';
  import Sun from '@lucide/svelte/icons/sun';
  import Moon from '@lucide/svelte/icons/moon';
  import Monitor from '@lucide/svelte/icons/monitor';
  import { Button } from '$lib/components/ui/button/index.js';
  import { persistTheme, type Mode } from '$lib/theme/index.js';
  import { cn } from '$lib/utils.js';

  /**
   * Cycling theme button: click advances light → dark → system → light.
   * `variant="bare"` renders a plain 26px circle (sidebar brand row); `"outline"` uses the shadcn outline button.
   * `class` merges onto the trigger; `iconClass` overrides glyph size (default: 16px outline / 14px bare).
   */
  let {
    class: className = '',
    variant = 'outline',
    iconClass
  }: {
    class?: string;
    variant?: 'outline' | 'bare';
    iconClass?: string;
  } = $props();

  const CYCLE: Mode[] = ['light', 'dark', 'system'];

  const CYCLE_META: Record<Mode, { icon: typeof Sun; label: string; next: string }> = {
    light: { icon: Sun, label: 'Light', next: 'Dark' },
    dark: { icon: Moon, label: 'Dark', next: 'System' },
    system: { icon: Monitor, label: 'System', next: 'Light' }
  };

  const currentMode = $derived(userPrefersMode.current ?? 'system');
  const meta = $derived(CYCLE_META[currentMode]);
  const ariaLabel = $derived(`Theme: ${meta.label} — click to switch to ${meta.next}`);
  const glyphClass = $derived(iconClass ?? (variant === 'bare' ? 'size-3.5' : 'size-4'));

  function cycleTheme(): void {
    const idx = CYCLE.indexOf(currentMode);
    const next = CYCLE[(idx + 1) % CYCLE.length];
    setMode(next);
    persistTheme(next);
  }
</script>

<!--
  The icon reflects the CURRENT mode; aria-label names the current mode and the
  next mode so screen-reader users know what clicking will do.
-->
{#if variant === 'bare'}
  <button
    type="button"
    aria-label={ariaLabel}
    data-theme-cycle-btn
    onclick={cycleTheme}
    class={cn(
      'flex size-[26px] shrink-0 items-center justify-center rounded-full',
      'bg-muted text-sidebar-foreground/70 hover:text-sidebar-foreground hover:opacity-60',
      'cursor-pointer border-0 transition-opacity',
      'outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring',
      className
    )}
  >
    {#key currentMode}
      {@const Icon = meta.icon}
      <Icon class={glyphClass} />
    {/key}
  </button>
{:else}
  <Button
    variant="outline"
    size="icon"
    aria-label={ariaLabel}
    onclick={cycleTheme}
    class={cn(className)}
  >
    {#key currentMode}
      {@const Icon = meta.icon}
      <Icon class={glyphClass} />
    {/key}
  </Button>
{/if}
