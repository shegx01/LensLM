<script lang="ts">
  import { setMode, userPrefersMode } from 'mode-watcher';
  import Sun from '@lucide/svelte/icons/sun';
  import Moon from '@lucide/svelte/icons/moon';
  import Monitor from '@lucide/svelte/icons/monitor';
  import { Button } from '$lib/components/ui/button/index.js';
  import { persistTheme, type Mode } from '$lib/theme/index.js';
  import { cn } from '$lib/utils.js';

  /**
   * Optional class string merged onto the trigger button.
   * Callers use this to vary sizing/positioning:
   *   - Onboarding (SystemCheck): class="size-9 rounded-lg"
   *   - Sidebar header: no override (defaults to Button size="icon")
   *   - Account footer: no override (call cycleTheme from outside)
   */
  let { class: className = '' }: { class?: string } = $props();

  const CYCLE: Mode[] = ['light', 'dark', 'system'];

  const CYCLE_META: Record<Mode, { icon: typeof Sun; label: string; next: string }> = {
    light: { icon: Sun, label: 'Light', next: 'Dark' },
    dark: { icon: Moon, label: 'Dark', next: 'System' },
    system: { icon: Monitor, label: 'System', next: 'Light' }
  };

  const currentMode = $derived(userPrefersMode.current ?? 'system');
  const meta = $derived(CYCLE_META[currentMode]);

  function cycleTheme(): void {
    const idx = CYCLE.indexOf(currentMode);
    const next = CYCLE[(idx + 1) % CYCLE.length];
    setMode(next);
    persistTheme(next);
  }
</script>

<!--
  Single cycling icon button: click advances light→dark→system→light.
  The icon reflects the CURRENT mode; aria-label names the current mode
  and the next mode so screen-reader users know what clicking will do.
  Accepts an optional `class` prop to override size/shape for different
  placement contexts (e.g. onboarding, sidebar header, footer menu).
-->
<Button
  variant="outline"
  size="icon"
  aria-label={`Theme: ${meta.label} — click to switch to ${meta.next}`}
  onclick={cycleTheme}
  class={cn(className)}
>
  {#key currentMode}
    {@const Icon = meta.icon}
    <Icon class="size-4" />
  {/key}
</Button>
