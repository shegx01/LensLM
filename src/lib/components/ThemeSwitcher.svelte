<script lang="ts">
  import { setMode, userPrefersMode } from 'mode-watcher';
  import Sun from '@lucide/svelte/icons/sun';
  import Moon from '@lucide/svelte/icons/moon';
  import Monitor from '@lucide/svelte/icons/monitor';
  import { Button } from '$lib/components/ui/button/index.js';
  import { persistTheme, setPersistErrorHandler, type Mode } from '$lib/theme/index.js';

  // Surface durable-write failures instead of silently diverging the store.
  let persistError = $state<string | null>(null);
  setPersistErrorHandler((err) => {
    persistError = err instanceof Error ? err.message : String(err);
  });

  const options: { value: Mode; label: string; icon: typeof Sun }[] = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor }
  ];

  const active = $derived(userPrefersMode.current);

  function select(mode: Mode): void {
    persistError = null;
    // Immediate live UI: mode-watcher applies the class + writes localStorage.
    setMode(mode);
    // Durable, debounced read-modify-write to AppConfig.theme (one-directional).
    persistTheme(mode);
  }
</script>

<div class="flex flex-col gap-2">
  <div
    role="group"
    aria-label="Theme"
    class="bg-muted inline-flex w-fit items-center gap-1 rounded-lg p-1"
  >
    {#each options as option (option.value)}
      {@const Icon = option.icon}
      <Button
        variant={active === option.value ? 'default' : 'ghost'}
        size="sm"
        aria-pressed={active === option.value}
        onclick={() => select(option.value)}
      >
        <Icon />
        {option.label}
      </Button>
    {/each}
  </div>
  {#if persistError}
    <p class="text-destructive text-xs" role="alert">
      Theme saved locally, but persisting failed: {persistError}
    </p>
  {/if}
</div>
