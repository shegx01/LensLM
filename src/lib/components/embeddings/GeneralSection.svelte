<!--
  GeneralSection — the "General" panel inside the global Preferences view.
  Wires to `get_config` / `set_config` via `updateConfig` — no new Rust command.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import Check from '@lucide/svelte/icons/check';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';
  import { ACCENTS, ACCENT_IDS, type AccentId } from '$lib/theme/accents.js';

  type MotionPref = 'system' | 'on' | 'off';
  const MOTION_OPTIONS: { value: MotionPref; label: string; hint: string }[] = [
    { value: 'system', label: 'System', hint: 'follows macOS Reduce Motion' },
    { value: 'on', label: 'On', hint: 'always animate' },
    { value: 'off', label: 'Off', hint: 'no motion' }
  ];

  let reopenLastNotebook = $state(true);
  let animations = $state<MotionPref>('system');
  let accent = $state<AccentId>(ACCENT_IDS[0]);
  let saving = $state(false);
  let saveError = $state<string | null>(null);

  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      // reopen_last_notebook defaults to true both in Rust and here; an absent key
      // in an older config.json is served as true by the serde default.
      reopenLastNotebook = cfg.reopen_last_notebook ?? true;
      animations = (['system', 'on', 'off'] as const).includes(cfg.animations as MotionPref)
        ? (cfg.animations as MotionPref)
        : 'system';
      accent = (ACCENT_IDS as readonly string[]).includes(cfg.accent)
        ? (cfg.accent as AccentId)
        : ACCENT_IDS[0];
    } catch {
      // Non-fatal: fall back to default ON.
    }
  });

  async function handleToggle(checked: boolean): Promise<void> {
    reopenLastNotebook = checked;
    saving = true;
    saveError = null;
    try {
      await updateConfig((cfg) => ({ ...cfg, reopen_last_notebook: checked }));
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save setting.';
      // Revert the optimistic update on failure.
      reopenLastNotebook = !checked;
    } finally {
      saving = false;
    }
  }

  async function selectAnimations(value: MotionPref): Promise<void> {
    if (value === animations) return;
    const previous = animations;
    animations = value;
    // Live-apply so the whole app's rail motion reacts before the write settles.
    document.documentElement.dataset.motion = value;
    saving = true;
    saveError = null;
    try {
      await updateConfig((cfg) => ({ ...cfg, animations: value }));
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save setting.';
      animations = previous;
      document.documentElement.dataset.motion = previous;
    } finally {
      saving = false;
    }
  }

  async function selectAccent(value: AccentId): Promise<void> {
    if (value === accent) return;
    const previous = accent;
    accent = value;
    // Live-apply the token layer so the whole app recolours before the write settles.
    document.documentElement.dataset.accent = value;
    saving = true;
    saveError = null;
    try {
      await updateConfig((cfg) => ({ ...cfg, accent: value }));
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save setting.';
      accent = previous;
      document.documentElement.dataset.accent = previous;
    } finally {
      saving = false;
    }
  }
</script>

<section class="flex flex-col" aria-label="General settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">General</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    General application behaviour and startup preferences.
  </p>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Startup
    </p>

    <label
      class="mt-3 flex cursor-pointer items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5 transition-colors hover:border-border/80"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground"
          >Reopen last notebook on launch</span
        >
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          When enabled, Lens automatically opens your most recently active notebook on startup
          instead of showing the empty workspace.
        </span>
      </span>
      <Switch
        checked={reopenLastNotebook}
        disabled={saving}
        aria-label="Reopen last notebook on launch"
        onCheckedChange={handleToggle}
      />
    </label>
  </div>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Accent
    </p>

    <div
      class="mt-3 flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">Accent color</span>
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          Tints buttons, highlights, and selected states across the whole app.
        </span>
      </span>
      <div role="radiogroup" aria-label="Accent color" class="flex shrink-0 items-center gap-2">
        {#each ACCENTS as sw (sw.id)}
          {@const selected = accent === sw.id}
          <button
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={sw.label}
            disabled={saving}
            title={sw.label}
            onclick={() => void selectAccent(sw.id)}
            class="flex size-7 items-center justify-center rounded-full transition-transform duration-150 hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
            style:background={sw.solid}
            style:box-shadow={selected ? `0 0 0 2px var(--card), 0 0 0 4px ${sw.solid}` : 'none'}
          >
            {#if selected}
              <Check class="size-3.5 text-white" strokeWidth={3} />
            {/if}
          </button>
        {/each}
      </div>
    </div>
  </div>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Animations
    </p>

    <div
      class="mt-3 flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">Animations</span>
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          Respect your system Reduce-Motion setting, or force motion on/off for LensLM.
        </span>
      </span>
      <div
        role="group"
        aria-label="Animations"
        class="bg-muted inline-flex w-fit shrink-0 items-center gap-1 rounded-lg p-1"
      >
        {#each MOTION_OPTIONS as option (option.value)}
          <Button
            variant={animations === option.value ? 'default' : 'ghost'}
            size="sm"
            disabled={saving}
            aria-pressed={animations === option.value}
            aria-label={`Animations: ${option.label} (${option.hint})`}
            onclick={() => void selectAnimations(option.value)}
          >
            {option.label}
          </Button>
        {/each}
      </div>
    </div>
  </div>

  {#if saveError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{saveError}</p>
  {/if}
</section>
