<!--
  GeneralSection — the "General" panel inside the global Preferences view.
  Wires to `get_config` / `set_config` via `updateConfig` — no new Rust command.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';

  let reopenLastNotebook = $state(true);
  let saving = $state(false);
  let saveError = $state<string | null>(null);

  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      // reopen_last_notebook defaults to true both in Rust and here; an absent key
      // in an older config.json is served as true by the serde default.
      reopenLastNotebook = cfg.reopen_last_notebook ?? true;
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

  {#if saveError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{saveError}</p>
  {/if}
</section>
