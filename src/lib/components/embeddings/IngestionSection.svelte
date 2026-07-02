<!--
  IngestionSection — the "Ingestion" panel inside the global Preferences view
  (issue #78: JS-render toggle). Mounted by PreferencesShell when the user
  selects the Ingestion nav item.

  Wires to the existing `get_config` / `set_config` commands via the shared
  `updateConfig` read-modify-write helper — no new Rust command.

  Tokens only — light + dark + every accent ([[theming-light-dark-accent]]).
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import type { AppConfig } from '$lib/theme/types.js';
  import { updateConfig } from '$lib/config.js';

  // ── State ──────────────────────────────────────────────────────────────────
  let jsRenderEnabled = $state(true);
  let saving = $state(false);
  let saveError = $state<string | null>(null);

  onMount(async () => {
    if (!isTauri()) return;
    try {
      const cfg = await invoke<AppConfig>('get_config');
      // js_render_enabled defaults to true both in Rust and here; an absent key
      // in an older config.json is served as true by the serde default.
      jsRenderEnabled = cfg.js_render_enabled ?? true;
    } catch {
      // Non-fatal: fall back to default ON.
    }
  });

  async function handleToggle(checked: boolean): Promise<void> {
    jsRenderEnabled = checked;
    saving = true;
    saveError = null;
    try {
      await updateConfig((cfg) => ({ ...cfg, js_render_enabled: checked }));
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save setting.';
      // Revert the optimistic update on failure.
      jsRenderEnabled = !checked;
    } finally {
      saving = false;
    }
  }
</script>

<section class="flex flex-col" aria-label="Ingestion settings">
  <!-- Title + subtitle -->
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Ingestion</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Controls how web pages and other sources are processed when added.
  </p>

  <!-- ── JS render toggle ── -->
  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Web page rendering
    </p>

    <label
      class="mt-3 flex cursor-pointer items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5 transition-colors hover:border-border/80"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground"
          >Enable JS rendering for web pages</span
        >
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          When a page returns near-empty text, Lens renders it in an offscreen view to capture
          JS-generated content. Disable if you prefer static-only extraction.
        </span>
      </span>
      <Switch
        checked={jsRenderEnabled}
        disabled={saving}
        aria-label="Enable JS rendering for web pages"
        onCheckedChange={handleToggle}
      />
    </label>
  </div>

  {#if saveError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{saveError}</p>
  {/if}
</section>
