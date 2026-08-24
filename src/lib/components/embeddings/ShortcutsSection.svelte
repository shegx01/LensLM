<!--
  ShortcutsSection — the editable "Shortcuts" panel inside the global Preferences view.
  Escape and Enter drive the capture widget itself, so the three actions bound to them
  are necessarily read-only.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import { fromEvent, toToken, type Binding } from '$lib/shortcuts/binding.js';
  import { findConflict, isValidForScope } from '$lib/shortcuts/conflicts.js';
  import { currentPlatform } from '$lib/shortcuts/platform.js';
  import { GROUP_ORDER, ROWS, SHORTCUTS_BY_ID, type ActionId } from '$lib/shortcuts/registry.js';
  import { toLensError } from '$lib/sources/lens-error.js';
  import ShortcutRow, { type RowChip } from './ShortcutRow.svelte';

  type Keymap = Partial<Record<ActionId, string>>;

  const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta']);

  const platform = currentPlatform();
  const QUALIFYING_MODIFIERS = platform === 'darwin' ? 'Command or Option' : 'Control or Alt';

  // One nullable slot, so AC-18's single-arm rule is structural rather than a per-row flag.
  let capture = $state<{ id: ActionId; candidate: Binding | null; message: string | null } | null>(
    null
  );
  let saveError = $state<string | null>(null);

  const keymap = $derived(appConfigStore.keymap);
  const hasOverrides = $derived(Object.keys(keymap).length > 0);

  const groups = $derived(
    GROUP_ORDER.map((group) => ({
      group,
      rows: ROWS.filter((row) => row.group === group).map((row) => ({
        ...row,
        chips: row.ids.map((id): RowChip => {
          const entry = SHORTCUTS_BY_ID[id];
          const override = keymap[id];
          return {
            id,
            action: entry.action,
            token: override ?? entry.defaultBinding,
            overridden: override !== undefined,
            remappable: entry.remappable
          };
        })
      }))
    })).filter((g) => g.rows.length > 0)
  );

  onMount(() => {
    void ensureLoaded();
  });

  function arm(id: ActionId): void {
    saveError = null;
    capture = { id, candidate: null, message: null };
  }

  function disarm(): void {
    capture = null;
  }

  function handleCapture(event: KeyboardEvent): void {
    const active = capture;
    if (active === null) return;
    // Tab is let through so an armed row can never trap focus — the cost is that Tab
    // is permanently unbindable.
    if (event.key === 'Tab') return;

    event.preventDefault();
    // AppShell's window listener stays live behind the settings view, so a Mod+K
    // candidate would otherwise also open the command palette over this panel.
    event.stopPropagation();

    if (MODIFIER_KEYS.has(event.key)) return;
    if (event.key === 'Escape') {
      disarm();
      return;
    }
    if (event.key === 'Enter') {
      void accept();
      return;
    }

    const binding = fromEvent(event, platform);
    if (binding === null) {
      capture = { id: active.id, candidate: null, message: 'That keystroke can’t be recorded.' };
      return;
    }

    const { scope } = SHORTCUTS_BY_ID[active.id];
    // Conflicts are reported before the modifier rule: every window-vs-player collision
    // is also modifier-less, and naming the occupying action is the more actionable message.
    const conflict = findConflict(scope, binding, active.id, keymap);
    let message: string | null = null;
    if (conflict !== null) {
      message = `Already used by “${conflict.action}”.`;
    } else if (!isValidForScope(scope, binding)) {
      message = `Global shortcuts need a ${QUALIFYING_MODIFIERS} modifier — Shift alone still types a character.`;
    }
    capture = { id: active.id, candidate: binding, message };
  }

  async function write(mutate: (current: Keymap) => Keymap): Promise<void> {
    saveError = null;
    try {
      await persist((cfg) => ({ ...cfg, keymap: mutate(cfg.keymap ?? {}) }));
    } catch (err) {
      saveError = toLensError(err).message;
    }
  }

  async function accept(): Promise<void> {
    const active = capture;
    if (active === null || active.candidate === null || active.message !== null) return;
    const { id } = active;
    const token = toToken(active.candidate);
    disarm();
    await write((current) => ({ ...current, [id]: token }));
  }

  function resetOne(id: ActionId): void {
    void write((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });
  }
</script>

<section class="flex flex-col" aria-label="Shortcuts settings">
  <div class="flex items-start justify-between gap-4">
    <div class="min-w-0">
      <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Shortcuts</h2>
      <p class="mt-1 text-[0.8rem] text-muted-foreground">
        Click a shortcut to record a new one. Enter saves and Escape cancels, so Enter, Escape and
        Tab can&rsquo;t themselves be recorded &mdash; the shortcuts that use them stay fixed.
      </p>
    </div>
    <Button
      variant="outline"
      size="sm"
      disabled={!hasOverrides}
      onclick={() => void write(() => ({}))}
    >
      Reset all
    </Button>
  </div>

  {#each groups as { group, rows } (group)}
    <div class="mt-6">
      <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
        {group}
      </p>
      <div class="mt-3 flex flex-col gap-2">
        {#each rows as row (row.label)}
          <ShortcutRow
            label={row.label}
            description={row.description}
            chips={row.chips}
            {platform}
            armedId={capture?.id ?? null}
            candidate={capture?.candidate ?? null}
            message={capture !== null && row.ids.includes(capture.id) ? capture.message : null}
            onarm={arm}
            oncapture={handleCapture}
            ondisarm={disarm}
            onreset={resetOne}
          />
        {/each}
      </div>
    </div>
  {/each}

  {#if saveError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">
      Couldn&rsquo;t save the shortcut. {saveError}
    </p>
  {:else if appConfigStore.persistError}
    <p class="mt-3 text-[0.75rem] text-destructive" role="alert">
      Shortcut saved, but Lens couldn&rsquo;t confirm it. {appConfigStore.persistError}
    </p>
  {/if}
</section>
