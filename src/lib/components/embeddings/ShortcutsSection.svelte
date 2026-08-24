<!--
  ShortcutsSection — the editable "Shortcuts" panel inside the global Preferences view.
  Escape and Enter drive the capture widget itself, so the three actions bound to them
  are necessarily read-only.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import { appConfigStore, ensureLoaded, persist } from '$lib/models/app-config.svelte.js';
  import {
    describe as describeBinding,
    fromEvent,
    toToken,
    type Binding
  } from '$lib/shortcuts/binding.js';
  import { findConflict, isValidForScope } from '$lib/shortcuts/conflicts.js';
  import { currentPlatform } from '$lib/shortcuts/platform.js';
  import { GROUP_ORDER, ROWS, SHORTCUTS_BY_ID, type ActionId } from '$lib/shortcuts/registry.js';
  import { toLensError } from '$lib/sources/lens-error.js';
  import ShortcutRow, { type RowChip } from './ShortcutRow.svelte';

  type Keymap = Partial<Record<ActionId, string>>;

  const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta']);

  const platform = currentPlatform();
  // Option is a composition modifier on macOS — Option+Q arrives as 'œ' and `fromEvent`
  // rejects it — so Command is the only qualifying modifier that can be recorded there.
  const QUALIFYING_MODIFIERS = platform === 'darwin' ? 'Command' : 'Control or Alt';
  const RECORDABLE_KEYS =
    'Use a letter, a number, Space, an arrow key, or [ or ], on its own or with modifiers.';

  // One nullable slot, so AC-18's single-arm rule is structural rather than a per-row flag.
  let capture = $state<{ id: ActionId; candidate: Binding | null; message: string | null } | null>(
    null
  );
  let saveError = $state<string | null>(null);
  let announcement = $state('');
  let ready = $state(false);
  let heading = $state<HTMLHeadingElement | null>(null);

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
    void ensureLoaded().finally(() => {
      ready = true;
    });
  });

  function announce(text: string): void {
    announcement = text;
  }

  function arm(id: ActionId): void {
    saveError = null;
    capture = { id, candidate: null, message: null };
  }

  function disarm(): void {
    capture = null;
  }

  function validate(id: ActionId, binding: Binding, map: Keymap): string | null {
    const { scope } = SHORTCUTS_BY_ID[id];
    // Conflicts are reported before the modifier rule: every window-vs-player collision
    // is also modifier-less, and naming the occupying action is the more actionable message.
    const conflict = findConflict(scope, binding, id, map);
    if (conflict !== null) return `Already used by “${conflict.action}”.`;
    if (!isValidForScope(scope, binding))
      return `Global shortcuts need a ${QUALIFYING_MODIFIERS} modifier — Shift alone still types a character.`;
    return null;
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
      const message = `That key can’t be recorded. ${RECORDABLE_KEYS}`;
      capture = { id: active.id, candidate: null, message };
      announce(message);
      return;
    }

    const message = validate(active.id, binding, keymap);
    capture = { id: active.id, candidate: binding, message };
    if (message !== null) announce(message);
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
    if (active === null || active.candidate === null) return;
    if (active.message !== null) {
      announce(`Not saved. ${active.message}`);
      return;
    }
    const { id, candidate } = active;
    const token = toToken(candidate);
    disarm();
    await write((current) => {
      // The check above ran against the store snapshot, but updateConfig re-reads get_config
      // and hands the mutator a different map — the binding has to clear THAT one.
      const blocked = validate(id, candidate, current);
      if (blocked !== null) throw new Error(blocked);
      return { ...current, [id]: token };
    });
    if (saveError === null) {
      announce(`${SHORTCUTS_BY_ID[id].action} is now ${describeBinding(token, platform)}.`);
    }
  }

  async function resetOne(id: ActionId): Promise<void> {
    await write((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });
    if (saveError === null) {
      const entry = SHORTCUTS_BY_ID[id];
      announce(`${entry.action} reset to ${describeBinding(entry.defaultBinding, platform)}.`);
    }
  }

  async function resetAll(): Promise<void> {
    // Both Chromium and WebKit blur a disabled element, and a successful reset disables this
    // button — park focus on the heading first so it never lands on <body>.
    heading?.focus();
    await write(() => ({}));
    if (saveError === null) announce('All shortcuts reset to their defaults.');
  }
</script>

<section class="flex flex-col" aria-label="Shortcuts settings">
  <div class="flex items-start justify-between gap-4">
    <div class="min-w-0">
      <h2
        bind:this={heading}
        tabindex="-1"
        class="text-xl font-extrabold tracking-[-0.4px] text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        Shortcuts
      </h2>
      <p class="mt-1 text-[0.8rem] text-muted-foreground">
        Click a shortcut to record a new one. Enter saves and Escape cancels, so Enter, Escape and
        Tab can&rsquo;t themselves be recorded &mdash; the shortcuts that use them stay fixed.
      </p>
    </div>
    <Button variant="outline" size="sm" disabled={!hasOverrides} onclick={() => void resetAll()}>
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
            disabled={!ready}
            onarm={arm}
            oncapture={handleCapture}
            ondisarm={disarm}
            onreset={(id) => void resetOne(id)}
          />
        {/each}
      </div>
    </div>
  {/each}

  <!-- Persistent region: a live node created together with its text does not re-announce. -->
  <span class="sr-only" role="status">{announcement}</span>

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
