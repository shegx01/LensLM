<!--
  ShortcutsSection — the "Shortcuts" panel inside the global Preferences view.
  Read-only render of `$lib/shortcuts/registry.ts` — no config read/write.
  Remapping is deferred to #239 (central dispatcher + AppConfig keymap).
-->
<script lang="ts">
  import { SHORTCUTS, type ShortcutEntry } from '$lib/shortcuts/registry.js';

  const GROUP_ORDER: ShortcutEntry['group'][] = ['Global', 'Chat', 'Audio player'];
  const groups = GROUP_ORDER.map((group) => ({
    group,
    items: SHORTCUTS.filter((item) => item.group === group)
  })).filter((g) => g.items.length > 0);
</script>

<section class="flex flex-col" aria-label="Shortcuts settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Shortcuts</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Keyboard shortcuts built into Lens today. Read-only — remapping is coming in a future update.
  </p>

  {#each groups as { group, items } (group)}
    <div class="mt-6">
      <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
        {group}
      </p>
      <div class="mt-3 flex flex-col gap-2">
        {#each items as item (item.action)}
          <div
            data-shortcut-row
            class="flex items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5"
          >
            <span class="min-w-0 flex-1">
              <span class="block text-[0.78rem] font-bold text-foreground">{item.action}</span>
              <span class="mt-0.5 block text-[0.68rem] text-muted-foreground"
                >{item.description}</span
              >
            </span>
            <span class="flex shrink-0 items-center gap-1.5">
              {#each item.keys as key, i (key)}
                {#if i > 0}<span class="text-[0.68rem] text-muted-foreground/50">/</span>{/if}
                <kbd class="shortcut-kbd">{key}</kbd>
              {/each}
            </span>
          </div>
        {/each}
      </div>
    </div>
  {/each}
</section>

<style>
  .shortcut-kbd {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--muted);
    font-family: inherit;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--foreground);
  }
</style>
