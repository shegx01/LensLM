<!--
  SettingsShell — presentational sidebar+content chrome for a sectioned settings surface,
  shared by the global Preferences view and the per-notebook settings. `onBack` and `label`
  are optional so a container that already provides a Back button / heading can omit them.

  The active nav item is tracked by a single indicator pill that SLIDES between rows on the
  rail spring, gated by --rail-motion.
-->
<script lang="ts">
  import type { Snippet } from 'svelte';
  import { cn } from '$lib/utils.js';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';

  type IconComponent = typeof ArrowLeft;

  export type NavItem = {
    id: string;
    label: string;
    icon: IconComponent;
    stub: boolean;
  };

  let {
    nav,
    active = $bindable(),
    onSelect,
    onBack,
    label,
    content
  }: {
    nav: NavItem[];
    active: string;
    onSelect?: (id: string) => void;
    onBack?: () => void;
    label?: string;
    content: Snippet<[string]>;
  } = $props();

  function select(id: string): void {
    active = id;
    onSelect?.(id);
  }

  let itemEls = $state<Record<string, HTMLButtonElement | undefined>>({});
  let ind = $state<{ top: number; height: number; shown: boolean }>({
    top: 0,
    height: 0,
    shown: false
  });
  // The indicator is placed instantly on first paint; sliding turns on only after,
  // so opening the panel doesn't animate the pill in from the top row.
  let animated = $state(false);

  // Track the active row's geometry so the indicator overlays it exactly. Reading
  // offsetTop/offsetHeight (rather than index × rowHeight) keeps it aligned when a
  // spacer like `about`'s `mt-auto` breaks the even rhythm.
  $effect(() => {
    void nav;
    const el = itemEls[active];
    if (!el) return;
    ind = { top: el.offsetTop, height: el.offsetHeight, shown: true };
    if (!animated) requestAnimationFrame(() => (animated = true));
  });
</script>

<div class="settings-shell flex h-full min-h-0 flex-1 overflow-hidden bg-background antialiased">
  <nav
    class="relative flex w-[212px] shrink-0 flex-col gap-0.5 overflow-y-auto border-r border-border/70 bg-muted/30 px-2.5 py-3.5"
    aria-label={label ? `${label} sections` : 'Settings sections'}
  >
    {#if onBack}
      <button
        type="button"
        onclick={onBack}
        class={cn(
          'mb-2 flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-left text-[0.78rem] font-semibold text-muted-foreground',
          'transition-[color,background-color,transform] duration-150 active:scale-[0.98]',
          'hover:bg-foreground/[0.04] hover:text-foreground',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
        )}
      >
        <ArrowLeft class="size-3.5 shrink-0" aria-hidden="true" />
        <span>Back</span>
      </button>
    {/if}

    {#if label}
      <p
        class="px-2.5 pb-2 text-[0.6rem] font-bold uppercase tracking-[0.09em] text-muted-foreground/60"
      >
        {label}
      </p>
    {/if}

    <span
      class="nav-ind pointer-events-none absolute left-2.5 right-2.5 rounded-[9px] bg-primary/10"
      class:is-shown={ind.shown}
      class:is-animated={animated}
      style="height: {ind.height}px; transform: translate3d(0, {ind.top}px, 0);"
      aria-hidden="true"
    >
      <span class="nav-ind-bar"></span>
    </span>

    {#each nav as item (item.id)}
      {@const isActive = active === item.id}
      <button
        bind:this={itemEls[item.id]}
        type="button"
        aria-current={isActive ? 'page' : undefined}
        aria-disabled={item.stub}
        onclick={() => {
          if (!item.stub) select(item.id);
        }}
        class={cn(
          'relative z-10 flex h-9 items-center gap-2.5 rounded-[9px] px-2.5 text-left text-[0.78rem] font-semibold',
          'transition-[color,background-color,transform] duration-150',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
          item.id === 'about' && 'mt-auto',
          isActive
            ? 'text-primary active:scale-[0.98]'
            : item.stub
              ? 'cursor-default text-muted-foreground/40'
              : 'text-muted-foreground hover:bg-foreground/[0.04] hover:text-foreground active:scale-[0.98]'
        )}
      >
        <item.icon
          class={cn('size-3.5 shrink-0 transition-transform duration-200', isActive && 'scale-105')}
          aria-hidden="true"
        />
        <span class="flex-1 truncate">{item.label}</span>
        {#if item.stub}
          <span
            class="rounded-full bg-muted-foreground/10 px-1.5 py-px text-[0.58rem] font-semibold uppercase tracking-[0.04em] text-muted-foreground/50"
            >Soon</span
          >
        {/if}
      </button>
    {/each}
  </nav>

  <div class="flex-1 overflow-y-auto px-10 py-8">
    {@render content(active)}
  </div>
</div>

<style>
  .nav-ind {
    top: 0;
    opacity: 0;
    transition: opacity 0.18s var(--ease-out, ease);
  }
  .nav-ind.is-shown {
    opacity: 1;
  }
  .nav-ind.is-animated {
    transition:
      transform calc(0.4s * var(--rail-motion, 1)) var(--ease-spring, ease),
      height calc(0.4s * var(--rail-motion, 1)) var(--ease-spring, ease),
      opacity 0.18s var(--ease-out, ease);
  }
  .nav-ind-bar {
    position: absolute;
    left: 0;
    top: 50%;
    height: 16px;
    width: 3px;
    border-radius: 999px;
    background: var(--primary);
    transform: translateY(-50%);
  }
</style>
