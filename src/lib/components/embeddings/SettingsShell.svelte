<!--
  SettingsShell — presentational sidebar+content chrome for a sectioned settings surface.
  Container-agnostic: renders in-place (global Preferences) or inside a Dialog (notebook sheet).
  `onBack` and `label` are optional so the shell can drop the Back button / section heading
  where a container (e.g. a modal with its own header + close) already provides them.
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
</script>

<div class="flex h-full min-h-0 flex-1 overflow-hidden bg-background">
  <nav
    class="flex w-[220px] shrink-0 flex-col gap-px overflow-y-auto border-r border-border px-2.5 py-3.5"
    aria-label={label ? `${label} sections` : 'Settings sections'}
  >
    {#if onBack}
      <button
        type="button"
        onclick={onBack}
        class={cn(
          'mb-2.5 flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-left text-[0.78rem] font-semibold text-muted-foreground transition-colors',
          'hover:bg-muted/50 hover:text-foreground',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
        )}
      >
        <ArrowLeft class="size-3.5 shrink-0" aria-hidden="true" />
        <span>Back</span>
      </button>
    {/if}

    {#if label}
      <p
        class="px-2.5 pb-2.5 text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70"
      >
        {label}
      </p>
    {/if}

    {#each nav as item (item.id)}
      {@const isActive = active === item.id}
      <button
        type="button"
        aria-current={isActive ? 'page' : undefined}
        aria-disabled={item.stub}
        onclick={() => {
          if (!item.stub) select(item.id);
        }}
        class={cn(
          'flex h-[34px] items-center gap-2.5 rounded-lg px-2.5 text-left text-[0.78rem] font-semibold transition-colors',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
          item.id === 'about' && 'mt-auto',
          isActive
            ? 'bg-primary/10 text-primary'
            : item.stub
              ? 'cursor-default text-muted-foreground/40'
              : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
        )}
      >
        <item.icon class="size-3.5 shrink-0" aria-hidden="true" />
        <span class="flex-1 truncate">{item.label}</span>
        {#if item.stub}
          <span class="text-[0.6rem] font-medium text-muted-foreground/40">Soon</span>
        {/if}
      </button>
    {/each}
  </nav>

  <div class="flex-1 overflow-y-auto px-10 py-8">
    {@render content(active)}
  </div>
</div>
