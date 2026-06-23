<script lang="ts">
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import Settings from '@lucide/svelte/icons/settings';
  import { cn } from '$lib/utils.js';
  import ThemeSwitcher from '$lib/components/ThemeSwitcher.svelte';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';

  /**
   * The display name of the logged-in user, read from AppConfig `user_name`.
   * Defaults to an empty string. The parent (NotebooksSidebar / AppShell) should
   * pass the value from `config.user_name` once it is available.
   *
   * Integration note: AppShell (Wave 3) must pass `userName={config.user_name}`.
   */
  let { userName = '' }: { userName?: string } = $props();

  // ---------------------------------------------------------------------------
  // Derived initials
  // ---------------------------------------------------------------------------

  const initials = $derived(
    userName
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((word) => word[0].toUpperCase())
      .join('') || '?'
  );

  // ---------------------------------------------------------------------------
  // Popover open state
  // ---------------------------------------------------------------------------

  let open = $state(false);
  let containerEl = $state<HTMLDivElement | null>(null);

  function toggle(): void {
    open = !open;
  }

  function handleFocusout(e: FocusEvent): void {
    const related = e.relatedTarget as Node | null;
    if (!containerEl?.contains(related)) {
      open = false;
    }
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      open = false;
    }
  }
</script>

<!--
  AccountFooter: initials avatar + user name + chevron. Clicking opens a small
  popover with Settings (disabled) and Switch theme (real). "Sign out" is
  intentionally OMITTED per spec (accounts are an MVP non-goal).
-->
<div
  bind:this={containerEl}
  class="relative"
  onfocusout={handleFocusout}
  onkeydown={handleKeydown}
  role="none"
>
  <!-- Trigger row -->
  <button
    type="button"
    aria-haspopup="menu"
    aria-expanded={open}
    aria-label={`Account menu for ${userName || 'user'}`}
    onclick={toggle}
    class={cn(
      'flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left',
      'text-sidebar-foreground transition-colors',
      'hover:bg-sidebar-accent/60 focus-visible:ring-2 focus-visible:ring-sidebar-ring outline-none',
      open && 'bg-sidebar-accent/60'
    )}
  >
    <!-- Initials avatar -->
    <div
      class="flex size-7 shrink-0 items-center justify-center rounded-full bg-sidebar-primary text-sidebar-primary-foreground text-xs font-semibold"
      aria-hidden="true"
    >
      {initials}
    </div>
    <span class="flex-1 truncate text-sm font-medium">{userName || 'Account'}</span>
    <ChevronUp
      class={cn('size-3.5 text-sidebar-foreground/50 transition-transform', open && 'rotate-180')}
    />
  </button>

  <!-- Popover menu — anchored above the footer, dismisses on blur/Esc -->
  {#if open}
    <div
      role="menu"
      aria-label="Account menu"
      data-account-menu
      class={cn(
        'absolute bottom-full left-0 right-0 mb-1.5 z-50',
        'rounded-xl border border-sidebar-border bg-sidebar shadow-lg',
        'overflow-hidden py-1'
      )}
    >
      <!-- Settings — disabled, tooltip "Available soon" -->
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            class={cn(
              'flex w-full cursor-not-allowed items-center gap-2.5 px-3 py-2',
              'text-sm text-sidebar-foreground/40 select-none'
            )}
            disabled
            aria-disabled="true"
            role="menuitem"
            data-settings-item
          >
            <Settings class="size-4 shrink-0" aria-hidden="true" />
            <span>Settings</span>
          </TooltipTrigger>
          <TooltipContent side="right">Available soon</TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <!-- Switch theme — real, embeds ThemeSwitcher as a menu row -->
      <div
        role="menuitem"
        data-switch-theme-item
        class="flex items-center gap-2.5 px-3 py-2 hover:bg-sidebar-accent/60 transition-colors cursor-pointer"
        tabindex="0"
        onkeydown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            const btn = (e.currentTarget as HTMLElement).querySelector('button');
            btn?.click();
          }
        }}
      >
        <!-- Embed ThemeSwitcher as a ghost button; suppress its border via class override -->
        <ThemeSwitcher
          class="size-6 rounded-md border-0 bg-transparent shadow-none hover:bg-transparent"
        />
        <span class="text-sm text-sidebar-foreground">Switch theme</span>
      </div>
    </div>
  {/if}
</div>
