<script lang="ts">
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import Settings from '@lucide/svelte/icons/settings';
  import Microscope from '@lucide/svelte/icons/microscope';
  import { cn } from '$lib/utils.js';
  import { getInitials, notebookStore } from '$lib/notebooks/index.js';
  import ThemeCycleButton from '$lib/components/ThemeCycleButton.svelte';
  import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider
  } from '$lib/components/ui/tooltip/index.js';

  let { userName = '' }: { userName?: string } = $props();

  const initials = $derived(getInitials(userName));

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

  // `focusout` alone misses clicks on non-focusable regions; capture-phase pointerdown covers them.
  $effect(() => {
    if (!open || typeof document === 'undefined') return;
    function onPointerDown(e: PointerEvent): void {
      if (containerEl && !containerEl.contains(e.target as Node)) {
        open = false;
      }
    }
    document.addEventListener('pointerdown', onPointerDown, true);
    return () => document.removeEventListener('pointerdown', onPointerDown, true);
  });
</script>

<div
  bind:this={containerEl}
  class="relative"
  onfocusout={handleFocusout}
  onkeydown={handleKeydown}
  role="none"
>
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
    <div
      class={cn(
        'flex size-[27px] shrink-0 items-center justify-center rounded-full',
        'bg-muted text-sidebar-foreground/80 text-[9px] font-bold',
        'ring-1 ring-inset ring-border',
        'shadow-[0_1px_3px_rgba(0,0,0,0.16)]'
      )}
      aria-hidden="true"
    >
      {initials}
    </div>
    <span class="flex-1 truncate text-sm font-medium">{userName || 'Account'}</span>
    <ChevronUp
      class={cn('size-3.5 text-sidebar-foreground/50 transition-transform', open && 'rotate-180')}
    />
  </button>

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
      <button
        type="button"
        class={cn(
          'flex w-full cursor-pointer items-center gap-2.5 px-3 py-2',
          'text-sm text-sidebar-foreground select-none',
          'hover:bg-sidebar-accent/60 transition-colors'
        )}
        role="menuitem"
        data-settings-item
        onclick={() => {
          notebookStore.settingsOpen = true;
          open = false;
        }}
      >
        <Settings class="size-4 shrink-0" aria-hidden="true" />
        <span>Settings</span>
      </button>

      <div
        role="menuitem"
        data-switch-theme-item
        class="flex items-center gap-2.5 px-3 py-2 hover:bg-sidebar-accent/60 transition-colors cursor-pointer"
        tabindex="0"
        onclick={(e) => {
          if ((e.target as HTMLElement).closest('button')) return;
          (e.currentTarget as HTMLElement).querySelector('button')?.click();
        }}
        onkeydown={(e) => {
          if ((e.target as HTMLElement).closest('button')) return;
          if (e.key === 'Enter' || e.key === ' ') {
            const btn = (e.currentTarget as HTMLElement).querySelector('button');
            btn?.click();
          }
        }}
      >
        <ThemeCycleButton
          class="size-6 rounded-md border-0 bg-transparent shadow-none hover:bg-transparent"
        />
        <span class="text-sm text-sidebar-foreground">Switch theme</span>
      </div>

      {#if import.meta.env.DEV}
        {@const noActive = notebookStore.activeNotebookId === null}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger
              class={cn(
                'flex w-full items-center gap-2.5 px-3 py-2 text-sm select-none',
                noActive
                  ? 'cursor-not-allowed text-sidebar-foreground/40'
                  : 'cursor-pointer text-sidebar-foreground hover:bg-sidebar-accent/60 transition-colors'
              )}
              role="menuitem"
              data-embeddings-inspector-item
              disabled={noActive}
              aria-disabled={noActive}
              onclick={() => {
                if (noActive) return;
                notebookStore.inspectorOpen = true;
                open = false;
              }}
            >
              <Microscope class="size-4 shrink-0" aria-hidden="true" />
              <span>Embeddings Inspector</span>
            </TooltipTrigger>
            {#if noActive}
              <TooltipContent side="right">No active notebook</TooltipContent>
            {/if}
          </Tooltip>
        </TooltipProvider>
      {/if}
    </div>
  {/if}
</div>
