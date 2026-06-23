<script lang="ts">
  import NotebooksSidebar from '$lib/components/notebooks/NotebooksSidebar.svelte';
  import { notebookStore } from '$lib/notebooks/index.js';

  /**
   * SidebarRail owns the left-rail collapse / hover-flyout behaviour so AppShell
   * keeps a stable grid layout.
   *
   * Layout contract (the parent grid column width is driven ONLY by
   * `notebookStore.sidebarCollapsed`: 88px collapsed, 256px expanded — hover never
   * changes it):
   *
   *   - EXPANDED (`sidebarCollapsed === false`): the panel sits in normal flow and
   *     fills the 256px grid cell. No overlay, no hover behaviour.
   *
   *   - COLLAPSED (`sidebarCollapsed === true`): the grid cell is a reserved 88px
   *     `relative` box. The panel is `absolute inset-y-0 left-0 z-50`, floating OVER
   *     the centre content. Its width animates 88px → 256px on hover, so the centre
   *     never reflows. `NotebooksSidebar` receives `collapsed={!hovered}`, swapping
   *     to the full expanded layout while the flyout is open.
   *
   * Only ONE `NotebooksSidebar` instance is ever rendered.
   */
  let {
    onnewnotebook,
    userName = ''
  }: {
    onnewnotebook?: () => void;
    userName?: string;
  } = $props();

  /**
   * Ephemeral hover flag — true while the pointer is over the collapsed flyout
   * panel (or any descendant). NEVER written to the store; only the collapse
   * button toggles persisted `sidebarCollapsed`.
   */
  let hovered = $state(false);

  const persistedCollapsed = $derived(notebookStore.sidebarCollapsed);
  // While persisted-collapsed: the panel shows the icon rail unless hovered.
  // While expanded: always the full layout. Hover is irrelevant when expanded.
  const sidebarCollapsed = $derived(persistedCollapsed && !hovered);
</script>

{#if persistedCollapsed}
  <!-- Reserved 88px cell. The floating panel overlays it; centre content never
       moves because this box keeps its width regardless of hover. -->
  <div data-sidebar-rail class="relative h-full w-full">
    <aside
      data-sidebar-flyout
      class={[
        'absolute inset-y-0 left-0 z-50 m-2 flex flex-col overflow-hidden rounded-xl',
        'border border-sidebar-border bg-sidebar text-sidebar-foreground shadow-sm',
        'transition-[width] duration-200 ease-out',
        // 88px cell − 2× m-2 (8px) gutter = 72px panel; expands to 256 − 16 = 240px.
        hovered ? 'w-[240px]' : 'w-[72px]'
      ].join(' ')}
      onpointerenter={() => (hovered = true)}
      onpointerleave={() => (hovered = false)}
    >
      <NotebooksSidebar collapsed={sidebarCollapsed} {onnewnotebook} {userName} />
    </aside>
  </div>
{:else}
  <!-- Expanded: normal flow, fills the 256px grid cell. No overlay. -->
  <aside
    data-sidebar-rail
    class="m-2 flex flex-col overflow-hidden rounded-xl border border-sidebar-border bg-sidebar text-sidebar-foreground shadow-sm"
  >
    <NotebooksSidebar collapsed={false} {onnewnotebook} {userName} />
  </aside>
{/if}
