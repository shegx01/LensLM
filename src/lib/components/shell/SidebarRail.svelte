<script lang="ts">
  import NotebooksSidebar from '$lib/components/notebooks/NotebooksSidebar.svelte';
  import { notebookStore } from '$lib/notebooks/index.js';

  /**
   * SidebarRail is the floating left-rail panel. Collapse/expand is controlled
   * ONLY by the collapse button (no hover behaviour). The parent grid column
   * width follows `notebookStore.sidebarCollapsed` (88px collapsed / 256px
   * expanded); this panel fills whatever cell width it is given.
   */
  let {
    onnewnotebook,
    userName = ''
  }: {
    onnewnotebook?: () => void;
    userName?: string;
  } = $props();

  const collapsed = $derived(notebookStore.sidebarCollapsed);
</script>

<aside
  data-sidebar-rail
  class="m-2 flex flex-col overflow-hidden rounded-xl border border-sidebar-border bg-sidebar text-sidebar-foreground shadow-sm"
>
  <NotebooksSidebar {collapsed} {onnewnotebook} {userName} />
</aside>
