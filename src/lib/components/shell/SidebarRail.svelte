<script lang="ts">
  import NotebooksSidebar from '$lib/components/notebooks/NotebooksSidebar.svelte';
  import { notebookStore } from '$lib/notebooks/index.js';

  /**
   * Floating left-rail panel. Collapse/expand is button-only (no hover behaviour).
   * Fills whatever cell width the parent grid assigns.
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
  class="m-2 flex flex-col overflow-hidden rounded-xl bg-card text-sidebar-foreground shadow-[var(--shadow-rail)]"
>
  <NotebooksSidebar {collapsed} {onnewnotebook} {userName} />
</aside>
