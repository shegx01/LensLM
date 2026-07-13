<!-- Notes export control (#25 C4): copy-all-to-clipboard + save-to-file (.md/.txt).
     Hidden outside Tauri (no filesystem/clipboard plugin host) and when there are no notes. -->
<script lang="ts">
  import { DropdownMenu } from 'bits-ui';
  import { isTauri } from '@tauri-apps/api/core';
  import Download from '@lucide/svelte/icons/download';
  import Copy from '@lucide/svelte/icons/copy';
  import FileText from '@lucide/svelte/icons/file-text';
  import { Button } from '$lib/components/ui/button/index.js';
  import { copyAllNotes, exportNotesToFile } from '$lib/notes/export.js';

  interface Props {
    notebookId: string;
    hasNotes: boolean;
  }

  let { notebookId, hasNotes }: Props = $props();

  const available = $derived(hasNotes && isTauri());
</script>

{#if available}
  <DropdownMenu.Root>
    <DropdownMenu.Trigger>
      {#snippet child({ props })}
        <Button {...props} variant="ghost" size="icon-sm" aria-label="Export notes">
          <Download class="size-4" strokeWidth={1.75} />
        </Button>
      {/snippet}
    </DropdownMenu.Trigger>
    <DropdownMenu.Portal>
      <DropdownMenu.Content
        align="end"
        sideOffset={6}
        class="z-50 min-w-44 rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-md"
      >
        <DropdownMenu.Item
          class="flex cursor-pointer select-none items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none data-highlighted:bg-accent data-highlighted:text-accent-foreground"
          onSelect={() => void copyAllNotes(notebookId)}
        >
          <Copy class="size-3.5" strokeWidth={1.75} />
          Copy all notes
        </DropdownMenu.Item>
        <DropdownMenu.Item
          class="flex cursor-pointer select-none items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none data-highlighted:bg-accent data-highlighted:text-accent-foreground"
          onSelect={() => void exportNotesToFile(notebookId, 'md')}
        >
          <FileText class="size-3.5" strokeWidth={1.75} />
          Export as .md
        </DropdownMenu.Item>
        <DropdownMenu.Item
          class="flex cursor-pointer select-none items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none data-highlighted:bg-accent data-highlighted:text-accent-foreground"
          onSelect={() => void exportNotesToFile(notebookId, 'txt')}
        >
          <FileText class="size-3.5" strokeWidth={1.75} />
          Export as .txt
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Portal>
  </DropdownMenu.Root>
{/if}
